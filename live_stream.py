# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public Aria Gen 2 USB streaming and a latest-RGB-only HLOC worker.

T_A_B maps coordinates in B to A. Alignment uses the corrected RGB capture
clock, never inference completion time. No pose is written back to the device.
SDK callbacks only snapshot RGB/buffer VIO; rectification and HLOC run in one
non-daemon worker. Callback sinks must be thread-safe, nonblocking enqueue-only
functions: the caller owns GUI dispatch and visualization throttling.
"""

from __future__ import annotations

import importlib
import ipaddress
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Literal, Protocol

import numpy as np
from aria_utils import Frame
from hloc_backend import HlocBackend
from map_utils import Camera, FloatArray, rigid_transform
from numpy.typing import NDArray
from pose_alignment import MappedPose, PoseAlignment, vio_status_valid, VioSample
from query_map import load_map, localize_frames

HIGH_FREQUENCY_VIO_QUEUE_SIZE: int = 4096
REGULAR_VIO_QUEUE_SIZE: int = 256


def _clear_exception_frames(error: BaseException) -> None:
    """Release unwound frame locals while preserving exception tracebacks and links."""
    pending = [error]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        traceback.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


class MatrixTransform(Protocol):
    def to_matrix(self) -> FloatArray: ...


class RegularVio(Protocol):
    capture_timestamp_ns: int
    frontend_session_uid: str
    pose_quality: object
    transform_odometry_bodyimu: MatrixTransform
    transform_bodyimu_device: MatrixTransform


class TrackingTimestamp(Protocol):
    def total_seconds(self) -> float: ...


class HighFrequencyVio(Protocol):
    tracking_timestamp: TrackingTimestamp
    session_uid: str
    quality_score: float
    transform_odometry_device: MatrixTransform


class RgbImage(Protocol):
    def to_numpy_array(self) -> NDArray[np.uint8]: ...


class RgbRecord(Protocol):
    capture_timestamp_ns: int


class CameraCalibration(Protocol):
    def get_image_size(self) -> tuple[int, int]: ...
    def get_transform_device_camera(self) -> MatrixTransform: ...
    def get_time_offset_sec_device_camera(self) -> float: ...


class DeviceCalibration(Protocol):
    def get_camera_calib(self, label: str) -> CameraCalibration | None: ...
    def get_device_version(self) -> str: ...


@dataclass(frozen=True)
class LiveStatus:
    state: str
    detail: str = ""
    timestamp_ns: int | None = None
    session_uid: str | None = None
    query_duration_sec: float | None = None
    num_inliers: int | None = None
    dropped_frames: int = 0
    initialization_durations_sec: dict[str, float] | None = None
    query_end_to_end_duration_sec: float | None = None


@dataclass(frozen=True)
class _RgbSnapshot:
    image: NDArray[np.uint8]
    capture_timestamp_ns: int
    index: int
    calibration: DeviceCalibration
    session_uid: str
    generation: int


class LiveLocalizer:
    """One odometry session, a bounded VIO history, and at most one pending RGB.

    on_pose receives EVERY accepted unique good VIO timestamp once aligned,
    including poses tagged stale. Feed ONE authoritative ordered VIO stream,
    through on_vio OR on_vio_high_frequency (or submit_vio), never both SDK
    queues: their independent deliveries can cross the shared watermark.
    run_live selects exactly one stream; no fusion or automatic switching occurs.
    on_image receives the rectified Frame selected
    by the worker (not every SDK RGB frame). on_status distinguishes missing
    calibration/VIO, localization failures, lost quality and fatal errors.
    A stopped localizer cannot be restarted. Call raise_if_failed from the main
    loop; exceptions at SDK/worker boundaries are retained with their cause.
    High-frequency quality >= 0.5 is the conservative default acceptance gate;
    nonfinite quality is always bad. Regular VIO requires pose_quality == GOOD.
    check_freshness marks missing host arrivals lost after vio_timeout_sec
    (default 3.0), without changing device-clock alignment or generating poses.
    Map/model preparation runs once on the worker and survives VIO resets.
    Status initialization_durations_sec reports backend preparation stages;
    query_end_to_end_duration_sec covers worker rectification through query
    completion, excluding pending-queue age and waiting for a VIO bracket.
    """

    def __init__(
        self,
        map_path: Path,
        on_pose: Callable[[MappedPose], None] | None = None,
        on_image: Callable[[Frame], None] | None = None,
        on_status: Callable[[LiveStatus], None] | None = None,
        *,
        fix_wait_duration_sec: float = 1.0,
        worker_join_duration_sec: float = 15.0,
        min_high_frequency_quality: float = 0.5,
        vio_timeout_sec: float = 3.0,
    ) -> None:
        if not np.isfinite(
            [fix_wait_duration_sec, worker_join_duration_sec, vio_timeout_sec]
        ).all():
            raise ValueError("Worker durations must be finite")
        if fix_wait_duration_sec < 0 or worker_join_duration_sec <= 0:
            raise ValueError(
                "Use nonnegative fix wait and positive worker join durations"
            )
        if vio_timeout_sec <= 0:
            raise ValueError("VIO arrival timeout must be positive")
        if not 0 <= min_high_frequency_quality <= 1:
            raise ValueError("High-frequency quality threshold must be in [0, 1]")
        self.map_path = Path(map_path)
        self.on_pose = on_pose
        self.on_image = on_image
        self.on_status = on_status
        self._fix_wait_duration_sec = fix_wait_duration_sec
        self._worker_join_duration_sec = worker_join_duration_sec
        self._min_high_frequency_quality = min_high_frequency_quality
        self._vio_timeout_sec = vio_timeout_sec
        self._condition = threading.Condition()
        self._alignment = PoseAlignment()
        self._calibration: DeviceCalibration | None = None
        self._session_uid: str | None = None
        self._retired_sessions: set[str] = set()
        self._generation = 0
        self._last_vio_timestamp_ns: int | None = None
        self._last_vio_arrival_time_sec: float | None = None
        self._vio_arrival_lost = False
        self._pending: _RgbSnapshot | None = None
        self._frame_index = 0
        self._dropped_frames = 0
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._initialization_durations_sec: dict[str, float] | None = None

    def start(self) -> LiveLocalizer:
        with self._condition:
            if self._stopped:
                raise RuntimeError("A stopped localizer cannot restart")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._work, name="aria-hloc-rgb", daemon=False
                )
                self._thread.start()
        return self

    def request_stop(self) -> None:
        """Signal the worker to exit without joining; safe before blocking cleanup."""
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify_all()

    def stop(self) -> None:
        self.request_stop()
        with self._condition:
            thread = self._thread
        if thread is not None:
            if thread is threading.current_thread():
                raise RuntimeError("stop must be called outside worker callbacks")
            thread.join(self._worker_join_duration_sec)
            if thread.is_alive():
                raise TimeoutError("HLOC worker did not stop within the join duration")
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            _clear_exception_frames(error)
            raise RuntimeError("Live localization failed") from error

    def check_freshness(self) -> None:
        """Host arrival watchdog, separate from the core's device-clock fix age.

        Call from the main loop. Initial waiting emits nothing; stopped VIO
        emits lost once per outage. No pose, reset, or SDK poll is synthesized.
        Replayed/retired/duplicate packets cannot clear or postpone an outage.
        """
        try:
            with self._condition:
                if (
                    self._stopped
                    or self._calibration is None
                    or self._last_vio_arrival_time_sec is None
                    or self._vio_arrival_lost
                ):
                    return
                idle_duration_sec = time.monotonic() - self._last_vio_arrival_time_sec
                if idle_duration_sec < self._vio_timeout_sec:
                    return
                self._vio_arrival_lost = True
                self._status(
                    "lost",
                    f"No new VIO for {idle_duration_sec:.3f} sec",
                    timestamp_ns=self._last_vio_timestamp_ns,
                )
        except Exception as error:
            self._fail(error)

    def _status(
        self,
        state: str,
        detail: str = "",
        *,
        timestamp_ns: int | None = None,
        query_duration_sec: float | None = None,
        query_end_to_end_duration_sec: float | None = None,
        num_inliers: int | None = None,
    ) -> None:
        if self.on_status is not None:
            with self._condition:
                if self._vio_arrival_lost and state in (
                    "calibrated",
                    "localizing",
                    "tracking",
                    "stale",
                ):
                    state = "lost"
                    detail = "VIO arrival timed out; waiting for a new good VIO sample"
                status = LiveStatus(
                    state=state,
                    detail=detail,
                    timestamp_ns=timestamp_ns,
                    query_duration_sec=query_duration_sec,
                    num_inliers=num_inliers,
                    session_uid=self._session_uid,
                    dropped_frames=self._dropped_frames,
                    initialization_durations_sec=(
                        dict(self._initialization_durations_sec)
                        if self._initialization_durations_sec is not None
                        else None
                    ),
                    query_end_to_end_duration_sec=query_end_to_end_duration_sec,
                )
            self.on_status(status)

    def _fail(self, error: Exception) -> None:
        # SDK/native callbacks need an explicit main-thread error channel.
        _clear_exception_frames(error)
        with self._condition:
            if self._error is not None:
                _clear_exception_frames(self._error)
                if error is not self._error:
                    self._error.add_note(
                        f"Additional live localization failure: {error}"
                    )
                return
            self._error = error
            self._stopped = True
            self._pending = None
            self._condition.notify_all()
        try:
            self._status("error", str(error))
        except Exception as status_error:
            error.add_note(f"Error status callback also failed: {status_error}")
            _clear_exception_frames(status_error)

    def on_calibration(self, calibration: DeviceCalibration) -> None:
        try:
            if calibration.get_camera_calib("camera-rgb") is None:
                raise ValueError("Streamed device calibration is missing camera-rgb")
            with self._condition:
                if self._stopped:
                    return
                self._calibration = calibration
            self._status("calibrated")
        except Exception as error:
            self._fail(error)

    def on_rgb(self, image_data: RgbImage, record: RgbRecord) -> None:
        try:
            image = np.asarray(image_data.to_numpy_array())
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected a decoded uint8 RGB image")
            if min(image.shape[:2]) <= 0:
                raise ValueError("RGB image is empty")
            timestamp_ns = int(record.capture_timestamp_ns)
            if timestamp_ns < 0:
                raise ValueError("RGB capture timestamp must be nonnegative")
            with self._condition:
                if self._stopped:
                    return
                if self._calibration is None or self._session_uid is None:
                    state = (
                        "waiting_calibration"
                        if self._calibration is None
                        else "waiting_vio"
                    )
                else:
                    if self._pending is not None:
                        self._dropped_frames += 1
                    self._pending = _RgbSnapshot(
                        image.copy(),
                        timestamp_ns,
                        self._frame_index,
                        self._calibration,
                        self._session_uid,
                        self._generation,
                    )
                    self._frame_index += 1
                    self._condition.notify_all()
                    return
            self._status(state, timestamp_ns=timestamp_ns)
        except Exception as error:
            self._fail(error)

    def on_vio(self, vio: RegularVio) -> None:
        try:
            quality_good = str(vio.pose_quality).split(".")[-1] == "GOOD" and (
                vio_status_valid(vio)
            )
            pose = (
                vio.transform_odometry_bodyimu.to_matrix()
                @ vio.transform_bodyimu_device.to_matrix()
                if quality_good
                else np.eye(4)
            )
            self.submit_vio(
                VioSample(
                    int(vio.capture_timestamp_ns),
                    pose,
                    str(vio.frontend_session_uid),
                    quality_good,
                )
            )
        except Exception as error:
            self._fail(error)

    def on_vio_high_frequency(self, vio: HighFrequencyVio) -> None:
        """The public SDK delivers individual OpenLoopTrajectoryPose, not a batch."""
        try:
            quality = float(vio.quality_score)
            quality_good = (
                np.isfinite(quality) and quality >= self._min_high_frequency_quality
            )
            pose = (
                vio.transform_odometry_device.to_matrix() if quality_good else np.eye(4)
            )
            self.submit_vio(
                VioSample(
                    round(vio.tracking_timestamp.total_seconds() * 1e9),
                    pose,
                    str(vio.session_uid),
                    bool(quality_good),
                )
            )
        except Exception as error:
            self._fail(error)

    def submit_vio(self, sample: VioSample) -> None:
        """Serialize one ordered VIO stream; duplicate/out-of-order clocks emit nothing."""
        with self._condition:
            if self._stopped or sample.session_uid in self._retired_sessions:
                return
            changed_session = sample.session_uid != self._session_uid
            # Let the core validate before altering adapter bookkeeping.
            pose = self._alignment.add_vio(sample)
            if changed_session:
                if self._session_uid is not None:
                    self._retired_sessions.add(self._session_uid)
                self._session_uid = sample.session_uid
                self._generation += 1
                self._pending = None
                self._last_vio_timestamp_ns = None
            if (
                self._last_vio_timestamp_ns is not None
                and sample.timestamp_ns <= self._last_vio_timestamp_ns
            ):
                return
            self._last_vio_timestamp_ns = sample.timestamp_ns
            self._last_vio_arrival_time_sec = time.monotonic()
            if changed_session or sample.quality_good:
                self._vio_arrival_lost = False
            state = self._alignment.state
            self._condition.notify_all()
            if changed_session:
                self._status("session_reset", timestamp_ns=sample.timestamp_ns)
            if pose is not None and self.on_pose is not None:
                self.on_pose(pose)
            self._status(state, timestamp_ns=sample.timestamp_ns)

    def _rectify(self, snapshot: _RgbSnapshot) -> Frame:
        calibration = importlib.import_module("projectaria_tools.core.calibration")
        source = snapshot.calibration.get_camera_calib("camera-rgb")
        if source is None:
            raise ValueError("Streamed device calibration is missing camera-rgb")
        offset_ns = round(float(source.get_time_offset_sec_device_camera()) * 1e9)
        height, width = snapshot.image.shape[:2]
        if tuple(source.get_image_size()) != (width, height):
            source = calibration.rescale_camera_calibration(
                source,
                (width, height),
                snapshot.calibration.get_device_version(),
                "camera-rgb",
            )
        target = calibration.get_linear_camera_calibration(
            1024, 768, 360.0, "camera-rgb"
        )
        image = calibration.distort_by_calibration(snapshot.image, target, source)
        principal = target.get_principal_point()
        camera = Camera(
            "camera-rgb",
            1024,
            768,
            360.0,
            float(principal[0]),
            float(principal[1]),
            source.get_transform_device_camera().to_matrix(),
        )
        frame = Frame(
            "camera-rgb",
            snapshot.index,
            snapshot.capture_timestamp_ns,
            np.asarray(image),
            camera,
            offset_ns,
        )
        if frame.corrected_timestamp_ns < 0:
            raise ValueError("Corrected RGB timestamp must be nonnegative")
        return frame

    def _next_rgb(self) -> _RgbSnapshot | None:
        with self._condition:
            self._condition.wait_for(lambda: self._stopped or self._pending is not None)
            if self._stopped:
                return None
            snapshot = self._pending
            self._pending = None
            return snapshot

    def _accept_fix(
        self, snapshot: _RgbSnapshot, frame: Frame, result: dict[str, object]
    ) -> bool:
        deadline_time_sec = time.monotonic() + self._fix_wait_duration_sec
        with self._condition:
            if self._stopped or snapshot.generation != self._generation:
                return False
            map_pose = rigid_transform(
                cast(list[list[float]], result["T_world_device"])
            )
            while not self._stopped and snapshot.generation == self._generation:
                if self._alignment.add_fix(
                    frame.corrected_timestamp_ns, map_pose, snapshot.session_uid
                ):
                    return True
                remaining_duration_sec = deadline_time_sec - time.monotonic()
                if remaining_duration_sec <= 0:
                    break
                self._condition.wait(remaining_duration_sec)
        return False

    def _work(self) -> None:
        try:
            with ExitStack() as cleanup:
                backend = HlocBackend()
                cleanup.callback(backend.close)
                load_map(self.map_path)
                initialization_durations_sec = backend.prepare_map(self.map_path)
                with self._condition:
                    self._initialization_durations_sec = dict(
                        initialization_durations_sec
                    )
                self._localize_loop(backend)
        except Exception as error:
            self._fail(error)

    def _localize_loop(self, backend: HlocBackend) -> None:
        while (snapshot := self._next_rgb()) is not None:
            start_time_sec = time.monotonic()
            frame = self._rectify(snapshot)
            with self._condition:
                if self._stopped or snapshot.generation != self._generation:
                    continue
                if self.on_image is not None:
                    self.on_image(frame)
                self._status("localizing", timestamp_ns=frame.corrected_timestamp_ns)
            result = localize_frames(backend, self.map_path, [frame])
            query_end_to_end_duration_sec = time.monotonic() - start_time_sec
            with self._condition:
                if self._stopped or snapshot.generation != self._generation:
                    continue
            accepted = bool(result["success"]) and self._accept_fix(
                snapshot, frame, result
            )
            with self._condition:
                if self._stopped or snapshot.generation != self._generation:
                    continue
                state = (
                    self._alignment.state
                    if accepted
                    else "fix_rejected"
                    if result["success"]
                    else "localization_failed"
                )
                self._status(
                    state,
                    ""
                    if accepted
                    else str(result.get("reason", "No VIO bracket for RGB capture")),
                    timestamp_ns=frame.corrected_timestamp_ns,
                    query_duration_sec=float(cast(float, result["query_duration_sec"])),
                    query_end_to_end_duration_sec=query_end_to_end_duration_sec,
                    num_inliers=int(
                        cast(
                            int,
                            result.get(
                                "num_inlier_landmarks", result.get("num_inliers", 0)
                            ),
                        )
                    ),
                )


def _explicit_ip(value: str, name: str) -> str:
    address = ipaddress.ip_address(value)
    if (
        address.is_unspecified
        or address.is_multicast
        or address.is_loopback
        or address.is_global
        or address.is_reserved
    ):
        raise ValueError(
            f"{name} must be an explicit local unicast USB-network address"
        )
    return str(address)


def run_live(
    device_ip: str,
    bind_address: str,
    profile: str,
    map_path: Path,
    on_pose: Callable[[MappedPose], None] | None = None,
    on_image: Callable[[Frame], None] | None = None,
    on_status: Callable[[LiveStatus], None] | None = None,
    stop_event: threading.Event | None = None,
    *,
    vio_stream: Literal["high-frequency", "regular"] = "high-frequency",
) -> None:
    """Explicit opt-in hardware entry point; use an existing profile (e.g. profile9).

    The host bind address is required and never defaults to all interfaces.
    start_streaming rotates certificates, so the receiver is created afterward.
    Every acquired resource gets an independent cleanup attempt, even if startup
    or another cleanup fails. Cleanup failures and callback failures are raised.
    The main loop checks VIO arrival freshness without polling the hardware.
    Exactly one VIO callback is registered: high-frequency by default, or the
    explicit regular fallback. Each uses its own bounded SDK FIFO, never a
    latest-only VIO queue. No automatic fallback or mixed-stream watermark exists.
    """
    if vio_stream not in ("high-frequency", "regular"):
        raise ValueError("vio_stream must be high-frequency or regular")
    device_ip = _explicit_ip(device_ip, "device_ip")
    bind_address = _explicit_ip(bind_address, "bind_address")
    if not profile.strip() or profile.endswith(".json"):
        raise ValueError("Specify an existing profile name, not a profile JSON")
    event = stop_event if stop_event is not None else threading.Event()
    if event.is_set():
        return
    sdk = importlib.import_module("aria.sdk_gen2")
    receiver_module = importlib.import_module("aria.stream_receiver")
    localizer = LiveLocalizer(map_path, on_pose, on_image, on_status)
    device = stream_receiver = None
    errors: list[Exception] = []
    try:
        localizer.start()
        localizer.raise_if_failed()
        client = sdk.DeviceClient()
        client.set_client_config(sdk.DeviceClientConfig())
        device = client.connect(sdk.DeviceTarget(ip=device_ip))
        config = sdk.HttpStreamingConfig()
        config.profile_name = profile
        config.streaming_interface = sdk.StreamingInterface.USB_NCM
        device.set_streaming_config(config)
        device.start_streaming()
        server_config = sdk.HttpServerConfig()
        server_config.address = bind_address
        server_config.port = 6768
        stream_receiver = receiver_module.StreamReceiver(enable_image_decoding=True)
        stream_receiver.set_server_config(server_config)
        stream_receiver.set_rgb_queue_size(1)
        stream_receiver.register_device_calib_callback(localizer.on_calibration)
        stream_receiver.register_rgb_callback(localizer.on_rgb)
        if vio_stream == "high-frequency":
            stream_receiver.set_vio_high_freq_queue_size(HIGH_FREQUENCY_VIO_QUEUE_SIZE)
            stream_receiver.register_vio_high_frequency_callback(
                localizer.on_vio_high_frequency
            )
        else:
            stream_receiver.set_vio_queue_size(REGULAR_VIO_QUEUE_SIZE)
            stream_receiver.register_vio_callback(localizer.on_vio)
        stream_receiver.start_server()
        while not event.wait(0.05):
            localizer.check_freshness()
            localizer.raise_if_failed()
        localizer.raise_if_failed()
    finally:
        # Unblock the non-daemon worker first, so an interrupt during a slow
        # device call below cannot leave it waiting forever.
        localizer.request_stop()
        interrupt: BaseException | None = None
        for cleanup in (
            device.stop_streaming if device is not None else None,
            stream_receiver.stop_server if stream_receiver is not None else None,
            localizer.stop,
        ):
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as error:
                    errors.append(error)
                except BaseException as error:  # noqa: B036
                    # e.g. a second Ctrl+C: finish the remaining cleanup first.
                    interrupt = interrupt or error
        if interrupt is not None:
            raise interrupt
        if errors:
            raise ExceptionGroup("Live streaming cleanup failed", errors)
