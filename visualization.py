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

"""Interactive map-frame pose visualization; rendering rate does not limit pose output."""

from __future__ import annotations

import importlib
import json
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image
from pose_alignment import MappedPose

GUI_INTERVAL_NS: int = 33_333_333


class RerunView:
    """Log a sparse map, current pose, bounded path and fix-age diagnostics."""

    def __init__(
        self, map_path: Path, recording_path: Path, *, viewer: str = "none"
    ) -> None:
        self.rr = importlib.import_module("rerun")
        blueprint = importlib.import_module("rerun.blueprint")
        # Load torch's libomp before pycolmap's (see HlocBackend._load_adapters).
        importlib.import_module("torch")
        colmap = importlib.import_module("pycolmap")
        self.lock = threading.Lock()
        self.origin_timestamp_ns: int | None = None
        self.viewer_elapsed_duration_ns: int | None = None
        self.session_offset_duration_ns = 0
        self.last_pose_timestamp_ns: int | None = None
        self.session_uid: str | None = None
        self.last_status_signature: str | None = None
        self.last_status_state: object = None
        self.path = deque(maxlen=600)
        model = colmap.Reconstruction(map_path / "sfm")
        points = np.asarray([point.xyz for point in model.points3D.values()])
        if not len(points):
            raise ValueError("Visualization requires a nonempty HLOC map")
        if viewer not in ("native", "none"):
            raise ValueError("Viewer must be native or none")
        self.recordings = []
        layout = blueprint.Blueprint(
            blueprint.Horizontal(
                blueprint.Spatial3DView(
                    origin="/map", name="HLOC map + mapped VIO pose"
                ),
                blueprint.Vertical(
                    blueprint.Spatial2DView(
                        origin="/camera", name="Recorded / streamed RGB"
                    ),
                    blueprint.TimeSeriesView(
                        origin="/diagnostics", name="Visual-fix age and tracking"
                    ),
                ),
                column_shares=[3, 2],
            )
        )
        try:
            self.recording = self.rr.RecordingStream("aria_hloc_mapped_pose")
            self.recordings.append(self.recording)
            self.recording.save(recording_path, default_blueprint=layout)
            if viewer == "native":
                interactive = self.rr.RecordingStream("aria_hloc_mapped_pose")
                self.recordings.append(interactive)
                interactive.spawn(default_blueprint=layout)
            self._log("map", self.rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            self._log(
                "map/landmarks",
                self.rr.Points3D(points, colors=[110, 145, 180], radii=0.015),
                static=True,
            )
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(
                    f"Rerun initialization cleanup also failed: {cleanup_error}"
                )
            raise

    def _log(self, entity: str, *components: object, static: bool = False) -> None:
        for recording in self.recordings:
            recording.log(entity, *components, static=static)

    def _begin_session(self, session_uid: str | None) -> None:
        self.session_offset_duration_ns = (
            0
            if self.viewer_elapsed_duration_ns is None
            else self.viewer_elapsed_duration_ns + 1
        )
        self.origin_timestamp_ns = None
        self.session_uid = session_uid
        self.last_pose_timestamp_ns = None
        self.last_status_signature = None
        self.last_status_state = None
        self.path.clear()

    def _set_time(self, timestamp_ns: int) -> None:
        if self.origin_timestamp_ns is None:
            self.origin_timestamp_ns = timestamp_ns
        elapsed_duration_ns = self.session_offset_duration_ns + max(
            0, timestamp_ns - self.origin_timestamp_ns
        )
        self.viewer_elapsed_duration_ns = max(
            elapsed_duration_ns, self.viewer_elapsed_duration_ns or 0
        )
        for recording in self.recordings:
            recording.set_time(
                "viewer_elapsed",
                duration=np.timedelta64(self.viewer_elapsed_duration_ns, "ns"),
            )

    def on_pose(self, pose: MappedPose) -> None:
        with self.lock:
            if pose.session_uid != self.session_uid:
                self._begin_session(pose.session_uid)
            if (
                self.last_pose_timestamp_ns is not None
                and pose.timestamp_ns - self.last_pose_timestamp_ns < GUI_INTERVAL_NS
            ):
                return
            self.last_pose_timestamp_ns = pose.timestamp_ns
            if self.last_status_state in ("lost", "error", "session_reset"):
                self.last_status_signature = None
            self.last_status_state = pose.status
            self._set_time(pose.timestamp_ns)
            color = [80, 220, 140] if pose.status == "tracking" else [150, 150, 150]
            position = pose.T_map_device[:3, 3]
            self.path.append(position.copy())
            self._log(
                "map/device_axes",
                self.rr.Transform3D(
                    translation=position, mat3x3=pose.T_map_device[:3, :3]
                ),
                self.rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=np.eye(3) * 0.18,
                    colors=[[255, 80, 80], [80, 255, 80], [80, 80, 255]],
                ),
            )
            self._log(
                "map/pose", self.rr.Points3D([position], colors=color, radii=0.07)
            )
            self._log(
                "map/trajectory", self.rr.LineStrips3D([list(self.path)], colors=color)
            )
            self._log("diagnostics/fix_age_sec", self.rr.Scalars(pose.age_ns / 1e9))
            self._log(
                "diagnostics/tracking",
                self.rr.Scalars(1 if pose.status == "tracking" else 0),
            )

    def on_image(self, image: np.ndarray, timestamp_ns: int) -> None:
        with self.lock:
            self._set_time(timestamp_ns)
            height, width = image.shape[:2]
            resized = Image.fromarray(image).resize(
                (640, max(1, round(height * 640 / width)))
            )
            self._log(
                "camera/image",
                self.rr.Image(np.asarray(resized)).compress(jpeg_quality=90),
            )

    def on_status(self, status: Mapping[str, object]) -> None:
        with self.lock:
            state = status.get("state")
            if state == "session_reset":
                session_uid = status.get("session_uid")
                self._begin_session(
                    session_uid if isinstance(session_uid, str) else None
                )
            signature = json.dumps(
                {
                    key: value
                    for key, value in status.items()
                    if key not in ("timestamp_ns", "arrival_timestamp_ns")
                },
                sort_keys=True,
                allow_nan=False,
            )
            query_update = "fix_accepted" in status or any(
                status.get(key) is not None
                for key in ("query_duration_sec", "num_inliers", "fix_timestamp_ns")
            )
            if (
                signature == self.last_status_signature
                and not query_update
                and state not in ("error", "session_reset")
            ):
                return
            self.last_status_signature = signature
            self.last_status_state = state
            timestamp_ns = status.get("timestamp_ns")
            if timestamp_ns is None:
                timestamp_ns = status.get("arrival_timestamp_ns")
            if isinstance(timestamp_ns, int):
                self._set_time(timestamp_ns)
            if state in ("lost", "session_reset", "error"):
                self.path.clear()
                self.last_pose_timestamp_ns = None
                for entity in ("map/pose", "map/trajectory", "map/device_axes"):
                    self._log(entity, self.rr.Clear(recursive=True))
                self._log("diagnostics/tracking", self.rr.Scalars(0))
            self._log(
                "diagnostics/status",
                self.rr.TextLog(json.dumps(dict(status), allow_nan=False)),
            )

    def close(self) -> None:
        failures = []
        for recording in self.recordings:
            try:
                recording.flush(timeout_sec=5)
            except Exception as error:
                failures.append(error)
            try:
                recording.disconnect()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("Rerun recording cleanup failed", failures)


class PoseWriter:
    """Publish every mapped VIO sample as JSONL, independently of GUI throttling."""

    def __init__(self, path: Path) -> None:
        self.target = path.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.count = 0

    def write(self, pose: MappedPose) -> None:
        value = {
            "timestamp_ns": pose.timestamp_ns,
            "session_uid": pose.session_uid,
            "frame": "map",
            "T_map_device": pose.T_map_device.tolist(),
            "fix_timestamp_ns": pose.fix_timestamp_ns,
            "fix_age_ns": pose.age_ns,
            "status": pose.status,
        }
        with self.lock:
            self.target.write(json.dumps(value, allow_nan=False) + "\n")
            self.count += 1

    def close(self) -> None:
        self.target.close()
