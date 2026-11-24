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

"""Public Project Aria Tools adapter. No device connection or MPS request is made."""

from __future__ import annotations

import bisect
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from map_utils import Camera, MAX_POSE_GAP_NS, SLAM_LABELS
from numpy.typing import NDArray

MAX_CALIBRATION_GAP_NS: int = 1_000_000_000
GEN1_SLAM_LABELS: tuple[str, ...] = ("camera-slam-left", "camera-slam-right")


@dataclass
class Frame:
    label: str
    index: int
    capture_timestamp_ns: int
    image: NDArray[np.uint8]
    camera: Camera
    time_offset_ns: int = 0

    @property
    def corrected_timestamp_ns(self) -> int:
        """PAT's camera-to-device time correction, without an exposure/readout heuristic."""
        return self.capture_timestamp_ns - self.time_offset_ns


class Recording:
    """Read one VRS with that recording's own online camera calibration."""

    def __init__(self, vrs: Path, online_calibration_path: Path) -> None:
        provider_module = importlib.import_module(
            "projectaria_tools.core.data_provider"
        )
        self.calibration_module = importlib.import_module(
            "projectaria_tools.core.calibration"
        )
        mps_module = importlib.import_module("projectaria_tools.core.mps")
        self.time_module = importlib.import_module("projectaria_tools.core.sensor_data")
        self.provider = provider_module.create_vrs_data_provider(str(vrs))
        if self.provider is None:
            raise ValueError(f"Could not open VRS: {vrs}")
        self.device_calibration = self.provider.get_device_calibration()
        if self.device_calibration is None:
            raise ValueError("VRS has no device calibration")
        self.calibrations = mps_module.read_online_calibration(
            str(online_calibration_path)
        )
        if not self.calibrations:
            raise ValueError("Online calibration is empty")
        self.calibration_timestamps_ns = [
            round(sample.tracking_timestamp.total_seconds() * 1e9)
            for sample in self.calibrations
        ]
        if any(
            a >= b
            for a, b in zip(
                self.calibration_timestamps_ns, self.calibration_timestamps_ns[1:]
            )
        ):
            raise ValueError(
                "Online calibration timestamps must be strictly increasing"
            )
        self.stream_ids = {}
        self.timestamps_ns = {}
        self.corrected_timestamps_ns = {}
        self.calibration_covered = {}

    def _slam_labels(self, available: set[str]) -> tuple[str, ...]:
        version = self.device_calibration.get_device_version()
        generations = self.calibration_module.DeviceVersion
        if version == generations.Gen1:
            labels, other = GEN1_SLAM_LABELS, SLAM_LABELS
        elif version == generations.Gen2:
            labels, other = SLAM_LABELS, GEN1_SLAM_LABELS
        else:
            raise ValueError(
                f"Unsupported device generation for SLAM cameras: {version}"
            )
        conflicting = available.intersection(other)
        if conflicting:
            raise ValueError(
                f"Device generation conflicts with SLAM camera labels: {sorted(conflicting)}"
            )
        return labels

    def select_labels(self, camera_set: str) -> tuple[str, ...]:
        """Require the generation's complete camera set and preserve original labels."""
        if camera_set not in ("rgb", "slam", "all"):
            raise ValueError(f"Unknown camera set: {camera_set}")
        available = set(self.device_calibration.get_camera_labels())
        slam = self._slam_labels(available) if camera_set != "rgb" else ()
        labels = {
            "rgb": ("camera-rgb",),
            "slam": slam,
            "all": slam + ("camera-rgb",),
        }[camera_set]
        if not set(labels).issubset(available):
            raise ValueError(
                f"Requested camera set {camera_set} is missing streams: {set(labels) - available}"
            )
        for label in labels:
            self._load_timestamps(label)
        return labels

    def _load_timestamps(self, label: str) -> None:
        stream = self.provider.get_stream_id_from_label(label)
        if stream is None or self.provider.get_num_data(stream) == 0:
            raise ValueError(f"VRS has no images for {label}")
        self.stream_ids[label] = stream
        timestamps = list(
            self.provider.get_timestamps_ns(
                stream, self.time_module.TimeDomain.DEVICE_TIME
            )
        )
        if not timestamps or any(a >= b for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError(
                f"Image timestamps must be nonempty and strictly increasing for {label}"
            )
        # Frames far from any online-calibration sample (e.g. before MPS SLAM
        # starts) stay addressable by index but are skipped by every consumer.
        covered = [
            self._calibration_gap_ns(t) <= MAX_CALIBRATION_GAP_NS for t in timestamps
        ]
        if not any(covered):
            raise ValueError(f"No {label} frames are covered by online calibration")
        corrected = [
            timestamp_ns - self.time_offset_ns(label, timestamp_ns, strict=False)
            for timestamp_ns in timestamps
        ]
        if any(a >= b for a, b in zip(corrected, corrected[1:])):
            raise ValueError(
                f"Corrected image timestamps must be strictly increasing for {label}"
            )
        self.timestamps_ns[label] = timestamps
        self.corrected_timestamps_ns[label] = corrected
        self.calibration_covered[label] = covered

    def has_calibration(self, label: str, index: int) -> bool:
        """Whether frame ``index`` of ``label`` has online calibration within the gap limit."""
        return self.calibration_covered[label][index]

    def _nearest_calibration(self, timestamp_ns: int) -> int:
        insertion = bisect.bisect_left(self.calibration_timestamps_ns, timestamp_ns)
        candidates = [
            i for i in (insertion - 1, insertion) if 0 <= i < len(self.calibrations)
        ]
        return min(
            candidates,
            key=lambda i: abs(self.calibration_timestamps_ns[i] - timestamp_ns),
        )

    def _calibration_gap_ns(self, timestamp_ns: int) -> int:
        index = self._nearest_calibration(timestamp_ns)
        return abs(self.calibration_timestamps_ns[index] - timestamp_ns)

    def _online_camera(
        self,
        label: str,
        timestamp_ns: int,
        strict: bool = True,
    ):
        index = self._nearest_calibration(timestamp_ns)
        if (
            strict
            and abs(self.calibration_timestamps_ns[index] - timestamp_ns)
            > MAX_CALIBRATION_GAP_NS
        ):
            raise ValueError(f"No online calibration near {timestamp_ns} ns")
        source = self.calibrations[index].get_camera_calib(label)
        if source is None:
            raise ValueError(f"Online calibration is missing {label}")
        return source

    def time_offset_ns(
        self,
        label: str,
        timestamp_ns: int,
        strict: bool = True,
    ) -> int:
        """Read the online camera offset in the same recording's clock domain.

        ``strict=False`` uses the nearest sample regardless of the gap limit; it is
        only used to order uncovered frames, which are never decoded.
        """
        camera = self._online_camera(label, timestamp_ns, strict)
        return round(float(camera.get_time_offset_sec_device_camera()) * 1e9)

    def _source_calibration(
        self, label: str, timestamp_ns: int, width: int, height: int
    ):
        source = self._online_camera(label, timestamp_ns)
        if tuple(source.get_image_size()) != (width, height):
            source = self.calibration_module.rescale_camera_calibration(
                source,
                (width, height),
                self.device_calibration.get_device_version(),
                label,
            )
        return source

    def frame(self, label: str, index: int) -> Frame:
        """Rectify using online calibration without rotating images upright."""
        data, record = self.provider.get_image_data_by_index(
            self.stream_ids[label], index
        )
        if not data.is_valid():
            raise ValueError(f"Invalid {label} image at index {index}")
        image = np.asarray(data.to_numpy_array())
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("Expected a uint8 grayscale or RGB camera image")
        height, width = image.shape[:2]
        timestamp_ns = int(record.capture_timestamp_ns)
        source = self._source_calibration(label, timestamp_ns, width, height)
        output_width, output_height = (
            (1024, 768) if label == "camera-rgb" else (512, 512)
        )
        focal = (
            360.0
            if label == "camera-rgb"
            else float(source.get_focal_lengths()[0]) * output_width / width
        )
        target = self.calibration_module.get_linear_camera_calibration(
            output_width, output_height, focal, label
        )
        rectified = self.calibration_module.distort_by_calibration(
            image, target, source
        )
        principal = target.get_principal_point()
        camera = Camera(
            label,
            output_width,
            output_height,
            focal,
            float(principal[0]),
            float(principal[1]),
            source.get_transform_device_camera().to_matrix(),
        )
        return Frame(
            label,
            index,
            timestamp_ns,
            np.asarray(rectified),
            camera,
            self.time_offset_ns(label, timestamp_ns),
        )

    def frameset(self, labels: tuple[str, ...], anchor_index: int) -> list[Frame]:
        """Synchronize corrected camera exposure times without VIO propagation."""
        anchor_timestamp_ns = self.corrected_timestamps_ns[labels[0]][anchor_index]
        frames = []
        for label in labels:
            timestamps = self.corrected_timestamps_ns[label]
            insertion = bisect.bisect_left(timestamps, anchor_timestamp_ns)
            indices = [
                i for i in (insertion - 1, insertion) if 0 <= i < len(timestamps)
            ]
            index = min(indices, key=lambda i: abs(timestamps[i] - anchor_timestamp_ns))
            frames.append(self.frame(label, index))
        if (
            max(frame.corrected_timestamp_ns for frame in frames)
            - min(frame.corrected_timestamp_ns for frame in frames)
            > MAX_POSE_GAP_NS
        ):
            raise ValueError(
                "SLAM frameset spans more than 2 ms; choose a synchronized recording"
            )
        return frames
