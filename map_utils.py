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

"""Map geometry and MPS input validation for the offline reference workflow."""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]
SLAM_LABELS: tuple[str, ...] = (
    "slam-front-left",
    "slam-front-right",
    "slam-side-left",
    "slam-side-right",
)
HLOC_COMMIT: str = "c13273bd0ecc2917a35910fd843712a1c6243193"
MAX_POSE_GAP_NS: int = 2_000_000


def file_sha256(path: Path) -> str:
    """Fingerprint a recorded input without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rigid_transform(value: Sequence[Sequence[float]] | FloatArray) -> FloatArray:
    """Require a finite, proper SE(3) transform; scale/reflection are not accepted."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if not (
        np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("Expected a rigid transform without scale or reflection")
    return matrix


def inverse_transform(matrix: FloatArray) -> FloatArray:
    """Invert T_target_source to T_source_target."""
    matrix = rigid_transform(matrix)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def transform_from_quaternion(
    quaternion: Sequence[float], xyz: Sequence[float]
) -> FloatArray:
    """Build T_world_device from MPS's [qw, qx, qy, qz] quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = xyz
    return rigid_transform(matrix)


def quaternion_wxyz(matrix: FloatArray) -> FloatArray:
    """Convert a validated rotation to COLMAP's quaternion order."""
    q_xyzw = Rotation.from_matrix(rigid_transform(matrix)[:3, :3]).as_quat()
    return q_xyzw[[3, 0, 1, 2]]


def rotation_error_deg(left: FloatArray, right: FloatArray) -> float:
    """Angular distance between two orientations."""
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    sine = (
        np.linalg.norm(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        / 2
    )
    return float(np.degrees(np.arctan2(sine, cosine)))


@dataclass
class Camera:
    """Rectified intrinsics in array/PAT convention and camera-to-device extrinsics."""

    label: str
    width: int
    height: int
    focal: float
    cx: float
    cy: float
    T_device_camera: FloatArray

    def __post_init__(self) -> None:
        if (
            min(self.width, self.height) <= 0
            or not np.isfinite([self.focal, self.cx, self.cy]).all()
            or self.focal <= 0
        ):
            raise ValueError("Invalid pinhole camera")
        self.T_device_camera = rigid_transform(self.T_device_camera)

    def colmap_params(self) -> list[float]:
        """COLMAP uses half-integer pixel centers; remapping stays in PAT convention."""
        return [self.focal, self.focal, self.cx + 0.5, self.cy + 0.5]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "focal": self.focal,
            "cx": self.cx,
            "cy": self.cy,
            "T_device_camera": self.T_device_camera.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Camera:
        return cls(
            str(value["label"]),
            int(cast(int, value["width"])),
            int(cast(int, value["height"])),
            float(cast(float, value["focal"])),
            float(cast(float, value["cx"])),
            float(cast(float, value["cy"])),
            rigid_transform(cast(list[list[float]], value["T_device_camera"])),
        )


@dataclass
class View:
    name: str
    sequence: int
    index: int
    capture_timestamp_ns: int
    camera: Camera
    T_world_camera: FloatArray
    time_offset_ns: int = 0

    @property
    def corrected_timestamp_ns(self) -> int:
        return self.capture_timestamp_ns - self.time_offset_ns

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sequence": self.sequence,
            "index": self.index,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "time_offset_ns": self.time_offset_ns,
            "corrected_timestamp_ns": self.corrected_timestamp_ns,
            "camera": self.camera.to_dict(),
            "T_world_camera": self.T_world_camera.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> View:
        name = str(value["name"])
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Map image names must be relative paths within the map")
        return cls(
            name,
            int(cast(int, value["sequence"])),
            int(cast(int, value["index"])),
            int(cast(int, value["capture_timestamp_ns"])),
            Camera.from_dict(cast(dict[str, object], value["camera"])),
            rigid_transform(cast(list[list[float]], value["T_world_camera"])),
            int(cast(int, value["time_offset_ns"])),
        )


class Trajectory:
    """One recording's device clock and one connected MPS world frame."""

    def __init__(self, csv_path: Path) -> None:
        with csv_path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        if not rows:
            raise ValueError("MPS trajectory is empty")
        self.timestamps_ns = [int(row["tracking_timestamp_us"]) * 1000 for row in rows]
        if any(a >= b for a, b in zip(self.timestamps_ns, self.timestamps_ns[1:])):
            raise ValueError("MPS timestamps must be strictly increasing")
        graphs = {row.get("graph_uid", "").strip() for row in rows}
        if len(graphs) != 1 or "" in graphs:
            raise ValueError(
                "Trajectory must have one nonempty graph_uid; select one connected MPS component"
            )
        self.graph_uid = graphs.pop()
        self.poses = [
            transform_from_quaternion(
                [float(row[f"q{axis}_world_device"]) for axis in "wxyz"],
                [float(row[f"t{axis}_world_device"]) for axis in "xyz"],
            )
            for row in rows
        ]

    def at(
        self, timestamp_ns: int, max_gap_ns: int = MAX_POSE_GAP_NS
    ) -> FloatArray | None:
        """Nearest pose within coverage; never extrapolate or cross recording clocks."""
        if (
            timestamp_ns < self.timestamps_ns[0]
            or timestamp_ns > self.timestamps_ns[-1]
        ):
            return None
        insertion = bisect.bisect_left(self.timestamps_ns, timestamp_ns)
        indices = [i for i in (insertion - 1, insertion) if 0 <= i < len(self.poses)]
        index = min(indices, key=lambda i: abs(self.timestamps_ns[i] - timestamp_ns))
        return (
            self.poses[index]
            if abs(self.timestamps_ns[index] - timestamp_ns) <= max_gap_ns
            else None
        )


def mps_slam_dir(path: Path) -> Path:
    """Accept an explicit slam directory or its immediate MPS output root."""
    directory = (
        path if (path / "closed_loop_trajectory.csv").is_file() else path / "slam"
    )
    for name in ("closed_loop_trajectory.csv", "online_calibration.jsonl"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {directory}")
    return directory


def mapping_pairs(views: Sequence[View], num_pairs: int = 20) -> list[tuple[str, str]]:
    """Bound nearest-neighbour work instead of allocating an NxN distance matrix."""
    if len(views) < 2 or num_pairs < 1:
        raise ValueError("Need at least two keyframes and a positive pair count")
    centres = np.asarray([view.T_world_camera[:3, 3] for view in views])
    tree = cKDTree(centres)
    pairs = set()
    for index, view in enumerate(views):
        distances, neighbours = tree.query(
            centres[index], k=min(len(views), num_pairs * 4 + 1)
        )
        accepted = 0
        for distance, neighbour in zip(
            np.atleast_1d(distances), np.atleast_1d(neighbours)
        ):
            if neighbour == index or distance > 4.0:
                continue
            dot = np.dot(
                view.T_world_camera[:3, 2], views[int(neighbour)].T_world_camera[:3, 2]
            )
            if dot < np.cos(np.radians(45)):
                continue
            pairs.add(tuple(sorted((view.name, views[int(neighbour)].name))))
            accepted += 1
            if accepted >= num_pairs:
                break
    if not pairs:
        raise ValueError(
            "No overlapping image pairs; check scan coverage and coordinate frames"
        )
    return sorted(pairs)


def covisibility_clusters(landmarks: dict[str, set[int]]) -> list[list[str]]:
    """Partition retrieved images by shared reconstructed landmarks."""
    parent = {name: name for name in landmarks}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    first = {}
    for name, points in landmarks.items():
        for point in points:
            if point in first:
                parent[root(name)] = root(first[point])
            else:
                first[point] = name
    groups = {}
    for name in landmarks:
        groups.setdefault(root(name), []).append(name)
    return sorted(
        [sorted(group) for group in groups.values()],
        key=lambda group: (-len(group), group),
    )


def unique_matches(
    matches: Iterable[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Deduplicate (camera, keypoint), discarding conflicting landmark assignments."""
    assignments = {}
    for camera_index, keypoint_index, landmark in matches:
        assignments.setdefault((camera_index, keypoint_index), set()).add(landmark)
    return [
        (camera, keypoint, next(iter(points)))
        for (camera, keypoint), points in sorted(assignments.items())
        if len(points) == 1
    ]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
