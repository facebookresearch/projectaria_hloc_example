#!/usr/bin/env python3
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

"""Build a metric HLOC map from one VRS or aligned MPS Multi-SLAM outputs."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from aria_utils import Recording
from hloc_backend import HlocBackend
from map_utils import (
    file_sha256,
    HLOC_COMMIT,
    mapping_pairs,
    mps_slam_dir,
    rotation_error_deg,
    Trajectory,
    View,
    write_json,
)
from PIL import Image


def resolve_mps(vrs_paths: Sequence[Path], mps_paths: Sequence[Path]) -> list[Path]:
    """Resolve explicit per-recording outputs or the public Multi-SLAM mapping file."""
    if not vrs_paths or len({path.resolve() for path in vrs_paths}) != len(vrs_paths):
        raise ValueError("Pass one or more distinct VRS recordings")
    if len(mps_paths) == len(vrs_paths) and not (
        len(mps_paths) == 1 and (mps_paths[0] / "vrs_to_multi_slam.json").is_file()
    ):
        return [mps_slam_dir(path) for path in mps_paths]
    if len(mps_paths) != 1:
        raise ValueError("Pass one MPS path per VRS, or one Multi-SLAM output root")
    root = mps_paths[0]
    mapping = json.loads((root / "vrs_to_multi_slam.json").read_text())
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in mapping.items()
    ):
        raise ValueError(
            "vrs_to_multi_slam.json must map VRS paths to nonempty output-folder strings"
        )
    results = []
    for vrs in vrs_paths:
        exact = [key for key in mapping if Path(key).resolve() == vrs.resolve()]
        matches = exact or [key for key in mapping if Path(key).name == vrs.name]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot uniquely associate {vrs.name} with Multi-SLAM output"
            )
        folder = Path(mapping[matches[0]])
        if folder.is_absolute() or ".." in folder.parts:
            raise ValueError("Multi-SLAM folder must stay inside the output root")
        results.append(mps_slam_dir(root / folder))
    return results


def extract_views(
    recording: Recording,
    trajectory: Trajectory,
    sequence: int,
    camera_set: str,
    output: Path,
    max_keyframes: int,
) -> list[View]:
    """Select motion-separated keyframes independently for each camera and recording.

    A nonzero ``max_keyframes`` keeps that many keyframes spread evenly over the
    recording instead of only its beginning.
    """
    views = []
    for label in recording.select_labels(camera_set):
        last_pose = None
        selected = []
        for index, timestamp_ns in enumerate(recording.corrected_timestamps_ns[label]):
            if not recording.has_calibration(label, index):
                continue
            pose = trajectory.at(timestamp_ns)
            if pose is None:
                continue
            if (
                last_pose is not None
                and np.linalg.norm(pose[:3, 3] - last_pose[:3, 3]) < 0.15
                and rotation_error_deg(last_pose, pose) < 10
            ):
                continue
            selected.append(index)
            last_pose = pose
        if max_keyframes and len(selected) > max_keyframes:
            picks = np.linspace(0, len(selected) - 1, max_keyframes).round()
            selected = [selected[int(i)] for i in picks]
        for index in selected:
            frame = recording.frame(label, index)
            pose = trajectory.at(frame.corrected_timestamp_ns)
            if pose is None:
                continue
            name = f"s{sequence:03d}/{label}/{index:08d}.png"
            path = output / "images" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.image).save(path)
            views.append(
                View(
                    name,
                    sequence,
                    index,
                    frame.capture_timestamp_ns,
                    frame.camera,
                    pose @ frame.camera.T_device_camera,
                    frame.time_offset_ns,
                )
            )
    return views


def build_map(
    vrs_paths: Sequence[Path],
    mps_paths: Sequence[Path],
    output: Path,
    camera_set: str = "slam",
    max_keyframes: int = 0,
) -> dict[str, object]:
    """Create a new map; an incomplete build never receives a valid map manifest."""
    if max_keyframes < 0:
        raise ValueError("max_keyframes must be nonnegative")
    if camera_set not in ("slam", "rgb", "all"):
        raise ValueError("Unknown camera set")
    slam_dirs = resolve_mps(vrs_paths, mps_paths)
    trajectories = [
        Trajectory(directory / "closed_loop_trajectory.csv") for directory in slam_dirs
    ]
    graphs = {trajectory.graph_uid for trajectory in trajectories}
    if len(graphs) != 1:
        raise ValueError(
            "Recordings do not share one graph_uid; run MPS Multi-SLAM and select a connected component"
        )
    source_hashes = [file_sha256(vrs) for vrs in vrs_paths]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError(
            "Mapping recordings must have distinct VRS SHA-256 values; a file was passed twice"
        )
    backend = HlocBackend()
    output.mkdir(parents=True, exist_ok=False)
    start_time_sec = time.monotonic()
    views = []
    for sequence, (vrs, slam, trajectory) in enumerate(
        zip(vrs_paths, slam_dirs, trajectories)
    ):
        recording = Recording(vrs, slam / "online_calibration.jsonl")
        views.extend(
            extract_views(
                recording, trajectory, sequence, camera_set, output, max_keyframes
            )
        )
        del recording
    stats = backend.build(output, views, mapping_pairs(views))
    stats["build_duration_sec"] = time.monotonic() - start_time_sec
    write_json(output / "build_stats.json", stats)
    write_json(
        output / "map.json",
        {
            "format_version": 1,
            "graph_uid": next(iter(graphs)),
            "hloc_commit": HLOC_COMMIT,
            "camera_set": camera_set,
            "source_vrs": [str(path.resolve()) for path in vrs_paths],
            "source_vrs_sha256": source_hashes,
            "views": [view.to_dict() for view in views],
        },
    )
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", nargs="+", type=Path, required=True)
    parser.add_argument("--mps", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-set", choices=("slam", "rgb", "all"), default="slam")
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=0,
        help="Per camera/recording; 0 means unlimited",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        result = build_map(
            args.vrs, args.mps, args.output_dir, args.camera_set, args.max_keyframes
        )
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
