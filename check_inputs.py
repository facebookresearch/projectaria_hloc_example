#!/usr/bin/env fbpython
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

"""Smoke-check recorded images and MPS inputs without HLOC or model downloads."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from aria_utils import Recording
from map_utils import mps_slam_dir, Trajectory


def check_inputs(vrs: Path, mps: Path) -> dict[str, object]:
    slam = mps_slam_dir(mps)
    trajectory = Trajectory(slam / "closed_loop_trajectory.csv")
    recording = Recording(vrs, slam / "online_calibration.jsonl")
    labels = recording.select_labels("all")
    frame_counts = {}
    for label in labels:
        timestamps = recording.corrected_timestamps_ns[label]
        candidates = [
            index
            for index, timestamp_ns in enumerate(timestamps)
            if trajectory.at(timestamp_ns) is not None
        ]
        if not candidates:
            raise ValueError(f"No {label} frames covered by the MPS trajectory")
        frame = recording.frame(label, candidates[len(candidates) // 2])
        if trajectory.at(frame.corrected_timestamp_ns) is None:
            raise ValueError(f"Decoded {label} frame is outside MPS coverage")
        frame_counts[label] = len(timestamps)
    return {
        "camera_frames": frame_counts,
        "trajectory_samples": len(trajectory.timestamps_ns),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", type=Path, default=Path("data/pilot/play_0/video.vrs"))
    parser.add_argument("--mps", type=Path, default=Path("data/pilot/play_0/mps"))
    args = parser.parse_args(argv)
    try:
        result = check_inputs(args.vrs, args.mps)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
