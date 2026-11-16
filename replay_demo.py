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

"""Replay recorded VIO with measured HLOC fixes, producing high-rate map-frame poses."""

from __future__ import annotations

import argparse
import bisect
import json
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import numpy as np
from aria_utils import Recording
from map_utils import file_sha256, mps_slam_dir, rigid_transform, write_json
from pose_alignment import PoseAlignment, vio_status_valid, VioSample
from query_map import load_map
from visualization import PoseWriter, RerunView


def read_vio(recording: Recording) -> list[VioSample]:
    """Prefer recorded high-frequency VIO, falling back to the frontend output stream."""
    provider = recording.provider
    stream = provider.get_stream_id_from_label("vio_high_frequency")
    high_frequency = stream is not None and provider.get_num_data(stream) > 0
    if not high_frequency:
        stream = provider.get_stream_id_from_label("vio")
    if stream is None or provider.get_num_data(stream) == 0:
        raise ValueError("Recording has no VIO stream")
    samples = []
    for index in range(provider.get_num_data(stream)):
        if high_frequency:
            value = provider.get_vio_high_freq_data_by_index(stream, index)
            quality = float(value.quality_score)
            sample = VioSample(
                round(value.tracking_timestamp.total_seconds() * 1e9),
                value.transform_odometry_device.to_matrix(),
                str(value.session_uid),
                bool(np.isfinite(quality) and quality >= 0.5),
            )
        else:
            value = provider.get_vio_data_by_index(stream, index)
            sample = VioSample(
                int(value.capture_timestamp_ns),
                value.transform_odometry_bodyimu.to_matrix()
                @ value.transform_bodyimu_device.to_matrix(),
                str(value.frontend_session_uid),
                str(value.pose_quality).split(".")[-1] == "GOOD"
                and vio_status_valid(value),
            )
        samples.append(sample)
    if len(samples) < 2 or any(
        a.timestamp_ns >= b.timestamp_ns for a, b in zip(samples, samples[1:])
    ):
        raise ValueError("Recorded VIO timestamps must be strictly increasing")
    return samples


def schedule_fixes(
    queries: Sequence[dict[str, object]], samples: Sequence[VioSample]
) -> list[tuple[int, dict[str, object], str]]:
    """Simulate one latest-frame worker using measured query durations, not instant fixes."""
    first_timestamp_ns, end_timestamp_ns = (
        samples[0].timestamp_ns,
        samples[-1].timestamp_ns,
    )
    candidates = sorted(
        [
            query
            for query in queries
            if first_timestamp_ns
            <= int(cast(int, query["corrected_timestamp_ns"]))
            <= end_timestamp_ns
        ],
        key=lambda query: int(cast(int, query["corrected_timestamp_ns"])),
    )
    timestamps_ns = [
        int(cast(int, query["corrected_timestamp_ns"])) for query in candidates
    ]
    vio_timestamps_ns = [sample.timestamp_ns for sample in samples]
    ready_timestamp_ns = first_timestamp_ns
    next_index = 0
    scheduled = []
    while next_index < len(candidates):
        latest_index = bisect.bisect_right(timestamps_ns, ready_timestamp_ns) - 1
        index = max(next_index, latest_index)
        query = candidates[index]
        capture_timestamp_ns = timestamps_ns[index]
        # Measured HLOC query time only; decode and rectification are excluded.
        duration_sec = float(cast(float, query["query_duration_sec"]))
        if not np.isfinite(duration_sec) or duration_sec < 0:
            raise ValueError("Cached query duration must be finite and nonnegative")
        ready_timestamp_ns = max(ready_timestamp_ns, capture_timestamp_ns) + round(
            duration_sec * 1e9
        )
        if ready_timestamp_ns > end_timestamp_ns:
            break
        vio_index = bisect.bisect_right(vio_timestamps_ns, capture_timestamp_ns) - 1
        scheduled.append((ready_timestamp_ns, query, samples[vio_index].session_uid))
        next_index = index + 1
    return scheduled


def replay(  # noqa: C901
    map_path: Path, vrs: Path, mps: Path, fixes_path: Path, output: Path
) -> dict[str, object]:
    """Use only cached visual fixes for alignment; MPS poses never enter propagation."""
    metadata = load_map(map_path)
    fixes = json.loads(fixes_path.read_text())
    if (
        not isinstance(fixes, dict)
        or fixes.get("map_graph_uid") != metadata["graph_uid"]
    ):
        raise ValueError("Cached fixes must identify the same map_graph_uid as the map")
    if fixes.get("query_vrs_sha256") != file_sha256(vrs):
        raise ValueError("Cached queries must match the replay VRS SHA-256")
    if fixes.get("mode") not in ("rgb", "slam"):
        raise ValueError("Cached query mode must be rgb or slam")
    queries = fixes.get("queries")
    if not isinstance(queries, list) or not all(
        isinstance(query, dict) for query in queries
    ):
        raise ValueError("Cached fixes must contain a list of query records")
    recording = Recording(vrs, mps_slam_dir(mps) / "online_calibration.jsonl")
    samples = read_vio(recording)
    labels = recording.select_labels("rgb")
    try:
        scheduled = schedule_fixes(queries, samples)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Cached query records have invalid timestamps or durations"
        ) from error
    output.mkdir(parents=True, exist_ok=False)
    resources = ExitStack()
    writer = PoseWriter(output / "poses.jsonl")
    resources.callback(writer.close)
    try:
        viewer = RerunView(map_path, output / "view.rrd")
    except BaseException:
        resources.close()
        raise
    resources.callback(viewer.close)
    aligner = PoseAlignment()
    # VIO at an arrival timestamp precedes the delayed visual fix at that timestamp.
    events = [(sample.timestamp_ns, 0, index) for index, sample in enumerate(samples)]
    events += [
        (timestamp_ns, 1, index) for index, (timestamp_ns, _, _) in enumerate(scheduled)
    ]
    events += [
        (timestamp_ns, 2, index)
        for index, timestamp_ns in enumerate(
            recording.corrected_timestamps_ns[labels[0]]
        )
        if samples[0].timestamp_ns <= timestamp_ns <= samples[-1].timestamp_ns
        and recording.has_calibration(labels[0], index)
    ]
    accepted_fixes = 0
    first_pose_timestamp_ns = None
    last_pose_timestamp_ns = None
    try:
        for timestamp_ns, kind, index in sorted(events):
            if kind == 0:
                pose = aligner.add_vio(samples[index])
                if pose is not None:
                    writer.write(pose)
                    viewer.on_pose(pose)
                    if first_pose_timestamp_ns is None:
                        first_pose_timestamp_ns = pose.timestamp_ns
                    last_pose_timestamp_ns = pose.timestamp_ns
            elif kind == 1:
                _, query, session_uid = scheduled[index]
                accepted = False
                if query.get("success"):
                    accepted = aligner.add_fix(
                        int(cast(int, query["corrected_timestamp_ns"])),
                        rigid_transform(
                            cast(list[list[float]], query["T_world_device"])
                        ),
                        session_uid,
                    )
                accepted_fixes += int(accepted)
                viewer.on_status(
                    {
                        "state": aligner.state,
                        "fix_accepted": accepted,
                        "arrival_timestamp_ns": timestamp_ns,
                        "fix_timestamp_ns": query["corrected_timestamp_ns"],
                        "source": "cached measured HLOC query",
                    }
                )
            else:
                frame = recording.frame(labels[0], index)
                viewer.on_image(frame.image, frame.corrected_timestamp_ns)
    finally:
        resources.close()
    if (
        writer.count < 2
        or first_pose_timestamp_ns is None
        or last_pose_timestamp_ns is None
    ):
        raise ValueError(
            "Replay produced fewer than two map-frame poses; check fix timestamps, sessions and VIO coverage"
        )
    stats: dict[str, object] = {
        "mode": "recorded VIO replay with cached HLOC solves and measured delays",
        "vio_samples": len(samples),
        "scheduled_queries": len(scheduled),
        "accepted_fixes": accepted_fixes,
        "mapped_pose_samples": writer.count,
        "first_mapped_pose_timestamp_ns": first_pose_timestamp_ns,
        "last_mapped_pose_timestamp_ns": last_pose_timestamp_ns,
        "time_to_first_mapped_pose_sec": (
            first_pose_timestamp_ns - samples[0].timestamp_ns
        )
        / 1e9,
        "mapped_pose_device_rate_hz": (writer.count - 1)
        * 1e9
        / (last_pose_timestamp_ns - first_pose_timestamp_ns),
        "gui_max_rate_hz": 30,
        "freshness_evaluated_on": "recorded device timestamps, not live wall-clock",
    }
    write_json(output / "replay_stats.json", stats)
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument("--mps", type=Path, required=True)
    parser.add_argument(
        "--fixes",
        type=Path,
        required=True,
        help="query_map.py JSON produced against this map",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        stats = replay(args.map, args.vrs, args.mps, args.fixes, args.output_dir)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
