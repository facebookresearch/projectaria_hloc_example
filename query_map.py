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

"""Evaluate recorded RGB queries or synchronized four-SLAM-camera gPnP queries."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import numpy as np
from aria_utils import Frame, Recording
from hloc_backend import HlocBackend
from map_utils import (
    file_sha256,
    HLOC_COMMIT,
    mps_slam_dir,
    rigid_transform,
    rotation_error_deg,
    Trajectory,
    View,
    write_json,
)


def _validate_source_hashes(hashes: object, num_sources: int) -> None:
    """Validate the optional parallel recording-fingerprint metadata."""
    if (
        not isinstance(hashes, list)
        or len(hashes) != num_sources
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for digest in hashes
        )
    ):
        raise ValueError(
            "Map source_vrs_sha256 must contain one lowercase SHA-256 per source"
        )


def _load_views(raw_views: list[dict[str, object]], num_sources: int) -> list[View]:
    """Require unique, well-formed views with strict recording namespace indices."""
    if any(
        type(view.get("sequence")) is not int
        or not 0 <= cast(int, view["sequence"]) < num_sources
        for view in raw_views
    ):
        raise ValueError("Map views must have integer source sequences in range")
    try:
        views = [View.from_dict(view) for view in raw_views]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid map view record") from error
    if not views or len({view.name for view in views}) != len(views):
        raise ValueError(
            "Map views must have unique image names and valid source sequences"
        )
    return views


def load_map(path: Path) -> dict[str, object]:
    """Require a completed reference map with a supported dependency snapshot."""
    data = json.loads((path / "map.json").read_text())
    if (
        not isinstance(data, dict)
        or type(data.get("format_version")) is not int
        or data.get("format_version") != 1
        or data.get("hloc_commit") != HLOC_COMMIT
    ):
        raise ValueError("Map has an unsupported format or HLOC snapshot")
    raw_views = data.get("views")
    sources = data.get("source_vrs")
    if not isinstance(raw_views, list) or not all(
        isinstance(view, dict) for view in raw_views
    ):
        raise ValueError("Map views must be a list of view records")
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(source, str) and source for source in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("Map source_vrs must contain distinct, nonempty path strings")
    if "source_vrs_sha256" in data:
        _validate_source_hashes(data["source_vrs_sha256"], len(sources))
    if not isinstance(data.get("graph_uid"), str) or not data["graph_uid"]:
        raise ValueError("Map must identify a nonempty MPS graph_uid")
    _load_views(raw_views, len(sources))
    for name in ("features.h5", "global.h5"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Map is missing {name}")
    if not (path / "sfm").is_dir():
        raise FileNotFoundError("Map is missing its COLMAP reconstruction")
    return data


def is_keyframe(
    frames: Sequence[Frame],
    source: Path,
    metadata: dict[str, object],
    *,
    source_sha256: str | None = None,
) -> bool:
    """Match all recording namespaces by content; legacy maps only match paths."""
    sources = cast(list[str], metadata["source_vrs"])
    hashes = metadata.get("source_vrs_sha256")
    if hashes is not None:
        digest = source_sha256 if source_sha256 is not None else file_sha256(source)
        sequences = {
            index
            for index, value in enumerate(cast(list[str], hashes))
            if value == digest
        }
    else:
        sequences = {
            index for index, path in enumerate(sources) if path == str(source.resolve())
        }
    if not sequences:
        return False
    selected = {
        (view.camera.label, view.index)
        for view in (
            View.from_dict(value)
            for value in cast(list[dict[str, object]], metadata["views"])
        )
        if view.sequence in sequences
    }
    return any((frame.label, frame.index) in selected for frame in frames)


def localize_frames(
    backend: HlocBackend, map_path: Path, frames: Sequence[Frame]
) -> dict[str, object]:
    """Localize decoded frames without staging images or feature files on disk."""
    start_time_sec = time.monotonic()
    result = backend.query(map_path, frames)
    result["query_duration_sec"] = time.monotonic() - start_time_sec
    return result


def query_map(
    map_path: Path,
    vrs: Path,
    mps: Path,
    mode: str = "rgb",
    num_queries: int = 50,
    stride: int = 1,
) -> dict[str, object]:
    """Query without a ground-truth retrieval prior; MPS poses are used only for diagnostics."""
    if mode not in ("rgb", "slam") or num_queries <= 0 or stride <= 0:
        raise ValueError("Use rgb/slam mode with positive query count and stride")
    metadata = load_map(map_path)
    query_sha256 = file_sha256(vrs)
    source_hashes = metadata.get("source_vrs_sha256")
    source_identity_check = "sha256" if source_hashes is not None else "legacy_path"
    query_in_mapping_recordings = (
        query_sha256 in cast(list[str], source_hashes)
        if source_hashes is not None
        else None
    )
    if source_hashes is None:
        logging.warning(
            "Legacy map uses path identity only; cannot establish cross-recording independence"
        )
    slam = mps_slam_dir(mps)
    trajectory = Trajectory(slam / "closed_loop_trajectory.csv")
    recording = Recording(vrs, slam / "online_calibration.jsonl")
    labels = recording.select_labels(mode)
    queries = []
    with ExitStack() as resources:
        backend = HlocBackend()
        resources.callback(backend.close)
        initialization_durations_sec = backend.prepare_map(map_path)
        for anchor_index in range(0, len(recording.timestamps_ns[labels[0]]), stride):
            start_time_sec = time.monotonic()
            frames = recording.frameset(labels, anchor_index)
            if is_keyframe(frames, vrs, metadata, source_sha256=query_sha256):
                continue
            result = localize_frames(backend, map_path, frames)
            result["query_end_to_end_duration_sec"] = time.monotonic() - start_time_sec
            timestamp_ns = frames[0].corrected_timestamp_ns
            result["capture_timestamp_ns"] = frames[0].capture_timestamp_ns
            result["corrected_timestamp_ns"] = timestamp_ns
            truth = (
                trajectory.at(timestamp_ns)
                if trajectory.graph_uid == metadata["graph_uid"]
                else None
            )
            if result["success"] and truth is not None:
                estimate = rigid_transform(
                    cast(list[list[float]], result["T_world_device"])
                )
                result["translation_error_m"] = float(
                    np.linalg.norm(estimate[:3, 3] - truth[:3, 3])
                )
                result["rotation_error_deg"] = rotation_error_deg(estimate, truth)
            queries.append(result)
            if len(queries) >= num_queries:
                break
    if not queries:
        raise ValueError(
            "No held-out query frames; choose a different recording or denser sampling"
        )
    successes = sum(bool(query["success"]) for query in queries)
    return {
        "initialization_durations_sec": initialization_durations_sec,
        "mode": mode,
        "map_graph_uid": metadata["graph_uid"],
        "query_vrs_sha256": query_sha256,
        "source_identity_check": source_identity_check,
        "query_in_mapping_recordings": query_in_mapping_recordings,
        "num_queries": len(queries),
        "num_successes": successes,
        "success_rate": successes / len(queries),
        "same_mps_graph": trajectory.graph_uid == metadata["graph_uid"],
        "queries": queries,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument("--mps", type=Path, required=True)
    parser.add_argument("--mode", choices=("rgb", "slam"), default="rgb")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.output.exists():
            raise FileExistsError("Query output already exists; choose a new file")
        result = query_map(
            args.map, args.vrs, args.mps, args.mode, args.num_queries, args.stride
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "queries"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
