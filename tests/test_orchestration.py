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

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import build_map as build
import numpy as np
import pytest
import query_map as query
from aria_utils import Frame
from map_utils import HLOC_COMMIT, SLAM_LABELS, Trajectory, transform_from_quaternion
from PIL import Image


@pytest.fixture
def slam_dirs(trajectory_file, tmp_path):
    paths = []
    for sequence in range(2):
        directory = tmp_path / f"mps{sequence}" / "slam"
        trajectory_file(
            f"mps{sequence}/slam/closed_loop_trajectory.csv",
            timestamps_us=(
                1000 + sequence * 10_000,
                3000 + sequence * 10_000,
                7000 + sequence * 10_000,
            ),
            translations=[(0, 0, 0), (0.2, 0, 0), (0.4, 0, 0)],
        )
        (directory / "online_calibration.jsonl").touch()
        paths.append(directory)
    return paths


@pytest.fixture
def frame_factory(camera):
    def make(index, timestamp_ns, label="camera-rgb", time_offset_ns=0):
        return Frame(
            label,
            index,
            timestamp_ns,
            np.full((3, 4), index, dtype=np.uint8),
            camera,
            time_offset_ns,
        )

    return make


@pytest.fixture
def map_directory(tmp_path, make_view):
    path = tmp_path / "map"
    path.mkdir()
    metadata = {
        "format_version": 1,
        "hloc_commit": HLOC_COMMIT,
        "graph_uid": "graph-A",
        "source_vrs": [str((tmp_path / "recording.vrs").resolve())],
        "views": [make_view("a.png").to_dict()],
    }
    (path / "map.json").write_text(json.dumps(metadata))
    (path / "sfm").mkdir()
    (path / "features.h5").touch()
    (path / "global.h5").touch()
    return path, metadata


def test_explicit_mps_roots_are_per_recording(slam_dirs, tmp_path):
    vrs = [tmp_path / "a.vrs", tmp_path / "b.vrs"]
    assert build.resolve_mps(vrs, [path.parent for path in slam_dirs]) == slam_dirs
    assert build.resolve_mps([vrs[0]], [slam_dirs[0]]) == slam_dirs[:1]


@pytest.mark.parametrize("case", ["no_vrs", "duplicate_vrs", "no_mps", "too_many_mps"])
def test_mps_input_cardinality_errors(tmp_path, case):
    vrs, mps = [tmp_path / "a.vrs", tmp_path / "b.vrs"], []
    if case == "no_vrs":
        vrs = []
    elif case == "duplicate_vrs":
        vrs = [tmp_path / "a.vrs", tmp_path / "." / "a.vrs"]
    elif case == "too_many_mps":
        mps = [tmp_path] * 3
    with pytest.raises(ValueError, match="distinct|one MPS"):
        build.resolve_mps(vrs, mps)


def test_multi_slam_exact_match_wins_over_ambiguous_basename(tmp_path, trajectory_file):
    vrs = [tmp_path / "first" / "a.vrs", tmp_path / "second" / "b.vrs"]
    root = tmp_path / "multi"
    for folder in ("first", "second"):
        trajectory_file(f"multi/{folder}/slam/closed_loop_trajectory.csv")
        (root / folder / "slam" / "online_calibration.jsonl").touch()
    mapping = {str(vrs[0]): "first", "/another/a.vrs": "wrong", "/old/b.vrs": "second"}
    (root / "vrs_to_multi_slam.json").write_text(json.dumps(mapping))
    assert build.resolve_mps(vrs, [root]) == [
        root / "first" / "slam",
        root / "second" / "slam",
    ]


@pytest.mark.parametrize(
    "mapping,error",
    [
        ([], "output-folder"),
        ({"/x/a.vrs": "one", "/y/a.vrs": "two", "b.vrs": "three"}, "uniquely"),
        ({"other.vrs": "one"}, "uniquely"),
        ({"a.vrs": "../outside", "b.vrs": "two"}, "inside"),
        ({"a.vrs": "/outside", "b.vrs": "two"}, "inside"),
    ],
)
def test_multi_slam_rejects_invalid_mapping(tmp_path, mapping, error):
    (tmp_path / "vrs_to_multi_slam.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match=error):
        build.resolve_mps([tmp_path / "a.vrs", tmp_path / "b.vrs"], [tmp_path])


def test_extract_views_motion_gating_actual_timestamps_and_camera_isolation(
    tmp_path, camera, frame_factory
):
    poses = {
        1_000_000: np.eye(4),
        2_000_000: transform_from_quaternion([1, 0, 0, 0], [0.1, 0, 0]),
        3_000_000: transform_from_quaternion([1, 0, 0, 0], [0.2, 0, 0]),
        4_000_000: transform_from_quaternion(
            [np.cos(np.radians(7.5)), 0, 0, np.sin(np.radians(7.5))], [0.2, 0, 0]
        ),
        5_000_000: transform_from_quaternion([1, 0, 0, 0], [1, 0, 0]),
    }
    recording = SimpleNamespace(
        select_labels=lambda mode: SLAM_LABELS[:2],
        timestamps_ns={label: [0, *poses] for label in SLAM_LABELS[:2]},
        corrected_timestamps_ns={label: [0, *poses] for label in SLAM_LABELS[:2]},
        has_calibration=lambda label, index: True,
        frame=lambda label, index: frame_factory(
            index, 1_000_000_000 if index == 5 else list(poses)[index - 1], label
        ),
    )
    trajectory = SimpleNamespace(at=lambda timestamp_ns: poses.get(timestamp_ns))
    views = build.extract_views(recording, trajectory, 7, "slam", tmp_path, 0)
    assert [view.index for view in views] == [1, 3, 4, 1, 3, 4]
    assert {view.sequence for view in views} == {7}
    assert views[0].name == "s007/slam-front-left/00000001.png"
    np.testing.assert_allclose(
        views[1].T_world_camera, poses[3_000_000] @ camera.T_device_camera
    )
    with Image.open(tmp_path / "images" / views[0].name) as image:
        np.testing.assert_array_equal(
            np.asarray(image), np.ones((3, 4), dtype=np.uint8)
        )
    limited = build.extract_views(recording, trajectory, 8, "slam", tmp_path, 1)
    assert len(limited) == 2
    assert [view.index for view in limited] == [1, 1]


@pytest.fixture
def fake_build_dependencies(monkeypatch, slam_dirs, frame_factory, tmp_path):
    for sequence in range(2):
        (tmp_path / f"recording{sequence}.vrs").write_bytes(
            f"recording {sequence}".encode()
        )
    backend = Mock()
    backend.build.return_value = {"num_landmarks": 25, "num_keyframes": 6}
    monkeypatch.setattr(build, "HlocBackend", Mock(return_value=backend))
    recordings = []

    def open_recording(vrs, calibration):
        sequence = int(vrs.stem[-1])
        timestamps_ns = Trajectory(
            slam_dirs[sequence] / "closed_loop_trajectory.csv"
        ).timestamps_ns
        recording = SimpleNamespace(
            select_labels=lambda mode: ("camera-rgb",),
            timestamps_ns={"camera-rgb": timestamps_ns},
            corrected_timestamps_ns={"camera-rgb": timestamps_ns},
            has_calibration=lambda label, index: True,
            frame=lambda label, index: frame_factory(
                index, timestamps_ns[index], label
            ),
        )
        recordings.append((vrs, calibration, recording))
        return recording

    monkeypatch.setattr(build, "Recording", open_recording)
    return backend, recordings


def test_build_map_orchestrates_recordings_without_merging_clocks(
    tmp_path, slam_dirs, fake_build_dependencies
):
    backend, recordings = fake_build_dependencies
    vrs = [tmp_path / "recording0.vrs", tmp_path / "recording1.vrs"]
    output = tmp_path / "new-map"
    result = build.build_map(vrs, slam_dirs, output, camera_set="rgb")
    assert result["num_landmarks"] == 25
    assert result["build_duration_sec"] >= 0
    assert [value[1] for value in recordings] == [
        path / "online_calibration.jsonl" for path in slam_dirs
    ]
    views = backend.build.call_args.args[1]
    assert [view.sequence for view in views] == [0, 0, 0, 1, 1, 1]
    assert [view.capture_timestamp_ns for view in views] == [
        1_000_000,
        3_000_000,
        7_000_000,
        11_000_000,
        13_000_000,
        17_000_000,
    ]
    assert len(backend.build.call_args.args[2]) > 0
    manifest = json.loads((output / "map.json").read_text())
    assert manifest["source_vrs"] == [str(path.resolve()) for path in vrs]
    assert manifest["source_vrs_sha256"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in vrs
    ]
    assert manifest["graph_uid"] == "graph-A"
    assert manifest["camera_set"] == "rgb"
    assert len(manifest["views"]) == 6
    assert json.loads((output / "build_stats.json").read_text()) == result


def test_disconnected_graphs_are_rejected_before_backend_or_output(
    monkeypatch, tmp_path, slam_dirs, trajectory_file
):
    trajectory_file("mps1/slam/closed_loop_trajectory.csv", graphs=["graph-B"] * 3)
    backend = Mock(side_effect=AssertionError("must not initialize inference"))
    monkeypatch.setattr(build, "HlocBackend", backend)
    output = tmp_path / "rejected-map"
    with pytest.raises(ValueError, match="share one graph_uid"):
        build.build_map([tmp_path / "a.vrs", tmp_path / "b.vrs"], slam_dirs, output)
    backend.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("kwargs", [{"max_keyframes": -1}, {"camera_set": "unknown"}])
def test_build_rejects_invalid_options_before_io(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build.build_map([], [], tmp_path / "no-output", **kwargs)
    assert not (tmp_path / "no-output").exists()


def test_failed_build_never_writes_valid_manifest(
    tmp_path, slam_dirs, fake_build_dependencies
):
    backend, _ = fake_build_dependencies
    backend.build.side_effect = ValueError("zero landmarks")
    output = tmp_path / "incomplete-map"
    with pytest.raises(ValueError, match="zero landmarks"):
        build.build_map(
            [tmp_path / "recording0.vrs"], slam_dirs[:1], output, camera_set="rgb"
        )
    assert not (output / "map.json").exists()
    assert not (output / "build_stats.json").exists()


def test_build_does_not_overwrite_existing_output(
    tmp_path, slam_dirs, fake_build_dependencies
):
    output = tmp_path / "existing-map"
    output.mkdir()
    sentinel = output / "map.json"
    sentinel.write_text("existing")
    with pytest.raises(FileExistsError):
        build.build_map([tmp_path / "recording0.vrs"], slam_dirs[:1], output)
    assert sentinel.read_text() == "existing"


def test_load_completed_map(map_directory):
    path, metadata = map_directory
    assert query.load_map(path) == metadata


@pytest.mark.parametrize(
    "case", ["list", "format", "snapshot", "empty_views", "duplicate_views"]
)
def test_load_map_rejects_invalid_manifest(map_directory, case):
    path, data = map_directory
    if case == "list":
        data = []
    elif case == "format":
        data["format_version"] = 2
    elif case == "snapshot":
        data["hloc_commit"] = "floating"
    elif case == "empty_views":
        data["views"] = []
    else:
        data["views"] *= 2
    (path / "map.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsupported|views|duplicate"):
        query.load_map(path)


@pytest.mark.parametrize("missing", ["features.h5", "global.h5", "sfm"])
def test_load_map_requires_all_inference_artifacts(map_directory, missing):
    path, _ = map_directory
    target = path / missing
    if missing == "sfm":
        target.rmdir()
    else:
        target.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        query.load_map(path)


def test_keyframe_exclusion_is_source_sequence_and_camera_specific(
    map_directory, tmp_path, frame_factory
):
    _, metadata = map_directory
    source = tmp_path / "recording.vrs"
    assert query.is_keyframe([frame_factory(0, 1_000_000)], source, metadata)
    assert not query.is_keyframe([frame_factory(1, 1_000_000)], source, metadata)
    assert not query.is_keyframe(
        [frame_factory(0, 1_000_000, "slam-front-left")], source, metadata
    )
    assert not query.is_keyframe(
        [frame_factory(0, 1_000_000)], tmp_path / "heldout.vrs", metadata
    )
    metadata["source_vrs"].insert(0, str((tmp_path / "other.vrs").resolve()))
    assert not query.is_keyframe([frame_factory(0, 1_000_000)], source, metadata)


@pytest.mark.parametrize("raises", [False, True])
def test_query_uses_decoded_pixels_without_filesystem_staging(
    monkeypatch, tmp_path, frame_factory, raises
):
    frames = [frame_factory(2, 1_000_000)]
    observed = []

    def run(map_path, actual_frames):
        observed.append(actual_frames)
        assert map_path == tmp_path
        assert actual_frames is frames
        np.testing.assert_array_equal(actual_frames[0].image, frames[0].image)
        if raises:
            raise RuntimeError("inference failed")
        return {"success": False, "stage_durations_sec": {"matching_duration_sec": 0.1}}

    monkeypatch.setattr(
        Image.Image,
        "save",
        Mock(side_effect=AssertionError("queries must not stage PNG files")),
    )
    backend = SimpleNamespace(query=run)
    if raises:
        with pytest.raises(RuntimeError, match="inference failed"):
            query.localize_frames(backend, tmp_path, frames)
    else:
        result = query.localize_frames(backend, tmp_path, frames)
        assert result["success"] is False
        assert result["query_duration_sec"] >= 0
        assert result["stage_durations_sec"]["matching_duration_sec"] == 0.1
    assert len(observed) == 1
    assert not (tmp_path / "images").exists()


@pytest.fixture
def fake_query_dependencies(monkeypatch, frame_factory, tmp_path):
    (tmp_path / "recording.vrs").write_bytes(b"synthetic mapping recording")
    (tmp_path / "heldout.vrs").write_bytes(b"synthetic held-out recording")
    timestamps_ns = [1_000_000, 3_000_000, 7_000_000]
    recording = SimpleNamespace(
        select_labels=Mock(return_value=("camera-rgb",)),
        timestamps_ns={"camera-rgb": timestamps_ns},
        corrected_timestamps_ns={"camera-rgb": timestamps_ns},
        has_calibration=lambda label, index: True,
        frameset=Mock(
            side_effect=lambda labels, index: [
                frame_factory(index, timestamps_ns[index])
            ]
        ),
    )
    monkeypatch.setattr(query, "Recording", Mock(return_value=recording))
    backend = Mock()
    pose = transform_from_quaternion([1, 0, 0, 0], [0.3, 0, 0])
    backend.prepare_map.return_value = {"map_load_duration_sec": 0.5}
    backend.query.side_effect = (
        lambda map_path, frames: {
            "success": True,
            "T_world_device": pose.tolist(),
        }
        if frames[0].index == 1
        else {"success": False, "reason": "no_matches"}
    )
    monkeypatch.setattr(query, "HlocBackend", Mock(return_value=backend))
    return recording, backend


@pytest.mark.parametrize("failure_stage", ["prepare", "decode", "query"])
def test_recorded_query_closes_resident_backend_on_failure(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, failure_stage
):
    path, _ = map_directory
    recording, backend = fake_query_dependencies
    failure = RuntimeError(f"{failure_stage} failed")
    if failure_stage == "prepare":
        backend.prepare_map.side_effect = failure
    elif failure_stage == "decode":
        recording.frameset.side_effect = failure
    else:
        backend.query.side_effect = failure
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        query.query_map(path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=2)
    backend.close.assert_called_once_with()
    backend.prepare_map.assert_called_once_with(path)
    if failure_stage == "prepare":
        backend.query.assert_not_called()
        recording.frameset.assert_not_called()


def test_recorded_query_surfaces_backend_cleanup_failure(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies
):
    path, _ = map_directory
    _, backend = fake_query_dependencies
    backend.close.side_effect = OSError("feature handle close failed")
    with pytest.raises(OSError, match="feature handle close failed"):
        query.query_map(path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=1)
    backend.close.assert_called_once_with()


def test_recorded_query_reports_decode_and_inference_duration_separately(
    monkeypatch, map_directory, slam_dirs, tmp_path, fake_query_dependencies
):
    path, _ = map_directory
    _, backend = fake_query_dependencies
    ticks = iter([0.0, 2.0, 5.0, 7.0])
    monkeypatch.setattr(query.time, "monotonic", lambda: next(ticks))
    result = query.query_map(
        path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=1
    )
    item = result["queries"][0]
    assert item["query_duration_sec"] == 3.0
    assert item["query_end_to_end_duration_sec"] == 7.0
    assert result["initialization_durations_sec"] == {"map_load_duration_sec": 0.5}
    backend.prepare_map.assert_called_once_with(path)
    backend.close.assert_called_once_with()


@pytest.mark.parametrize("same_graph", [True, False])
def test_query_uses_heldout_frames_and_truth_only_for_same_graph_diagnostics(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, same_graph
):
    path, metadata = map_directory
    recording, backend = fake_query_dependencies
    if not same_graph:
        metadata["graph_uid"] = "different-map-frame"
        (path / "map.json").write_text(json.dumps(metadata))
    result = query.query_map(
        path, tmp_path / "recording.vrs", slam_dirs[0], num_queries=2
    )
    assert result["num_queries"] == 2
    assert result["num_successes"] == 1
    assert result["success_rate"] == 0.5
    assert result["same_mps_graph"] is same_graph
    assert [value["capture_timestamp_ns"] for value in result["queries"]] == [
        3_000_000,
        7_000_000,
    ]
    assert backend.query.call_count == 2
    success, failure = result["queries"]
    if same_graph:
        assert success["translation_error_m"] == pytest.approx(0.1)
        assert success["rotation_error_deg"] == pytest.approx(0)
    else:
        assert "translation_error_m" not in success
        assert "rotation_error_deg" not in success
    assert "translation_error_m" not in failure
    assert all(len(call.args) == 2 for call in backend.query.call_args_list)
    backend.prepare_map.assert_called_once_with(path)
    backend.close.assert_called_once_with()
    assert result["initialization_durations_sec"] == {"map_load_duration_sec": 0.5}
    assert all(
        item["query_end_to_end_duration_sec"] >= item["query_duration_sec"]
        for item in result["queries"]
    )
    recording.select_labels.assert_called_once_with("rgb")


def test_query_honors_stride_and_query_limit(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies
):
    path, _ = map_directory
    recording, backend = fake_query_dependencies
    result = query.query_map(
        path, tmp_path / "recording.vrs", slam_dirs[0], num_queries=1, stride=2
    )
    assert result["num_queries"] == 1
    assert result["queries"][0]["capture_timestamp_ns"] == 7_000_000
    assert [call.args[1] for call in recording.frameset.call_args_list] == [0, 2]
    assert backend.query.call_count == 1


def test_query_stops_immediately_at_requested_count(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies
):
    path, _ = map_directory
    recording, _ = fake_query_dependencies
    result = query.query_map(
        path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=1
    )
    assert result["num_queries"] == 1
    assert recording.frameset.call_count == 1


def test_query_rejects_recording_with_only_mapping_keyframes(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, make_view
):
    path, metadata = map_directory
    metadata["views"] = [make_view(str(i), index=i).to_dict() for i in range(3)]
    (path / "map.json").write_text(json.dumps(metadata))
    _, backend = fake_query_dependencies
    with pytest.raises(ValueError, match="No held-out"):
        query.query_map(path, tmp_path / "recording.vrs", slam_dirs[0])
    backend.query.assert_not_called()
    backend.close.assert_called_once_with()


@pytest.mark.parametrize(
    "kwargs", [{"mode": "all"}, {"num_queries": 0}, {"stride": 0}, {"stride": -1}]
)
def test_query_rejects_bad_options_before_files(tmp_path, kwargs):
    with pytest.raises(ValueError, match="positive"):
        query.query_map(tmp_path, tmp_path, tmp_path, **kwargs)


@pytest.mark.parametrize("rig_execution", ["fast", "", None])
def test_query_rejects_unknown_rig_execution_before_files(tmp_path, rig_execution):
    with pytest.raises(ValueError, match="rig_execution"):
        query.query_map(tmp_path, tmp_path, tmp_path, rig_execution=rig_execution)


@pytest.mark.parametrize("rig_execution", ["auto", "serial", "parallel"])
def test_query_cli_passes_rig_execution_to_the_backend(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, rig_execution
):
    path, _ = map_directory
    output = tmp_path / "rig.json"
    query.main(
        [
            "--map",
            str(path),
            "--vrs",
            str(tmp_path / "heldout.vrs"),
            "--mps",
            str(slam_dirs[0]),
            "--num-queries",
            "1",
            "--rig-execution",
            rig_execution,
            "--output",
            str(output),
        ]
    )
    query.HlocBackend.assert_called_once_with(rig_execution=rig_execution)
    assert json.loads(output.read_text())["rig_execution_requested"] == rig_execution


def test_query_cli_rejects_unknown_rig_execution(capsys):
    with pytest.raises(SystemExit) as error:
        query.main(
            ["--map", "m", "--vrs", "v", "--mps", "p", "--output", "o"]
            + ["--rig-execution", "fast"]
        )
    assert error.value.code == 2
    assert "--rig-execution" in capsys.readouterr().err


@pytest.mark.parametrize("module", [build, query])
def test_cli_requires_arguments(module):
    with pytest.raises(SystemExit) as error:
        module.main([])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid geometry"),
        OSError("unreadable input"),
        ImportError("optional HLOC absent"),
        subprocess.CalledProcessError(1, "git"),
    ],
)
def test_build_cli_reports_domain_and_dependency_errors(monkeypatch, capsys, error):
    monkeypatch.setattr(build, "build_map", Mock(side_effect=error))
    with pytest.raises(SystemExit) as failure:
        build.main(["--vrs", "a.vrs", "--mps", "mps", "--output-dir", "map"])
    assert failure.value.code == 2
    assert str(error) in capsys.readouterr().err


def test_build_cli_passes_options_and_prints_statistics(monkeypatch, capsys):
    run = Mock(return_value={"num_landmarks": 25})
    monkeypatch.setattr(build, "build_map", run)
    build.main(
        [
            "--vrs",
            "a.vrs",
            "b.vrs",
            "--mps",
            "a",
            "b",
            "--output-dir",
            "map",
            "--camera-set",
            "all",
            "--max-keyframes",
            "3",
        ]
    )
    run.assert_called_once_with(
        [Path("a.vrs"), Path("b.vrs")], [Path("a"), Path("b")], Path("map"), "all", 3
    )
    assert json.loads(capsys.readouterr().out) == {"num_landmarks": 25}


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad map"),
        OSError("missing input"),
        ImportError("HLOC absent"),
        subprocess.CalledProcessError(1, "git"),
    ],
)
def test_query_cli_reports_failures_without_output(
    monkeypatch, tmp_path, capsys, error
):
    monkeypatch.setattr(query, "query_map", Mock(side_effect=error))
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as failure:
        query.main(
            ["--map", "map", "--vrs", "a.vrs", "--mps", "mps", "--output", str(output)]
        )
    assert failure.value.code == 2
    assert str(error) in capsys.readouterr().err
    assert not output.exists()


def test_query_cli_writes_full_results_but_prints_only_summary(
    monkeypatch, tmp_path, capsys
):
    result = {"num_queries": 1, "queries": [{"success": False}]}
    run = Mock(return_value=result)
    monkeypatch.setattr(query, "query_map", run)
    output = tmp_path / "new" / "result.json"
    query.main(
        [
            "--map",
            "map",
            "--vrs",
            "a.vrs",
            "--mps",
            "mps",
            "--mode",
            "slam",
            "--num-queries",
            "2",
            "--stride",
            "3",
            "--output",
            str(output),
        ]
    )
    run.assert_called_once_with(
        Path("map"), Path("a.vrs"), Path("mps"), "slam", 2, 3, "auto"
    )
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out) == {"num_queries": 1}


def test_query_cli_never_overwrites_existing_results(monkeypatch, tmp_path, capsys):
    run = Mock(side_effect=AssertionError("must not run inference"))
    monkeypatch.setattr(query, "query_map", run)
    output = tmp_path / "existing.json"
    output.write_text("existing")
    with pytest.raises(SystemExit) as failure:
        query.main(
            ["--map", "map", "--vrs", "a.vrs", "--mps", "mps", "--output", str(output)]
        )
    assert failure.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert output.read_text() == "existing"
    run.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("views", None),
        ("views", [None]),
        ("source_vrs", None),
        ("source_vrs", []),
        ("source_vrs", [""]),
        ("source_vrs", [1]),
        ("source_vrs", ["same", "same"]),
        ("graph_uid", None),
        ("graph_uid", 1),
        ("graph_uid", ""),
        ("format_version", True),
    ],
)
def test_map_manifest_rejects_wrong_field_types_and_empty_identifiers(
    map_directory, field, value
):
    path, metadata = map_directory
    metadata[field] = value
    (path / "map.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        query.load_map(path)


@pytest.mark.parametrize("field", ["views", "source_vrs", "graph_uid"])
def test_map_manifest_missing_fields_are_domain_errors(map_directory, field):
    path, metadata = map_directory
    del metadata[field]
    (path / "map.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        query.load_map(path)


@pytest.mark.parametrize("field", ["camera", "T_world_camera", "time_offset_ns"])
def test_map_loader_translates_incomplete_view_records_to_value_error(
    map_directory, field
):
    path, metadata = map_directory
    del metadata["views"][0][field]
    (path / "map.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Invalid map view record"):
        query.load_map(path)


@pytest.mark.parametrize("folder", [None, 4, {}, [], ""])
def test_single_vrs_multi_slam_rejects_nonstring_or_empty_output_folder(
    tmp_path, folder
):
    (tmp_path / "vrs_to_multi_slam.json").write_text(json.dumps({"a.vrs": folder}))
    with pytest.raises(ValueError, match="output-folder strings"):
        build.resolve_mps([tmp_path / "a.vrs"], [tmp_path])


def test_one_recording_can_be_selected_from_multi_slam_root(tmp_path, trajectory_file):
    trajectory_file("multi/selected/slam/closed_loop_trajectory.csv")
    root = tmp_path / "multi"
    (root / "selected" / "slam" / "online_calibration.jsonl").touch()
    (root / "vrs_to_multi_slam.json").write_text(
        json.dumps({"/old/a.vrs": "selected", "/old/b.vrs": "not-selected"})
    )
    assert build.resolve_mps([tmp_path / "a.vrs"], [root]) == [
        root / "selected" / "slam"
    ]


def test_extract_views_maps_corrected_time_but_serializes_raw_capture(
    tmp_path, slam_dirs, frame_factory, camera
):
    trajectory = Trajectory(slam_dirs[0] / "closed_loop_trajectory.csv")
    corrected_ns = trajectory.timestamps_ns
    offset_ns = 10_000_000
    raw_ns = [value + offset_ns for value in corrected_ns]
    recording = SimpleNamespace(
        select_labels=lambda mode: ("camera-rgb",),
        timestamps_ns={"camera-rgb": raw_ns},
        corrected_timestamps_ns={"camera-rgb": corrected_ns},
        has_calibration=lambda label, index: True,
        frame=lambda label, index: frame_factory(
            index, raw_ns[index], label, offset_ns
        ),
    )
    assert all(trajectory.at(value) is None for value in raw_ns)
    views = build.extract_views(recording, trajectory, 0, "rgb", tmp_path, 0)
    assert [view.capture_timestamp_ns for view in views] == raw_ns
    assert [view.corrected_timestamp_ns for view in views] == corrected_ns
    assert [view.time_offset_ns for view in views] == [offset_ns] * 3
    for view, pose in zip(views, trajectory.poses):
        np.testing.assert_allclose(view.T_world_camera, pose @ camera.T_device_camera)
        assert view.to_dict()["corrected_timestamp_ns"] == view.corrected_timestamp_ns


@pytest.mark.parametrize(
    "cached_timestamp_ns",
    [-17_000_000, 3_000_000],
    ids=["cache-outside", "actual-frame-outside"],
)
def test_extract_views_never_uses_raw_pose_when_corrected_time_is_uncovered(
    tmp_path, slam_dirs, frame_factory, cached_timestamp_ns
):
    trajectory = Trajectory(slam_dirs[0] / "closed_loop_trajectory.csv")
    assert trajectory.at(3_000_000) is not None
    frame = Mock(return_value=frame_factory(0, 3_000_000, time_offset_ns=20_000_000))
    recording = SimpleNamespace(
        select_labels=lambda mode: ("camera-rgb",),
        timestamps_ns={"camera-rgb": [3_000_000]},
        corrected_timestamps_ns={"camera-rgb": [cached_timestamp_ns]},
        has_calibration=lambda label, index: True,
        frame=frame,
    )
    assert build.extract_views(recording, trajectory, 0, "rgb", tmp_path, 0) == []
    assert frame.call_count == (1 if cached_timestamp_ns == 3_000_000 else 0)
    assert not (tmp_path / "images").exists()


@pytest.mark.parametrize("offset_ns", [10_000_000, -10_000_000])
def test_query_ground_truth_uses_corrected_time_and_preserves_raw_capture(
    monkeypatch,
    map_directory,
    slam_dirs,
    tmp_path,
    fake_query_dependencies,
    frame_factory,
    offset_ns,
):
    path, _ = map_directory
    recording, backend = fake_query_dependencies
    raw_ns = 3_000_000 + offset_ns
    frame = frame_factory(1, raw_ns, time_offset_ns=offset_ns)
    recording.timestamps_ns = {"camera-rgb": [raw_ns]}
    recording.corrected_timestamps_ns = {"camera-rgb": [3_000_000]}
    recording.frameset.side_effect = lambda labels, index: [frame]
    trajectory = Trajectory(slam_dirs[0] / "closed_loop_trajectory.csv")
    lookup = Mock(wraps=trajectory.at)
    monkeypatch.setattr(trajectory, "at", lookup)
    monkeypatch.setattr(query, "Trajectory", lambda csv: trajectory)
    result = query.query_map(
        path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=1
    )
    diagnostic = result["queries"][0]
    lookup.assert_called_once_with(3_000_000)
    assert diagnostic["capture_timestamp_ns"] == raw_ns
    assert diagnostic["corrected_timestamp_ns"] == 3_000_000
    assert diagnostic["translation_error_m"] == pytest.approx(0.1)
    assert diagnostic["rotation_error_deg"] == pytest.approx(0)
    assert backend.query.call_args.args[1] == [frame]


def test_query_omits_truth_errors_when_only_raw_timestamp_has_pose(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, frame_factory
):
    path, _ = map_directory
    recording, _ = fake_query_dependencies
    recording.timestamps_ns = {"camera-rgb": [3_000_000]}
    recording.corrected_timestamps_ns = {"camera-rgb": [-17_000_000]}
    recording.frameset.side_effect = lambda labels, index: [
        frame_factory(1, 3_000_000, time_offset_ns=20_000_000)
    ]
    result = query.query_map(
        path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=1
    )
    diagnostic = result["queries"][0]
    assert diagnostic["success"] is True
    assert diagnostic["capture_timestamp_ns"] == 3_000_000
    assert diagnostic["corrected_timestamp_ns"] == -17_000_000
    assert "translation_error_m" not in diagnostic
    assert "rotation_error_deg" not in diagnostic


@pytest.mark.parametrize("mode", ["rgb", "slam"])
def test_query_cache_records_actual_vrs_hash_and_selected_mode(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, mode
):
    path, metadata = map_directory
    recording, _ = fake_query_dependencies
    vrs = tmp_path / "heldout.vrs"
    result = query.query_map(path, vrs, slam_dirs[0], mode=mode, num_queries=1)
    assert result["query_vrs_sha256"] == hashlib.sha256(vrs.read_bytes()).hexdigest()
    assert result["mode"] == mode
    assert result["map_graph_uid"] == metadata["graph_uid"]
    recording.select_labels.assert_called_once_with(mode)


@pytest.mark.parametrize(
    "hashes",
    [
        None,
        {},
        "a" * 64,
        [],
        ["a" * 64, "b" * 64],
        [None],
        [3],
        [""],
        ["a" * 63],
        ["a" * 65],
        ["A" * 64],
        ["g" * 64],
    ],
)
def test_map_rejects_malformed_source_fingerprints_before_inference(
    map_directory, hashes
):
    path, metadata = map_directory
    metadata["source_vrs_sha256"] = hashes
    (path / "map.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source_vrs_sha256"):
        query.load_map(path)


@pytest.mark.parametrize("sequence", [-1, 1, -0.5, 0.9, True, False, "0", None])
def test_map_view_sequence_must_index_recording_namespace(map_directory, sequence):
    path, metadata = map_directory
    metadata["views"][0]["sequence"] = sequence
    (path / "map.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sequence"):
        query.load_map(path)


def test_builder_hashes_each_recording_once_before_initializing_backend(
    monkeypatch, tmp_path, slam_dirs, fake_build_dependencies
):
    backend, _ = fake_build_dependencies
    paths = [tmp_path / "recording0.vrs", tmp_path / "recording1.vrs"]
    hashed = Mock(wraps=build.file_sha256)
    monkeypatch.setattr(build, "file_sha256", hashed)

    def construct():
        assert [call.args[0] for call in hashed.call_args_list] == paths
        return backend

    build.HlocBackend.side_effect = construct
    output = tmp_path / "fingerprinted-map"
    build.build_map(paths, slam_dirs, output, camera_set="rgb")
    manifest = json.loads((output / "map.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["source_vrs_sha256"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ]
    assert hashed.call_count == 2


def test_builder_missing_recording_fails_before_backend(
    monkeypatch, tmp_path, slam_dirs
):
    backend = Mock(side_effect=AssertionError("must not initialize inference"))
    monkeypatch.setattr(build, "HlocBackend", backend)
    with pytest.raises(FileNotFoundError):
        build.build_map(
            [tmp_path / "missing.vrs"], slam_dirs[:1], tmp_path / "missing-map"
        )
    backend.assert_not_called()
    assert not (tmp_path / "missing-map").exists()


def test_portable_renamed_recording_excludes_keyframes_and_hashes_once(
    monkeypatch, map_directory, slam_dirs, tmp_path, fake_query_dependencies
):
    path, metadata = map_directory
    recording, backend = fake_query_dependencies
    original = tmp_path / "recording.vrs"
    moved = tmp_path / "renamed.vrs"
    moved.write_bytes(original.read_bytes())
    metadata["source_vrs"] = ["/producer-only/recording.vrs"]
    metadata["source_vrs_sha256"] = [hashlib.sha256(original.read_bytes()).hexdigest()]
    (path / "map.json").write_text(json.dumps(metadata))
    hashed = Mock(wraps=query.file_sha256)
    monkeypatch.setattr(query, "file_sha256", hashed)
    result = query.query_map(path, moved, slam_dirs[0], num_queries=2)
    hashed.assert_called_once_with(moved)
    assert result["source_identity_check"] == "sha256"
    assert result["query_in_mapping_recordings"] is True
    assert result["query_vrs_sha256"] == metadata["source_vrs_sha256"][0]
    assert [call.args[1][0].index for call in backend.query.call_args_list] == [1, 2]
    assert recording.frameset.call_count == 3


def test_same_producer_path_with_different_bytes_is_not_mapping_recording(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies
):
    path, metadata = map_directory
    _, backend = fake_query_dependencies
    source = tmp_path / "recording.vrs"
    metadata["source_vrs_sha256"] = [hashlib.sha256(source.read_bytes()).hexdigest()]
    source.write_bytes(b"different recording at the same pathname")
    (path / "map.json").write_text(json.dumps(metadata))
    result = query.query_map(path, source, slam_dirs[0], num_queries=1)
    assert result["query_in_mapping_recordings"] is False
    assert result["source_identity_check"] == "sha256"
    assert backend.query.call_args.args[1][0].index == 0


def test_legacy_unmatched_path_reports_unknown_independence_and_warns_once(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, caplog
):
    path, _ = map_directory
    result = query.query_map(
        path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=3
    )
    assert result["source_identity_check"] == "legacy_path"
    assert result["query_in_mapping_recordings"] is None
    warnings = [
        record
        for record in caplog.records
        if "cannot establish cross-recording independence" in record.message
    ]
    assert len(warnings) == 1


def test_fingerprint_keyframe_lookup_matches_all_duplicate_content_namespaces(
    monkeypatch, tmp_path, make_view, frame_factory
):
    source = tmp_path / "copy.vrs"
    source.write_bytes(b"same source in two mapping namespaces")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "source_vrs": ["/producer/a.vrs", "/producer/b.vrs", "/producer/c.vrs"],
        "source_vrs_sha256": [digest, digest, "0" * 64],
        "views": [
            make_view("a.png", sequence=0, index=0).to_dict(),
            make_view("b.png", sequence=1, index=5).to_dict(),
            make_view("c.png", sequence=2, index=7).to_dict(),
        ],
    }
    hashed = Mock(wraps=query.file_sha256)
    monkeypatch.setattr(query, "file_sha256", hashed)
    assert query.is_keyframe([frame_factory(5, 1_000_000)], source, metadata)
    hashed.assert_called_once_with(source)
    assert query.is_keyframe(
        [frame_factory(0, 1_000_000)], source, metadata, source_sha256=digest
    )
    assert not query.is_keyframe(
        [frame_factory(7, 1_000_000)], source, metadata, source_sha256=digest
    )
    assert not query.is_keyframe(
        [frame_factory(5, 1_000_000, "slam-front-left")],
        source,
        metadata,
        source_sha256=digest,
    )
    assert hashed.call_count == 1


def test_builder_rejects_duplicate_recording_bytes_under_new_name(
    tmp_path, slam_dirs, fake_build_dependencies
):
    backend, _ = fake_build_dependencies
    copy = tmp_path / "recording9.vrs"
    copy.write_bytes((tmp_path / "recording0.vrs").read_bytes())
    output = tmp_path / "duplicate-map"
    with pytest.raises(ValueError, match="distinct VRS SHA-256"):
        build.build_map([tmp_path / "recording0.vrs", copy], slam_dirs, output)
    build.HlocBackend.assert_not_called()
    assert not output.exists()


def test_extract_views_skips_uncalibrated_frames_and_spreads_keyframe_cap(
    tmp_path, frame_factory
):
    timestamps_ns = list(range(10))
    poses = {
        t: transform_from_quaternion([1, 0, 0, 0], [0.2 * t, 0, 0])
        for t in timestamps_ns
    }
    decoded = []

    def frame(label, index):
        decoded.append(index)
        return frame_factory(index, timestamps_ns[index], label)

    recording = SimpleNamespace(
        select_labels=lambda mode: ("camera-rgb",),
        timestamps_ns={"camera-rgb": timestamps_ns},
        corrected_timestamps_ns={"camera-rgb": timestamps_ns},
        has_calibration=lambda label, index: index != 0,
        frame=frame,
    )
    trajectory = SimpleNamespace(at=lambda timestamp_ns: poses.get(timestamp_ns))
    views = build.extract_views(recording, trajectory, 0, "rgb", tmp_path, 3)
    assert decoded == [1, 5, 9]
    assert [view.index for view in views] == [1, 5, 9]


def test_query_records_unusable_anchor_and_skips_uncalibrated_anchor(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies, frame_factory
):
    path, _ = map_directory
    recording, backend = fake_query_dependencies
    recording.has_calibration = lambda label, index: index != 0

    def frameset(labels, index):
        if index == 2:
            raise ValueError("SLAM frameset spans more than 2 ms")
        return [frame_factory(index, recording.timestamps_ns["camera-rgb"][index])]

    recording.frameset = Mock(side_effect=frameset)
    result = query.query_map(path, tmp_path / "heldout.vrs", slam_dirs[0])
    assert [call.args[1] for call in recording.frameset.call_args_list] == [1, 2]
    assert result["num_queries"] == 2
    failed = result["queries"][1]
    assert failed["success"] is False
    assert failed["reason"] == "invalid_frames"
    assert failed["anchor_index"] == 2
    assert failed["corrected_timestamp_ns"] == 7_000_000
    assert backend.query.call_count == 1


def test_invalid_frames_consume_requested_query_quota(
    map_directory, slam_dirs, tmp_path, fake_query_dependencies
):
    path, _ = map_directory
    recording, backend = fake_query_dependencies
    recording.has_calibration = lambda label, index: index != 0
    recording.frameset = Mock(side_effect=ValueError("unsynchronized companion"))
    result = query.query_map(
        path, tmp_path / "heldout.vrs", slam_dirs[0], num_queries=1
    )
    assert result["num_queries"] == 1
    assert result["num_successes"] == 0
    assert result["queries"][0]["reason"] == "invalid_frames"
    assert result["queries"][0]["anchor_index"] == 1
    assert [call.args[1] for call in recording.frameset.call_args_list] == [1]
    backend.query.assert_not_called()
