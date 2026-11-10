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

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import replay_demo as replay
from aria_utils import Frame
from map_utils import file_sha256, transform_from_quaternion
from pose_alignment import PoseAlignment, VioSample
from visualization import PoseWriter


def transform_value(matrix):
    return SimpleNamespace(to_matrix=lambda: matrix)


@pytest.mark.parametrize("high_frequency", [True, False])
def test_read_vio_uses_official_fields_and_composes_regular_bodyimu_extrinsics(
    high_frequency,
):
    odometry = transform_from_quaternion([1, 0, 0, 1], [1, 2, 3])
    device = transform_from_quaternion([1, 0, 0, 0], [0.1, 0.2, 0.3])
    provider = Mock()
    provider.get_stream_id_from_label.side_effect = (
        lambda label: "high"
        if label == "vio_high_frequency" and high_frequency
        else ("regular" if label == "vio" else None)
    )
    provider.get_num_data.return_value = 2
    provider.get_vio_high_freq_data_by_index.side_effect = (
        lambda stream, index: SimpleNamespace(
            tracking_timestamp=timedelta(milliseconds=index + 1),
            transform_odometry_device=transform_value(odometry),
            session_uid="high-session",
            quality_score=0.5,
        )
    )
    provider.get_vio_data_by_index.side_effect = lambda stream, index: SimpleNamespace(
        capture_timestamp_ns=(index + 1) * 1_000_000,
        transform_odometry_bodyimu=transform_value(odometry),
        transform_bodyimu_device=transform_value(device),
        frontend_session_uid="regular-session",
        pose_quality="PoseQuality.GOOD",
    )
    samples = replay.read_vio(SimpleNamespace(provider=provider))
    assert [sample.timestamp_ns for sample in samples] == [1_000_000, 2_000_000]
    assert all(
        sample.session_uid == ("high-session" if high_frequency else "regular-session")
        for sample in samples
    )
    assert all(sample.quality_good is True for sample in samples)
    np.testing.assert_allclose(
        samples[0].T_odometry_device, odometry if high_frequency else odometry @ device
    )
    if high_frequency:
        provider.get_vio_data_by_index.assert_not_called()
    else:
        provider.get_vio_high_freq_data_by_index.assert_not_called()


@pytest.mark.parametrize("quality", [0.499, float("nan"), float("inf")])
def test_bad_high_frequency_quality_is_python_bool_ignored_by_aligner(quality):
    provider = Mock()
    provider.get_stream_id_from_label.return_value = "high"
    provider.get_num_data.return_value = 2
    provider.get_vio_high_freq_data_by_index.side_effect = (
        lambda stream, index: SimpleNamespace(
            tracking_timestamp=timedelta(milliseconds=index),
            transform_odometry_device=transform_value(np.full((4, 4), np.nan)),
            session_uid="session-a",
            quality_score=quality,
        )
    )
    samples = replay.read_vio(SimpleNamespace(provider=provider))
    aligner = PoseAlignment()
    assert all(sample.quality_good is False for sample in samples)
    assert [aligner.add_vio(sample) for sample in samples] == [None, None]
    assert aligner.state == "lost"


def test_empty_high_frequency_stream_falls_back_to_regular_bad_quality():
    provider = Mock()
    provider.get_stream_id_from_label.side_effect = lambda label: label
    provider.get_num_data.side_effect = (
        lambda stream: 0 if stream == "vio_high_frequency" else 2
    )
    provider.get_vio_data_by_index.side_effect = lambda stream, index: SimpleNamespace(
        capture_timestamp_ns=index * 1_000_000,
        transform_odometry_bodyimu=transform_value(np.eye(4)),
        transform_bodyimu_device=transform_value(np.eye(4)),
        frontend_session_uid="regular",
        pose_quality="PoseQuality.BAD",
    )
    samples = replay.read_vio(SimpleNamespace(provider=provider))
    assert all(sample.quality_good is False for sample in samples)
    provider.get_vio_high_freq_data_by_index.assert_not_called()


@pytest.mark.parametrize(
    "case", ["missing", "empty", "single", "duplicate", "descending"]
)
def test_read_vio_requires_stream_and_strict_time_coverage(case):
    provider = Mock()
    provider.get_stream_id_from_label.return_value = (
        None if case == "missing" else "stream"
    )
    provider.get_num_data.return_value = (
        0 if case == "empty" else (1 if case == "single" else 2)
    )
    timestamps_us = [1000, 1000] if case == "duplicate" else [2000, 1000]
    provider.get_vio_high_freq_data_by_index.side_effect = (
        lambda stream, index: SimpleNamespace(
            tracking_timestamp=timedelta(microseconds=timestamps_us[index]),
            transform_odometry_device=transform_value(np.eye(4)),
            session_uid="session",
            quality_score=1.0,
        )
    )
    with pytest.raises(ValueError, match="VIO stream|increasing"):
        replay.read_vio(SimpleNamespace(provider=provider))


@pytest.fixture
def vio_samples():
    return [
        VioSample(
            index * 10_000_000,
            transform_from_quaternion([1, 0, 0, 0], [index * 0.1, 0, 0]),
            "session-a",
        )
        for index in range(13)
    ]


def query_fix(timestamp_ns, duration_sec, success=True):
    return {
        "success": success,
        "corrected_timestamp_ns": timestamp_ns,
        "capture_timestamp_ns": timestamp_ns + 1_000_000,
        "query_duration_sec": duration_sec,
        "T_world_device": transform_from_quaternion([1, 0, 0, 1], [5, -2, 1]).tolist(),
    }


def test_schedule_measured_single_worker_selects_latest_and_drops_stale_backlog(
    vio_samples,
):
    queries = [
        query_fix(30_000_000, 0.01),
        query_fix(0, 0.035),
        query_fix(10_000_000, 0.1),
        query_fix(20_000_000, 0.1),
        query_fix(40_000_000, 0.01),
        query_fix(110_000_000, 0.02),
        query_fix(50_000_000, 0, False),
        query_fix(1_000_000_000, 0),
    ]
    scheduled = replay.schedule_fixes(queries, vio_samples)
    assert [entry[0] for entry in scheduled] == [
        35_000_000,
        45_000_000,
        55_000_000,
        55_000_000,
    ]
    assert [entry[1]["corrected_timestamp_ns"] for entry in scheduled] == [
        0,
        30_000_000,
        40_000_000,
        50_000_000,
    ]
    assert all(entry[2] == "session-a" for entry in scheduled)
    assert all(entry[0] > entry[1]["corrected_timestamp_ns"] for entry in scheduled)


@pytest.mark.parametrize("duration_sec", [-0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("success", [True, False])
def test_schedule_rejects_invalid_measured_duration(vio_samples, duration_sec, success):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replay.schedule_fixes([query_fix(0, duration_sec, success)], vio_samples)


def test_failed_query_consumes_worker_time_but_never_becomes_a_pose_fix(vio_samples):
    failed = query_fix(0, 0.05, False)
    del failed["T_world_device"]
    scheduled = replay.schedule_fixes(
        [
            failed,
            query_fix(10_000_000, 0),
            query_fix(20_000_000, 0),
            query_fix(40_000_000, 0.01),
        ],
        vio_samples,
    )
    assert len(scheduled) == 2
    assert scheduled[0][0] == 50_000_000
    assert scheduled[0][1] is failed
    assert scheduled[1][0] == 60_000_000
    assert scheduled[1][1]["corrected_timestamp_ns"] == 40_000_000
    assert scheduled[1][1]["success"] is True


def test_schedule_retains_capture_session_and_zero_duration(vio_samples):
    vio_samples[6:] = [
        VioSample(sample.timestamp_ns, sample.T_odometry_device, "session-b")
        for sample in vio_samples[6:]
    ]
    scheduled = replay.schedule_fixes(
        [query_fix(50_000_000, 0.03), query_fix(90_000_000, 0)], vio_samples
    )
    assert [(arrival_ns, session) for arrival_ns, _, session in scheduled] == [
        (80_000_000, "session-a"),
        (90_000_000, "session-b"),
    ]
    assert replay.schedule_fixes([], vio_samples) == []


@pytest.fixture
def replay_inputs(monkeypatch, tmp_path, camera, vio_samples):
    map_path = tmp_path / "map"
    slam = tmp_path / "mps" / "slam"
    slam.mkdir(parents=True)
    (slam / "closed_loop_trajectory.csv").write_text(
        "POISON: MPS poses must never enter alignment"
    )
    (slam / "online_calibration.jsonl").touch()
    fixes_path = tmp_path / "fixes.json"
    vrs = tmp_path / "input.vrs"
    vrs.write_bytes(b"synthetic VRS recording A")
    cache_identity = {
        "map_graph_uid": "map-graph",
        "mode": "rgb",
        "query_vrs_sha256": file_sha256(vrs),
    }
    fixes_path.write_text(
        json.dumps({**cache_identity, "queries": [query_fix(0, 0.02)]})
    )
    monkeypatch.setattr(
        replay, "load_map", Mock(return_value={"graph_uid": "map-graph"})
    )
    provider = Mock()
    provider.get_stream_id_from_label.return_value = "high"
    provider.get_num_data.return_value = len(vio_samples)
    provider.get_vio_high_freq_data_by_index.side_effect = (
        lambda stream, index: SimpleNamespace(
            tracking_timestamp=timedelta(
                microseconds=vio_samples[index].timestamp_ns // 1000
            ),
            transform_odometry_device=transform_value(
                vio_samples[index].T_odometry_device
            ),
            session_uid=vio_samples[index].session_uid,
            quality_score=1.0,
        )
    )
    timestamps_ns = [5_000_000, 50_000_000, 200_000_000]
    recording = SimpleNamespace(
        provider=provider,
        select_labels=Mock(return_value=("camera-rgb",)),
        corrected_timestamps_ns={"camera-rgb": timestamps_ns},
        has_calibration=lambda label, index: True,
        frame=Mock(
            side_effect=lambda label, index: Frame(
                label,
                index,
                timestamps_ns[index] + 1_000_000,
                np.zeros((2, 3), dtype=np.uint8),
                camera,
                1_000_000,
            )
        ),
    )
    open_recording = Mock(return_value=recording)
    monkeypatch.setattr(replay, "Recording", open_recording)
    viewer = Mock(spec_set=["on_pose", "on_image", "on_status", "close"])
    open_viewer = Mock(return_value=viewer)
    monkeypatch.setattr(replay, "RerunView", open_viewer)
    writers = []

    def open_writer(path):
        writer = PoseWriter(path)
        writers.append(writer)
        return writer

    monkeypatch.setattr(replay, "PoseWriter", open_writer)
    return SimpleNamespace(
        map_path=map_path,
        vrs=vrs,
        cache_identity=cache_identity,
        mps=slam.parent,
        slam=slam,
        fixes=fixes_path,
        output=tmp_path / "output",
        recording=recording,
        open_recording=open_recording,
        viewer=viewer,
        open_viewer=open_viewer,
        writers=writers,
    )


def run_replay(inputs):
    return replay.replay(
        inputs.map_path, inputs.vrs, inputs.mps, inputs.fixes, inputs.output
    )


@pytest.mark.parametrize("mode", ["rgb", "slam"])
def test_replay_real_alignment_outputs_every_vio_pose_after_delayed_fix(
    replay_inputs, vio_samples, mode
):
    inputs = replay_inputs
    cache = json.loads(inputs.fixes.read_text())
    cache["mode"] = mode
    inputs.fixes.write_text(json.dumps(cache))
    stats = run_replay(inputs)
    rows = [
        json.loads(line)
        for line in (inputs.output / "poses.jsonl").read_text().splitlines()
    ]
    assert stats["vio_samples"] == 13
    assert stats["scheduled_queries"] == stats["accepted_fixes"] == 1
    assert stats["mapped_pose_samples"] == len(rows) == 10
    assert stats["first_mapped_pose_timestamp_ns"] == 30_000_000
    assert stats["last_mapped_pose_timestamp_ns"] == 120_000_000
    assert stats["time_to_first_mapped_pose_sec"] == pytest.approx(0.03)
    assert stats["mapped_pose_device_rate_hz"] == pytest.approx(100)
    assert stats["gui_max_rate_hz"] == 30
    map_odometry = np.array(query_fix(0, 0)["T_world_device"])
    for row, sample in zip(rows, vio_samples[3:]):
        np.testing.assert_allclose(
            row["T_map_device"], map_odometry @ sample.T_odometry_device
        )
        assert row["fix_age_ns"] == sample.timestamp_ns
    inputs.open_recording.assert_called_once_with(
        inputs.vrs, inputs.slam / "online_calibration.jsonl"
    )
    assert inputs.recording.frame.call_count == 2
    assert [call.args[1] for call in inputs.viewer.on_image.call_args_list] == [
        5_000_000,
        50_000_000,
    ]
    assert inputs.viewer.on_pose.call_count == 10
    assert json.loads((inputs.output / "replay_stats.json").read_text()) == stats
    inputs.viewer.close.assert_called_once()
    assert inputs.writers[0].target.closed


def test_wrong_map_graph_rejected_before_provider_or_sinks(replay_inputs):
    inputs = replay_inputs
    inputs.fixes.write_text(json.dumps({"map_graph_uid": "other", "queries": []}))
    inputs.vrs.unlink()
    with pytest.raises(ValueError, match="map_graph_uid"):
        run_replay(inputs)
    inputs.open_recording.assert_not_called()
    inputs.open_viewer.assert_not_called()
    assert not inputs.output.exists()


@pytest.mark.parametrize(
    "queries",
    [
        [],
        [query_fix(0, 0, False)],
        [query_fix(0, 1)],
        [query_fix(120_000_000, 0)],
        [query_fix(110_000_000, 0)],
    ],
)
def test_zero_or_single_pose_replay_fails_without_success_stats(replay_inputs, queries):
    inputs = replay_inputs
    inputs.fixes.write_text(json.dumps({**inputs.cache_identity, "queries": queries}))
    with pytest.raises(ValueError, match="fewer than two"):
        run_replay(inputs)
    inputs.viewer.close.assert_called_once()
    assert inputs.writers[0].target.closed
    assert not (inputs.output / "replay_stats.json").exists()


def test_delayed_fix_from_retired_session_is_rejected(replay_inputs, vio_samples):
    for index in range(6, len(vio_samples)):
        sample = vio_samples[index]
        vio_samples[index] = VioSample(
            sample.timestamp_ns, sample.T_odometry_device, "session-b"
        )
    replay_inputs.fixes.write_text(
        json.dumps({**replay_inputs.cache_identity, "queries": [query_fix(0, 0.08)]})
    )
    with pytest.raises(ValueError, match="fewer than two"):
        run_replay(replay_inputs)
    status = replay_inputs.viewer.on_status.call_args.args[0]
    assert status["fix_accepted"] is False
    assert status["state"] == "lost"
    assert replay_inputs.writers[0].count == 0


@pytest.mark.parametrize("failure", ["on_pose", "close"])
def test_replay_closes_pose_file_even_when_viewer_fails(replay_inputs, failure):
    getattr(replay_inputs.viewer, failure).side_effect = OSError("viewer failed")
    with pytest.raises(OSError, match="viewer failed"):
        run_replay(replay_inputs)
    replay_inputs.viewer.close.assert_called_once()
    assert replay_inputs.writers[0].target.closed
    assert not (replay_inputs.output / "replay_stats.json").exists()


def test_replay_closes_writer_if_viewer_construction_fails(replay_inputs):
    replay_inputs.open_viewer.side_effect = ImportError("rerun unavailable")
    with pytest.raises(ImportError, match="rerun unavailable"):
        run_replay(replay_inputs)
    assert replay_inputs.writers[0].target.closed


def test_replay_does_not_overwrite_output_directory(replay_inputs):
    replay_inputs.output.mkdir()
    sentinel = replay_inputs.output / "existing"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        run_replay(replay_inputs)
    assert sentinel.read_text() == "unchanged"
    replay_inputs.open_viewer.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        OSError("read failed"),
        ValueError("invalid input"),
        ImportError("PAT unavailable"),
    ],
)
def test_replay_cli_errors_are_actionable(monkeypatch, capsys, error):
    monkeypatch.setattr(replay, "replay", Mock(side_effect=error))
    with pytest.raises(SystemExit) as failure:
        replay.main(
            [
                "--map",
                "map",
                "--vrs",
                "a.vrs",
                "--mps",
                "mps",
                "--fixes",
                "fixes.json",
                "--output-dir",
                "output",
            ]
        )
    assert failure.value.code == 2
    assert str(error) in capsys.readouterr().err


def test_replay_cli_passes_explicit_paths_and_prints_stats(monkeypatch, capsys):
    run = Mock(return_value={"mapped_pose_samples": 100})
    monkeypatch.setattr(replay, "replay", run)
    replay.main(
        [
            "--map",
            "map",
            "--vrs",
            "a.vrs",
            "--mps",
            "mps",
            "--fixes",
            "fixes.json",
            "--output-dir",
            "output",
        ]
    )
    run.assert_called_once_with(
        Path("map"), Path("a.vrs"), Path("mps"), Path("fixes.json"), Path("output")
    )
    assert json.loads(capsys.readouterr().out) == {"mapped_pose_samples": 100}
    with pytest.raises(SystemExit) as failure:
        replay.main([])
    assert failure.value.code == 2


def test_replay_failed_solve_uses_measured_time_without_applying_missing_pose(
    replay_inputs,
):
    failed = query_fix(0, 0.02, False)
    del failed["T_world_device"]
    replay_inputs.fixes.write_text(
        json.dumps(
            {
                **replay_inputs.cache_identity,
                "queries": [failed, query_fix(30_000_000, 0.01)],
            }
        )
    )
    stats = run_replay(replay_inputs)
    assert stats["scheduled_queries"] == 2
    assert stats["accepted_fixes"] == 1
    assert stats["first_mapped_pose_timestamp_ns"] == 50_000_000
    assert stats["mapped_pose_samples"] == 8
    assert [
        call.args[0]["fix_accepted"]
        for call in replay_inputs.viewer.on_status.call_args_list
    ] == [False, True]


@pytest.mark.parametrize(
    "fixes",
    [
        [],
        {"map_graph_uid": "map-graph"},
        {"map_graph_uid": "map-graph", "queries": None},
        {"map_graph_uid": "map-graph", "queries": [None]},
        {"map_graph_uid": "map-graph", "queries": [{"success": True}]},
    ],
)
def test_replay_rejects_malformed_cache_before_creating_sinks(replay_inputs, fixes):
    if isinstance(fixes, dict):
        fixes = {**replay_inputs.cache_identity, **fixes}
    replay_inputs.fixes.write_text(json.dumps(fixes))
    with pytest.raises(ValueError):
        run_replay(replay_inputs)
    replay_inputs.open_viewer.assert_not_called()
    assert not replay_inputs.output.exists()


def test_replay_cache_from_different_vrs_same_graph_and_clock_is_rejected_before_fusion(
    monkeypatch, replay_inputs
):
    inputs = replay_inputs
    common = b"same recording prefix" * 110_000
    inputs.vrs.write_bytes(common + b"A")
    other_vrs = inputs.vrs.with_name("other.vrs")
    other_vrs.write_bytes(common + b"B")
    cache = {
        **inputs.cache_identity,
        "query_vrs_sha256": file_sha256(inputs.vrs),
        "queries": [query_fix(0, 0.02)],
    }
    inputs.fixes.write_text(json.dumps(cache))
    assert file_sha256(other_vrs) != cache["query_vrs_sha256"]
    schedule = Mock(
        side_effect=AssertionError("must not schedule another recording's fixes")
    )
    align = Mock(side_effect=AssertionError("must not fuse another recording's pose"))
    monkeypatch.setattr(replay, "schedule_fixes", schedule)
    monkeypatch.setattr(replay, "PoseAlignment", align)
    with pytest.raises(ValueError, match="SHA-256"):
        replay.replay(
            inputs.map_path, other_vrs, inputs.mps, inputs.fixes, inputs.output
        )
    inputs.open_recording.assert_not_called()
    schedule.assert_not_called()
    align.assert_not_called()
    inputs.open_viewer.assert_not_called()
    assert not inputs.output.exists()


@pytest.mark.parametrize("fingerprint", [None, "", "not-a-sha256"])
def test_replay_requires_matching_recording_fingerprint_before_provider(
    replay_inputs, fingerprint
):
    cache = json.loads(replay_inputs.fixes.read_text())
    if fingerprint is None:
        del cache["query_vrs_sha256"]
    else:
        cache["query_vrs_sha256"] = fingerprint
    replay_inputs.fixes.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="SHA-256"):
        run_replay(replay_inputs)
    replay_inputs.open_recording.assert_not_called()
    assert not replay_inputs.output.exists()


@pytest.mark.parametrize("mode", [None, "all", "RGB", 3])
def test_replay_rejects_missing_or_invalid_cached_mode_before_provider(
    replay_inputs, mode
):
    cache = json.loads(replay_inputs.fixes.read_text())
    if mode is None:
        del cache["mode"]
    else:
        cache["mode"] = mode
    replay_inputs.fixes.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="rgb or slam"):
        run_replay(replay_inputs)
    replay_inputs.open_recording.assert_not_called()
    assert not replay_inputs.output.exists()


def test_replay_disk_write_failure_closes_both_owned_sinks(monkeypatch, replay_inputs):
    original_writer = replay.PoseWriter

    def failing_writer(path):
        writer = original_writer(path)
        writer.write = Mock(side_effect=OSError("disk full"))
        return writer

    monkeypatch.setattr(replay, "PoseWriter", failing_writer)
    with pytest.raises(OSError, match="disk full"):
        run_replay(replay_inputs)
    replay_inputs.viewer.close.assert_called_once()
    assert replay_inputs.writers[0].target.closed
    assert not (replay_inputs.output / "replay_stats.json").exists()
