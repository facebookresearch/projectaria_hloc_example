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

"""Two-map/third-query contracts with synthetic inputs, not Pilot measurements."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import call, Mock

import build_map as build
import h5py
import numpy as np
import pytest
import query_map as query
from aria_utils import Frame, Recording
from map_utils import file_sha256, Trajectory, View


def recording_with_frames(camera, timestamps_us, pixel):
    frames = [
        Frame(
            "camera-rgb",
            index,
            timestamp_us * 1000,
            np.full((camera.height, camera.width, 3), pixel, dtype=np.uint8),
            camera,
        )
        for index, timestamp_us in enumerate(timestamps_us)
    ]
    recording = Mock(spec=Recording)
    recording.select_labels.return_value = ("camera-rgb",)
    recording.timestamps_ns = {
        "camera-rgb": [frame.capture_timestamp_ns for frame in frames]
    }
    recording.corrected_timestamps_ns = recording.timestamps_ns
    recording.frame.side_effect = lambda label, index: frames[index]
    recording.frameset.side_effect = lambda labels, index: [frames[index]]
    return recording, frames


@pytest.fixture
def pilot_inputs(tmp_path, trajectory_file, camera):
    inputs = {}
    for number in (0, 2, 1):
        name = f"play_{number}"
        folder = tmp_path / "data/pilot" / name
        timestamps_us = (
            [101_000, 135_000, 169_000]
            if number == 1
            else [
                1000 + number * 10_000,
                3000 + number * 10_000,
                7000 + number * 10_000,
            ]
        )
        translations = [
            (5.0 + index, 0, 0) if number == 1 else (0.2 * index, 0.1 * number, 0)
            for index in range(3)
        ]
        csv = trajectory_file(
            f"data/pilot/{name}/mps/slam/closed_loop_trajectory.csv",
            timestamps_us=timestamps_us,
            translations=translations,
        )
        (csv.parent / "online_calibration.jsonl").write_text(
            f"synthetic {name} calibration"
        )
        vrs = folder / "video.vrs"
        vrs.write_bytes(f"synthetic distinct {name} VRS".encode())
        extrinsics = camera.T_device_camera.copy()
        extrinsics[0, 3] += number * 0.01
        calibrated_camera = replace(
            camera, focal=camera.focal + number, T_device_camera=extrinsics
        )
        frame_timestamps_us = (
            [101_000 + index * 1000 for index in range(69)]
            if number == 1
            else timestamps_us
        )
        recording, frames = recording_with_frames(
            calibrated_camera, frame_timestamps_us, number
        )
        inputs[name] = SimpleNamespace(
            vrs=vrs,
            mps=folder / "mps",
            csv=csv,
            recording=recording,
            frames=frames,
            camera=calibrated_camera,
        )
    return inputs


@pytest.fixture
def mock_pipeline(monkeypatch, pilot_inputs, write_h5):
    recordings = {source.vrs: source.recording for source in pilot_inputs.values()}
    mapping_reader = Mock(side_effect=lambda vrs, calibration: recordings[vrs])
    query_reader = Mock(side_effect=lambda vrs, calibration: recordings[vrs])
    monkeypatch.setattr(build, "Recording", mapping_reader)
    monkeypatch.setattr(query, "Recording", query_reader)
    mapping_backend = Mock()

    def build_artifacts(output, views, pairs):
        descriptors = {
            view.name: [1.0, index + 1.0] for index, view in enumerate(views)
        }
        write_h5(output / "global.h5", descriptors)
        write_h5(
            output / "features.h5",
            {view.name: [[0, 0]] for view in views},
            dataset="keypoints",
        )
        (output / "sfm").mkdir()
        return {"num_keyframes": len(views)}

    mapping_backend.build.side_effect = build_artifacts
    builder = Mock(return_value=mapping_backend)
    monkeypatch.setattr(build, "HlocBackend", builder)
    query_backend = Mock()
    query_backend.prepare_map.return_value = {"map_load_duration_sec": 0.0}
    pose = np.eye(4)
    pose[0, 3] = 5.1
    query_backend.query.side_effect = lambda map_path, frames: (
        {"success": False, "reason": "synthetic_no_matches"}
        if frames[0].index == 34
        else {"success": True, "T_world_device": pose.tolist()}
    )
    monkeypatch.setattr(query, "HlocBackend", Mock(return_value=query_backend))
    return SimpleNamespace(
        mapping_reader=mapping_reader,
        query_reader=query_reader,
        mapping_backend=mapping_backend,
        query_backend=query_backend,
        builder=builder,
    )


@pytest.fixture
def multi_map(tmp_path, pilot_inputs, mock_pipeline):
    output = tmp_path / "maps/pilot_play02_rgb"
    mapping = [pilot_inputs[name] for name in ("play_0", "play_2")]
    build.build_map(
        [source.vrs for source in mapping],
        [source.mps for source in mapping],
        output,
        camera_set="rgb",
        max_keyframes=0,
    )
    return output


def test_two_mapping_namespaces_preserve_source_order_and_calibration(
    multi_map, pilot_inputs, mock_pipeline
):
    mapping = [pilot_inputs[name] for name in ("play_0", "play_2")]
    metadata = query.load_map(multi_map)
    expected_hashes = [file_sha256(source.vrs) for source in mapping]
    assert len(set(expected_hashes)) == 2
    assert metadata["source_vrs_sha256"] == expected_hashes
    assert metadata["source_vrs"] == [str(source.vrs.resolve()) for source in mapping]
    assert file_sha256(pilot_inputs["play_1"].vrs) not in expected_hashes
    assert metadata["camera_set"] == "rgb"
    assert metadata["graph_uid"] == "graph-A"
    assert mock_pipeline.mapping_reader.call_args_list == [
        call(source.vrs, source.mps / "slam/online_calibration.jsonl")
        for source in mapping
    ]
    views = [View.from_dict(value) for value in metadata["views"]]
    assert [view.sequence for view in views] == [0, 0, 0, 1, 1, 1]
    for sequence, source in enumerate(mapping):
        selected = [view for view in views if view.sequence == sequence]
        assert [view.name for view in selected] == [
            f"s{sequence:03d}/camera-rgb/{index:08d}.png" for index in range(3)
        ]
        assert [view.capture_timestamp_ns for view in selected] == [
            frame.capture_timestamp_ns for frame in source.frames
        ]
        source.recording.select_labels.assert_called_once_with("rgb")
        for view, pose in zip(selected, Trajectory(source.csv).poses):
            assert view.camera.focal == source.camera.focal
            np.testing.assert_allclose(
                view.camera.T_device_camera, source.camera.T_device_camera
            )
            np.testing.assert_allclose(
                view.T_world_camera, pose @ source.camera.T_device_camera
            )
    pairs = mock_pipeline.mapping_backend.build.call_args.args[2]
    assert any(
        left.startswith("s000/") and right.startswith("s001/") for left, right in pairs
    )
    with h5py.File(multi_map / "global.h5", "r") as descriptors:
        assert sorted(descriptors.keys()) == ["s000", "s001"]
        assert all(view.name in descriptors for view in views)


@pytest.mark.parametrize("same_graph", [True, False])
def test_third_recording_query_cache_uses_its_mps_without_changing_the_map(
    tmp_path,
    multi_map,
    pilot_inputs,
    mock_pipeline,
    trajectory_file,
    capsys,
    same_graph,
):
    source = pilot_inputs["play_1"]
    if not same_graph:
        trajectory_file(
            "data/pilot/play_1/mps/slam/closed_loop_trajectory.csv",
            timestamps_us=[101_000, 135_000, 169_000],
            graphs=["query-other-frame"] * 3,
            translations=[(5, 0, 0), (6, 0, 0), (7, 0, 0)],
        )
    before = {
        name: file_sha256(multi_map / name)
        for name in ("map.json", "global.h5", "features.h5")
    }
    output = tmp_path / "queries/pilot_play1_multi_rgb.json"
    query.main(
        [
            "--map",
            str(multi_map),
            "--vrs",
            str(source.vrs),
            "--mps",
            str(source.mps),
            "--mode",
            "rgb",
            "--num-queries",
            "100",
            "--stride",
            "34",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert result["source_identity_check"] == "sha256"
    assert result["query_in_mapping_recordings"] is False
    assert result["query_vrs_sha256"] == file_sha256(source.vrs)
    assert result["map_graph_uid"] == "graph-A"
    assert result["same_mps_graph"] is same_graph
    assert result["num_queries"] == 3
    assert result["num_successes"] == 2
    assert "queries" not in json.loads(capsys.readouterr().out)
    mock_pipeline.query_reader.assert_called_once_with(
        source.vrs, source.mps / "slam/online_calibration.jsonl"
    )
    source.recording.select_labels.assert_called_once_with("rgb")
    calls = mock_pipeline.query_backend.query.call_args_list
    assert [item.args[1][0].index for item in calls] == [0, 34, 68]
    assert all(
        len(item.args) == 2 and not item.kwargs and item.args[0] == multi_map
        for item in calls
    )
    assert all(item.args[1][0].camera is source.camera for item in calls)
    first, failure, last = result["queries"]
    assert first["T_world_device"] == last["T_world_device"]
    assert failure["success"] is False
    assert failure["reason"] == "synthetic_no_matches"
    assert "translation_error_m" not in failure
    if same_graph:
        assert first["translation_error_m"] == pytest.approx(0.1)
        assert last["translation_error_m"] == pytest.approx(1.9)
    else:
        assert "translation_error_m" not in first and "rotation_error_deg" not in first
        assert "translation_error_m" not in last and "rotation_error_deg" not in last
    mock_pipeline.query_backend.prepare_map.assert_called_once_with(multi_map)
    mock_pipeline.query_backend.close.assert_called_once_with()
    mock_pipeline.mapping_backend.build.assert_called_once()
    assert before == {name: file_sha256(multi_map / name) for name in before}


@pytest.mark.parametrize("second_graph", ["disconnected-frame", ""])
def test_second_mapping_graph_is_checked_before_backend_or_recording_decode(
    monkeypatch, tmp_path, pilot_inputs, mock_pipeline, trajectory_file, second_graph
):
    mapping = [pilot_inputs[name] for name in ("play_0", "play_2")]
    trajectory_file(
        "data/pilot/play_2/mps/slam/closed_loop_trajectory.csv",
        timestamps_us=[21_000, 23_000, 27_000],
        graphs=[second_graph] * 3,
    )
    trajectories = Mock(wraps=build.Trajectory)
    monkeypatch.setattr(build, "Trajectory", trajectories)
    output = tmp_path / "maps/rejected_play02"
    with pytest.raises(ValueError, match="graph_uid"):
        build.build_map(
            [source.vrs for source in mapping],
            [source.mps for source in mapping],
            output,
            camera_set="rgb",
        )
    assert trajectories.call_args_list == [call(source.csv) for source in mapping]
    mock_pipeline.builder.assert_not_called()
    mock_pipeline.mapping_reader.assert_not_called()
    assert not output.exists()
