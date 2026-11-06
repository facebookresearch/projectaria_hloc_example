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

import map_utils as maps
import numpy as np
import pytest


def test_rigid_inverse_and_quaternion_roundtrip():
    pose = maps.transform_from_quaternion([2, 0, 0, 2], [1, -2, 3])
    np.testing.assert_allclose(
        pose[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12
    )
    np.testing.assert_allclose(
        maps.inverse_transform(pose) @ pose, np.eye(4), atol=1e-12
    )
    roundtrip = maps.transform_from_quaternion(maps.quaternion_wxyz(pose), pose[:3, 3])
    np.testing.assert_allclose(roundtrip, pose, atol=1e-12)
    assert maps.rotation_error_deg(np.eye(4), pose) == pytest.approx(90)
    assert maps.rotation_error_deg(pose, pose) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("case", ["shape", "nan", "scale", "reflection", "bottom_row"])
def test_reject_nonrigid_transform(case):
    pose = np.eye(4)
    if case == "shape":
        pose = pose[:3]
    elif case == "nan":
        pose[0, 3] = np.nan
    elif case == "scale":
        pose[0, 0] = 2
    elif case == "reflection":
        pose[0, 0] = -1
    else:
        pose[3, 0] = 1
    with pytest.raises(ValueError, match="transform"):
        maps.rigid_transform(pose)
    with pytest.raises(ValueError):
        maps.inverse_transform(pose)


@pytest.mark.parametrize(
    "quaternion", [[0, 0, 0, 0], [1, 0, 0], [float("nan"), 0, 0, 1]]
)
def test_invalid_quaternion(quaternion):
    with pytest.raises(ValueError, match="quaternion"):
        maps.transform_from_quaternion(quaternion, [0, 0, 0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 0),
        ("height", -1),
        ("focal", 0),
        ("focal", float("inf")),
        ("cx", float("nan")),
        ("cy", float("inf")),
    ],
)
def test_invalid_camera(camera, field, value):
    data = camera.to_dict()
    data[field] = value
    with pytest.raises(ValueError, match="camera"):
        maps.Camera.from_dict(data)


def test_view_and_camera_serialization_preserves_pat_pixels(camera, make_view):
    view = make_view("s000/camera-rgb/00000000.png", xyz=(1, 2, 3))
    restored = maps.View.from_dict(json.loads(json.dumps(view.to_dict())))
    assert restored.name == view.name
    assert restored.camera.colmap_params() == [25, 25, 16, 12]
    assert restored.camera.cx == 15.5
    np.testing.assert_array_equal(
        restored.camera.T_device_camera, camera.T_device_camera
    )
    np.testing.assert_array_equal(restored.T_world_camera, view.T_world_camera)
    assert restored.capture_timestamp_ns == view.capture_timestamp_ns


@pytest.mark.parametrize("offset_ns", [0, 250_000, -250_000])
def test_view_roundtrip_retains_raw_time_and_signed_camera_offset(make_view, offset_ns):
    view = make_view()
    view.time_offset_ns = offset_ns
    data = json.loads(json.dumps(view.to_dict()))
    assert data["capture_timestamp_ns"] == 1_000_000
    assert data["time_offset_ns"] == offset_ns
    assert data["corrected_timestamp_ns"] == 1_000_000 - offset_ns
    restored = maps.View.from_dict(data)
    assert restored.capture_timestamp_ns == 1_000_000
    assert restored.time_offset_ns == offset_ns
    assert restored.corrected_timestamp_ns == 1_000_000 - offset_ns


@pytest.mark.parametrize(
    "name", ["/outside.png", "../outside.png", "a/../../outside.png"]
)
def test_view_path_cannot_escape_map(make_view, name):
    data = make_view().to_dict()
    data["name"] = name
    with pytest.raises(ValueError, match="relative paths"):
        maps.View.from_dict(data)


def test_trajectory_clock_units_nearest_pose_and_no_extrapolation(trajectory_file):
    trajectory = maps.Trajectory(trajectory_file())
    assert trajectory.graph_uid == "graph-A"
    assert trajectory.timestamps_ns == [1_000_000, 3_000_000, 7_000_000]
    assert trajectory.at(999_999) is None
    assert trajectory.at(7_000_001) is None
    np.testing.assert_array_equal(trajectory.at(3_000_000)[:3, 3], [1, 0, 0])
    np.testing.assert_array_equal(trajectory.at(2_000_000)[:3, 3], [0, 0, 0])
    assert trajectory.at(5_000_000, max_gap_ns=1_999_999) is None
    assert trajectory.at(5_000_000, max_gap_ns=2_000_000) is not None


@pytest.mark.parametrize(
    "timestamps_us,graphs,error",
    [
        ([], [], "empty"),
        ([1000, 1000], ["a", "a"], "increasing"),
        ([2000, 1000], ["a", "a"], "increasing"),
        ([1000, 2000], ["a", "b"], "graph_uid"),
        ([1000], [" "], "graph_uid"),
    ],
)
def test_trajectory_rejects_disconnected_or_invalid_clock(
    trajectory_file, timestamps_us, graphs, error
):
    with pytest.raises(ValueError, match=error):
        maps.Trajectory(trajectory_file(timestamps_us=timestamps_us, graphs=graphs))


def test_recording_clocks_are_never_merged(trajectory_file):
    first = maps.Trajectory(trajectory_file("first.csv", timestamps_us=(1000, 2000)))
    second = maps.Trajectory(
        trajectory_file("second.csv", timestamps_us=(10_000, 11_000))
    )
    assert first.graph_uid == second.graph_uid
    assert first.at(10_000_000) is None
    assert second.at(1_000_000) is None


@pytest.mark.parametrize("nested", [False, True])
def test_mps_directory_resolution(tmp_path, nested):
    slam = tmp_path / "slam" if nested else tmp_path
    slam.mkdir(exist_ok=True)
    for name in ("closed_loop_trajectory.csv", "online_calibration.jsonl"):
        (slam / name).touch()
    assert maps.mps_slam_dir(tmp_path) == slam


@pytest.mark.parametrize(
    "missing", ["closed_loop_trajectory.csv", "online_calibration.jsonl"]
)
def test_mps_missing_inputs(tmp_path, missing):
    slam = tmp_path / "slam"
    slam.mkdir()
    for name in ("closed_loop_trajectory.csv", "online_calibration.jsonl"):
        if name != missing:
            (slam / name).touch()
    with pytest.raises(FileNotFoundError, match=missing):
        maps.mps_slam_dir(tmp_path)


def test_mapping_pairs_filter_orientation_distance_and_deduplicate(make_view):
    views = [
        make_view("a"),
        make_view("b", xyz=(0.2, 0, 0)),
        make_view("c", xyz=(4, 0, 0)),
        make_view("far", xyz=(20, 0, 0)),
        make_view("back", xyz=(0.1, 0, 0), yaw_deg=180),
    ]
    assert maps.mapping_pairs(views, num_pairs=2) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert maps.mapping_pairs(views[:2], num_pairs=1) == [("a", "b")]


@pytest.mark.parametrize("count,num_pairs", [(0, 1), (1, 1), (2, 0)])
def test_mapping_pairs_invalid_input(make_view, count, num_pairs):
    with pytest.raises(ValueError, match="at least two"):
        maps.mapping_pairs([make_view(str(i)) for i in range(count)], num_pairs)


@pytest.mark.parametrize("xyz,yaw_deg", [((4.01, 0, 0), 0), ((0.2, 0, 0), 46)])
def test_mapping_pairs_no_overlap(make_view, xyz, yaw_deg):
    with pytest.raises(ValueError, match="No overlapping"):
        maps.mapping_pairs([make_view("a"), make_view("b", xyz, yaw_deg)])


def test_covisibility_is_transitive_and_deterministic():
    assert maps.covisibility_clusters(
        {"d": {10}, "b": {1, 2}, "c": {2, 3}, "a": {1}, "isolated": set()}
    ) == [["a", "b", "c"], ["d"], ["isolated"]]
    assert maps.covisibility_clusters({}) == []


def test_matches_discard_conflicts_but_keep_multi_camera_observations():
    matches = [(1, 0, 7), (0, 2, 4), (0, 1, 7), (0, 1, 7), (0, 2, 8)]
    assert maps.unique_matches(iter(matches)) == [(0, 1, 7), (1, 0, 7)]
    assert maps.unique_matches([]) == []


def test_json_is_finite_and_roundtrippable(tmp_path):
    path = tmp_path / "stats.json"
    maps.write_json(path, {"success": False, "count": 2})
    assert json.loads(path.read_text()) == {"success": False, "count": 2}
    assert path.read_text().endswith("\n")
    with pytest.raises(ValueError):
        maps.write_json(path, {"error": float("nan")})


@pytest.mark.parametrize("size", [0, 17, 2 * 1024 * 1024 + 13])
def test_file_sha256_matches_stdlib_across_empty_and_multiple_chunks(tmp_path, size):
    payload = (bytes(range(256)) * ((size + 255) // 256))[:size]
    path = tmp_path / "recording.vrs"
    path.write_bytes(payload)
    assert maps.file_sha256(path) == hashlib.sha256(payload).hexdigest()
    assert path.read_bytes() == payload


def test_file_sha256_missing_input_reports_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        maps.file_sha256(tmp_path / "missing.vrs")
