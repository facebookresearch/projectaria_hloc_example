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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import visualization as viz
from map_utils import transform_from_quaternion
from pose_alignment import MappedPose


@pytest.fixture
def rerun_modules(monkeypatch):
    streams = []

    def stream(application_id):
        result = Mock(
            spec_set=["save", "spawn", "log", "set_time", "flush", "disconnect"]
        )
        streams.append(result)
        return result

    def component(kind):
        def build(*args, **kwargs):
            value = SimpleNamespace(kind=kind, args=args, kwargs=kwargs)
            if kind == "Image":
                value.compress = Mock(
                    side_effect=lambda jpeg_quality: SimpleNamespace(
                        kind="EncodedImage",
                        args=args,
                        kwargs={"jpeg_quality": jpeg_quality},
                    )
                )
            return value

        return Mock(side_effect=build)

    rr = SimpleNamespace(
        RecordingStream=Mock(side_effect=stream),
        ViewCoordinates=SimpleNamespace(RIGHT_HAND_Z_UP="right-hand-z-up"),
    )
    for kind in (
        "Points3D",
        "Transform3D",
        "Arrows3D",
        "LineStrips3D",
        "Scalars",
        "Image",
        "Clear",
        "TextLog",
    ):
        setattr(rr, kind, component(kind))
    blueprint = SimpleNamespace(
        **{
            kind: component(kind)
            for kind in (
                "Blueprint",
                "Horizontal",
                "Vertical",
                "Spatial3DView",
                "Spatial2DView",
                "TimeSeriesView",
            )
        }
    )
    model = SimpleNamespace(
        points3D={
            1: SimpleNamespace(xyz=np.array([1, 2, 3])),
            2: SimpleNamespace(xyz=np.array([4, 5, 6])),
        }
    )
    colmap = SimpleNamespace(Reconstruction=Mock(return_value=model))
    modules = {
        "rerun": rr,
        "rerun.blueprint": blueprint,
        "torch": SimpleNamespace(),
        "pycolmap": colmap,
    }
    original = viz.importlib.import_module
    monkeypatch.setattr(
        viz.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original(name),
    )
    return SimpleNamespace(rr=rr, streams=streams, model=model, colmap=colmap)


@pytest.fixture
def view(rerun_modules, tmp_path):
    return viz.RerunView(tmp_path / "map", tmp_path / "view.rrd")


@pytest.fixture
def mapped_pose():
    def make(timestamp_ns=0, session_uid="session-a", status="tracking", age_ns=0):
        pose = transform_from_quaternion([1, 0, 0, 1], [timestamp_ns / 1e9, 2, 3])
        return MappedPose(
            timestamp_ns, pose, timestamp_ns - age_ns, session_uid, age_ns, status
        )

    return make


def entity_logs(stream, entity):
    return [call for call in stream.log.call_args_list if call.args[0] == entity]


@pytest.mark.parametrize("viewer,stream_count", [("none", 1), ("native", 2)])
def test_map_is_static_and_logged_to_each_explicit_sink(
    rerun_modules, tmp_path, viewer, stream_count
):
    view = viz.RerunView(tmp_path / "map", tmp_path / "view.rrd", viewer=viewer)
    assert len(rerun_modules.streams) == stream_count
    rerun_modules.colmap.Reconstruction.assert_called_once_with(
        tmp_path / "map" / "sfm"
    )
    rerun_modules.streams[0].save.assert_called_once()
    assert rerun_modules.streams[0].save.call_args.args == (tmp_path / "view.rrd",)
    for stream in rerun_modules.streams:
        landmarks = entity_logs(stream, "map/landmarks")[0]
        assert landmarks.kwargs == {"static": True}
        np.testing.assert_array_equal(landmarks.args[1].args[0], [[1, 2, 3], [4, 5, 6]])
        assert entity_logs(stream, "map")[0].args[1] == "right-hand-z-up"
    if viewer == "native":
        rerun_modules.streams[1].spawn.assert_called_once()
    view.close()
    for stream in rerun_modules.streams:
        stream.flush.assert_called_once_with(timeout_sec=5)
        stream.disconnect.assert_called_once_with()


@pytest.mark.parametrize("case", ["viewer", "empty_map"])
def test_invalid_view_initialization_does_not_leak_streams(
    rerun_modules, tmp_path, case
):
    if case == "empty_map":
        rerun_modules.model.points3D = {}
    with pytest.raises(ValueError):
        viz.RerunView(
            tmp_path / "map",
            tmp_path / "view.rrd",
            viewer="invalid" if case == "viewer" else "none",
        )
    assert all(stream.disconnect.called for stream in rerun_modules.streams)


def test_render_throttling_never_throttles_jsonl_pose_output(
    view, rerun_modules, tmp_path, mapped_pose
):
    writer = viz.PoseWriter(tmp_path / "poses.jsonl")
    for index in range(121):
        pose = mapped_pose(index * 10_000_000)
        writer.write(pose)
        view.on_pose(pose)
    writer.close()
    rows = [
        json.loads(line) for line in (tmp_path / "poses.jsonl").read_text().splitlines()
    ]
    assert writer.count == len(rows) == 121
    assert [row["timestamp_ns"] for row in rows] == [
        index * 10_000_000 for index in range(121)
    ]
    assert all(
        row["frame"] == "map" and row["session_uid"] == "session-a" for row in rows
    )
    rendered = entity_logs(rerun_modules.streams[0], "map/pose")
    assert len(rendered) == 31
    times_ns = [
        int(call.kwargs["duration"].astype("timedelta64[ns]").astype(np.int64))
        for call in rerun_modules.streams[0].set_time.call_args_list
    ]
    assert all(
        right - left >= viz.GUI_INTERVAL_NS
        for left, right in zip(times_ns, times_ns[1:])
    )
    np.testing.assert_allclose(
        rows[-1]["T_map_device"], mapped_pose(1_200_000_000).T_map_device
    )
    assert writer.target.closed


@pytest.mark.parametrize(
    "status,color,tracking",
    [("tracking", [80, 220, 140], 1), ("stale", [150, 150, 150], 0)],
)
def test_device_transform_is_map_from_device_and_reports_fix_age(
    view, rerun_modules, mapped_pose, status, color, tracking
):
    pose = mapped_pose(2_000_000_000, status=status, age_ns=1_500_000_000)
    view.on_pose(pose)
    stream = rerun_modules.streams[0]
    transform = entity_logs(stream, "map/device_axes")[0].args[1]
    np.testing.assert_array_equal(
        transform.kwargs["translation"], pose.T_map_device[:3, 3]
    )
    np.testing.assert_array_equal(transform.kwargs["mat3x3"], pose.T_map_device[:3, :3])
    assert entity_logs(stream, "map/pose")[0].args[1].kwargs["colors"] == color
    assert entity_logs(stream, "diagnostics/fix_age_sec")[0].args[1].args == (1.5,)
    assert entity_logs(stream, "diagnostics/tracking")[0].args[1].args == (tracking,)
    assert stream.set_time.call_args.kwargs["duration"] == np.timedelta64(0, "ns")


def test_session_reset_restarts_throttle_and_bounded_path(view, mapped_pose):
    for index in range(605):
        view.on_pose(mapped_pose(index * viz.GUI_INTERVAL_NS))
    assert len(view.path) == 600
    view.on_pose(mapped_pose(0, session_uid="session-b"))
    assert view.session_uid == "session-b"
    assert len(view.path) == 1
    np.testing.assert_array_equal(view.path[0], [0, 2, 3])


@pytest.mark.parametrize("state", ["lost", "session_reset", "error"])
def test_lost_status_clears_pose_geometry_but_not_static_map(
    view, rerun_modules, mapped_pose, state
):
    view.on_pose(mapped_pose())
    stream = rerun_modules.streams[0]
    stream.log.reset_mock()
    view.on_status({"state": state, "arrival_timestamp_ns": 2_000_000})
    assert len(view.path) == 0
    cleared = [
        call.args[0]
        for call in stream.log.call_args_list
        if call.args[1].kind == "Clear"
    ]
    assert cleared == ["map/pose", "map/trajectory", "map/device_axes"]
    assert all(
        call.args[1].kwargs["recursive"]
        for call in stream.log.call_args_list
        if call.args[1].kind == "Clear"
    )
    assert not entity_logs(stream, "map/landmarks")
    assert (
        json.loads(entity_logs(stream, "diagnostics/status")[0].args[1].args[0])[
            "state"
        ]
        == state
    )


def test_image_resize_preserves_aspect_and_uses_device_elapsed_time(
    view, rerun_modules
):
    image = np.full((3, 4, 3), 17, dtype=np.uint8)
    view.on_image(image, 1_000_000)
    view.on_image(image, 4_000_000)
    stream = rerun_modules.streams[0]
    logged = entity_logs(stream, "camera/image")[-1].args[1].args[0]
    assert logged.shape == (480, 640, 3)
    assert np.all(logged == 17)
    assert entity_logs(stream, "camera/image")[-1].args[1].kind == "EncodedImage"
    assert entity_logs(stream, "camera/image")[-1].args[1].kwargs == {
        "jpeg_quality": 90
    }
    assert stream.set_time.call_args.kwargs["duration"] == np.timedelta64(
        3_000_000, "ns"
    )


def test_status_accepts_zero_device_timestamp_and_no_timestamp(view, rerun_modules):
    view.on_status(
        {"state": "tracking", "timestamp_ns": 0, "arrival_timestamp_ns": 5_000_000}
    )
    assert view.origin_timestamp_ns == 0
    stream = rerun_modules.streams[0]
    stream.set_time.reset_mock()
    view.on_status({"state": "tracking"})
    stream.set_time.assert_not_called()
    assert len(entity_logs(stream, "diagnostics/status")) == 1


def test_pose_writer_is_exclusive_finite_and_closed(tmp_path, mapped_pose):
    path = tmp_path / "poses.jsonl"
    writer = viz.PoseWriter(path)
    with pytest.raises(FileExistsError):
        viz.PoseWriter(path)
    pose = mapped_pose(10_000_000, age_ns=2_000_000)
    writer.write(pose)
    pose.T_map_device[0, 3] = np.nan
    with pytest.raises(ValueError):
        writer.write(pose)
    assert writer.count == 1
    writer.close()
    row = json.loads(path.read_text())
    assert row["fix_timestamp_ns"] == 8_000_000
    assert row["fix_age_ns"] == 2_000_000
    assert row["status"] == "tracking"
    assert writer.target.closed
    with pytest.raises(ValueError):
        writer.write(mapped_pose())


@pytest.mark.parametrize("failure", ["flush", "disconnect"])
def test_close_disconnects_all_sinks_even_if_cleanup_fails(
    rerun_modules, tmp_path, failure
):
    view = viz.RerunView(tmp_path / "map", tmp_path / "view.rrd", viewer="native")
    error = OSError("sink failed")
    getattr(rerun_modules.streams[0], failure).side_effect = error
    with pytest.raises(ExceptionGroup, match="cleanup failed") as group:
        view.close()
    assert group.value.exceptions == (error,)
    assert all(stream.disconnect.called for stream in rerun_modules.streams)
    assert all(stream.flush.called for stream in rerun_modules.streams)


@pytest.mark.parametrize("failure", ["save", "spawn"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_rerun_initialization_error_closes_registered_streams_and_keeps_original_error(
    rerun_modules, tmp_path, failure, cleanup_fails
):
    original_stream = rerun_modules.rr.RecordingStream.side_effect
    error = OSError("initialize failed")

    def stream(application_id):
        result = original_stream(application_id)
        if (failure == "save" and len(rerun_modules.streams) == 1) or (
            failure == "spawn" and len(rerun_modules.streams) == 2
        ):
            getattr(result, failure).side_effect = error
        if cleanup_fails:
            result.flush.side_effect = OSError("cleanup failed")
        return result

    rerun_modules.rr.RecordingStream.side_effect = stream
    with pytest.raises(OSError, match="initialize failed") as raised:
        viz.RerunView(tmp_path / "map", tmp_path / "view.rrd", viewer="native")
    assert raised.value is error
    assert all(stream.disconnect.called for stream in rerun_modules.streams)
    if cleanup_fails:
        assert any("cleanup also failed" in note for note in error.__notes__)


def test_session_clock_restart_stitches_monotonic_viewer_time_without_changing_jsonl(
    view, rerun_modules, mapped_pose, tmp_path
):
    writer = viz.PoseWriter(tmp_path / "clock-poses.jsonl")
    poses = [
        mapped_pose(9_000_000_000),
        mapped_pose(10_000_000_000),
        mapped_pose(10_000_000, session_uid="session-b"),
        mapped_pose(50_000_000, session_uid="session-b"),
    ]
    for pose in poses[:2]:
        writer.write(pose)
        view.on_pose(pose)
    writer.write(poses[2])
    view.on_pose(poses[2])
    assert len(view.path) == 1
    assert view.origin_timestamp_ns == 10_000_000
    view.on_image(np.zeros((2, 2), dtype=np.uint8), 5_000_000)
    writer.write(poses[3])
    view.on_pose(poses[3])
    writer.close()
    calls = rerun_modules.streams[0].set_time.call_args_list
    elapsed_ns = [
        int(call.kwargs["duration"].astype("timedelta64[ns]").astype(np.int64))
        for call in calls
    ]
    assert all(call.args == ("viewer_elapsed",) for call in calls)
    assert elapsed_ns == [0, 1_000_000_000, 1_000_000_001, 1_000_000_001, 1_040_000_001]
    assert len(view.path) == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "clock-poses.jsonl").read_text().splitlines()
    ]
    assert [row["timestamp_ns"] for row in rows] == [
        pose.timestamp_ns for pose in poses
    ]


def test_session_reset_status_begins_new_clock_before_first_new_pose(
    view, rerun_modules, mapped_pose
):
    view.on_pose(mapped_pose(1_000_000_000))
    view.on_pose(mapped_pose(1_100_000_000))
    view.on_status(
        {"state": "session_reset", "session_uid": "session-b", "timestamp_ns": 0}
    )
    assert view.origin_timestamp_ns == 0
    assert view.viewer_elapsed_duration_ns == 100_000_001
    assert len(view.path) == 0
    view.on_pose(mapped_pose(1_000_000, session_uid="session-b"))
    assert view.viewer_elapsed_duration_ns == 101_000_001
    assert len(view.path) == 1
    assert entity_logs(rerun_modules.streams[0], "diagnostics/tracking")[-1].args[
        1
    ].args == (1,)


@pytest.mark.parametrize("state", ["lost", "error"])
def test_loss_plots_tracking_zero_and_next_valid_pose_resumes_immediately(
    view, rerun_modules, mapped_pose, state
):
    view.on_pose(mapped_pose(100_000_000))
    view.on_status({"state": state, "timestamp_ns": 110_000_000})
    stream = rerun_modules.streams[0]
    assert entity_logs(stream, "diagnostics/tracking")[-1].args[1].args == (0,)
    view.on_pose(mapped_pose(111_000_000))
    assert len(view.path) == 1
    assert len(entity_logs(stream, "map/pose")) == 3
    assert entity_logs(stream, "diagnostics/tracking")[-1].args[1].args == (1,)
    view.on_status({"state": state, "timestamp_ns": 112_000_000})
    assert len(view.path) == 0
    assert entity_logs(stream, "diagnostics/tracking")[-1].args[1].args == (0,)


def test_full_rate_stable_status_dedup_keeps_query_error_and_state_updates(
    view, rerun_modules
):
    for index in range(803):
        view.on_status(
            {
                "state": "tracking",
                "timestamp_ns": index * 1_000_000,
                "query_duration_sec": None,
                "num_inliers": None,
            }
        )
    stream = rerun_modules.streams[0]
    assert len(entity_logs(stream, "diagnostics/status")) == 1
    query = {"state": "tracking", "query_duration_sec": 0.25, "num_inliers": 25}
    view.on_status(query)
    view.on_status(query)
    view.on_status({"state": "stale", "timestamp_ns": 900_000_000})
    view.on_status({"state": "error", "detail": "failed"})
    view.on_status({"state": "error", "detail": "failed"})
    statuses = [
        json.loads(call.args[1].args[0])
        for call in entity_logs(stream, "diagnostics/status")
    ]
    assert len(statuses) == 6
    assert statuses[1:3] == [query, query]
    assert [status["state"] for status in statuses[3:]] == ["stale", "error", "error"]
