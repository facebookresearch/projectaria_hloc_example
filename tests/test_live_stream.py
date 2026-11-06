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

import gc
import threading
import traceback
import weakref
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import live_stream as live
import numpy as np
import pytest
from pose_alignment import VioSample


def transform(x: float = 0.0):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def sample(timestamp_ms=100, x=1.0, session="A", quality=True):
    return VioSample(timestamp_ms * 1_000_000, transform(x), session, quality)


def regular(timestamp_ms=100, x=1.0, session="A", quality="GOOD"):
    return SimpleNamespace(
        capture_timestamp_ns=timestamp_ms * 1_000_000,
        frontend_session_uid=session,
        pose_quality=quality,
        transform_odometry_bodyimu=SimpleNamespace(
            to_matrix=lambda: transform(x - 0.25)
        ),
        transform_bodyimu_device=SimpleNamespace(to_matrix=lambda: transform(0.25)),
    )


def high_frequency(timestamp_ms=100, x=1.0, session="A", quality=1.0):
    return SimpleNamespace(
        tracking_timestamp=timedelta(milliseconds=timestamp_ms),
        session_uid=session,
        quality_score=quality,
        transform_odometry_device=SimpleNamespace(to_matrix=lambda: transform(x)),
    )


@pytest.fixture
def harness(monkeypatch, tmp_path):
    source = Mock()
    source.get_image_size.return_value = (8, 6)
    source.get_time_offset_sec_device_camera.return_value = 0.03
    source.get_transform_device_camera.return_value.to_matrix.return_value = transform(
        0.25
    )
    device_calibration = Mock()
    device_calibration.get_camera_calib.return_value = source
    device_calibration.get_device_version.return_value = "Gen2"
    pat = Mock()
    pat.rescale_camera_calibration.return_value = source
    pat.get_linear_camera_calibration.return_value.get_principal_point.return_value = [
        511.5,
        383.5,
    ]
    pat.distort_by_calibration.side_effect = (
        lambda image, target, original: image.copy()
    )
    monkeypatch.setattr(live.importlib, "import_module", Mock(return_value=pat))
    backend = Mock()
    backend.prepare_map.return_value = {
        "map_load_duration_sec": 1.0,
        "global_features_load_duration_sec": 2.0,
        "models_load_duration_sec": 3.0,
    }
    backend_constructor = Mock(return_value=backend)
    monkeypatch.setattr(live, "HlocBackend", backend_constructor)
    map_loader = Mock()
    monkeypatch.setattr(live, "load_map", map_loader)
    query = Mock(
        return_value={
            "success": True,
            "T_world_device": transform(10).tolist(),
            "num_inlier_landmarks": 42,
            "query_duration_sec": 6.0,
        }
    )
    monkeypatch.setattr(live, "localize_frames", query)
    poses, images, statuses = [], [], []
    condition = threading.Condition()

    def on_status(status):
        with condition:
            statuses.append(status)
            condition.notify_all()

    localizer = live.LiveLocalizer(
        tmp_path / "map",
        poses.append,
        images.append,
        on_status,
        fix_wait_duration_sec=0.2,
        worker_join_duration_sec=2.0,
    )
    result = SimpleNamespace(
        localizer=localizer,
        poses=poses,
        images=images,
        statuses=statuses,
        condition=condition,
        source=source,
        calibration=device_calibration,
        pat=pat,
        backend=backend,
        backend_constructor=backend_constructor,
        query=query,
        map_loader=map_loader,
    )
    yield result
    # A failing callback/worker test still joins its non-daemon thread.
    try:
        localizer.stop()
    except RuntimeError:
        pass


def rgb(harness, corrected_timestamp_ms=150, pixels=None):
    pixels = np.full((3, 4, 3), 7, dtype=np.uint8) if pixels is None else pixels
    data = SimpleNamespace(to_numpy_array=lambda: pixels)
    record = SimpleNamespace(
        capture_timestamp_ns=(corrected_timestamp_ms + 30) * 1_000_000
    )
    harness.localizer.on_rgb(data, record)
    return pixels


def wait_status(harness, predicate):
    with harness.condition:
        assert harness.condition.wait_for(
            lambda: any(predicate(status) for status in harness.statuses), timeout=2.0
        ), harness.statuses


def seed(harness):
    harness.localizer.on_calibration(harness.calibration)
    harness.localizer.submit_vio(sample(100, 1))
    harness.localizer.submit_vio(sample(200, 2))


def wait_fix(harness, timestamp_ms=150):
    wait_status(
        harness,
        lambda status: (
            status.state == "tracking"
            and status.query_duration_sec is not None
            and status.timestamp_ns == timestamp_ms * 1_000_000
        ),
    )


def test_bootstrap_uses_rgb_capture_time_not_inference_finish_and_emits_every_vio(
    harness,
):
    seed(harness)
    assert harness.poses == []
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    for timestamp_ms in (210, 220, 230):
        harness.localizer.submit_vio(sample(timestamp_ms, timestamp_ms / 100))
    assert [pose.timestamp_ns for pose in harness.poses] == [
        210_000_000,
        220_000_000,
        230_000_000,
    ]
    for pose, x in zip(harness.poses, (10.6, 10.7, 10.8)):
        assert pose.T_map_device[0, 3] == pytest.approx(x)
        assert pose.fix_timestamp_ns == 150_000_000
        assert pose.age_ns == pose.timestamp_ns - 150_000_000
        assert pose.status == "tracking"
    harness.map_loader.assert_called_once_with(harness.localizer.map_path)
    harness.backend.prepare_map.assert_called_once_with(harness.localizer.map_path)
    harness.query.assert_called_once_with(
        harness.backend, harness.localizer.map_path, harness.images
    )
    assert harness.statuses[-4].num_inliers == 42


def test_resident_backend_prepares_once_across_queries_and_session_reset(harness):
    seed(harness)
    harness.localizer.start()
    harness.localizer.start()
    for timestamp_ms in (120, 180):
        rgb(harness, timestamp_ms)
        wait_fix(harness, timestamp_ms)
    harness.localizer.submit_vio(sample(100, 1, session="B"))
    harness.localizer.submit_vio(sample(200, 2, session="B"))
    rgb(harness, 150)
    wait_fix(harness, 150)
    harness.backend_constructor.assert_called_once_with()
    harness.map_loader.assert_called_once_with(harness.localizer.map_path)
    harness.backend.prepare_map.assert_called_once_with(harness.localizer.map_path)
    assert harness.query.call_count == 3
    assert all(
        call.args[:2] == (harness.backend, harness.localizer.map_path)
        for call in harness.query.call_args_list
    )
    harness.backend.close.assert_not_called()
    harness.localizer.stop()
    harness.localizer.stop()
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()


def test_worker_timing_separates_init_query_and_rectification_from_queue_and_fix_wait(
    harness, monkeypatch
):
    clock = Mock(return_value=1.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    seed(harness)
    assert harness.statuses[-1].initialization_durations_sec is None
    rgb(harness)
    original_rectify = harness.localizer._rectify
    original_accept_fix = harness.localizer._accept_fix
    result = dict(harness.query.return_value)

    def rectify(snapshot):
        assert clock.return_value == 100.0
        clock.return_value = 102.0
        return original_rectify(snapshot)

    def query(*args):
        clock.return_value = 108.0
        return result

    def accept_fix(*args):
        clock.return_value = 200.0
        return original_accept_fix(*args)

    monkeypatch.setattr(harness.localizer, "_rectify", rectify)
    monkeypatch.setattr(harness.localizer, "_accept_fix", accept_fix)
    harness.query.side_effect = query
    clock.return_value = 100.0
    harness.localizer.start()
    wait_fix(harness)
    harness.localizer.stop()
    status = harness.statuses[-1]
    assert status.query_duration_sec == 6.0
    assert status.query_end_to_end_duration_sec == 8.0
    assert status.timestamp_ns == 150_000_000
    assert (
        status.initialization_durations_sec == harness.backend.prepare_map.return_value
    )
    status.initialization_durations_sec["models_load_duration_sec"] = 99.0
    harness.localizer._status("tracking")
    assert harness.statuses[-1].initialization_durations_sec == {
        "map_load_duration_sec": 1.0,
        "global_features_load_duration_sec": 2.0,
        "models_load_duration_sec": 3.0,
    }
    assert harness.statuses[-1].query_end_to_end_duration_sec is None


def test_prepare_does_not_block_vio_or_revive_retired_session_or_watchdog(
    harness, monkeypatch
):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    seed(harness)
    entered, release, delivered = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    initialization_durations_sec = dict(harness.backend.prepare_map.return_value)

    def prepare(map_path):
        entered.set()
        assert release.wait(2.0)
        return initialization_durations_sec

    def deliver_new_session():
        harness.localizer.on_vio(regular(100, session="B"))
        harness.localizer.on_vio(regular(200, session="B"))
        harness.localizer.on_vio(regular(300, session="A"))
        rgb(harness, 160)
        delivered.set()

    harness.backend.prepare_map.side_effect = prepare
    harness.localizer.start()
    vio_thread = threading.Thread(target=deliver_new_session)
    try:
        assert entered.wait(2.0)
        rgb(harness, 150)
        vio_thread.start()
        assert delivered.wait(1.0), "Model preparation blocked the VIO thread"
        harness.query.assert_not_called()
        clock.return_value = 103.0
        harness.localizer.check_freshness()
        assert harness.statuses[-1].state == "lost"
        assert harness.localizer._session_uid == "B"
        release.set()
        wait_status(
            harness,
            lambda status: status.query_duration_sec is not None,
        )
    finally:
        release.set()
        if vio_thread.ident is not None:
            vio_thread.join(2.0)
    assert not vio_thread.is_alive()
    harness.localizer.stop()
    assert harness.statuses[-1].state == "lost"
    assert harness.statuses[-1].session_uid == "B"
    assert (
        harness.statuses[-1].initialization_durations_sec
        == initialization_durations_sec
    )
    assert harness.localizer._vio_arrival_lost
    assert harness.localizer._session_uid == "B"
    assert harness.poses == []
    assert [frame.corrected_timestamp_ns for frame in harness.images] == [160_000_000]
    harness.backend.prepare_map.assert_called_once_with(harness.localizer.map_path)
    harness.backend.close.assert_called_once_with()


@pytest.mark.parametrize("stage", ["constructor", "map_validation", "prepare"])
def test_worker_startup_failure_surfaces_and_closes_acquired_backend(harness, stage):
    error = ValueError(f"Backend {stage} failed")
    operation = {
        "constructor": harness.backend_constructor,
        "map_validation": harness.map_loader,
        "prepare": harness.backend.prepare_map,
    }[stage]
    operation.side_effect = error
    harness.localizer.start()
    wait_status(harness, lambda status: status.state == "error")
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.stop()
    assert captured.value.__cause__ is error
    assert not harness.localizer._thread.is_alive()
    if stage == "constructor":
        harness.backend.close.assert_not_called()
    else:
        harness.backend.close.assert_called_once_with()
    harness.query.assert_not_called()
    assert harness.statuses[-1].initialization_durations_sec is None


def test_prepare_and_close_failures_both_reach_main_thread(harness):
    prepare_error = ValueError("prepare failed")
    close_error = OSError("close failed")
    harness.backend.prepare_map.side_effect = prepare_error
    harness.backend.close.side_effect = close_error
    harness.localizer.start()
    wait_status(harness, lambda status: status.state == "error")
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.stop()
    assert captured.value.__cause__ is close_error
    assert close_error.__context__ is prepare_error
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()


def test_close_failure_on_clean_stop_is_not_an_uncaught_worker_exception(harness):
    error = OSError("close failed")
    seed(harness)
    harness.backend.close.side_effect = error
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.stop()
    assert captured.value.__cause__ is error
    assert harness.statuses[-1].state == "error"
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()


def test_failed_forward_releases_model_but_retains_original_exception(harness):
    error = ValueError("Model forward failed")

    class FakeModel:
        def forward(self):
            raise error

    model = FakeModel()
    model_reference = weakref.ref(model)
    harness.backend.model = model
    del model

    def query(backend, map_path, frames):
        return backend.model.forward()

    def close():
        del harness.backend.model

    harness.query.side_effect = query
    harness.backend.close.side_effect = close
    seed(harness)
    harness.localizer.start()
    rgb(harness)
    wait_status(harness, lambda status: status.state == "error")
    original_traceback = error.__traceback__
    original_locations = traceback.extract_tb(original_traceback)
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.stop()
    gc.collect()
    assert model_reference() is None
    assert captured.value.__cause__ is error
    assert error.__traceback__ is original_traceback
    assert traceback.extract_tb(error.__traceback__) == original_locations
    assert "forward" in [location.name for location in original_locations]
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()
    with pytest.raises(RuntimeError, match="Live localization failed") as repeated:
        harness.localizer.raise_if_failed()
    assert repeated.value.__cause__ is error


def test_clear_exception_frames_preserves_complete_cyclic_exception_graph():
    root = ValueError("primary")
    cause = RuntimeError("cause")
    context = OSError("context")
    child = TypeError("group child")
    interrupt = KeyboardInterrupt("base group child")
    group = BaseExceptionGroup("cleanup", [child, interrupt])
    errors = [root, cause, context, group, child, interrupt]

    def capture(error):
        payload = Mock()
        reference = weakref.ref(payload)
        try:
            raise error
        except type(error):
            return reference

    references = [capture(error) for error in errors]
    root.__cause__ = cause
    root.__context__ = context
    cause.__context__ = root
    context.__cause__ = group
    group.__cause__ = cause
    root.add_note("Original primary note")
    tracebacks = [error.__traceback__ for error in errors]
    locations = [traceback.extract_tb(value) for value in tracebacks]
    assert all(reference() is not None for reference in references)
    live._clear_exception_frames(root)
    gc.collect()
    assert all(reference() is None for reference in references)
    assert root.__cause__ is cause and root.__context__ is context
    assert cause.__context__ is root
    assert context.__cause__ is group and group.__cause__ is cause
    assert group.exceptions == (child, interrupt)
    assert root.__notes__ == ["Original primary note"]
    for error, original_traceback, original_locations in zip(
        errors, tracebacks, locations
    ):
        assert error.__traceback__ is original_traceback
        assert traceback.extract_tb(error.__traceback__) == original_locations


def test_clear_exception_frames_skips_executing_frame_then_clears_after_return():
    error = RuntimeError("Active callback")

    def active_callback():
        payload = Mock()
        reference = weakref.ref(payload)
        try:
            raise error
        except RuntimeError:
            original_traceback = error.__traceback__
            live._clear_exception_frames(error)
            assert reference() is payload
            assert error.__traceback__ is original_traceback
            return reference

    reference = active_callback()
    assert reference() is not None
    live._clear_exception_frames(error)
    gc.collect()
    assert reference() is None


def test_callback_failure_keeps_cause_and_reports_later_close_failure(harness):
    seed(harness)
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    callback_error = ValueError("calibration callback failed")
    cleanup_error = OSError("backend close failed")
    cleanup_references = []

    def close():
        payload = Mock()
        cleanup_references.append(weakref.ref(payload))
        raise cleanup_error

    harness.backend.close.side_effect = close
    harness.calibration.get_camera_calib.side_effect = callback_error
    harness.localizer.on_calibration(harness.calibration)
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.stop()
    gc.collect()
    assert cleanup_references[0]() is None
    assert captured.value.__cause__ is callback_error
    assert any("backend close failed" in note for note in callback_error.__notes__)
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()


def test_stop_during_prepare_closes_without_query_or_post_stop_status(harness):
    seed(harness)
    entered, release = threading.Event(), threading.Event()
    initialization_durations_sec = dict(harness.backend.prepare_map.return_value)

    def prepare(map_path):
        entered.set()
        assert release.wait(2.0)
        return initialization_durations_sec

    harness.backend.prepare_map.side_effect = prepare
    harness.localizer.start()
    try:
        assert entered.wait(2.0)
        rgb(harness)
        harness.localizer._worker_join_duration_sec = 0.01
        with pytest.raises(TimeoutError, match="did not stop"):
            harness.localizer.stop()
        harness.backend.close.assert_not_called()
        statuses_before_release = list(harness.statuses)
    finally:
        release.set()
        harness.localizer._worker_join_duration_sec = 2.0
        harness.localizer.stop()
    harness.backend.close.assert_called_once_with()
    harness.query.assert_not_called()
    assert harness.images == []
    assert harness.statuses == statuses_before_release
    assert not harness.localizer._thread.is_alive()


def test_streamed_factory_calibration_rescales_and_keeps_pat_pixels_and_offset(harness):
    seed(harness)
    harness.localizer.start()
    pixels = rgb(harness)
    wait_fix(harness)
    frame = harness.images[0]
    assert frame.capture_timestamp_ns == 180_000_000
    assert frame.corrected_timestamp_ns == 150_000_000
    assert frame.time_offset_ns == 30_000_000
    assert (frame.camera.width, frame.camera.height, frame.camera.focal) == (
        1024,
        768,
        360.0,
    )
    assert (frame.camera.cx, frame.camera.cy) == (511.5, 383.5)
    assert frame.camera.colmap_params()[2:] == [512.0, 384.0]
    np.testing.assert_array_equal(frame.camera.T_device_camera, transform(0.25))
    np.testing.assert_array_equal(frame.image, pixels)
    harness.pat.rescale_camera_calibration.assert_called_once_with(
        harness.source, (4, 3), "Gen2", "camera-rgb"
    )
    harness.pat.get_linear_camera_calibration.assert_called_once_with(
        1024, 768, 360.0, "camera-rgb"
    )
    harness.pat.distort_by_calibration.assert_called_once()
    # Map metadata never supplies query intrinsics or extrinsics.
    harness.map_loader.return_value = {"views": [{"camera": "another device"}]}


def test_no_rescale_when_actual_image_matches_calibration(harness):
    harness.source.get_image_size.return_value = (4, 3)
    seed(harness)
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    harness.pat.rescale_camera_calibration.assert_not_called()


def test_waiting_calibration_and_vio_never_infer_or_invent_a_map_pose(harness):
    rgb(harness)
    assert harness.statuses[-1].state == "waiting_calibration"
    harness.localizer.on_calibration(harness.calibration)
    rgb(harness)
    assert harness.statuses[-1].state == "waiting_vio"
    harness.query.assert_not_called()
    assert harness.images == harness.poses == []


def test_worker_waits_for_future_vio_without_extrapolating(harness):
    harness.localizer.on_calibration(harness.calibration)
    harness.localizer.submit_vio(sample(100, 1))
    harness.localizer.start()
    rgb(harness)
    wait_status(harness, lambda status: status.state == "localizing")
    assert harness.poses == []
    harness.localizer.submit_vio(sample(200, 2))
    wait_fix(harness)
    harness.localizer.submit_vio(sample(300, 3))
    assert harness.poses[-1].T_map_device[0, 3] == pytest.approx(11.5)


@pytest.mark.parametrize("timestamp_ms", [50, 250])
def test_missing_brackets_reject_fix_not_extrapolate(harness, timestamp_ms):
    seed(harness)
    harness.localizer.start()
    rgb(harness, timestamp_ms)
    wait_status(harness, lambda status: status.state == "fix_rejected")
    harness.localizer.submit_vio(sample(300, 3))
    assert harness.poses == []


def test_latest_one_slot_drops_intermediate_frames_and_never_queries_on_sdk_thread(
    harness,
):
    seed(harness)
    entered, release = threading.Event(), threading.Event()
    query_timestamps_ns, thread_ids = [], []
    original_result = harness.query.return_value

    def query(backend, path, frames):
        thread_ids.append(threading.get_ident())
        query_timestamps_ns.append(frames[0].corrected_timestamp_ns)
        if len(query_timestamps_ns) == 1:
            entered.set()
            assert release.wait(2.0)
        return dict(original_result)

    harness.query.side_effect = query
    harness.localizer.start()
    try:
        rgb(harness, 120)
        assert entered.wait(2.0)
        rgb(harness, 140)
        rgb(harness, 160)
        pixels = rgb(harness, 180)
        pixels[:] = 99
        assert query_timestamps_ns == [120_000_000]
        assert harness.localizer._dropped_frames == 2
        release.set()
        wait_fix(harness, 180)
    finally:
        release.set()
    assert query_timestamps_ns == [120_000_000, 180_000_000]
    assert all(thread_id != threading.get_ident() for thread_id in thread_ids)
    assert [frame.index for frame in harness.images] == [0, 3]
    assert np.all(harness.images[-1].image == 7)
    assert harness.statuses[-1].dropped_frames == 2


def test_session_reset_bad_quality_discards_inflight_old_fix_and_retired_vio(harness):
    seed(harness)
    entered, release = threading.Event(), threading.Event()
    original_result = harness.query.return_value

    def query(backend, path, frames):
        if frames[0].corrected_timestamp_ns == 150_000_000:
            entered.set()
            assert release.wait(2.0)
            # Obsolete results are discarded before their pose is consumed.
            return {"success": True, "T_world_device": [[float("nan")]]}
        return dict(original_result)

    harness.query.side_effect = query
    harness.localizer.start()
    try:
        rgb(harness)
        assert entered.wait(2.0)
        rgb(harness, 180)  # queued old-session RGB must be discarded, too
        harness.localizer.submit_vio(sample(10, session="B", quality=False))
        assert harness.statuses[-1].state == "lost"
        harness.localizer.submit_vio(sample(100, 1, session="B"))
        harness.localizer.submit_vio(sample(200, 2, session="B"))
        harness.localizer.submit_vio(sample(300, 3, session="A"))
        assert harness.localizer._session_uid == "B"
        assert harness.poses == []
        rgb(harness, 160)
        release.set()
        wait_fix(harness, 160)
    finally:
        release.set()
    assert [
        call.args[2][0].corrected_timestamp_ns for call in harness.query.call_args_list
    ] == [150_000_000, 160_000_000]
    harness.localizer.submit_vio(sample(300, 3, session="B"))
    assert len(harness.poses) == 1
    assert harness.poses[0].session_uid == "B"
    assert harness.poses[0].fix_timestamp_ns == 160_000_000


@pytest.mark.parametrize("kind", ["regular", "high-frequency"])
def test_selected_vio_adapter_preserves_composition_and_duplicate_filtering(
    harness, kind
):
    seed(harness)
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    callback, make_vio = (
        (harness.localizer.on_vio, regular)
        if kind == "regular"
        else (harness.localizer.on_vio_high_frequency, high_frequency)
    )
    callback(make_vio(210, 2.1))
    callback(make_vio(210, 999))
    callback(make_vio(220, 2.2))
    callback(make_vio(220, 999))
    callback(make_vio(205, 999))
    assert [pose.timestamp_ns for pose in harness.poses] == [210_000_000, 220_000_000]
    np.testing.assert_allclose(
        [pose.T_map_device[0, 3] for pose in harness.poses], [10.6, 10.7]
    )


@pytest.mark.parametrize("kind", ["regular", "high_frequency"])
def test_bad_quality_stops_pose_output_without_consuming_invalid_matrix(harness, kind):
    seed(harness)
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    if kind == "regular":
        vio = regular(210, quality="NOT_GOOD")
        vio.transform_odometry_bodyimu.to_matrix = Mock(
            side_effect=AssertionError("bad matrix")
        )
        harness.localizer.on_vio(vio)
    else:
        vio = high_frequency(210, quality=float("nan"))
        vio.transform_odometry_device.to_matrix = Mock(
            side_effect=AssertionError("bad matrix")
        )
        harness.localizer.on_vio_high_frequency(vio)
    harness.localizer.raise_if_failed()
    assert harness.statuses[-1].state == "lost"
    assert harness.poses == []
    harness.localizer.on_vio(regular(220, 2.2))
    assert harness.poses[-1].status == "tracking"


def test_pose_delivery_precedes_concurrent_session_reset(harness):
    seed(harness)
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    entered, release, reset_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    deliveries = []
    original_status = harness.localizer.on_status

    def on_pose(pose):
        entered.set()
        assert release.wait(2.0)
        deliveries.append(("pose", pose.session_uid))

    def on_status(status):
        deliveries.append((status.state, status.session_uid))
        original_status(status)

    def reset_session():
        harness.localizer.on_vio(regular(10, session="B", quality="BAD"))
        reset_done.set()

    harness.localizer.on_pose = on_pose
    harness.localizer.on_status = on_status
    pose_thread = threading.Thread(
        target=harness.localizer.on_vio, args=(regular(210, 2.1),)
    )
    reset_thread = threading.Thread(target=reset_session)
    try:
        pose_thread.start()
        assert entered.wait(2.0)
        reset_thread.start()
        assert not reset_done.wait(0.01)
    finally:
        release.set()
        pose_thread.join(2.0)
        if reset_thread.ident is not None:
            reset_thread.join(2.0)
    assert not pose_thread.is_alive() and not reset_thread.is_alive()
    assert deliveries == [
        ("pose", "A"),
        ("tracking", "A"),
        ("session_reset", "B"),
        ("lost", "B"),
    ]
    harness.localizer.raise_if_failed()


def test_stale_pose_is_still_emitted_and_bad_new_session_clears_it(harness):
    seed(harness)
    harness.localizer.start()
    rgb(harness)
    wait_fix(harness)
    harness.localizer.submit_vio(sample(121_000, 3))
    assert harness.poses[-1].status == "stale"
    assert harness.statuses[-1].state == "stale"
    before = len(harness.poses)
    harness.localizer.on_vio_high_frequency(
        high_frequency(10, session="B", quality=0.0)
    )
    harness.localizer.on_vio_high_frequency(high_frequency(20, session="B"))
    assert len(harness.poses) == before
    assert harness.statuses[-1].state == "lost"


def test_unsuccessful_hloc_result_has_status_and_no_fix(harness):
    seed(harness)
    harness.query.return_value = {
        "success": False,
        "reason": "insufficient_inliers",
        "query_duration_sec": 6.0,
    }
    harness.localizer.start()
    rgb(harness)
    wait_status(harness, lambda status: status.state == "localization_failed")
    assert harness.statuses[-1].detail == "insufficient_inliers"
    assert harness.poses == []


@pytest.mark.parametrize(
    "source",
    [
        "rgb",
        "calibration",
        "regular",
        "high_frequency",
        "worker",
        "status",
        "image",
        "pose",
    ],
)
def test_callback_and_worker_exceptions_surface_on_main_with_cause(harness, source):
    error = ValueError("intentional callback failure")
    if source == "rgb":
        harness.localizer.on_rgb(
            SimpleNamespace(to_numpy_array=Mock(side_effect=error)), SimpleNamespace()
        )
    elif source == "calibration":
        harness.calibration.get_camera_calib.side_effect = error
        harness.localizer.on_calibration(harness.calibration)
    elif source == "regular":
        vio = regular()
        vio.transform_odometry_bodyimu.to_matrix = Mock(side_effect=error)
        harness.localizer.on_vio(vio)
    elif source == "high_frequency":
        vio = high_frequency()
        vio.transform_odometry_device.to_matrix = Mock(side_effect=error)
        harness.localizer.on_vio_high_frequency(vio)
    elif source == "status":
        harness.localizer.on_status = Mock(side_effect=error)
        harness.localizer.on_calibration(harness.calibration)
    else:
        seed(harness)
        if source == "worker":
            harness.query.side_effect = error
        elif source == "image":
            harness.localizer.on_image = Mock(side_effect=error)
        harness.localizer.start()
        rgb(harness)
        if source == "pose":
            wait_fix(harness)
            harness.localizer.on_pose = Mock(side_effect=error)
            harness.localizer.on_vio(regular(300, 3))
        else:
            wait_status(harness, lambda status: status.state == "error")
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.raise_if_failed()
    assert captured.value.__cause__ is error
    with pytest.raises(RuntimeError):
        harness.localizer.stop()
    if harness.localizer._thread is not None:
        assert not harness.localizer._thread.is_alive()
        harness.backend.close.assert_called_once_with()
    else:
        harness.backend.close.assert_not_called()


def test_worker_status_callback_failure_closes_backend_with_original_cause(harness):
    seed(harness)
    error = ValueError("Worker status callback failed")
    original_status = harness.localizer.on_status

    def on_status(status):
        if status.state == "localizing":
            raise error
        original_status(status)

    harness.localizer.on_status = on_status
    harness.localizer.start()
    rgb(harness)
    wait_status(harness, lambda status: status.state == "error")
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.stop()
    assert captured.value.__cause__ is error
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()
    harness.query.assert_not_called()


def test_rectification_failure_closes_backend_with_original_cause(harness):
    seed(harness)
    error = ValueError("Rectification failed")
    harness.pat.distort_by_calibration.side_effect = error
    harness.localizer.start()
    rgb(harness)
    wait_status(harness, lambda status: status.state == "error")
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.stop()
    assert captured.value.__cause__ is error
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()
    harness.query.assert_not_called()


def test_stop_interrupts_bracket_wait_and_never_emits_post_stop(harness):
    harness.localizer.on_calibration(harness.calibration)
    harness.localizer.submit_vio(sample(100))
    harness.localizer._fix_wait_duration_sec = 60.0
    harness.localizer.start()
    rgb(harness)
    wait_status(harness, lambda status: status.state == "localizing")
    harness.localizer.stop()
    assert not harness.localizer._thread.is_alive()
    harness.localizer.on_vio(regular(200))
    rgb(harness)
    assert harness.poses == []
    assert harness.localizer._pending is None
    harness.localizer.stop()
    with pytest.raises(RuntimeError, match="cannot restart"):
        harness.localizer.start()


def test_join_timeout_is_not_hidden_and_thread_can_be_joined_after_release(harness):
    seed(harness)
    entered, release = threading.Event(), threading.Event()

    def query(*args):
        entered.set()
        assert release.wait(2.0)
        return {"success": False, "query_duration_sec": 0.0}

    harness.query.side_effect = query
    harness.localizer._worker_join_duration_sec = 0.01
    harness.localizer.start()
    try:
        rgb(harness)
        assert entered.wait(2.0)
        assert not harness.localizer._thread.daemon
        with pytest.raises(TimeoutError, match="did not stop"):
            harness.localizer.stop()
    finally:
        release.set()
        harness.localizer._worker_join_duration_sec = 2.0
        harness.localizer.stop()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fix_wait_duration_sec": -1},
        {"fix_wait_duration_sec": float("inf")},
        {"worker_join_duration_sec": 0},
        {"worker_join_duration_sec": float("nan")},
        {"min_high_frequency_quality": -0.1},
        {"min_high_frequency_quality": 1.1},
        {"min_high_frequency_quality": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        live.LiveLocalizer(tmp_path, **kwargs)


@pytest.fixture
def sdk_harness(monkeypatch, tmp_path):
    sdk, receiver_module, localizer = Mock(), Mock(), Mock()
    device = sdk.DeviceClient.return_value.connect.return_value
    stream_receiver = receiver_module.StreamReceiver.return_value
    stop_event = threading.Event()
    stream_receiver.start_server.side_effect = stop_event.set
    importer = Mock(
        side_effect=lambda name: {
            "aria.sdk_gen2": sdk,
            "aria.stream_receiver": receiver_module,
        }[name]
    )
    monkeypatch.setattr(live.importlib, "import_module", importer)
    constructor = Mock(return_value=localizer)
    monkeypatch.setattr(live, "LiveLocalizer", constructor)
    return SimpleNamespace(
        sdk=sdk,
        receiver_module=receiver_module,
        localizer=localizer,
        device=device,
        receiver=stream_receiver,
        event=stop_event,
        importer=importer,
        constructor=constructor,
        map_path=tmp_path / "map",
    )


def run_sdk(harness, **kwargs):
    options = {
        "device_ip": "169.254.1.2",
        "bind_address": "169.254.1.1",
        "profile": "profile9",
        "map_path": harness.map_path,
        "stop_event": harness.event,
    }
    options.update(kwargs)
    live.run_live(**options)


def test_run_live_uses_exact_ip_usb_existing_profile_and_high_frequency_only(
    sdk_harness,
):
    run_sdk(sdk_harness)
    sdk_harness.sdk.DeviceTarget.assert_called_once_with(ip="169.254.1.2")
    sdk_harness.sdk.DeviceClient.return_value.usb_network_devices.assert_not_called()
    config = sdk_harness.sdk.HttpStreamingConfig.return_value
    assert config.profile_name == "profile9"
    assert config.streaming_interface == sdk_harness.sdk.StreamingInterface.USB_NCM
    server_config = sdk_harness.sdk.HttpServerConfig.return_value
    assert server_config.address == "169.254.1.1"
    assert server_config.port == 6768
    sdk_harness.receiver.set_rgb_queue_size.assert_called_once_with(1)
    for name, callback in (
        ("register_device_calib_callback", "on_calibration"),
        ("register_rgb_callback", "on_rgb"),
        ("register_vio_high_frequency_callback", "on_vio_high_frequency"),
    ):
        getattr(sdk_harness.receiver, name).assert_called_once_with(
            getattr(sdk_harness.localizer, callback)
        )
    sdk_harness.receiver.register_vio_callback.assert_not_called()
    sdk_harness.receiver.set_vio_queue_size.assert_not_called()
    sdk_harness.receiver.set_vio_high_freq_queue_size.assert_called_once_with(
        live.HIGH_FREQUENCY_VIO_QUEUE_SIZE
    )
    assert 1 < live.HIGH_FREQUENCY_VIO_QUEUE_SIZE <= 50_000
    sdk_harness.device.stop_streaming.assert_called_once()
    sdk_harness.receiver.stop_server.assert_called_once()
    sdk_harness.localizer.stop.assert_called_once()


def test_run_live_regular_fallback_registers_exactly_one_vio_callback(sdk_harness):
    run_sdk(sdk_harness, vio_stream="regular")
    sdk_harness.receiver.register_vio_callback.assert_called_once_with(
        sdk_harness.localizer.on_vio
    )
    sdk_harness.receiver.register_vio_high_frequency_callback.assert_not_called()
    sdk_harness.receiver.set_vio_high_freq_queue_size.assert_not_called()
    sdk_harness.receiver.set_vio_queue_size.assert_called_once_with(
        live.REGULAR_VIO_QUEUE_SIZE
    )
    assert 1 < live.REGULAR_VIO_QUEUE_SIZE <= 50_000
    sdk_harness.receiver.register_rgb_callback.assert_called_once_with(
        sdk_harness.localizer.on_rgb
    )
    sdk_harness.receiver.register_device_calib_callback.assert_called_once_with(
        sdk_harness.localizer.on_calibration
    )


@pytest.mark.parametrize("vio_stream", ["", "mixed", "auto", "high_frequency", None])
def test_invalid_vio_stream_fails_before_sdk_import_or_resource_start(
    sdk_harness, vio_stream
):
    with pytest.raises(ValueError, match="vio_stream"):
        run_sdk(sdk_harness, vio_stream=vio_stream)
    sdk_harness.importer.assert_not_called()
    sdk_harness.constructor.assert_not_called()
    sdk_harness.sdk.DeviceClient.assert_not_called()


def test_unregistered_regular_future_timestamp_cannot_discard_high_frequency_tail(
    harness, sdk_harness, monkeypatch
):
    seed(harness)
    localizer = harness.localizer
    with localizer._condition:
        assert localizer._alignment.add_fix(150_000_000, transform(10), "A")
    sdk_harness.constructor.return_value = localizer
    monkeypatch.setattr(localizer, "start", Mock(return_value=localizer))
    callbacks = {}

    def register_regular(callback):
        callbacks["regular"] = callback

    def register_high_frequency(callback):
        callbacks["high-frequency"] = callback

    def deliver_sdk_samples():
        # Model independent SDK queue delivery: regular future sample is first.
        if "regular" in callbacks:
            callbacks["regular"](regular(1000, 10))
        callbacks["high-frequency"](high_frequency(210, 2.1))
        callbacks["high-frequency"](high_frequency(220, 2.2))
        sdk_harness.event.set()

    sdk_harness.receiver.register_vio_callback.side_effect = register_regular
    sdk_harness.receiver.register_vio_high_frequency_callback.side_effect = (
        register_high_frequency
    )
    sdk_harness.receiver.start_server.side_effect = deliver_sdk_samples
    run_sdk(sdk_harness)
    assert set(callbacks) == {"high-frequency"}
    sdk_harness.receiver.register_vio_callback.assert_not_called()
    assert localizer._last_vio_timestamp_ns == 220_000_000
    assert [pose.timestamp_ns for pose in harness.poses] == [210_000_000, 220_000_000]
    np.testing.assert_allclose(
        [pose.T_map_device[0, 3] for pose in harness.poses], [10.6, 10.7]
    )
    harness.query.assert_not_called()
    harness.map_loader.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "connect",
        "set_config",
        "start_device",
        "receiver_constructor",
        "start_server",
        "callback",
        "stop_device",
        "stop_receiver",
        "stop_worker",
    ],
)
def test_sdk_failures_cleanup_every_acquired_resource_independently(
    sdk_harness, failure
):
    error = RuntimeError("SDK failure")
    operations = {
        "connect": sdk_harness.sdk.DeviceClient.return_value.connect,
        "set_config": sdk_harness.device.set_streaming_config,
        "start_device": sdk_harness.device.start_streaming,
        "receiver_constructor": sdk_harness.receiver_module.StreamReceiver,
        "start_server": sdk_harness.receiver.start_server,
        "callback": sdk_harness.localizer.raise_if_failed,
        "stop_device": sdk_harness.device.stop_streaming,
        "stop_receiver": sdk_harness.receiver.stop_server,
        "stop_worker": sdk_harness.localizer.stop,
    }
    if failure == "callback":
        operations[failure].side_effect = [None, error]
    else:
        operations[failure].side_effect = error
    with pytest.raises((RuntimeError, ExceptionGroup)):
        run_sdk(sdk_harness)
    sdk_harness.localizer.stop.assert_called_once()
    if failure != "connect":
        sdk_harness.device.stop_streaming.assert_called_once()
    if failure in (
        "start_server",
        "callback",
        "stop_device",
        "stop_receiver",
        "stop_worker",
    ):
        sdk_harness.receiver.stop_server.assert_called_once()


def test_sdk_start_failure_joins_and_closes_real_localizer_backend(
    harness, sdk_harness
):
    error = RuntimeError("Device start failed")
    sdk_harness.constructor.return_value = harness.localizer
    sdk_harness.device.start_streaming.side_effect = error
    with pytest.raises(RuntimeError, match="Device start failed") as captured:
        run_sdk(sdk_harness)
    assert captured.value is error
    harness.backend_constructor.assert_called_once_with()
    harness.backend.prepare_map.assert_called_once_with(harness.localizer.map_path)
    harness.backend.close.assert_called_once_with()
    assert not harness.localizer._thread.is_alive()
    harness.query.assert_not_called()


def test_multiple_stop_failures_are_reported_together(sdk_harness):
    errors = [RuntimeError("device"), TimeoutError("receiver"), TimeoutError("worker")]
    sdk_harness.device.stop_streaming.side_effect = errors[0]
    sdk_harness.receiver.stop_server.side_effect = errors[1]
    sdk_harness.localizer.stop.side_effect = errors[2]
    with pytest.raises(ExceptionGroup) as captured:
        run_sdk(sdk_harness)
    assert list(captured.value.exceptions) == errors


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_ip": ""},
        {"device_ip": "0.0.0.0"},
        {"device_ip": "224.0.0.1"},
        {"bind_address": "0.0.0.0"},
        {"bind_address": "::"},
        {"bind_address": "127.0.0.1"},
        {"bind_address": "8.8.8.8"},
        {"profile": ""},
        {"profile": "custom.json"},
    ],
)
def test_invalid_or_public_network_configuration_never_imports_sdk(sdk_harness, kwargs):
    with pytest.raises(ValueError):
        run_sdk(sdk_harness, **kwargs)
    sdk_harness.importer.assert_not_called()


def test_freshness_timeout_boundary_and_resume_without_synthetic_poses(
    harness, monkeypatch
):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    seed(harness)
    with harness.localizer._condition:
        assert harness.localizer._alignment.add_fix(150_000_000, transform(10), "A")
    harness.localizer.on_vio(regular(210, 2.1))
    before = len(harness.statuses)
    pose_count = len(harness.poses)
    clock.return_value = 102.999
    harness.localizer.check_freshness()
    assert len(harness.statuses) == before
    clock.return_value = 103.0
    harness.localizer.check_freshness()
    status = harness.statuses[-1]
    assert (status.state, status.session_uid, status.timestamp_ns) == (
        "lost",
        "A",
        210_000_000,
    )
    assert "3.000 sec" in status.detail
    assert len(harness.poses) == pose_count
    assert harness.localizer._alignment.state == "tracking"
    harness.localizer.check_freshness()
    assert len(harness.statuses) == before + 1
    clock.return_value = 104.0
    harness.localizer.on_vio(regular(220, 2.2, quality="BAD"))
    assert harness.statuses[-1].state == "lost"
    assert harness.localizer._vio_arrival_lost
    assert len(harness.poses) == pose_count
    clock.return_value = 105.0
    harness.localizer.on_vio_high_frequency(high_frequency(230, 2.3))
    assert harness.statuses[-1].state == "tracking"
    assert not harness.localizer._vio_arrival_lost
    assert len(harness.poses) == pose_count + 1
    assert harness.poses[-1].T_map_device[0, 3] == pytest.approx(10.8)
    assert harness.poses[-1].fix_timestamp_ns == 150_000_000
    clock.return_value = 107.999
    harness.localizer.check_freshness()
    assert harness.statuses[-1].state == "tracking"
    clock.return_value = 108.0
    harness.localizer.check_freshness()
    assert harness.statuses[-1].state == "lost"


def test_retired_duplicate_and_out_of_order_vio_do_not_refresh_arrival(
    harness, monkeypatch
):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    seed(harness)
    clock.return_value = 101.0
    harness.localizer.on_vio(regular(10, session="B", quality="BAD"))
    harness.localizer.on_vio_high_frequency(high_frequency(20, session="B"))
    clock.return_value = 103.99
    harness.localizer.on_vio(regular(300, session="A"))
    harness.localizer.on_vio_high_frequency(high_frequency(20, session="B"))
    harness.localizer.on_vio(regular(15, session="B"))
    assert harness.localizer._last_vio_arrival_time_sec == 101.0
    assert harness.localizer._session_uid == "B"
    clock.return_value = 104.0
    harness.localizer.check_freshness()
    assert harness.statuses[-1].state == "lost"
    assert harness.localizer._vio_arrival_lost
    harness.localizer.on_vio(regular(310, session="A"))
    harness.localizer.on_vio(regular(20, session="B"))
    assert harness.localizer._vio_arrival_lost
    harness.localizer.on_vio(regular(30, session="B"))
    assert not harness.localizer._vio_arrival_lost
    assert harness.statuses[-1].state == "lost"  # No cross-session alignment reuse.
    assert harness.poses == []


@pytest.mark.parametrize("state", ["calibrated", "localizing", "tracking", "stale"])
def test_worker_or_calibration_status_cannot_regreen_timed_out_vio(
    harness, monkeypatch, state
):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    seed(harness)
    clock.return_value = 103.0
    harness.localizer.check_freshness()
    harness.localizer._status(state, timestamp_ns=150_000_000)
    assert harness.statuses[-1].state == "lost"
    assert "new good VIO" in harness.statuses[-1].detail
    harness.localizer._status("error", "fatal")
    assert harness.statuses[-1].state == "error"


@pytest.mark.parametrize("waiting", ["calibration", "vio", "stopped"])
def test_freshness_preserves_initial_waiting_and_ignores_clean_stop(
    harness, monkeypatch, waiting
):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    if waiting == "calibration":
        harness.localizer.on_vio(regular())
        rgb(harness)
        assert harness.statuses[-1].state == "waiting_calibration"
    elif waiting == "vio":
        harness.localizer.on_calibration(harness.calibration)
        rgb(harness)
        assert harness.statuses[-1].state == "waiting_vio"
    else:
        seed(harness)
        harness.localizer.stop()
    before = len(harness.statuses)
    clock.return_value = 1000.0
    harness.localizer.check_freshness()
    assert len(harness.statuses) == before
    assert not harness.localizer._vio_arrival_lost


def test_freshness_status_callback_failure_is_fatal_with_cause(harness, monkeypatch):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    seed(harness)
    error = RuntimeError("freshness callback failure")
    harness.localizer.on_status = Mock(side_effect=error)
    clock.return_value = 103.0
    harness.localizer.check_freshness()
    with pytest.raises(RuntimeError, match="Live localization failed") as captured:
        harness.localizer.raise_if_failed()
    assert captured.value.__cause__ is error


@pytest.mark.parametrize("vio_timeout_sec", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_vio_timeout_is_rejected(tmp_path, vio_timeout_sec):
    with pytest.raises(ValueError):
        live.LiveLocalizer(tmp_path, vio_timeout_sec=vio_timeout_sec)


def test_configurable_freshness_timeout_without_status_sink(harness, monkeypatch):
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(live.time, "monotonic", clock)
    seed(harness)
    harness.localizer._vio_timeout_sec = 5.0
    harness.localizer.on_status = None
    clock.return_value = 104.99
    harness.localizer.check_freshness()
    assert not harness.localizer._vio_arrival_lost
    clock.return_value = 105.0
    harness.localizer.check_freshness()
    assert harness.localizer._vio_arrival_lost
    harness.localizer.raise_if_failed()


def test_run_live_checks_freshness_each_main_loop_iteration(sdk_harness):
    sdk_harness.receiver.start_server.side_effect = None
    sdk_harness.event = Mock()
    sdk_harness.event.is_set.return_value = False
    sdk_harness.event.wait.side_effect = [False, False, True]
    run_sdk(sdk_harness)
    assert sdk_harness.localizer.check_freshness.call_count == 2
    assert sdk_harness.localizer.raise_if_failed.call_count == 4
    assert [call.args for call in sdk_harness.event.wait.call_args_list] == [
        (0.05,),
        (0.05,),
        (0.05,),
    ]
    sdk_harness.device.stop_streaming.assert_called_once()
    sdk_harness.receiver.stop_server.assert_called_once()


def test_already_stopped_never_imports_sdk_or_connects(sdk_harness):
    sdk_harness.event.set()
    run_sdk(sdk_harness)
    sdk_harness.importer.assert_not_called()
