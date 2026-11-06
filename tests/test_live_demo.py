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
import queue
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import live_demo as live
import numpy as np
import pytest
from aria_utils import Frame
from live_stream import LiveStatus
from pose_alignment import MappedPose
from visualization import PoseWriter


@pytest.fixture
def mapped_poses():
    return [
        MappedPose(
            index * 10_000_000,
            np.eye(4),
            0,
            "session-a",
            index * 10_000_000,
            "tracking",
        )
        for index in range(121)
    ]


@pytest.fixture
def sinks(tmp_path):
    writer = PoseWriter(tmp_path / "poses.jsonl")
    viewer = Mock(spec_set=["on_pose", "on_image", "on_status", "close"])
    yield writer, viewer
    writer.close()


def test_queue_keeps_every_mapped_pose_and_dispatches_corrected_image_status(
    sinks, mapped_poses, camera
):
    writer, viewer = sinks
    stop = threading.Event()
    outputs = live.QueuedOutputs(writer, viewer, stop)
    caller_thread = threading.get_ident()
    consumer_threads = []
    viewer.on_pose.side_effect = lambda pose: consumer_threads.append(
        threading.get_ident()
    )
    image = np.zeros((2, 3), dtype=np.uint8)
    frame = Frame(camera.label, 0, 10_000_000, image, camera, 2_000_000)
    status = LiveStatus("stale", timestamp_ns=8_000_000, dropped_frames=3)
    try:
        for pose in mapped_poses:
            outputs.on_pose(pose)
        outputs.on_image(frame)
        outputs.on_status(status)
    finally:
        outputs.close()
    writer.close()
    rows = [
        json.loads(line) for line in Path(writer.target.name).read_text().splitlines()
    ]
    assert writer.count == len(rows) == 121
    assert [row["timestamp_ns"] for row in rows] == [
        pose.timestamp_ns for pose in mapped_poses
    ]
    assert viewer.on_pose.call_count == 121
    assert all(value != caller_thread for value in consumer_threads)
    assert len(set(consumer_threads)) == 1
    assert viewer.on_image.call_args.args[0] is image
    assert viewer.on_image.call_args.args[1] == 8_000_000
    viewer.on_status.assert_called_once_with(asdict(status))
    assert not outputs.thread.is_alive()
    assert not stop.is_set()


@pytest.mark.parametrize("failure", ["writer", "viewer"])
def test_output_consumer_failure_stops_stream_and_is_raised_with_original_cause(
    sinks, mapped_poses, failure
):
    writer, viewer = sinks
    error = OSError("output failed")
    if failure == "writer":
        writer.write = Mock(side_effect=error)
    else:
        viewer.on_pose.side_effect = error
    stop = threading.Event()
    outputs = live.QueuedOutputs(writer, viewer, stop)
    outputs.on_pose(mapped_poses[0])
    try:
        assert stop.wait(timeout=2)
        with pytest.raises(RuntimeError, match="consumer failed") as callback_error:
            outputs.on_status(LiveStatus("tracking"))
        assert callback_error.value.__cause__ is error
    finally:
        with pytest.raises(RuntimeError, match="output failed") as close_error:
            outputs.close()
    assert close_error.value.__cause__ is error
    assert not outputs.thread.is_alive()


def test_full_queue_stops_explicitly_and_drains_all_accepted_poses(
    monkeypatch, sinks, mapped_poses
):
    writer, viewer = sinks
    monkeypatch.setattr(
        live,
        "queue",
        SimpleNamespace(
            Queue=lambda maxsize: queue.Queue(maxsize=1),
            Full=queue.Full,
            Empty=queue.Empty,
        ),
    )
    entered, release, stop = threading.Event(), threading.Event(), threading.Event()
    original_write = writer.write

    def blocked_write(pose):
        entered.set()
        assert release.wait(timeout=2)
        original_write(pose)

    writer.write = blocked_write
    outputs = live.QueuedOutputs(writer, viewer, stop)
    try:
        outputs.on_pose(mapped_poses[0])
        assert entered.wait(timeout=2)
        outputs.on_pose(mapped_poses[1])
        with pytest.raises(RuntimeError, match="queue is full"):
            outputs.on_pose(mapped_poses[2])
        assert stop.is_set()
    finally:
        release.set()
        outputs.close()
    assert writer.count == 2
    assert viewer.on_pose.call_count == 2
    assert not outputs.thread.is_alive()


def test_close_reports_nonterminating_consumer_without_blocking_test(
    monkeypatch, sinks
):
    thread = Mock()
    thread.is_alive.return_value = True
    monkeypatch.setattr(
        live,
        "threading",
        SimpleNamespace(Thread=Mock(return_value=thread), Event=threading.Event),
    )
    writer, viewer = sinks
    outputs = live.QueuedOutputs(writer, viewer, threading.Event())
    with pytest.raises(RuntimeError, match="did not stop"):
        outputs.close()
    thread.join.assert_called_once()
    assert outputs.closing.is_set()
    assert outputs.queue.empty()


def test_closed_queue_refuses_late_callback_without_dropping_silently(
    sinks, mapped_poses
):
    writer, viewer = sinks
    outputs = live.QueuedOutputs(writer, viewer, threading.Event())
    outputs.close()
    with pytest.raises(RuntimeError, match="closing"):
        outputs.on_pose(mapped_poses[0])
    assert writer.count == 0
    assert outputs.queue.empty()
    assert not outputs.thread.is_alive()


@pytest.fixture
def cli_inputs(monkeypatch, tmp_path):
    output = tmp_path / "output"
    argv = [
        "live_demo",
        "--map",
        str(tmp_path / "map"),
        "--device-ip",
        "usb-device",
        "--bind-address",
        "usb-host",
        "--output-dir",
        str(output),
        "--viewer",
        "none",
        "--profile",
        "profile-custom",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    viewer = Mock(spec_set=["on_pose", "on_image", "on_status", "close"])
    open_viewer = Mock(return_value=viewer)
    monkeypatch.setattr(live, "RerunView", open_viewer)
    writers = []

    def open_writer(path):
        writer = PoseWriter(path)
        writers.append(writer)
        return writer

    monkeypatch.setattr(live, "PoseWriter", open_writer)
    run = Mock()
    monkeypatch.setattr(live, "run_live", run)
    return SimpleNamespace(
        output=output,
        map_path=tmp_path / "map",
        viewer=viewer,
        open_viewer=open_viewer,
        writers=writers,
        run=run,
    )


def test_live_cli_mock_sdk_publishes_every_pose_and_closes_owned_sinks(
    cli_inputs, mapped_poses, capsys
):
    inputs = cli_inputs

    def stream(
        device_ip,
        bind_address,
        profile,
        map_path,
        on_pose,
        on_image,
        on_status,
        stop,
        *,
        vio_stream,
    ):
        assert vio_stream == "high-frequency"
        assert (device_ip, bind_address, profile, map_path) == (
            "usb-device",
            "usb-host",
            "profile-custom",
            inputs.map_path,
        )
        for pose in mapped_poses:
            on_pose(pose)
        on_status(LiveStatus("tracking"))

    inputs.run.side_effect = stream
    live.main()
    assert inputs.writers[0].count == 121
    assert inputs.writers[0].target.closed
    inputs.viewer.close.assert_called_once()
    assert inputs.viewer.on_pose.call_count == 121
    assert json.loads(capsys.readouterr().out) == {
        "mapped_pose_samples": 121,
        "output": str(inputs.output),
    }
    inputs.open_viewer.assert_called_once_with(
        inputs.map_path, inputs.output / "view.rrd", viewer="none"
    )


@pytest.mark.parametrize("error", [KeyboardInterrupt(), OSError("SDK failed")])
def test_live_cli_interrupt_or_sdk_failure_always_cleans_up(cli_inputs, error):
    cli_inputs.run.side_effect = error
    if isinstance(error, KeyboardInterrupt):
        live.main()
    else:
        with pytest.raises(OSError, match="SDK failed"):
            live.main()
    assert cli_inputs.run.call_args.args[-1].is_set() or not isinstance(
        error, KeyboardInterrupt
    )
    assert cli_inputs.writers[0].target.closed
    cli_inputs.viewer.close.assert_called_once()


@pytest.mark.parametrize("failure", ["viewer", "queue"])
def test_live_cli_construction_failure_closes_previously_opened_resources(
    monkeypatch, cli_inputs, failure
):
    if failure == "viewer":
        cli_inputs.open_viewer.side_effect = ImportError("construction failed")
    else:
        monkeypatch.setattr(
            live, "QueuedOutputs", Mock(side_effect=ImportError("construction failed"))
        )
    with pytest.raises(ImportError, match="construction failed"):
        live.main()
    assert cli_inputs.writers[0].target.closed
    if failure == "queue":
        cli_inputs.viewer.close.assert_called_once()
    cli_inputs.run.assert_not_called()


def test_live_cli_consumer_error_is_not_reported_as_success(
    cli_inputs, mapped_poses, capsys
):
    cli_inputs.viewer.on_pose.side_effect = OSError("viewer failed")

    def stream(*args, **kwargs):
        assert kwargs["vio_stream"] == "high-frequency"
        args[4](mapped_poses[0])
        assert args[-1].wait(timeout=2)

    cli_inputs.run.side_effect = stream
    with pytest.raises(RuntimeError, match="output failed"):
        live.main()
    assert cli_inputs.writers[0].target.closed
    cli_inputs.viewer.close.assert_called_once()
    assert not capsys.readouterr().out


def test_live_cli_explicit_regular_stream_is_forwarded(cli_inputs, monkeypatch):
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--vio-stream", "regular"])
    live.main()
    assert cli_inputs.run.call_args.kwargs == {"vio_stream": "regular"}


def test_live_cli_does_not_overwrite_results_or_accept_missing_required_args(
    monkeypatch, cli_inputs
):
    cli_inputs.output.mkdir()
    sentinel = cli_inputs.output / "existing"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        live.main()
    assert sentinel.read_text() == "unchanged"
    cli_inputs.run.assert_not_called()
    monkeypatch.setattr(sys, "argv", ["live_demo"])
    with pytest.raises(SystemExit) as failure:
        live.main()
    assert failure.value.code == 2
