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

"""Stream Aria VIO into a HLOC map frame; publish JSONL and visualize with Rerun."""

from __future__ import annotations

import argparse
import json
import queue
import threading
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import cast

from aria_utils import Frame
from live_stream import LiveStatus, run_live
from pose_alignment import MappedPose
from visualization import PoseWriter, RerunView


class QueuedOutputs:
    """Bounded callback handoff: disk and GUI work never runs on SDK callback threads."""

    def __init__(
        self, writer: PoseWriter, viewer: RerunView, stop_event: threading.Event
    ) -> None:
        self.writer = writer
        self.viewer = viewer
        self.stop_event = stop_event
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=50_000)
        self.closing = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._consume, name="mapped-pose-output")
        self.thread.start()

    def _put(self, kind: str, value: object) -> None:
        if self.closing.is_set():
            raise RuntimeError("Output consumer is closing")
        if self.error is not None:
            raise RuntimeError("Output consumer failed") from self.error
        try:
            self.queue.put_nowait((kind, value))
        except queue.Full as error:
            self.stop_event.set()
            raise RuntimeError(
                "Mapped pose output queue is full; refusing to silently drop poses"
            ) from error

    def on_pose(self, pose: MappedPose) -> None:
        self._put("pose", pose)

    def on_image(self, frame: Frame) -> None:
        self._put("image", frame)

    def on_status(self, status: LiveStatus) -> None:
        self._put("status", status)

    def _consume(self) -> None:
        try:
            while True:
                try:
                    kind, value = self.queue.get(timeout=0.1)
                except queue.Empty:
                    if self.closing.is_set():
                        return
                    continue
                if kind == "pose":
                    pose = cast(MappedPose, value)
                    self.writer.write(pose)
                    self.viewer.on_pose(pose)
                elif kind == "image":
                    frame = cast(Frame, value)
                    self.viewer.on_image(frame.image, frame.corrected_timestamp_ns)
                else:
                    self.viewer.on_status(asdict(cast(LiveStatus, value)))
        except Exception as error:
            # This owned thread reports fatal output failure; the stream must stop.
            self.error = error
            self.stop_event.set()

    def close(self) -> None:
        self.closing.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("Output thread did not stop")
        if self.error is not None:
            raise RuntimeError("Mapped pose output failed") from self.error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument(
        "--device-ip",
        required=True,
        help="Explicit paired Aria Gen 2 USB device address",
    )
    parser.add_argument(
        "--bind-address",
        required=True,
        help="Host's USB-NCM interface address; no wildcard/public bind",
    )
    parser.add_argument(
        "--profile", default="profile9", help="Existing profile supplying RGB and VIO"
    )
    parser.add_argument(
        "--vio-stream",
        choices=("high-frequency", "regular"),
        default="high-frequency",
        help="One authoritative stream; regular is an explicit lower-rate fallback",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--viewer", choices=("native", "none"), default="native")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stop_event = threading.Event()
    with ExitStack() as cleanup:
        writer = PoseWriter(args.output_dir / "poses.jsonl")
        cleanup.callback(writer.close)
        viewer = RerunView(args.map, args.output_dir / "view.rrd", viewer=args.viewer)
        cleanup.callback(viewer.close)
        outputs = QueuedOutputs(writer, viewer, stop_event)
        cleanup.callback(outputs.close)
        try:
            run_live(
                args.device_ip,
                args.bind_address,
                args.profile,
                args.map,
                outputs.on_pose,
                outputs.on_image,
                outputs.on_status,
                stop_event,
                vio_stream=args.vio_stream,
            )
        except KeyboardInterrupt:
            stop_event.set()
    print(
        json.dumps(
            {"mapped_pose_samples": writer.count, "output": str(args.output_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
