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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import check_inputs as checker
import pytest


@pytest.mark.parametrize("failure", [None, "no_coverage", "decoded_uncovered"])
def test_checks_recorded_cameras_without_inference(tmp_path, monkeypatch, failure):
    calls = []

    class Recording:
        def __init__(self, vrs, online):
            calls.append((vrs, online))
            self.corrected_timestamps_ns = {
                "camera-rgb": [10, 20, 30],
                "slam-front-left": [10, 20, 30],
            }

        def select_labels(self, mode):
            assert mode == "all"
            return tuple(self.corrected_timestamps_ns)

        def frame(self, label, index):
            calls.append((label, index))
            return SimpleNamespace(
                corrected_timestamp_ns=200 if failure == "decoded_uncovered" else 20
            )

    trajectory = SimpleNamespace(
        timestamps_ns=[10, 20, 30],
        at=lambda timestamp_ns: None
        if failure == "no_coverage" or timestamp_ns == 200
        else "pose",
    )
    monkeypatch.setattr(checker, "mps_slam_dir", lambda path: path / "slam")
    monkeypatch.setattr(checker, "Trajectory", lambda path: trajectory)
    monkeypatch.setattr(checker, "Recording", Recording)
    if failure:
        with pytest.raises(ValueError, match="covered|coverage"):
            checker.check_inputs(tmp_path / "scan.vrs", tmp_path / "mps")
    else:
        result = checker.check_inputs(tmp_path / "scan.vrs", tmp_path / "mps")
        assert result == {
            "camera_frames": {"camera-rgb": 3, "slam-front-left": 3},
            "trajectory_samples": 3,
        }
        assert calls[1:] == [("camera-rgb", 1), ("slam-front-left", 1)]


def test_input_check_cli_explicit_testing_sequence(monkeypatch, capsys):
    run = Mock(return_value={"trajectory_samples": 7})
    monkeypatch.setattr(checker, "check_inputs", run)
    checker.main(
        ["--vrs", "data/pilot/play_1/video.vrs", "--mps", "data/pilot/play_1/mps"]
    )
    run.assert_called_once_with(
        Path("data/pilot/play_1/video.vrs"), Path("data/pilot/play_1/mps")
    )
    assert json.loads(capsys.readouterr().out) == {"trajectory_samples": 7}


def test_input_check_cli_success_and_failure(monkeypatch, capsys):
    run = Mock(return_value={"trajectory_samples": 3})
    monkeypatch.setattr(checker, "check_inputs", run)
    checker.main([])
    run.assert_called_once_with(
        Path("data/pilot/play_0/video.vrs"), Path("data/pilot/play_0/mps")
    )
    assert json.loads(capsys.readouterr().out) == {"trajectory_samples": 3}

    def fail(*paths):
        raise ValueError("missing sample")

    monkeypatch.setattr(checker, "check_inputs", fail)
    with pytest.raises(SystemExit) as caught:
        checker.main([])
    assert caught.value.code == 2
    assert "missing sample" in capsys.readouterr().err
