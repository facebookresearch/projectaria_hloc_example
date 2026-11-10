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

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import aria_utils as aria
import numpy as np
import pytest
from map_utils import SLAM_LABELS


@pytest.fixture
def pat(monkeypatch, camera):
    source = Mock()
    source.get_image_size.return_value = (32, 24)
    source.get_focal_lengths.return_value = [20, 20]
    source.get_time_offset_sec_device_camera.return_value = 0.0
    source.get_transform_device_camera.return_value.to_matrix.return_value = (
        camera.T_device_camera
    )
    sample = SimpleNamespace(
        tracking_timestamp=timedelta(milliseconds=1),
        get_camera_calib=Mock(return_value=source),
    )
    device = Mock()
    device.get_camera_labels.return_value = SLAM_LABELS + ("camera-rgb",)
    device.get_device_version.return_value = "Gen2"
    provider = Mock()
    provider.get_device_calibration.return_value = device
    provider.get_stream_id_from_label.side_effect = lambda label: label
    provider.get_num_data.return_value = 3
    provider.get_timestamps_ns.return_value = [1_000_000, 2_000_000, 3_000_000]
    data = Mock()
    data.is_valid.return_value = True
    data.to_numpy_array.return_value = np.arange(32 * 24, dtype=np.uint8).reshape(
        24, 32
    )
    provider.get_image_data_by_index.return_value = (
        data,
        SimpleNamespace(capture_timestamp_ns=1_000_000),
    )
    calibration = Mock()
    calibration.rescale_camera_calibration.return_value = source
    calibration.get_linear_camera_calibration.return_value.get_principal_point.return_value = [
        255.5,
        255.5,
    ]
    calibration.distort_by_calibration.side_effect = (
        lambda image, target, original: image.copy()
    )
    modules = {
        "projectaria_tools.core.data_provider": SimpleNamespace(
            create_vrs_data_provider=Mock(return_value=provider)
        ),
        "projectaria_tools.core.calibration": calibration,
        "projectaria_tools.core.mps": SimpleNamespace(
            read_online_calibration=Mock(return_value=[sample])
        ),
        "projectaria_tools.core.sensor_data": SimpleNamespace(
            TimeDomain=SimpleNamespace(DEVICE_TIME="DEVICE_TIME")
        ),
    }
    original_import = aria.importlib.import_module
    imports = Mock(
        side_effect=lambda name: modules[name]
        if name in modules
        else original_import(name)
    )
    monkeypatch.setattr(aria.importlib, "import_module", imports)
    return SimpleNamespace(
        modules=modules,
        provider=provider,
        device=device,
        data=data,
        source=source,
        sample=sample,
        calibration=calibration,
        imports=imports,
    )


@pytest.fixture
def recording(pat, tmp_path):
    return aria.Recording(tmp_path / "input.vrs", tmp_path / "online.jsonl")


def test_recording_reads_only_its_explicit_local_inputs(pat, recording, tmp_path):
    pat.modules[
        "projectaria_tools.core.data_provider"
    ].create_vrs_data_provider.assert_called_once_with(str(tmp_path / "input.vrs"))
    pat.modules[
        "projectaria_tools.core.mps"
    ].read_online_calibration.assert_called_once_with(str(tmp_path / "online.jsonl"))
    assert recording.calibration_timestamps_ns == [1_000_000]
    assert recording.stream_ids == {}


@pytest.mark.parametrize(
    "case,error",
    [
        ("provider", "Could not open"),
        ("device", "device calibration"),
        ("empty", "empty"),
        ("duplicates", "increasing"),
        ("descending", "increasing"),
    ],
)
def test_recording_rejects_invalid_inputs(pat, tmp_path, case, error):
    mps = pat.modules["projectaria_tools.core.mps"]
    if case == "provider":
        pat.modules[
            "projectaria_tools.core.data_provider"
        ].create_vrs_data_provider.return_value = None
    elif case == "device":
        pat.provider.get_device_calibration.return_value = None
    elif case == "empty":
        mps.read_online_calibration.return_value = []
    elif case == "duplicates":
        mps.read_online_calibration.return_value = [pat.sample, pat.sample]
    else:
        mps.read_online_calibration.return_value = [
            SimpleNamespace(tracking_timestamp=timedelta(milliseconds=2)),
            pat.sample,
        ]
    with pytest.raises(ValueError, match=error):
        aria.Recording(tmp_path / "bad.vrs", tmp_path / "bad.jsonl")


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("rgb", ("camera-rgb",)),
        ("slam", SLAM_LABELS),
        ("all", SLAM_LABELS + ("camera-rgb",)),
    ],
)
def test_select_labels_requires_all_streams_and_device_clock(
    recording, pat, mode, expected
):
    assert recording.select_labels(mode) == expected
    assert set(recording.stream_ids) == set(expected)
    for label in expected:
        assert recording.timestamps_ns[label] == [1_000_000, 2_000_000, 3_000_000]
        assert (
            recording.corrected_timestamps_ns[label] == recording.timestamps_ns[label]
        )
    assert all(
        call.args[1] == "DEVICE_TIME"
        for call in pat.provider.get_timestamps_ns.call_args_list
    )


@pytest.mark.parametrize(
    "case,error",
    [
        ("missing", "missing streams"),
        ("no_stream", "no images"),
        ("zero_images", "no images"),
        ("duplicate_clock", "increasing"),
        ("descending_clock", "increasing"),
    ],
)
def test_select_labels_does_not_silently_drop_cameras(recording, pat, case, error):
    if case == "missing":
        pat.device.get_camera_labels.return_value = SLAM_LABELS[:-1]
    elif case == "no_stream":
        pat.provider.get_stream_id_from_label.side_effect = None
        pat.provider.get_stream_id_from_label.return_value = None
    elif case == "zero_images":
        pat.provider.get_num_data.return_value = 0
    elif case == "duplicate_clock":
        pat.provider.get_timestamps_ns.return_value = [1, 1]
    else:
        pat.provider.get_timestamps_ns.return_value = [2, 1]
    with pytest.raises(ValueError, match=error):
        recording.select_labels("slam")


def test_online_calibration_nearest_sample_and_rescale(recording, pat):
    later = SimpleNamespace(get_camera_calib=Mock(return_value=pat.source))
    recording.calibrations.append(later)
    recording.calibration_timestamps_ns.append(3_000_000)
    assert recording._source_calibration("camera-rgb", 2_900_000, 32, 24) is pat.source
    later.get_camera_calib.assert_called_once_with("camera-rgb")
    pat.calibration.rescale_camera_calibration.assert_not_called()
    assert recording._source_calibration("camera-rgb", 1_000_000, 64, 48) is pat.source
    pat.calibration.rescale_camera_calibration.assert_called_once_with(
        pat.source, (64, 48), "Gen2", "camera-rgb"
    )


def test_online_calibration_gap_boundary_and_missing_camera(recording, pat):
    assert (
        recording._source_calibration(
            "camera-rgb", 1_000_000 + aria.MAX_CALIBRATION_GAP_NS, 32, 24
        )
        is pat.source
    )
    with pytest.raises(ValueError, match="No online calibration"):
        recording._source_calibration(
            "camera-rgb", 1_000_001 + aria.MAX_CALIBRATION_GAP_NS, 32, 24
        )
    pat.sample.get_camera_calib.return_value = None
    with pytest.raises(ValueError, match="missing camera-rgb"):
        recording._source_calibration("camera-rgb", 1_000_000, 32, 24)


@pytest.mark.parametrize(
    "label,size,focal",
    [("camera-rgb", (1024, 768), 360.0), ("slam-front-left", (512, 512), 320.0)],
)
def test_frame_rectification_keeps_recorded_pose_and_pixel_convention(
    recording, pat, camera, label, size, focal
):
    recording.select_labels("all")
    frame = recording.frame(label, 0)
    assert (frame.label, frame.index, frame.capture_timestamp_ns) == (
        label,
        0,
        1_000_000,
    )
    assert (frame.camera.width, frame.camera.height, frame.camera.focal) == (
        *size,
        focal,
    )
    assert frame.camera.cx == 255.5
    assert frame.camera.colmap_params()[2] == 256.0
    np.testing.assert_array_equal(frame.camera.T_device_camera, camera.T_device_camera)
    np.testing.assert_array_equal(frame.image, pat.data.to_numpy_array.return_value)
    pat.calibration.get_linear_camera_calibration.assert_called_once_with(
        *size, focal, label
    )
    pat.calibration.distort_by_calibration.assert_called_once()


@pytest.mark.parametrize(
    "case", ["invalid", "float", "one_dimensional", "four_dimensional"]
)
def test_frame_rejects_invalid_pixels(recording, pat, case):
    recording.select_labels("rgb")
    if case == "invalid":
        pat.data.is_valid.return_value = False
    elif case == "float":
        pat.data.to_numpy_array.return_value = np.zeros((24, 32), dtype=np.float32)
    elif case == "one_dimensional":
        pat.data.to_numpy_array.return_value = np.zeros(32, dtype=np.uint8)
    else:
        pat.data.to_numpy_array.return_value = np.zeros((1, 24, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid|uint8"):
        recording.frame("camera-rgb", 0)


def test_rgb_image_array_is_supported(recording, pat):
    recording.select_labels("rgb")
    pat.data.to_numpy_array.return_value = np.zeros((24, 32, 3), dtype=np.uint8)
    assert recording.frame("camera-rgb", 1).image.shape == (24, 32, 3)


def test_frameset_uses_nearest_image_in_each_recorded_camera(
    recording, monkeypatch, camera
):
    recording.select_labels("slam")
    recording.timestamps_ns = {
        SLAM_LABELS[0]: [10_000_000, 20_000_000],
        SLAM_LABELS[1]: [19_900_000, 30_000_000],
        SLAM_LABELS[2]: [10_000_000, 20_100_000],
        SLAM_LABELS[3]: [19_800_000],
    }
    recording.corrected_timestamps_ns = recording.timestamps_ns.copy()
    frame = Mock(
        side_effect=lambda label, index: aria.Frame(
            label,
            index,
            recording.timestamps_ns[label][index],
            np.zeros((2, 2), dtype=np.uint8),
            camera,
        )
    )
    monkeypatch.setattr(recording, "frame", frame)
    frames = recording.frameset(SLAM_LABELS, 1)
    assert [value.index for value in frames] == [1, 0, 1, 0]
    assert [value.capture_timestamp_ns for value in frames] == [
        20_000_000,
        19_900_000,
        20_100_000,
        19_800_000,
    ]


@pytest.mark.parametrize("span_ns,accepted", [(2_000_000, True), (2_000_001, False)])
def test_frameset_checks_actual_capture_span_not_timestamp_lookup(
    recording, monkeypatch, camera, span_ns, accepted
):
    recording.select_labels("slam")
    monkeypatch.setattr(
        recording,
        "frame",
        lambda label, index: aria.Frame(
            label,
            index,
            1_000_000 + (span_ns if label == SLAM_LABELS[-1] else 0),
            np.zeros((2, 2), dtype=np.uint8),
            camera,
        ),
    )
    if accepted:
        assert len(recording.frameset(SLAM_LABELS, 0)) == 4
    else:
        with pytest.raises(ValueError, match="more than 2 ms"):
            recording.frameset(SLAM_LABELS, 0)


@pytest.mark.parametrize(
    "offset_sec,offset_ns",
    [(0.0, 0), (0.002, 2_000_000), (-0.003, -3_000_000), (1.6e-9, 2)],
)
def test_frame_keeps_raw_timestamp_and_applies_signed_online_offset(
    recording, pat, offset_sec, offset_ns
):
    pat.source.get_time_offset_sec_device_camera.return_value = offset_sec
    recording.select_labels("rgb")
    frame = recording.frame("camera-rgb", 0)
    assert frame.capture_timestamp_ns == 1_000_000
    assert frame.time_offset_ns == offset_ns
    assert frame.corrected_timestamp_ns == 1_000_000 - offset_ns
    assert recording.corrected_timestamps_ns["camera-rgb"] == [
        value - offset_ns for value in recording.timestamps_ns["camera-rgb"]
    ]


def test_corrected_cache_uses_nearest_online_sample_for_each_raw_timestamp(
    recording, pat
):
    pat.source.get_time_offset_sec_device_camera.return_value = 0.0002
    later_camera = SimpleNamespace(get_time_offset_sec_device_camera=lambda: 0.0004)
    recording.calibrations.append(
        SimpleNamespace(get_camera_calib=lambda label: later_camera)
    )
    recording.calibration_timestamps_ns.append(3_000_000)
    pat.provider.get_timestamps_ns.return_value = [1_000_000, 2_900_000, 3_000_000]
    recording.select_labels("rgb")
    assert recording.timestamps_ns["camera-rgb"] == [1_000_000, 2_900_000, 3_000_000]
    assert recording.corrected_timestamps_ns["camera-rgb"] == [
        800_000,
        2_500_000,
        2_600_000,
    ]


@pytest.mark.parametrize("later_offset_sec", [0.001, 0.002])
def test_corrected_clock_must_remain_strictly_increasing(
    recording, pat, later_offset_sec
):
    later_camera = SimpleNamespace(
        get_time_offset_sec_device_camera=lambda: later_offset_sec
    )
    recording.calibrations.append(
        SimpleNamespace(get_camera_calib=lambda label: later_camera)
    )
    recording.calibration_timestamps_ns.append(2_000_000)
    pat.provider.get_timestamps_ns.return_value = [1_000_000, 2_000_000]
    with pytest.raises(ValueError, match="Corrected image timestamps"):
        recording.select_labels("rgb")
    assert "camera-rgb" not in recording.corrected_timestamps_ns


def test_nonempty_image_count_cannot_hide_empty_timestamp_list(recording, pat):
    pat.provider.get_timestamps_ns.return_value = []
    with pytest.raises(ValueError, match="nonempty"):
        recording.select_labels("rgb")


@pytest.mark.parametrize(
    "corrected_sync",
    [True, False],
    ids=["raw-skew-corrected-sync", "raw-sync-corrected-skew"],
)
def test_frameset_synchronization_uses_signed_corrected_times(
    recording, pat, monkeypatch, camera, corrected_sync
):
    offsets_ns = (
        [0, 3_000_000, 6_000_000, 9_000_000]
        if corrected_sync
        else [0, 0, 0, -3_000_000]
    )
    offsets = dict(zip(SLAM_LABELS, offsets_ns))
    raw = {
        label: [10_000_000 + (offsets[label] if corrected_sync else 0)]
        for label in SLAM_LABELS
    }
    pat.sample.get_camera_calib.side_effect = lambda label: SimpleNamespace(
        get_time_offset_sec_device_camera=lambda: offsets[label] / 1e9
    )
    pat.provider.get_timestamps_ns.side_effect = lambda stream, domain: raw[stream]
    recording.select_labels("slam")
    monkeypatch.setattr(
        recording,
        "frame",
        lambda label, index: aria.Frame(
            label,
            index,
            raw[label][index],
            np.zeros((2, 2), dtype=np.uint8),
            camera,
            offsets[label],
        ),
    )
    if corrected_sync:
        frames = recording.frameset(SLAM_LABELS, 0)
        assert [frame.corrected_timestamp_ns for frame in frames] == [10_000_000] * 4
        assert (
            max(frame.capture_timestamp_ns for frame in frames)
            - min(frame.capture_timestamp_ns for frame in frames)
            == 9_000_000
        )
    else:
        assert len({values[0] for values in recording.timestamps_ns.values()}) == 1
        with pytest.raises(ValueError, match="more than 2 ms"):
            recording.frameset(SLAM_LABELS, 0)


def test_frameset_nearest_selection_uses_corrected_not_raw_clock(
    recording, pat, monkeypatch, camera
):
    raw = {label: [10_000_000] for label in SLAM_LABELS}
    raw[SLAM_LABELS[1]] = [9_000_000, 15_000_000]
    offsets = {
        label: 5_000_000 if label == SLAM_LABELS[1] else 0 for label in SLAM_LABELS
    }
    pat.sample.get_camera_calib.side_effect = lambda label: SimpleNamespace(
        get_time_offset_sec_device_camera=lambda: offsets[label] / 1e9
    )
    pat.provider.get_timestamps_ns.side_effect = lambda stream, domain: raw[stream]
    recording.select_labels("slam")
    monkeypatch.setattr(
        recording,
        "frame",
        lambda label, index: aria.Frame(
            label,
            index,
            raw[label][index],
            np.zeros((2, 2), dtype=np.uint8),
            camera,
            offsets[label],
        ),
    )
    frames = recording.frameset(SLAM_LABELS, 0)
    assert [frame.index for frame in frames] == [0, 1, 0, 0]
    assert [frame.corrected_timestamp_ns for frame in frames] == [10_000_000] * 4
    assert frames[1].capture_timestamp_ns == 15_000_000


def test_frames_outside_online_calibration_are_marked_not_fatal(recording, pat):
    uncovered_ns = 1_000_001 + aria.MAX_CALIBRATION_GAP_NS
    pat.provider.get_timestamps_ns.return_value = [1_000_000, 2_000_000, uncovered_ns]
    assert recording.select_labels("rgb") == ("camera-rgb",)
    assert [recording.has_calibration("camera-rgb", i) for i in range(3)] == [
        True,
        True,
        False,
    ]
    assert recording.corrected_timestamps_ns["camera-rgb"][2] == uncovered_ns
    with pytest.raises(ValueError, match="No online calibration"):
        recording.time_offset_ns("camera-rgb", uncovered_ns)


def test_stream_without_any_calibrated_frame_is_rejected(recording, pat):
    start_ns = 1_000_001 + aria.MAX_CALIBRATION_GAP_NS
    pat.provider.get_timestamps_ns.return_value = [start_ns, start_ns + 1_000_000]
    with pytest.raises(ValueError, match="No camera-rgb frames are covered"):
        recording.select_labels("rgb")
