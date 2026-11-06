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

"""Synthetic ABI/numerical tests, not neural accuracy or performance benchmarks."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import resident_hloc as resident


def features(count=3, size=(32, 24)):
    return {
        "keypoints": np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        "descriptors": np.arange(256 * count, dtype=np.float32).reshape(256, count),
        "scores": np.ones(count, dtype=np.float32),
        "image_size": np.array(size),
    }


@pytest.mark.parametrize("max_bytes", [0, -5, 1.0, True, None])
def test_cache_rejects_invalid_bounds(max_bytes):
    with pytest.raises(ValueError, match="positive integer"):
        resident.FeatureCache(max_bytes)


def test_cache_owns_readonly_arrays_and_true_lru_bytes():
    source = {"keypoints": np.arange(8, dtype=np.float32)}
    loader = Mock(side_effect=lambda _: source)
    cache = resident.FeatureCache(max_bytes=64)
    a = cache.get("a", loader)
    cache.get("b", loader)
    assert cache.bytes == 64
    assert a["keypoints"].base is None
    assert not np.shares_memory(a["keypoints"], source["keypoints"])
    source["keypoints"][:] = 99
    np.testing.assert_array_equal(a["keypoints"], np.arange(8))
    assert cache.get("a", loader) is a
    cache.get("c", loader)
    assert list(cache.entries) == ["a", "c"]  # b, not a, was least recently used.
    cache.get("b", loader)
    assert list(cache.entries) == ["c", "b"]
    assert loader.call_count == 4
    assert cache.bytes == sum(
        value.nbytes for entry in cache.entries.values() for value in entry.values()
    )
    with pytest.raises(ValueError, match="read-only"):
        a["keypoints"][0] = 1
    with pytest.raises(TypeError):
        a["new"] = np.zeros(1)
    snapshot = cache.snapshot()
    assert snapshot["cache_hits"] == 1
    assert snapshot["cache_misses"] == 4
    assert snapshot["cache_evictions"] == 2
    assert json.loads(json.dumps(snapshot)) == snapshot
    cache.clear()
    assert cache.bytes == 0 and not cache.entries
    assert cache.snapshot()["cache_hits"] == 0
    assert snapshot["cache_bytes"] == 64
    cache.clear()


def test_cache_oversized_items_bypass_without_evicting_and_failed_load_not_cached():
    cache = resident.FeatureCache(16)
    loader = Mock(
        side_effect=lambda name: {
            "value": np.zeros(4 if name == "small" else 5, dtype=np.float32)
        }
    )
    small = cache.get("small", loader)
    first = cache.get("large", loader)
    second = cache.get("large", loader)
    assert first is not second and not first["value"].flags["W"]
    assert list(cache.entries) == ["small"]
    assert cache.get("small", loader) is small
    assert cache.bytes == 16 and cache.oversize_bypasses == 2
    assert cache.evictions == 0
    failing_loader = Mock(side_effect=OSError("read failed"))
    with pytest.raises(OSError, match="read failed"):
        cache.get("broken", failing_loader)
    assert "broken" not in cache.entries
    assert cache.misses == 4


@pytest.fixture
def map_fixture(tmp_path):
    names = [f"r{i:02}.png" for i in range(23)]
    with h5py.File(tmp_path / "global.h5", "w") as source:
        for i, name in enumerate(names):
            source.create_group(name)["global_descriptor"] = np.array(
                [10.0, i], dtype=np.float32
            )
    with h5py.File(tmp_path / "features.h5", "w") as source:
        for name in names:
            group = source.create_group(name)
            for key, array in features().items():
                group[key] = array
    points = [SimpleNamespace(point3D_id=i, has_point3D=lambda: True) for i in range(3)]
    model = SimpleNamespace(
        images={
            i: SimpleNamespace(name=name, points2D=points)
            for i, name in reversed(list(enumerate(names)))
        },
        points3D={i: SimpleNamespace(xyz=np.array([i, 1, 2])) for i in range(3)},
        num_points3D=lambda: 3,
    )
    colmap = SimpleNamespace(Reconstruction=Mock(return_value=model))
    return tmp_path, colmap, model


def test_map_load_once_managed_h5_cache_and_old_retrieval_ranking(
    map_fixture, monkeypatch
):
    path, colmap, _ = map_fixture
    opened = []
    original_file = resident.h5py.File

    def tracked_file(*args, **kwargs):
        assert args[1] == "r"
        stream = original_file(*args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(resident.h5py, "File", tracked_file)
    session = resident.ResidentMap(path, colmap, 65536)
    assert len(opened) == 2 and not opened[0].id.valid
    assert opened[1].id.valid
    colmap.Reconstruction.assert_called_once_with(path / "sfm")
    np.testing.assert_allclose(
        np.linalg.norm(session.descriptors, axis=1), 1, atol=1e-7
    )
    assert not session.descriptors.flags["W"]
    assert session.retrieve([np.array([2.0, 0], dtype=np.float32)])["q0.png"] == [
        f"r{i:02}.png" for i in range(20)
    ]
    assert session.retrieve([np.array([0.0, 3], dtype=np.float32)])["q0.png"] == [
        f"r{i:02}.png" for i in reversed(range(3, 23))
    ]
    first = session.features("r00.png")
    assert session.features("r00.png") is first
    assert session.cache.hits == 1 and session.cache.misses == 1
    assert len(opened) == 2
    assert session.landmarks["r00.png"] == {0, 1, 2}
    np.testing.assert_array_equal(session.point_ids["r00.png"], [0, 1, 2])
    assert not session.point_ids["r00.png"].flags["W"]
    session.close()
    assert not opened[1].id.valid
    assert session.model is None and not session.images and not session.point_ids
    assert (
        not session.references
        and not session.landmarks
        and session.descriptors.size == 0
    )
    assert session.cache.bytes == 0
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.features("r00.png")
    with pytest.raises(RuntimeError, match="closed"):
        session.retrieve([])
    with pytest.raises(RuntimeError, match="closed"):
        session._read_features("r00.png")


@pytest.mark.parametrize(
    "descriptor",
    [[0.0, 0.0], [np.nan, 1.0], [1.0, np.inf], [1.0], [[1.0, 0.0]], [1e308, 1e308]],
)
def test_map_query_retrieval_rejects_invalid_descriptor(map_fixture, descriptor):
    path, colmap, _ = map_fixture
    session = resident.ResidentMap(path, colmap, 1024)
    with pytest.raises(ValueError, match="descriptor"):
        session.retrieve([np.asarray(descriptor)])
    session.close()


def test_retrieval_score_range_guard_still_applies(map_fixture):
    path, colmap, _ = map_fixture
    session = resident.ResidentMap(path, colmap, 1024)
    session.descriptors = session.descriptors * 2
    with pytest.raises(ValueError, match="scores"):
        session.retrieve([np.array([1.0, 0.0])])
    session.descriptors[0, 0] = np.inf
    with pytest.raises(ValueError, match="scores"):
        session.retrieve([np.array([1.0, 0.0])])
    session.close()


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((1, 2)),
        np.array([[np.nan, 1]]),
        np.ones((1, 1, 2)),
        np.empty((0, 2)),
        np.ones((1, 2), dtype=np.int64),
        np.array([[1e308, 1e308]]),
    ],
)
def test_map_normalization_rejects_malformed_zero_or_nonfinite_descriptors(bad):
    with pytest.raises(ValueError, match="descriptor"):
        resident.normalized_descriptors(bad)


@pytest.mark.parametrize(
    "failure",
    ["empty", "duplicate_names", "absent_landmark", "absent_global", "absent_local"],
)
def test_map_initialization_failure_releases_partial_resources(
    map_fixture, monkeypatch, failure
):
    path, colmap, model = map_fixture
    if failure == "empty":
        model.num_points3D = lambda: 0
    elif failure == "duplicate_names":
        model.images[1].name = model.images[0].name
    elif failure == "absent_landmark":
        model.points3D.pop(1)
    elif failure == "absent_global":
        (path / "global.h5").unlink()
    else:
        (path / "features.h5").unlink()
    # Track the otherwise inaccessible failed constructor via its cleanup call.
    sessions = []
    original_close = resident.ResidentMap.close

    def track_close(session):
        sessions.append(session)
        original_close(session)

    monkeypatch.setattr(resident.ResidentMap, "close", track_close)
    with pytest.raises((ValueError, FileNotFoundError)):
        resident.ResidentMap(path, colmap, 4096)
    assert len(sessions) == 1
    assert sessions[0].model is None and sessions[0].feature_file is None
    assert not sessions[0].cache.entries


def test_local_feature_h5_mismatch_and_invalid_values_not_cached(map_fixture):
    path, colmap, model = map_fixture
    model.images[0].points2D = model.images[0].points2D[:-1]
    session = resident.ResidentMap(path, colmap, 4096)
    with pytest.raises(ValueError, match="align"):
        session.features("r00.png")
    assert not session.cache.entries
    session.close()
    with h5py.File(path / "features.h5", "a") as source:
        source["r01.png"]["keypoints"][0, 0] = np.nan
    session = resident.ResidentMap(path, colmap, 4096)
    with pytest.raises(ValueError, match="finite"):
        session.features("r01.png")
    assert not session.cache.entries
    session.close()


class FakeTensor:
    """The public tensor conversion ABI only; no inference is simulated."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.device = "cpu"

    def __getitem__(self, item):
        return FakeTensor(self.array[item])

    def __len__(self):
        return len(self.array)

    @property
    def shape(self):
        return self.array.shape

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def float(self):
        return FakeTensor(self.array.astype(np.float32))

    def to(self, device, non_blocking=False):
        self.device = device
        return self


class FakeTorch:
    def __init__(self, cuda=False):
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda, synchronize=Mock(), empty_cache=Mock()
        )
        self.grad_depth = 0
        self.from_numpy = Mock(side_effect=FakeTensor)
        self.empty = Mock(
            side_effect=lambda shape: FakeTensor(np.empty(shape, dtype=np.float32))
        )

    @contextmanager
    def no_grad(self):
        self.grad_depth += 1
        try:
            yield
        finally:
            self.grad_depth -= 1


class FakeModel:
    def __init__(self, torch, output):
        self.torch = torch
        self.output = output
        self.eval_calls = 0
        self.devices = []
        self.calls = []
        self.failure = None

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, data):
        assert self.torch.grad_depth == 1
        self.calls.append(data)
        if self.failure is not None:
            raise self.failure
        return self.output


@pytest.fixture
def neural_abi(monkeypatch):
    torch = FakeTorch()
    local = FakeModel(
        torch,
        {
            key: [FakeTensor(value)]
            for key, value in features().items()
            if key != "image_size"
        },
    )
    global_model = FakeModel(torch, {"global_descriptor": FakeTensor([[3.0, 4.0]])})
    matcher_model = FakeModel(
        torch,
        {
            "matches0": FakeTensor([[2, -1, 0]]),
            "matching_scores0": FakeTensor([[0.91, 0, 0.81]]),
        },
    )
    default_conf = {
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",
    }
    extractor = SimpleNamespace(
        ImageDataset=SimpleNamespace(default_conf=default_conf),
        resize_image=Mock(),
        extractors=SimpleNamespace(__name__="hloc.extractors"),
        confs={
            "superpoint_aachen": {
                "model": {"name": "superpoint", "nms_radius": 3, "max_keypoints": 4096},
                "preprocessing": {"grayscale": True, "resize_max": 1024},
            },
            "netvlad": {
                "model": {"name": "netvlad"},
                "preprocessing": {"resize_max": 1024},
            },
        },
        dynamic_load=Mock(
            side_effect=lambda root, name: lambda conf: local
            if name == "superpoint"
            else global_model
        ),
    )
    matcher = SimpleNamespace(
        matchers=SimpleNamespace(__name__="hloc.matchers"),
        confs={
            "superpoint+lightglue": {
                "model": {"name": "lightglue", "features": "superpoint"}
            }
        },
        dynamic_load=Mock(return_value=lambda conf: matcher_model),
    )
    original_import = resident.importlib.import_module
    monkeypatch.setattr(
        resident.importlib,
        "import_module",
        lambda name: torch if name == "torch" else original_import(name),
    )
    return torch, extractor, matcher, local, global_model, matcher_model


def test_models_are_one_time_eval_no_grad_cpu_with_direct_match_outputs(neural_abi):
    torch, extractor, matcher, local, global_model, matcher_model = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    assert (
        extractor.dynamic_load.call_count == 2 and matcher.dynamic_load.call_count == 1
    )
    assert [call.args[1] for call in extractor.dynamic_load.call_args_list] == [
        "superpoint",
        "netvlad",
    ]
    for model in (local, global_model, matcher_model):
        assert model.eval_calls == 1 and model.devices == ["cpu"]
    for _ in range(2):
        query, descriptor = session.extract(np.zeros((24, 32), dtype=np.uint8))
        np.testing.assert_array_equal(query["keypoints"], features()["keypoints"])
        np.testing.assert_array_equal(query["descriptors"], features()["descriptors"])
        np.testing.assert_array_equal(query["image_size"], [32, 24])
        np.testing.assert_array_equal(descriptor, [3, 4])
        np.testing.assert_array_equal(
            session.match(query, features()), [[0, 2], [2, 0]]
        )
    assert (
        extractor.dynamic_load.call_count == 2 and matcher.dynamic_load.call_count == 1
    )
    assert local.calls[0]["image"].shape == (1, 1, 24, 32)
    assert global_model.calls[0]["image"].shape == (1, 3, 24, 32)
    assert local.calls[0]["image"].array.dtype == np.float32
    data = matcher_model.calls[-1]
    assert data["descriptors0"].shape == (1, 256, 3)
    assert data["image0"].shape == (1, 1, 24, 32)
    np.testing.assert_array_equal(data["image_size0"].array, [[32, 24]])
    assert data["image_size0"].array.dtype == np.float32
    assert torch.grad_depth == 0
    torch.cuda.synchronize.assert_not_called()
    session.close()
    session.close()
    assert session.local_model is session.global_model is session.match_model is None
    torch.cuda.empty_cache.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        session.extract(np.zeros((24, 32), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="closed"):
        session.match(features(), features())


def test_match_copies_readonly_cached_arrays_before_tensor_conversion(neural_abi):
    torch, extractor, matcher, _, _, matcher_model = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    cache = resident.FeatureCache(65536)
    reference = cache.get("a", lambda _: features())
    original = reference["descriptors"].copy()
    session.match(features(), reference)
    data = matcher_model.calls[-1]
    data["descriptors1"].array[:] = 0
    np.testing.assert_array_equal(reference["descriptors"], original)
    assert not reference["descriptors"].flags["W"]
    assert not np.shares_memory(data["descriptors1"].array, reference["descriptors"])
    assert all(call.args[0].flags["W"] for call in torch.from_numpy.call_args_list)
    session.close()


def test_cuda_selected_and_synchronized_without_global_empty_cache(neural_abi):
    torch, extractor, matcher, local, global_model, matcher_model = neural_abi
    torch.cuda.is_available = lambda: True
    session = resident.ResidentModels(extractor, matcher)
    assert session.device == "cuda"
    assert all(
        model.devices == ["cuda"] for model in (local, global_model, matcher_model)
    )
    torch.cuda.synchronize.assert_called_once()
    session.synchronize()
    assert torch.cuda.synchronize.call_count == 2
    session.close()
    torch.cuda.empty_cache.assert_not_called()


@pytest.mark.parametrize("stage", ["local", "global", "matcher"])
def test_model_initialization_failure_clears_partial_model_references(
    neural_abi, monkeypatch, stage
):
    _, extractor, matcher, local, global_model, matcher_model = neural_abi
    failed = {"local": local, "global": global_model, "matcher": matcher_model}[stage]
    failed.to = Mock(side_effect=RuntimeError("device initialization failed"))
    sessions = []
    original_close = resident.ResidentModels.close

    def tracked_close(session):
        sessions.append(session)
        original_close(session)

    monkeypatch.setattr(resident.ResidentModels, "close", tracked_close)
    with pytest.raises(RuntimeError, match="device initialization failed"):
        resident.ResidentModels(extractor, matcher)
    assert len(sessions) == 1
    assert (
        sessions[0].local_model
        is sessions[0].global_model
        is sessions[0].match_model
        is None
    )


@pytest.mark.parametrize("stage", ["local", "global", "matcher"])
def test_inference_errors_propagate_without_path_fallback(neural_abi, stage):
    _, extractor, matcher, local, global_model, matcher_model = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    failed = {"local": local, "global": global_model, "matcher": matcher_model}[stage]
    failed.failure = RuntimeError("forward failed")
    with pytest.raises(RuntimeError, match="forward failed"):
        if stage == "matcher":
            session.match(features(), features())
        else:
            session.extract(np.zeros((24, 32), dtype=np.uint8))
    session.close()


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"matches0": FakeTensor([[0, 1]])},
        {"matches0": FakeTensor([[0, 1, 3]])},
        {"matches0": FakeTensor([[0, -2, 1]])},
        {"matches0": FakeTensor([[0.0, 1.0, 2.0]])},
        None,
    ],
)
def test_invalid_match_predictions_are_deterministic(neural_abi, output):
    _, extractor, matcher, _, _, matcher_model = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    matcher_model.output = output
    with pytest.raises(ValueError, match="matcher"):
        session.match(features(), features())
    session.close()


def test_empty_keypoints_skip_match_forward(neural_abi):
    _, extractor, matcher, _, _, matcher_model = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    assert session.match(features(0), features()).shape == (0, 2)
    assert session.match(features(), features(0)).shape == (0, 2)
    assert not matcher_model.calls
    session.close()


@pytest.mark.parametrize(
    "prediction",
    [
        None,
        [],
        {1: [FakeTensor([1])]},
        {"x": []},
        {"x": [None]},
        {"x": [FakeTensor([1]), FakeTensor([2])]},
    ],
)
def test_prediction_array_boundary_rejects_malformed_batch(prediction):
    with pytest.raises(ValueError, match="prediction"):
        resident.prediction_arrays(prediction)


@pytest.mark.parametrize(
    "case",
    [
        "missing_keypoints",
        "bad_descriptor_shape",
        "bad_keypoints_shape",
        "too_many",
        "invalid_size",
        "fractional_size",
        "scores_shape",
        "nan",
        "nonnumeric",
    ],
)
def test_local_features_validate_dimensions_and_finiteness(case):
    local = features(4097 if case == "too_many" else 3)
    if case == "missing_keypoints":
        local.pop("keypoints")
    elif case == "bad_descriptor_shape":
        local["descriptors"] = local["descriptors"].T
    elif case == "bad_keypoints_shape":
        local["keypoints"] = np.zeros((3, 3))
    elif case == "invalid_size":
        local["image_size"] = np.array([0, 24])
    elif case == "fractional_size":
        local["image_size"] = np.array([32.5, 24])
    elif case == "scores_shape":
        local["scores"] = np.zeros((3, 1))
    elif case == "nan":
        local["keypoints"][0, 0] = np.nan
    elif case == "nonnumeric":
        local["scores"] = np.array(["bad"] * 3)
    with pytest.raises(ValueError):
        resident.validate_features(local)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2,), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
    ],
)
def test_preprocess_rejects_empty_or_malformed_images(neural_abi, image):
    _, extractor, _, _, _, _ = neural_abi
    with pytest.raises(ValueError, match="uint8"):
        resident.preprocess_image(image, {}, extractor)


def test_preprocess_grayscale_matches_png_decoder_truncation_and_global_rgb(neural_abi):
    _, extractor, _, _, _, _ = neural_abi
    # OpenCV libpng read_image grayscale oracle: saturated primaries and gray.
    # Green is 149, not cvtColor/PIL's rounded 150; no float grayscale rounding.
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [127, 127, 127]]], dtype=np.uint8
    )
    local = resident.preprocess_image(rgb, {"grayscale": True}, extractor)
    np.testing.assert_array_equal(
        local["image"], np.array([[[76, 149, 29, 127]]], dtype=np.float32) / 255
    )
    global_data = resident.preprocess_image(rgb, {}, extractor)
    np.testing.assert_array_equal(
        global_data["image"], rgb.astype(np.float32).transpose(2, 0, 1) / 255
    )
    mono = np.array([[0, 1, 255], [12, 63, 125]], dtype=np.uint8)
    global_mono = resident.preprocess_image(mono, {}, extractor)
    np.testing.assert_array_equal(
        global_mono["image"], np.repeat(mono[None], 3, axis=0).astype(np.float32) / 255
    )
    extractor.resize_image.assert_not_called()
    np.testing.assert_array_equal(local["original_size"], [4, 1])


def test_resize_uses_pinned_upstream_float32_area_and_original_pixel_centers(
    neural_abi,
):
    _, extractor, matcher, local, _, _ = neural_abi
    image = np.arange(24, dtype=np.uint8).reshape(4, 6)
    expected = image.astype(np.float32).reshape(2, 2, 3, 2).mean(axis=(1, 3))

    def resize(array, size, interpolation):
        assert (
            array.dtype == np.float32 and size == (3, 2) and interpolation == "cv2_area"
        )
        np.testing.assert_array_equal(array, image.astype(np.float32))
        return expected.copy()

    extractor.resize_image.side_effect = resize
    extractor.confs["superpoint_aachen"]["preprocessing"]["resize_max"] = 3
    # Keep global preprocessing unresized to test the separate configurations.
    session = resident.ResidentModels(extractor, matcher)
    local.output["scales"] = [FakeTensor(np.ones(3, dtype=np.float32))]
    query, _ = session.extract(image)
    np.testing.assert_array_equal(
        local.calls[0]["image"].array, expected[None, None] / 255
    )
    np.testing.assert_array_equal(
        query["keypoints"], (features()["keypoints"] + 0.5) * 2 - 0.5
    )
    np.testing.assert_array_equal(query["descriptors"], features()["descriptors"])
    np.testing.assert_array_equal(query["scales"], np.full(3, 2))
    np.testing.assert_array_equal(query["image_size"], [6, 4])
    session.close()


def test_forced_resize_rounds_shape_and_delegates_interpolation(neural_abi):
    _, extractor, _, _, _, _ = neural_abi
    extractor.resize_image.return_value = np.zeros((2, 6), dtype=np.float32)
    resident.preprocess_image(
        np.zeros((2, 5), dtype=np.uint8),
        {"grayscale": True, "resize_max": 6, "resize_force": True},
        extractor,
    )
    assert extractor.resize_image.call_args.args[1:] == ((6, 2), "cv2_area")


@pytest.mark.parametrize("stage", ["missing_local", "missing_global", "invalid_global"])
def test_extraction_prediction_failures_are_deterministic(neural_abi, stage):
    _, extractor, matcher, local, global_model, _ = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    if stage == "missing_local":
        local.output = {}
    elif stage == "missing_global":
        global_model.output = {}
    else:
        global_model.output = {"global_descriptor": FakeTensor([[np.nan, 1.0]])}
    with pytest.raises(ValueError, match="prediction"):
        session.extract(np.zeros((24, 32), dtype=np.uint8))
    session.close()


def test_stage_timing_units_accumulation_and_sync_overhead(monkeypatch):
    clock = iter([10.0, 10.125, 11.0, 11.5])
    monkeypatch.setattr(resident.time, "perf_counter", lambda: next(clock))
    synchronize = Mock()
    timings = resident.StageTimings(synchronize)
    with timings.stage("matching"):
        pass
    with pytest.raises(RuntimeError, match="stage failed"):
        with timings.stage("matching"):
            raise RuntimeError("stage failed")
    assert timings.durations_sec == {"matching_duration_sec": 0.625}
    assert synchronize.call_count == 4
