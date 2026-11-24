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
import signal
import threading
import time
from contextlib import contextmanager, ExitStack
from functools import partial
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


class FakeStream:
    def __init__(self, device):
        self.device = device
        self.synchronize = Mock()


class FakeCuda:
    """The CUDA calls the resident models make; device and stream are thread-local."""

    def __init__(self, available):
        self.is_available = lambda: available
        self.synchronize = Mock()
        self.empty_cache = Mock()
        self.current_device = Mock(return_value=1)
        self.Stream = Mock(side_effect=lambda device: FakeStream(device))
        self.local = threading.local()

    @contextmanager
    def device(self, index):
        previous = getattr(self.local, "device", None)
        self.local.device = index
        try:
            yield
        finally:
            self.local.device = previous

    @contextmanager
    def stream(self, stream):
        previous = getattr(self.local, "stream", None)
        self.local.stream = stream
        try:
            yield
        finally:
            self.local.stream = previous

    def active(self):
        return getattr(self.local, "device", None), getattr(self.local, "stream", None)


class FakeTorch:
    def __init__(self, cuda=False):
        self.cuda = FakeCuda(cuda)
        self.grad = threading.local()
        self.from_numpy = Mock(side_effect=FakeTensor)
        self.empty = Mock(
            side_effect=lambda shape: FakeTensor(np.empty(shape, dtype=np.float32))
        )

    @property
    def grad_depth(self):
        return getattr(self.grad, "depth", 0)

    @contextmanager
    def no_grad(self):
        self.grad.depth = self.grad_depth + 1
        try:
            yield
        finally:
            self.grad.depth -= 1


class FakeModel:
    def __init__(self, torch, output):
        self.torch = torch
        self.output = output
        self.eval_calls = 0
        self.devices = []
        self.calls = []
        self.contexts = []
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
        self.contexts.append(self.torch.cuda.active())
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
    assert session.device == "cuda" and session.device_index == 1
    assert all(
        model.devices == ["cuda"] for model in (local, global_model, matcher_model)
    )
    torch.cuda.synchronize.assert_called_once_with(1)
    torch.cuda.current_device.return_value = 0
    session.synchronize()
    assert torch.cuda.synchronize.call_count == 2
    # Later syncs keep the weights' device even if the caller's default changed.
    torch.cuda.synchronize.assert_called_with(1)
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


def test_stage_failure_survives_a_failing_cleanup_sync_and_is_timed(monkeypatch):
    clock = iter([10.0, 10.25, 11.0])
    monkeypatch.setattr(resident.time, "perf_counter", lambda: next(clock))
    synchronize = Mock(side_effect=[None, RuntimeError("device sync failed")])
    timings = resident.StageTimings(synchronize)
    with pytest.raises(ValueError, match="worker failed") as raised:
        with timings.stage("extraction"):
            raise ValueError("worker failed")
    assert any("device sync failed" in note for note in raised.value.__notes__)
    assert timings.durations_sec == {"extraction_duration_sec": 0.25}
    synchronize.side_effect = [None, RuntimeError("success-path sync failed")]
    with pytest.raises(RuntimeError, match="success-path sync failed"):
        with timings.stage("matching"):
            pass
    assert "matching_duration_sec" not in timings.durations_sec


def test_fatal_stage_cleanup_keeps_the_body_error_and_records_elapsed(monkeypatch):
    clock = iter([20.0, 20.5])
    monkeypatch.setattr(resident.time, "perf_counter", lambda: next(clock))
    synchronize = Mock(side_effect=[None, KeyboardInterrupt("sync interrupted")])
    timings = resident.StageTimings(synchronize)
    with pytest.raises(KeyboardInterrupt) as raised:
        with timings.stage("matching"):
            raise ValueError("worker failed")
    assert isinstance(raised.value.__context__, ValueError)
    assert timings.durations_sec == {"matching_duration_sec": 0.5}


@pytest.mark.parametrize(
    "requested,device,cameras,mode,workers,reason",
    [
        ("auto", "cuda", 2, "parallel", 2, "auto"),
        ("auto", "cuda", 4, "parallel", 4, "auto"),
        ("parallel", "cuda", 5, "parallel", 4, "parallel"),
        ("auto", "cpu", 2, "serial", 1, "requires_cuda"),
        ("parallel", "cpu", 4, "serial", 1, "requires_cuda"),
        ("serial", "cuda", 4, "serial", 1, "requested"),
        ("auto", "cuda", 1, "serial", 1, "single_camera"),
        ("parallel", "cuda", 0, "serial", 1, "single_camera"),
    ],
)
def test_rig_plan_runs_cameras_in_parallel_only_for_cuda_rigs(
    requested, device, cameras, mode, workers, reason
):
    assert resident.rig_execution_plan(requested, device, cameras) == {
        "requested": requested,
        "mode": mode,
        "workers": workers,
        "cameras": cameras,
        "device": device,
        "reason": reason,
    }


@pytest.mark.parametrize("requested", ["fast", "", None, "SERIAL"])
def test_rig_plan_rejects_unknown_modes(requested):
    with pytest.raises(ValueError, match="rig_execution"):
        resident.rig_execution_plan(requested, "cuda", 2)


@pytest.fixture
def cuda_session(neural_abi):
    torch, extractor, matcher, *_ = neural_abi
    torch.cuda.is_available = lambda: True
    session = resident.ResidentModels(extractor, matcher)
    yield torch, session
    session.close()


def live_rig_workers():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("aria-hloc-rig")
    ]


def test_parallel_tasks_overlap_keep_order_and_use_one_stream_per_worker(
    cuda_session,
):
    torch, session = cuda_session
    barrier = threading.Barrier(2, timeout=10)

    def camera(name):
        # Both cameras must be in flight at once, or the barrier times out.
        barrier.wait()
        return name, threading.current_thread().name, torch.cuda.active()

    syncs = torch.cuda.synchronize.call_count
    results = session.run_parallel(
        [partial(camera, "left"), partial(camera, "right")], 2
    )
    assert [name for name, _, _ in results] == ["left", "right"]
    streams = [stream for _, _, (_, stream) in results]
    assert streams == session._rig_streams and streams[0] is not streams[1]
    assert all(device == 1 for _, _, (device, _) in results)
    assert all(stream.device == 1 for stream in streams)
    assert all(name.startswith("aria-hloc-rig") for _, name, _ in results)
    assert [stream.synchronize.call_count for stream in streams] == [1, 1]
    assert torch.cuda.synchronize.call_count == syncs + 2
    torch.cuda.synchronize.assert_called_with(1)
    assert torch.cuda.active() == (None, None)
    assert not live_rig_workers()
    again = session.run_parallel([lambda: torch.cuda.active()[1]] * 2, 2)
    assert again == streams
    assert torch.cuda.Stream.call_count == 2


def test_more_tasks_than_workers_partition_statically_onto_bounded_streams(
    cuda_session,
):
    torch, session = cuda_session
    tasks = [
        partial(lambda index: (index, torch.cuda.active()[1]), index)
        for index in range(4)
    ]
    results = session.run_parallel(tasks, 2)
    left, right = session._rig_streams
    assert results == [(0, left), (1, right), (2, left), (3, right)]
    assert torch.cuda.Stream.call_count == 2


def test_parallel_failure_drains_workers_and_raises_lowest_camera_task(cuda_session):
    torch, session = cuda_session
    later_failed = threading.Event()
    finished = []

    def first():
        assert later_failed.wait(10)
        raise RuntimeError("camera 0 failed")

    def second():
        later_failed.set()
        raise ValueError("camera 1 failed")

    def third():
        assert later_failed.wait(10)
        finished.append("camera 2")
        return "ok"

    syncs = torch.cuda.synchronize.call_count
    with pytest.raises(RuntimeError, match="camera 0 failed") as raised:
        session.run_parallel([first, second, third], 3)
    assert any("camera task 1 also failed" in note for note in raised.value.__notes__)
    assert finished == ["camera 2"]
    assert [stream.synchronize.call_count for stream in session._rig_streams] == [
        1,
        1,
        1,
    ]
    assert torch.cuda.synchronize.call_count == syncs + 2
    assert not live_rig_workers()
    # A Python task error leaves the session usable; device faults are not simulated.
    assert session.run_parallel([lambda: "left", lambda: "right"], 2) == [
        "left",
        "right",
    ]


def test_static_partition_raises_lowest_failing_task_not_first_worker(cuda_session):
    _, session = cuda_session
    ran = []

    def task(index):
        ran.append(index)
        if index in (1, 2):
            raise RuntimeError(f"task {index} failed")
        return index

    with pytest.raises(RuntimeError, match="task 1 failed") as raised:
        session.run_parallel([partial(task, index) for index in range(5)], 2)
    assert any("camera task 2 also failed" in note for note in raised.value.__notes__)
    # Worker 0 stops at task 2 (skipping 4); worker 1 stops at task 1 (skipping 3).
    assert sorted(ran) == [0, 1, 2]


@pytest.mark.parametrize("failure", ["task", "none"])
def test_stream_and_device_sync_failures_never_mask_the_primary_error(
    cuda_session, failure
):
    torch, session = cuda_session
    session.run_parallel([lambda: 0, lambda: 1], 2)
    left, right = session._rig_streams
    right.synchronize.side_effect = RuntimeError("stream sync failed")

    def failing_device_sync(*_):
        # Pre-launch sync succeeds; the post-join sync fails.
        if torch.cuda.synchronize.call_count > syncs + 1:
            raise RuntimeError("device sync failed")

    syncs = torch.cuda.synchronize.call_count
    torch.cuda.synchronize.side_effect = failing_device_sync
    tasks = [
        Mock(side_effect=ValueError("model forward failed"))
        if failure == "task"
        else Mock(return_value=0),
        Mock(return_value=1),
    ]
    expected = "model forward failed" if failure == "task" else "stream sync failed"
    with pytest.raises(Exception, match=expected) as raised:
        session.run_parallel(tasks, 2)
    notes = raised.value.__notes__
    assert any("device sync failed" in note for note in notes)
    if failure == "task":
        assert isinstance(raised.value, ValueError)
        assert any("stream sync failed" in note for note in notes)
    assert left.synchronize.call_count == right.synchronize.call_count == 2
    assert not live_rig_workers()


def test_device_context_failure_raises_without_partial_results(cuda_session):
    torch, session = cuda_session
    task = Mock(return_value="never returned")
    torch.cuda.device = Mock(side_effect=RuntimeError("device context failed"))
    with pytest.raises(RuntimeError, match="device context failed"):
        session.run_parallel([task, task], 2)
    task.assert_not_called()
    assert not live_rig_workers()


def test_fatal_worker_error_still_drains_every_stream_and_the_device(cuda_session):
    torch, session = cuda_session
    stopped = threading.Event()
    finished = []

    def camera_0():
        assert stopped.wait(10)
        raise ValueError("camera 0 failed")

    def camera_1():
        stopped.set()
        raise SystemExit("worker stopped")

    def camera_2():
        assert stopped.wait(10)
        finished.append(2)
        return 2

    syncs = torch.cuda.synchronize.call_count
    with pytest.raises(SystemExit, match="worker stopped") as raised:
        session.run_parallel([camera_0, camera_1, camera_2], 3)
    assert any("camera task 0 also failed" in note for note in raised.value.__notes__)
    assert finished == [2]
    streams = session._rig_streams
    assert [stream.synchronize.call_count for stream in streams] == [1, 1, 1]
    assert torch.cuda.synchronize.call_count == syncs + 2
    assert not live_rig_workers()


def test_fatal_worker_keeps_its_stream_sync_failure_as_a_note(cuda_session):
    _, session = cuda_session
    session.run_parallel([lambda: 0, lambda: 1], 2)
    left, right = session._rig_streams
    right.synchronize.side_effect = RuntimeError("stream sync failed")
    with pytest.raises(SystemExit) as raised:
        session.run_parallel([lambda: 0, Mock(side_effect=SystemExit("stop"))], 2)
    assert any("stream sync failed" in note for note in raised.value.__notes__)
    assert left.synchronize.call_count == right.synchronize.call_count == 2
    assert not live_rig_workers()


def test_fatal_stream_sync_keeps_the_workers_task_error_as_a_note(cuda_session):
    _, session = cuda_session
    session.run_parallel([lambda: 0, lambda: 1], 2)
    left, right = session._rig_streams
    left.synchronize.side_effect = KeyboardInterrupt("stream sync interrupted")
    with pytest.raises(KeyboardInterrupt) as raised:
        session.run_parallel(
            [Mock(side_effect=ValueError("camera 0 failed")), Mock()], 2
        )
    assert any("camera task 0 also failed" in note for note in raised.value.__notes__)
    assert right.synchronize.call_count == 2
    assert not live_rig_workers()


def test_fatal_device_sync_keeps_completed_worker_errors_as_notes(cuda_session):
    torch, session = cuda_session
    syncs = torch.cuda.synchronize.call_count

    def interrupted_after_join(*_):
        if torch.cuda.synchronize.call_count > syncs + 1:
            raise KeyboardInterrupt("device sync interrupted")

    torch.cuda.synchronize.side_effect = interrupted_after_join
    with pytest.raises(KeyboardInterrupt, match="device sync interrupted") as raised:
        session.run_parallel(
            [Mock(side_effect=ValueError("camera 0 failed")), Mock()], 2
        )
    assert any("camera task 0 also failed" in note for note in raised.value.__notes__)
    assert [stream.synchronize.call_count for stream in session._rig_streams] == [1, 1]
    assert not live_rig_workers()


def test_failed_stream_creation_adds_no_partial_streams(cuda_session):
    torch, session = cuda_session
    torch.cuda.Stream.side_effect = [FakeStream(1), RuntimeError("no stream")]
    task = Mock()
    with pytest.raises(RuntimeError, match="no stream"):
        session.run_parallel([task, task], 2)
    task.assert_not_called()
    assert session._rig_streams == []
    torch.cuda.Stream.side_effect = lambda device: FakeStream(device)
    assert session.run_parallel([lambda: 1, lambda: 2], 2) == [1, 2]
    existing = list(session._rig_streams)
    torch.cuda.Stream.side_effect = [FakeStream(1), RuntimeError("no third stream")]
    with pytest.raises(RuntimeError, match="no third stream"):
        session.run_parallel([task] * 4, 4)
    task.assert_not_called()
    # Existing streams stay valid; none of the new ones is kept.
    assert session._rig_streams == existing
    session.close()
    assert session._rig_streams == []


def test_parallel_execution_is_refused_on_cpu(neural_abi):
    _, extractor, matcher, *_ = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    task = Mock()
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        session.run_parallel([task, task], 2)
    task.assert_not_called()
    session.close()


@pytest.mark.parametrize("count,workers", [(2, 0), (2, 3), (5, 5), (2, True), (2, 2.0)])
def test_parallel_workers_are_bounded_by_cameras_and_four(cuda_session, count, workers):
    torch, session = cuda_session
    task = Mock()
    with pytest.raises(ValueError, match="between 1"):
        session.run_parallel([task] * count, workers)
    task.assert_not_called()
    torch.cuda.Stream.assert_not_called()


def test_parallel_execution_is_refused_after_close(cuda_session):
    _, session = cuda_session
    session.close()
    task = Mock()
    with pytest.raises(RuntimeError, match="closed"):
        session.run_parallel([task, task], 2)
    task.assert_not_called()


def test_extract_and_match_run_no_grad_inside_each_worker_stream(
    cuda_session, neural_abi
):
    torch, session = cuda_session
    _, _, _, local, global_model, matcher_model = neural_abi
    image = np.zeros((24, 32), dtype=np.uint8)
    query, descriptor = session.extract(image)
    extracted = session.run_parallel([partial(session.extract, image)] * 2, 2)
    for features_, descriptor_ in extracted:
        for key, value in query.items():
            np.testing.assert_array_equal(features_[key], value)
        np.testing.assert_array_equal(descriptor_, descriptor)
    matched = session.run_parallel([partial(session.match, query, features())] * 2, 2)
    for match in matched:
        np.testing.assert_array_equal(match, [[0, 2], [2, 0]])
    # Each concurrent pair gets its own input dict and tensors (the matcher mutates it).
    left, right = matcher_model.calls[-2:]
    assert left is not right
    assert not np.shares_memory(left["descriptors1"].array, right["descriptors1"].array)
    streams = {id(stream) for stream in session._rig_streams}
    for model in (local, global_model, matcher_model):
        workers = model.contexts[-2:]
        assert {device for device, _ in workers} == {1}
        assert {id(stream) for _, stream in workers} == streams
    assert local.contexts[0] == (None, None)
    assert torch.grad_depth == 0


class SignalSenders:
    """Real SIGINT senders that the fixture cancels and joins before restoring."""

    def __init__(self):
        self.cancel = threading.Event()
        self.threads = []
        self.releases = []

    def send(self, started, count=1, release=None, after_sec=0.2):
        """Send real SIGINTs to the main thread once every camera task has started."""

        def run():
            for event in started:
                while not event.wait(0.01):
                    if self.cancel.is_set():
                        return
            for _ in range(count):
                if self.cancel.wait(0.05):
                    return
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
            if release is not None:
                self.cancel.wait(after_sec)
                release.set()

        if release is not None:
            self.releases.append(release)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.threads.append(thread)
        return thread

    def close(self):
        self.cancel.set()
        for release in self.releases:
            release.set()
        for thread in self.threads:
            thread.join(10)
        return [thread for thread in self.threads if thread.is_alive()]


@pytest.fixture
def main_thread_signals():
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("SIGINT is delivered to the main thread only")
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    senders = SignalSenders()
    try:
        yield senders
    finally:
        # A test that fails early must not leave a SIGINT for later tests: stop the
        # senders, release blocked cameras and join them before restoring.
        senders.cancel.set()
        while True:
            try:
                signal.signal(signal.SIGINT, lambda signum, frame: None)
                break
            except KeyboardInterrupt:
                pass
        alive = senders.close()
        signal.signal(signal.SIGINT, previous)
        assert not alive, "SIGINT sender threads did not stop"


def test_signal_senders_stop_without_signalling_when_cameras_never_start(
    main_thread_signals,
):
    received = []
    signal.signal(signal.SIGINT, lambda signum, frame: received.append(signum))
    senders = SignalSenders()
    release = threading.Event()
    senders.send([threading.Event()], count=2, release=release)
    assert senders.close() == []
    assert release.is_set()
    assert received == []


def blocked_camera(index, started, release, finished):
    def run():
        started[index].set()
        assert release.wait(10)
        finished.append(index)
        return index

    return run


@pytest.mark.parametrize("count", [1, 2])
def test_sigint_while_waiting_lets_every_camera_finish_before_raising(
    cuda_session, main_thread_signals, count
):
    torch, session = cuda_session
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    finished = []
    closed_after = []
    tasks = [blocked_camera(i, started, release, finished) for i in range(2)]
    sender = main_thread_signals.send(started, count, release)
    with pytest.raises(KeyboardInterrupt) as raised:
        with ExitStack() as resources:
            # The CLI closes its session as the interrupt unwinds.
            resources.callback(session.close)
            resources.callback(
                lambda: closed_after.append(
                    (
                        sorted(finished),
                        [s.synchronize.call_count for s in session._rig_streams],
                        [t.name for t in live_rig_workers()],
                    )
                )
            )
            session.run_parallel(tasks, 2)
    sender.join(10)
    assert closed_after == [([0, 1], [1, 1], [])]
    assert session._rig_streams == []
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    notes = getattr(raised.value, "__notes__", [])
    assert sum("1 more Ctrl+C press" in note for note in notes) == count - 1


def test_sigint_after_a_camera_failure_keeps_the_interrupt_and_notes_it(
    cuda_session, main_thread_signals
):
    _, session = cuda_session
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    finished = []

    def failing():
        started[0].set()
        raise ValueError("camera 0 failed")

    sender = main_thread_signals.send(started, 1, release)
    with pytest.raises(KeyboardInterrupt) as raised:
        session.run_parallel(
            [failing, blocked_camera(1, started, release, finished)], 2
        )
    sender.join(10)
    assert finished == [1]
    assert any("camera task 0 also failed" in note for note in raised.value.__notes__)
    assert [stream.synchronize.call_count for stream in session._rig_streams] == [1, 1]
    assert not live_rig_workers()


def held_presses():
    return signal.getsignal(signal.SIGINT).__self__.count


def wait_until(condition):
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.parametrize("target", ["main", "other_thread"])
def test_sigint_between_submit_and_registration_is_held_until_cameras_finish(
    cuda_session, main_thread_signals, monkeypatch, target
):
    _, session = cuda_session
    finished = []

    class InterruptingPool(resident.ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            # The signal lands after the Future exists but before it is recorded.
            # A thread other than main runs only the C handler; Python then runs
            # the SIGINT handler in the main thread, as for an unmasked host thread.
            if target == "main":
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
            else:
                sender = threading.Thread(
                    target=lambda: signal.pthread_kill(
                        threading.get_ident(), signal.SIGINT
                    )
                )
                sender.start()
                sender.join()
            presses = len(submitted) + 1
            wait_until(lambda: held_presses() == presses)
            submitted.append(future)
            return future

    submitted = []

    monkeypatch.setattr(resident, "ThreadPoolExecutor", InterruptingPool)

    def camera(index):
        time.sleep(0.1)
        finished.append(index)
        return index

    with pytest.raises(KeyboardInterrupt) as raised:
        session.run_parallel([partial(camera, 0), partial(camera, 1)], 2)
    assert sorted(finished) == [0, 1]
    assert all(future.done() for future in submitted)
    assert not live_rig_workers()
    assert any("1 more Ctrl+C press" in note for note in raised.value.__notes__)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_submit_failure_keeps_its_error_when_the_cleanup_sync_fails(
    cuda_session, monkeypatch
):
    torch, session = cuda_session
    syncs = []

    def synchronize(*_):
        syncs.append(len(syncs))
        if len(syncs) == 2:
            raise RuntimeError("device sync failed")

    class FailingPool(resident.ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            raise RuntimeError("cannot start camera thread")

    torch.cuda.synchronize.side_effect = synchronize
    monkeypatch.setattr(resident, "ThreadPoolExecutor", FailingPool)
    task = Mock()
    with pytest.raises(RuntimeError, match="cannot start camera thread") as raised:
        session.run_parallel([task, task], 2)
    task.assert_not_called()
    assert len(syncs) == 2
    assert any("device sync failed" in note for note in raised.value.__notes__)


def test_submit_failure_waits_for_cameras_already_started(cuda_session, monkeypatch):
    _, session = cuda_session
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    finished = []
    submitted = []

    class FailingPool(resident.ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            if submitted:
                threading.Timer(0.2, release.set).start()
                raise RuntimeError("cannot start camera thread")
            submitted.append(fn)
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(resident, "ThreadPoolExecutor", FailingPool)
    tasks = [blocked_camera(i, started, release, finished) for i in range(2)]
    with pytest.raises(RuntimeError, match="cannot start camera thread"):
        session.run_parallel(tasks, 2)
    assert finished == [0]
    assert not live_rig_workers()


def test_sigint_during_a_submit_failure_is_noted_on_the_failure(
    cuda_session, main_thread_signals, monkeypatch
):
    _, session = cuda_session
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    finished = []
    submitted = []

    class FailingPool(resident.ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            if submitted:
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
                wait_until(lambda: held_presses() == 1)
                threading.Timer(0.2, release.set).start()
                raise RuntimeError("cannot start camera thread")
            submitted.append(fn)
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(resident, "ThreadPoolExecutor", FailingPool)
    tasks = [blocked_camera(i, started, release, finished) for i in range(2)]
    with pytest.raises(RuntimeError, match="cannot start camera thread") as raised:
        session.run_parallel(tasks, 2)
    assert finished == [0]
    assert not live_rig_workers()
    assert any("Ctrl+C pressed 1 time" in note for note in raised.value.__notes__)
    assert any(
        "handler raised KeyboardInterrupt()" in note for note in raised.value.__notes__
    )
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.parametrize("raises", [False, True])
def test_held_ctrl_c_reaches_a_custom_handler_when_a_submit_failure_propagates(
    cuda_session, main_thread_signals, monkeypatch, raises
):
    _, session = cuda_session
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    finished = []
    submitted = []
    calls = []

    def handler(signum, frame):
        calls.append((signum, sorted(finished)))
        if raises:
            raise SystemExit("custom stop")

    class FailingPool(resident.ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            if submitted:
                signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
                wait_until(lambda: held_presses() == 1)
                threading.Timer(0.2, release.set).start()
                raise RuntimeError("cannot start camera thread")
            submitted.append(fn)
            return super().submit(fn, *args, **kwargs)

    signal.signal(signal.SIGINT, handler)
    monkeypatch.setattr(resident, "ThreadPoolExecutor", FailingPool)
    tasks = [blocked_camera(i, started, release, finished) for i in range(2)]
    with pytest.raises(RuntimeError, match="cannot start camera thread") as raised:
        session.run_parallel(tasks, 2)
    # The coordinator error stays primary; the handler still ran once, after the
    # started camera finished.
    assert calls == [(signal.SIGINT, [0])]
    assert signal.getsignal(signal.SIGINT) is handler
    notes = raised.value.__notes__
    assert any("Ctrl+C pressed 1 time" in note for note in notes)
    assert any("handler raised SystemExit" in note for note in notes) is raises
    assert not live_rig_workers()


def test_interrupt_before_launch_starts_no_camera(cuda_session):
    torch, session = cuda_session
    original = signal.getsignal(signal.SIGINT)
    torch.cuda.synchronize.side_effect = KeyboardInterrupt
    task = Mock()
    with pytest.raises(KeyboardInterrupt):
        session.run_parallel([task, task], 2)
    task.assert_not_called()
    assert not live_rig_workers()
    assert signal.getsignal(signal.SIGINT) is original


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_cleanup_sync_after_an_interrupt_never_replaces_it(
    cuda_session, main_thread_signals, failure
):
    torch, session = cuda_session
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    finished = []
    syncs = []

    def synchronize(*_):
        syncs.append(len(syncs))
        if len(syncs) == 2:
            raise failure("device sync interrupted")

    torch.cuda.synchronize.side_effect = synchronize
    sender = main_thread_signals.send(started, 1, release)
    with pytest.raises(KeyboardInterrupt) as raised:
        session.run_parallel(
            [blocked_camera(i, started, release, finished) for i in range(2)], 2
        )
    sender.join(10)
    assert sorted(finished) == [0, 1]
    expected = (
        "Ctrl+C pressed 1 time"
        if failure is KeyboardInterrupt
        else "synchronization failed"
    )
    assert any(expected in note for note in raised.value.__notes__)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_sigint_during_the_final_device_sync_is_held(cuda_session, main_thread_signals):
    torch, session = cuda_session
    syncs = []

    def synchronize(*_):
        syncs.append(len(syncs))
        if len(syncs) == 2:
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
            wait_until(lambda: held_presses() == 1)

    torch.cuda.synchronize.side_effect = synchronize
    with pytest.raises(KeyboardInterrupt):
        session.run_parallel([lambda: 0, lambda: 1], 2)
    assert len(syncs) == 2
    assert [stream.synchronize.call_count for stream in session._rig_streams] == [1, 1]


def test_sigint_handler_is_left_alone_off_the_main_thread_or_when_ignored(
    cuda_session, main_thread_signals
):
    _, session = cuda_session
    outcome = []
    worker = threading.Thread(
        target=lambda: outcome.append(session.run_parallel([lambda: 0, lambda: 1], 2))
    )
    worker.start()
    worker.join(10)
    assert outcome == [[0, 1]]
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    assert session.run_parallel([lambda: 2, lambda: 3], 2) == [2, 3]
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN


@pytest.mark.parametrize("raises", [False, True])
def test_custom_sigint_handler_runs_once_after_cameras_finish(
    cuda_session, main_thread_signals, raises
):
    _, session = cuda_session
    started = [threading.Event(), threading.Event()]
    release = threading.Event()
    finished = []
    calls = []

    def handler(signum, frame):
        calls.append((signum, sorted(finished)))
        if raises:
            raise SystemExit("custom stop")

    signal.signal(signal.SIGINT, handler)
    sender = main_thread_signals.send(started, 2, release)
    tasks = [blocked_camera(i, started, release, finished) for i in range(2)]
    if raises:
        with pytest.raises(SystemExit, match="custom stop") as raised:
            session.run_parallel(tasks, 2)
        assert any("1 more Ctrl+C press" in note for note in raised.value.__notes__)
    else:
        assert session.run_parallel(tasks, 2) == [0, 1]
    sender.join(10)
    assert calls == [(signal.SIGINT, [0, 1])]
    assert signal.getsignal(signal.SIGINT) is handler


def test_inputs_follow_the_initialized_device_after_a_caller_switch(
    cuda_session, neural_abi
):
    torch, session = cuda_session
    _, _, _, local, global_model, matcher_model = neural_abi
    assert session.tensor_device == "cuda:1"
    torch.cuda.current_device.return_value = 0
    image = np.zeros((24, 32), dtype=np.uint8)
    query, _ = session.extract(image)
    session.match(query, features())
    session.run_parallel([partial(session.extract, image)] * 2, 2)
    assert {call["image"].device for call in local.calls} == {"cuda:1"}
    assert {call["image"].device for call in global_model.calls} == {"cuda:1"}
    assert {call["descriptors0"].device for call in matcher_model.calls} == {"cuda:1"}
    assert all(
        model.devices == ["cuda"] for model in (local, global_model, matcher_model)
    )


def test_cpu_inputs_stay_on_the_cpu(neural_abi):
    _, extractor, matcher, local, _, matcher_model = neural_abi
    session = resident.ResidentModels(extractor, matcher)
    assert session.tensor_device == "cpu"
    query, _ = session.extract(np.zeros((24, 32), dtype=np.uint8))
    session.match(query, features())
    assert local.calls[-1]["image"].device == "cpu"
    assert matcher_model.calls[-1]["descriptors1"].device == "cpu"
    session.close()
