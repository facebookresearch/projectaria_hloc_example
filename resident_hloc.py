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

"""Resident, read-only HLOC query resources; neural packages are runtime imports.

One serialized worker owns each session. A map is an immutable snapshot: stop
queries before replacing its files, then explicitly close/reinitialize the
backend. Neither map files nor upstream modules are modified here. Within one
rig query, per-camera extraction and matching may run on bounded worker threads
with one CUDA stream each; the call returns only after every worker finished.
"""

from __future__ import annotations

import importlib
import signal
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import (
    FrameType,
    MappingProxyType,
    ModuleType,
    SimpleNamespace,
    TracebackType,
)
from typing import cast, Protocol, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray

FeatureArray = NDArray[np.generic]
Features = Mapping[str, FeatureArray]
DEFAULT_CACHE_MAX_BYTES: int = 256 * 1024 * 1024
MAX_KEYPOINTS: int = 4096
RIG_EXECUTION_MODES: tuple[str, ...] = ("auto", "serial", "parallel")
MAX_RIG_WORKERS: int = 4
TaskResult = TypeVar("TaskResult")


class CpuPrediction(Protocol):
    """Only the public tensor conversion used at the optional neural boundary."""

    def cpu(self) -> CpuPrediction: ...

    def numpy(self) -> FeatureArray: ...


class CudaStream(Protocol):
    """The public stream method rig workers use before returning."""

    def synchronize(self) -> None: ...


def prediction_arrays(prediction: object) -> dict[str, FeatureArray]:
    if not isinstance(prediction, Mapping):
        raise ValueError("Feature prediction must be a mapping")
    arrays = {}
    for key, value in prediction.items():
        if not isinstance(key, str):
            raise ValueError("Feature prediction names must be strings")
        try:
            if len(value) != 1:
                raise ValueError("Expected a single-image feature prediction")
            arrays[key] = np.array(
                cast(CpuPrediction, value[0]).cpu().numpy(), copy=True
            )
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError("Invalid feature tensor prediction") from error
    return arrays


class FeatureCache:
    """Byte-bounded true LRU of owned, read-only CPU arrays (not GPU tensors).

    The bound counts numerical array storage, not the reconstruction/global
    index, model weights, or Python container overhead.
    """

    def __init__(self, max_bytes: int = DEFAULT_CACHE_MAX_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Feature cache max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.entries: OrderedDict[str, Features] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversize_bypasses = 0

    def get(self, name: str, loader: Callable[[str], Features]) -> Features:
        if name in self.entries:
            self.hits += 1
            self.entries.move_to_end(name)
            return self.entries[name]
        self.misses += 1
        arrays = {
            key: np.array(value, copy=True) for key, value in loader(name).items()
        }
        for array in arrays.values():
            array.setflags(write=False)
        features = MappingProxyType(arrays)
        size_bytes = sum(array.nbytes for array in arrays.values())
        if size_bytes > self.max_bytes:
            self.oversize_bypasses += 1
            return features
        while self.bytes + size_bytes > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= sum(array.nbytes for array in removed.values())
            self.evictions += 1
        self.entries[name] = features
        self.bytes += size_bytes
        return features

    def snapshot(self) -> dict[str, int]:
        """Return independent JSON-safe counters, never live mutable state."""
        return {
            "cache_bytes": self.bytes,
            "cache_max_bytes": self.max_bytes,
            "cache_entries": len(self.entries),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_evictions": self.evictions,
            "cache_oversize_bypasses": self.oversize_bypasses,
        }

    def clear(self) -> None:
        self.entries.clear()
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.oversize_bypasses = 0


def validate_features(features: Features) -> None:
    """Require the pinned SuperPoint D-by-N descriptor/keypoint convention."""
    try:
        keypoints = features["keypoints"]
        descriptors = features["descriptors"]
        size = features["image_size"]
    except KeyError as error:
        raise ValueError("Missing local feature prediction") from error
    if any(
        not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()
        for value in features.values()
    ):
        raise ValueError("Local features must be finite numerical arrays")
    count = len(keypoints) if keypoints.ndim else -1
    if (
        keypoints.shape != (count, 2)
        or not 0 <= count <= MAX_KEYPOINTS
        or descriptors.ndim != 2
        or descriptors.shape[0] != 256
        or descriptors.shape[1] != count
        or size.shape != (2,)
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or np.any(size != np.floor(size))
    ):
        raise ValueError("Invalid local feature dimensions")
    if "scores" in features and features["scores"].shape != (count,):
        raise ValueError("Invalid local feature score dimensions")


def normalized_descriptors(descriptors: FeatureArray) -> FeatureArray:
    if (
        descriptors.ndim != 2
        or min(descriptors.shape) <= 0
        or not np.issubdtype(descriptors.dtype, np.floating)
    ):
        raise ValueError("Invalid retrieval descriptor dimensions or dtype")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    if (
        not np.isfinite(descriptors).all()
        or not np.isfinite(norms).all()
        or np.any(norms == 0)
    ):
        raise ValueError("Map retrieval descriptors are invalid")
    normalized = descriptors / norms
    normalized.setflags(write=False)
    return normalized


class ResidentMap:
    """One reconstruction/global index and one managed read-only local HDF5 handle."""

    def __init__(
        self, map_path: Path, colmap: ModuleType, cache_max_bytes: int
    ) -> None:
        self.path = map_path
        self.cache = FeatureCache(cache_max_bytes)
        self.model = None
        self.feature_file: h5py.File | None = None
        self.images = {}
        self.point_ids = {}
        self.landmarks = {}
        self.references: list[str] = []
        self.descriptors = np.empty((0, 0), dtype=np.float32)
        self.initialization_durations_sec: dict[str, float] = {}
        try:
            self._load(colmap)
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def _load(self, colmap: ModuleType) -> None:
        start_time_sec = time.perf_counter()
        self.model = colmap.Reconstruction(self.path / "sfm")
        if self.model.num_points3D() == 0:
            raise ValueError("Map has no reconstructed landmarks")
        self.images = {image.name: image for image in self.model.images.values()}
        if not self.images or len(self.images) != len(self.model.images):
            raise ValueError("Map image names must be nonempty and unique")
        for name, image in self.images.items():
            ids = np.array(
                [
                    point.point3D_id if point.has_point3D() else -1
                    for point in image.points2D
                ],
                dtype=np.int64,
            )
            if any(
                int(point) not in self.model.points3D for point in ids if point >= 0
            ):
                raise ValueError("Map observation references an absent landmark")
            ids.setflags(write=False)
            self.point_ids[name] = ids
            self.landmarks[name] = {int(point) for point in ids if point >= 0}
        self.initialization_durations_sec["reconstruction_duration_sec"] = (
            time.perf_counter() - start_time_sec
        )
        start_time_sec = time.perf_counter()
        self.references = sorted(self.images)
        with h5py.File(self.path / "global.h5", "r") as source:
            descriptors = np.stack(
                [
                    np.asarray(source[name]["global_descriptor"])
                    for name in self.references
                ]
            )
        self.descriptors = normalized_descriptors(descriptors)
        self.feature_file = h5py.File(self.path / "features.h5", "r")
        self.initialization_durations_sec["map_features_duration_sec"] = (
            time.perf_counter() - start_time_sec
        )

    def retrieve(self, descriptors: Sequence[FeatureArray]) -> dict[str, list[str]]:
        if self.model is None:
            raise RuntimeError("Resident map is closed")
        retrieved = {}
        for index, descriptor in enumerate(descriptors):
            if not np.issubdtype(descriptor.dtype, np.floating):
                raise ValueError("Query retrieval descriptor is invalid")
            norm = np.linalg.norm(descriptor)
            if (
                descriptor.shape != (self.descriptors.shape[1],)
                or not np.isfinite(descriptor).all()
                or not np.isfinite(norm)
                or norm == 0
            ):
                raise ValueError("Query retrieval descriptor is invalid")
            scores = np.einsum(
                "ij,j->i", self.descriptors, descriptor / norm, optimize=False
            )
            if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.0001):
                raise ValueError("Retrieval scores are invalid")
            retrieved[f"q{index}.png"] = [
                self.references[int(i)] for i in np.argsort(-scores)[:20]
            ]
        return retrieved

    def features(self, name: str) -> Features:
        if self.feature_file is None:
            raise RuntimeError("Resident map is closed")
        return self.cache.get(name, self._read_features)

    def _read_features(self, name: str) -> Features:
        if self.feature_file is None:
            raise RuntimeError("Resident map is closed")
        group = self.feature_file[name]
        features = {key: np.asarray(value) for key, value in group.items()}
        validate_features(features)
        if len(features["keypoints"]) != len(self.point_ids[name]):
            raise ValueError(
                "Map keypoints do not align with reconstruction observations"
            )
        return features

    def close(self) -> None:
        feature_file, self.feature_file = self.feature_file, None
        self.cache.clear()
        self.model = None
        self.images.clear()
        self.point_ids.clear()
        self.landmarks.clear()
        self.references.clear()
        self.descriptors = np.empty((0, 0), dtype=np.float32)
        if feature_file is not None:
            feature_file.close()


def preprocess_image(
    image: NDArray[np.uint8], conf: Mapping[str, object], extractor: ModuleType
) -> dict[str, FeatureArray]:
    """ImageDataset arithmetic after read_image, without a PNG encode/decode.

    OpenCV's libpng IMREAD_GRAYSCALE truncates with 15-bit coefficients, unlike
    cvtColor/PIL rounding. Frames formerly staged as metadata-free RGB PNGs use
    this same conversion; monochrome PNGs expand to RGB for NetVLAD.
    """
    if (
        image.dtype != np.uint8
        or image.ndim not in (2, 3)
        or min(image.shape[:2]) <= 0
        or (image.ndim == 3 and image.shape[2] != 3)
    ):
        raise ValueError("Expected a nonempty uint8 grayscale or RGB image")
    config = SimpleNamespace(**{**extractor.ImageDataset.default_conf, **conf})
    grayscale = config.grayscale
    if grayscale and image.ndim == 3:
        rgb = image.astype(np.uint32)
        image = (
            (9797 * rgb[..., 0] + 19234 * rgb[..., 1] + 3737 * rgb[..., 2]) >> 15
        ).astype(np.uint8)
    elif not grayscale and image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]
    resize_max = config.resize_max
    if resize_max and (config.resize_force or max(size) > resize_max):
        scale = resize_max / max(size)
        size_new = tuple(int(round(value * scale)) for value in size)
        image = extractor.resize_image(image, size_new, config.interpolation)
    image = image[None] if grayscale else image.transpose((2, 0, 1))
    return {"image": image / 255.0, "original_size": np.array(size)}


def rig_execution_plan(
    requested: str, device: str, num_cameras: int
) -> dict[str, object]:
    """Resolve how one query runs its cameras; only CUDA rigs run in parallel."""
    if requested not in RIG_EXECUTION_MODES:
        raise ValueError(f"rig_execution must be one of {RIG_EXECUTION_MODES}")
    if num_cameras < 2:
        mode, reason = "serial", "single_camera"
    elif requested == "serial":
        mode, reason = "serial", "requested"
    elif device != "cuda":
        mode, reason = "serial", "requires_cuda"
    else:
        mode, reason = "parallel", requested
    return {
        "requested": requested,
        "mode": mode,
        "workers": min(num_cameras, MAX_RIG_WORKERS) if mode == "parallel" else 1,
        "cameras": num_cameras,
        "device": str(device),
        "reason": reason,
    }


def collect_failures(
    futures: Sequence[Future[None]],
    failed: Sequence[Sequence[tuple[int, Exception]]],
    cleanup: Sequence[Sequence[Exception]],
) -> list[tuple[BaseException, str]]:
    """Every worker error with its note, in raise priority: fatal, task, cleanup."""
    notes: list[tuple[BaseException, str]] = [
        (error, f"Rig worker stopped: {error!r}")
        for error in (future.exception() for future in futures)
        if error is not None
    ]
    notes += [
        (error, f"Rig camera task {index} also failed: {error!r}")
        for index, error in sorted(
            (failure for errors in failed for failure in errors),
            key=lambda failure: failure[0],
        )
    ]
    notes += [
        (error, f"Rig stream/device synchronization failed: {error!r}")
        for errors in cleanup
        for error in errors
    ]
    return notes


def annotate(
    primary: BaseException, notes: Sequence[tuple[BaseException, str]]
) -> None:
    for error, note in notes:
        if error is not primary:
            primary.add_note(note)


class DeferredInterrupt:
    """Hold Ctrl+C while rig camera tasks run, then hand it to the original handler.

    CPython runs SIGINT handlers in the main thread whichever thread the OS
    delivers the signal to, so a temporary Python handler holds every Ctrl+C,
    including repeated ones. It is installed only from the main thread over a
    Python-level handler (the default and Jupyter's), restored on exit, and the
    original handler is then called once. Ignored or default-action SIGINT,
    non-Python handlers and other threads, where no KeyboardInterrupt can arrive,
    are left alone.
    """

    def __init__(self) -> None:
        self.count = 0
        self.previous: Callable[[int, FrameType | None], object] | None = None

    def __enter__(self) -> DeferredInterrupt:
        if threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGINT)
            if callable(previous):
                self.previous = previous
                signal.signal(signal.SIGINT, self._hold)
        return self

    def _hold(self, signum: int, frame: FrameType | None) -> None:
        self.count += 1

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.previous is not None:
            signal.signal(signal.SIGINT, self.previous)
        if self.count and error is not None:
            error.add_note(
                f"Ctrl+C pressed {self.count} time(s) while rig camera tasks finished"
            )
            # The original handler still runs once; the coordinator error stays primary.
            try:
                self._call_previous()
            except (Exception, KeyboardInterrupt, SystemExit) as handler_error:
                error.add_note(
                    f"The original Ctrl+C handler raised {handler_error!r} after "
                    "rig camera tasks finished"
                )

    def _call_previous(self) -> None:
        if self.previous is not None:
            self.previous(signal.SIGINT, None)

    def deliver(self, notes: Sequence[tuple[BaseException, str]]) -> None:
        """Call the restored handler once for held presses; note worker failures."""
        if not self.count:
            return
        try:
            self._call_previous()
        except BaseException as error:
            annotate(error, notes)
            if self.count > 1:
                error.add_note(
                    f"{self.count - 1} more Ctrl+C press(es) arrived while rig camera "
                    "tasks finished"
                )
            raise


class ResidentModels:
    """Pinned upstream extractor/matcher instances, retained across map queries."""

    def __init__(self, extractor: ModuleType, matcher: ModuleType) -> None:
        self.extractor = extractor
        self.local_model = self.global_model = self.match_model = None
        self.device_index: int | None = None
        self._rig_streams: list[CudaStream] = []
        try:
            self.torch = importlib.import_module("torch")
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                self.device_index = int(self.torch.cuda.current_device())
            # Inputs follow the weights' device, not the caller's later default.
            self.tensor_device = (
                f"cuda:{self.device_index}" if self.device == "cuda" else "cpu"
            )
            self.local_conf = deepcopy(extractor.confs["superpoint_aachen"])
            self.global_conf = deepcopy(extractor.confs["netvlad"])
            self.local_model = extractor.dynamic_load(
                extractor.extractors, self.local_conf["model"]["name"]
            )(deepcopy(self.local_conf["model"]))
            self.local_model.eval().to(self.device)
            self.global_model = extractor.dynamic_load(
                extractor.extractors, self.global_conf["model"]["name"]
            )(deepcopy(self.global_conf["model"]))
            self.global_model.eval().to(self.device)
            self.match_model = matcher.dynamic_load(
                matcher.matchers, matcher.confs["superpoint+lightglue"]["model"]["name"]
            )(deepcopy(matcher.confs["superpoint+lightglue"]["model"]))
            self.match_model.eval().to(self.device)
            self.synchronize()
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize(self.device_index)

    def _streams(self, count: int) -> list[CudaStream]:
        """Persistent per-worker streams for this resident session on its model device.

        Reusing them bounds this session's cached allocator blocks to at most four
        streams; a new session after a map transition gets its own streams. If
        creating a stream fails, existing streams are kept and no new one is added.
        """
        if len(self._rig_streams) < count:
            created = [
                self.torch.cuda.Stream(device=self.device_index)
                for _ in range(count - len(self._rig_streams))
            ]
            self._rig_streams.extend(created)
        return self._rig_streams[:count]

    def run_parallel(
        self, tasks: Sequence[Callable[[], TaskResult]], workers: int
    ) -> list[TaskResult]:
        """Run independent per-camera tasks on bounded threads, one CUDA stream each.

        Worker k runs tasks k, k + workers, ... on the device holding the weights;
        results keep task order. Every submitted task finishes, its stream and the
        device are synchronized, and its worker exits before this returns or raises.
        A Ctrl+C meanwhile is held until then (see `DeferredInterrupt`). A fatal
        worker error is raised first, else the lowest-index task failure; others
        become notes.
        """
        self._check_parallel(len(tasks), workers)
        streams = self._streams(workers)
        results: list[TaskResult | None] = [None] * len(tasks)
        failed: list[list[tuple[int, Exception]]] = [[] for _ in range(workers)]
        cleanup: list[list[Exception]] = [[] for _ in range(workers + 1)]
        futures: list[Future[None]] = []
        interrupt = DeferredInterrupt()
        with interrupt:
            self.synchronize()
            pool = ThreadPoolExecutor(workers, thread_name_prefix="aria-hloc-rig")
            try:
                for slot in range(workers):
                    jobs = [
                        (index, tasks[index])
                        for index in range(slot, len(tasks), workers)
                    ]
                    futures.append(
                        pool.submit(
                            self._run_worker,
                            streams[slot],
                            jobs,
                            results,
                            failed[slot],
                            cleanup[slot],
                        )
                    )
                for future in futures:
                    future.exception()
                pool.shutdown(wait=True)
            except BaseException as error:
                # A camera task that already started must not outlive this call,
                # or the caller could release the session under it.
                self._quiesce(pool, futures, cleanup[workers])
                annotate(error, collect_failures(futures, failed, cleanup))
                raise
            try:
                self.synchronize()
            except Exception as error:
                cleanup[workers].append(error)
            except BaseException as error:
                annotate(error, collect_failures(futures, failed, cleanup))
                raise
        notes = collect_failures(futures, failed, cleanup)
        interrupt.deliver(notes)
        if notes:
            annotate(notes[0][0], notes[1:])
            raise notes[0][0]
        return cast(list[TaskResult], results)

    def _check_parallel(self, num_tasks: int, workers: int) -> None:
        if (
            self.local_model is None
            or self.global_model is None
            or self.match_model is None
        ):
            raise RuntimeError("Resident models are closed")
        if self.device != "cuda":
            raise RuntimeError("Parallel rig execution requires a CUDA device")
        if type(workers) is not int or not 1 <= workers <= min(
            num_tasks, MAX_RIG_WORKERS
        ):
            raise ValueError("Rig workers must be between 1 and min(cameras, 4)")

    def _quiesce(
        self,
        pool: ThreadPoolExecutor,
        futures: Sequence[Future[None]],
        cleanup: list[Exception],
    ) -> None:
        """Wait for every submitted task and pool worker, then drain the device."""
        for future in futures:
            future.exception()
        pool.shutdown(wait=True)
        try:
            self.synchronize()
        except Exception as error:
            cleanup.append(error)

    def _run_worker(
        self,
        stream: CudaStream,
        jobs: Sequence[tuple[int, Callable[[], TaskResult]]],
        results: list[TaskResult | None],
        failed: list[tuple[int, Exception]],
        cleanup: list[Exception],
    ) -> None:
        """Run one worker's tasks in order on its own stream, always draining it.

        Errors are recorded in the caller's lists; a fatal BaseException still
        propagates, after this worker's stream has been synchronized.
        """
        try:
            device = self.torch.cuda.device(self.device_index)
            with device, self.torch.cuda.stream(stream):
                try:
                    for index, task in jobs:
                        try:
                            results[index] = task()
                        except Exception as error:
                            failed.append((index, error))
                            break
                finally:
                    try:
                        stream.synchronize()
                    except Exception as error:
                        cleanup.append(error)
        except Exception as error:
            cleanup.append(error)

    def _extract_prediction(
        self,
        model: Callable[[dict[str, object]], object],
        data: Mapping[str, FeatureArray],
    ) -> dict[str, FeatureArray]:
        return prediction_arrays(
            model(
                {
                    "image": self.torch.from_numpy(data["image"][None]).to(
                        self.tensor_device, non_blocking=True
                    )
                }
            )
        )

    def extract(self, image: NDArray[np.uint8]) -> tuple[Features, FeatureArray]:
        if self.local_model is None or self.global_model is None:
            raise RuntimeError("Resident models are closed")
        local_data = preprocess_image(
            image, self.local_conf["preprocessing"], self.extractor
        )
        global_data = preprocess_image(
            image, self.global_conf["preprocessing"], self.extractor
        )
        with self.torch.no_grad():
            features = self._extract_prediction(self.local_model, local_data)
            global_prediction = self._extract_prediction(self.global_model, global_data)
            try:
                descriptor = global_prediction["global_descriptor"]
            except KeyError as error:
                raise ValueError("Missing global feature prediction") from error
        original_size = local_data["original_size"]
        features["image_size"] = original_size
        if "keypoints" in features:
            size = np.array(local_data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            features["keypoints"] = (features["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in features:
                features["scales"] *= scales.mean()
        validate_features(features)
        if (
            descriptor.ndim != 1
            or not len(descriptor)
            or not np.issubdtype(descriptor.dtype, np.floating)
            or not np.isfinite(descriptor).all()
        ):
            raise ValueError("Invalid global feature prediction")
        return features, descriptor

    def _pair_data(self, features: Features, suffix: str) -> dict[str, object]:
        # A copy avoids exposing the read-only CPU cache to writable tensor views.
        data = {
            key + suffix: self.torch.from_numpy(np.array(value, copy=True)).float()[
                None
            ]
            for key, value in features.items()
        }
        data = {
            key: value
            if key.startswith("image")
            else value.to(self.tensor_device, non_blocking=True)
            for key, value in data.items()
        }
        size = tuple(int(value) for value in features["image_size"])[::-1]
        data["image" + suffix] = self.torch.empty((1, 1) + size)
        return data

    def match(self, query: Features, reference: Features) -> NDArray[np.int64]:
        if self.match_model is None:
            raise RuntimeError("Resident models are closed")
        validate_features(query)
        validate_features(reference)
        if not len(query["keypoints"]) or not len(reference["keypoints"]):
            return np.empty((0, 2), dtype=np.int64)
        with self.torch.no_grad():
            prediction = self.match_model(
                {**self._pair_data(query, "0"), **self._pair_data(reference, "1")}
            )
            if not isinstance(prediction, Mapping) or "matches0" not in prediction:
                raise ValueError("Missing matcher prediction")
            matches = prediction_arrays({"matches0": prediction["matches0"]})[
                "matches0"
            ]
        if (
            matches.shape != (len(query["keypoints"]),)
            or not np.issubdtype(matches.dtype, np.integer)
            or np.any(matches < -1)
            or np.any(matches >= len(reference["keypoints"]))
        ):
            raise ValueError("Invalid matcher indices")
        indices = np.flatnonzero(matches != -1)
        return np.stack((indices, matches[indices]), axis=-1).astype(np.int64)

    def close(self) -> None:
        self.local_model = self.global_model = self.match_model = None
        self._rig_streams.clear()


class StageTimings:
    """Wall-clock stages include CUDA synchronization overhead when applicable."""

    def __init__(self, synchronize: Callable[[], None]) -> None:
        self.synchronize = synchronize
        self.durations_sec: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start_time_sec = time.perf_counter()
        self.synchronize()
        try:
            yield
        except BaseException as error:
            # Keep the stage's own failure; a failing cleanup sync becomes a note.
            try:
                self.synchronize()
            except Exception as cleanup_error:
                error.add_note(
                    f"{name} stage synchronization failed: {cleanup_error!r}"
                )
            finally:
                self._record(name, start_time_sec)
            raise
        self.synchronize()
        self._record(name, start_time_sec)

    def _record(self, name: str, start_time_sec: float) -> None:
        key = name + "_duration_sec"
        self.durations_sec[key] = (
            self.durations_sec.get(key, 0.0) + time.perf_counter() - start_time_sec
        )
