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

"""Pinned public HLOC features/triangulation and COLMAP absolute pose estimation."""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast, Protocol

import h5py
import numpy as np
from aria_utils import Frame
from map_utils import (
    Camera,
    covisibility_clusters,
    FloatArray,
    HLOC_COMMIT,
    inverse_transform,
    quaternion_wxyz,
    unique_matches,
    View,
)
from resident_hloc import (
    DEFAULT_CACHE_MAX_BYTES,
    FeatureCache,
    Features,
    ResidentMap,
    ResidentModels,
    StageTimings,
)

MIN_INLIER_LANDMARKS: int = 20
MIN_INLIER_RATIO: float = 0.2
# Parallel loading on Linux; macOS subprocess loaders hang at shutdown. Containers
# with a small /dev/shm can set ARIA_HLOC_DATA_LOADER_WORKERS=0.
DATA_LOADER_WORKERS: int = int(
    os.environ.get(
        "ARIA_HLOC_DATA_LOADER_WORKERS", "0" if sys.platform == "darwin" else "2"
    )
)


class PoseTransform(Protocol):
    """Minimal matrix interface of an optional native pose backend."""

    def matrix(self) -> FloatArray: ...


def validate_hloc_checkout(source: Path) -> None:
    """Require the released upstream snapshot, not a floating branch or internal adapter."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != HLOC_COMMIT:
        raise ValueError(f"HLOC must be checked out at {HLOC_COMMIT}; found {commit}")


def colmap_camera(camera: Camera) -> object:
    module = importlib.import_module("pycolmap")
    return module.Camera(
        model="PINHOLE",
        width=camera.width,
        height=camera.height,
        params=camera.colmap_params(),
    )


def estimate_pose(
    cameras: Sequence[Camera],
    points2d: FloatArray,
    points3d: FloatArray,
    camera_indices: Sequence[int],
    landmark_ids: Sequence[int],
) -> dict[str, object]:
    """Solve a camera/rig pose without VIO; count both observations and distinct landmarks."""
    if (
        len(points2d) != len(points3d)
        or len(points2d) != len(camera_indices)
        or len(points2d) != len(landmark_ids)
    ):
        raise ValueError("Correspondence arrays have inconsistent lengths")
    if not cameras or any(
        index < 0 or index >= len(cameras) for index in camera_indices
    ):
        raise ValueError("Invalid correspondence camera index")
    if (
        points2d.shape != (len(points2d), 2)
        or points3d.shape != (len(points2d), 3)
        or not np.isfinite(points2d).all()
        or not np.isfinite(points3d).all()
    ):
        raise ValueError("Expected finite 2D/3D correspondences")
    if len(set(landmark_ids)) < 6:
        return {"success": False, "reason": "insufficient_correspondences"}
    module = importlib.import_module("pycolmap")
    calibration = [colmap_camera(camera) for camera in cameras]
    estimation = {"ransac": {"max_error": 12.0, "min_inlier_ratio": MIN_INLIER_RATIO}}
    if len(cameras) == 1:
        result = module.estimate_and_refine_absolute_pose(
            points2d + 0.5,
            points3d,
            calibration[0],
            estimation,
            {"refine_focal_length": False, "refine_extra_params": False},
        )
    else:
        cams_from_rig = [
            module.Rigid3d(inverse_transform(camera.T_device_camera)[:3])
            for camera in cameras
        ]
        result = module.estimate_and_refine_generalized_absolute_pose(
            points2d + 0.5,
            points3d,
            camera_indices,
            cams_from_rig,
            calibration,
            estimation["ransac"],
            {"refine_focal_length": False, "refine_extra_params": False},
        )
    if result is None:
        return {"success": False, "reason": "pose_estimation_failed"}
    return pose_result(result, cameras, landmark_ids)


def pose_result(
    result: dict[str, object], cameras: Sequence[Camera], landmark_ids: Sequence[int]
) -> dict[str, object]:
    mask = np.asarray(result["inlier_mask"], dtype=bool)
    if mask.shape != (len(landmark_ids),):
        raise ValueError("COLMAP returned an invalid inlier mask")
    ids = np.asarray(landmark_ids)
    unique_inliers = len(set(ids[mask]))
    observation_ratio = float(mask.mean())
    landmark_ratio = unique_inliers / len(set(landmark_ids))
    success = (
        unique_inliers >= MIN_INLIER_LANDMARKS
        and min(observation_ratio, landmark_ratio) >= MIN_INLIER_RATIO
    )
    output: dict[str, object] = {
        "success": success,
        "num_inlier_observations": int(mask.sum()),
        "num_inlier_landmarks": unique_inliers,
        "observation_inlier_ratio": observation_ratio,
        "landmark_inlier_ratio": landmark_ratio,
    }
    if success:
        key = "cam_from_world" if len(cameras) == 1 else "rig_from_world"
        transform = cast(PoseTransform, result[key])
        T_camera_or_rig_world = np.eye(4)
        T_camera_or_rig_world[:3] = transform.matrix()
        T_world_camera_or_rig = inverse_transform(T_camera_or_rig_world)
        T_world_device = (
            T_world_camera_or_rig @ inverse_transform(cameras[0].T_device_camera)
            if len(cameras) == 1
            else T_world_camera_or_rig
        )
        output["T_world_device"] = T_world_device.tolist()
    else:
        output["reason"] = "insufficient_inliers"
    return output


def write_reference_model(directory: Path, views: Sequence[View]) -> None:
    """Write known camera poses in COLMAP's camera-from-world convention."""
    directory.mkdir()
    camera_lines, image_lines = [], []
    for image_id, view in enumerate(views, start=1):
        camera = view.camera
        params = " ".join(str(value) for value in camera.colmap_params())
        camera_lines.append(
            f"{image_id} PINHOLE {camera.width} {camera.height} {params}"
        )
        transform = inverse_transform(view.T_world_camera)
        pose = np.concatenate((quaternion_wxyz(transform), transform[:3, 3]))
        pose_string = " ".join(str(value) for value in pose)
        image_lines.append(f"{image_id} {pose_string} {image_id} {view.name}\n")
    (directory / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (directory / "images.txt").write_text("\n".join(image_lines) + "\n")
    (directory / "points3D.txt").write_text("")


class HlocBackend:
    """Offline HLOC wrappers plus one serialized, immutable resident query session.

    prepare_map is explicit initialization; a query may initialize a fresh backend
    for compatibility, but never silently switches or reloads an active map.
    Stop the worker before replacing map files. close is terminal; create a new
    backend after closing or initialization failure.
    """

    def __init__(self, *, cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES) -> None:
        self._map: ResidentMap | None = None
        self._models: ResidentModels | None = None
        self._closed = False
        self._cache_max_bytes = FeatureCache(cache_max_bytes).max_bytes
        try:
            self._load_adapters()
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def _load_adapters(self) -> None:
        # hloc imports pycolmap at import time. On macOS the pycolmap and torch
        # wheels each bundle libomp; loading pycolmap's copy first makes the
        # first torch inference segfault. Loading torch first avoids that.
        importlib.import_module("torch")
        hloc = importlib.import_module("hloc")
        if hloc.__file__ is None:
            raise ValueError("HLOC must have a source checkout")
        validate_hloc_checkout(Path(hloc.__file__).resolve().parent.parent)
        self.extractor = importlib.import_module("hloc.extract_features")
        self.matcher = importlib.import_module("hloc.match_features")
        self.triangulator = importlib.import_module("hloc.triangulation")
        self.io = importlib.import_module("hloc.utils.io")
        self.colmap = importlib.import_module("pycolmap")
        if self.colmap.__version__ != "4.2.1":
            raise ValueError("This example requires pycolmap==4.2.1")
        parameters = inspect.signature(self.extractor.main).parameters
        if "num_workers" not in parameters and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            raise ValueError(
                "HLOC checkout is missing the worker patch; run "
                "`git -C Hierarchical-Localization apply ../patches/hloc-workers.patch`"
            )
        self.feature_conf = self.extractor.confs["superpoint_aachen"]
        self.match_conf = self.matcher.confs["superpoint+lightglue"]

    def __enter__(self) -> HlocBackend:
        if self._closed:
            raise RuntimeError("HLOC backend is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _release_resident(self) -> None:
        map_session, self._map = self._map, None
        models, self._models = self._models, None
        try:
            if map_session is not None:
                map_session.close()
        finally:
            if models is not None:
                models.close()

    def close(self) -> None:
        self._closed = True
        self._release_resident()

    def prepare_map(self, map_path: Path) -> dict[str, float]:
        """Load once; same path is a no-op, an explicit different path reinitializes.

        Returned initialization stage durations are seconds. Failed initialization
        closes all partial resources and permanently closes this backend.
        """
        if self._closed:
            raise RuntimeError("HLOC backend is closed")
        # abspath is a lexical identity check, not a per-query stat/hash/read.
        map_path = map_path.absolute()
        if self._map is not None and self._map.path == map_path:
            return {}
        start_time_sec = time.perf_counter()
        try:
            self._release_resident()
            self._map = ResidentMap(map_path, self.colmap, self._cache_max_bytes)
            model_start_time_sec = time.perf_counter()
            self._models = ResidentModels(self.extractor, self.matcher)
            return {
                **self._map.initialization_durations_sec,
                "models_duration_sec": time.perf_counter() - model_start_time_sec,
                "initialization_duration_sec": time.perf_counter() - start_time_sec,
            }
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def extract(
        self, images: Path, names: list[str], output: Path
    ) -> tuple[Path, Path]:
        features = self.extractor.main(
            self.feature_conf,
            images,
            image_list=names,
            feature_path=output / "features.h5",
            as_half=False,
            num_workers=DATA_LOADER_WORKERS,
        )
        global_features = self.extractor.main(
            self.extractor.confs["netvlad"],
            images,
            image_list=names,
            feature_path=output / "global.h5",
            as_half=False,
            num_workers=DATA_LOADER_WORKERS,
        )
        return features, global_features

    def build(
        self, output: Path, views: Sequence[View], pairs: Sequence[tuple[str, str]]
    ) -> dict[str, object]:
        write_reference_model(output / "reference", views)
        pairs_path = output / "pairs.txt"
        pairs_path.write_text("".join(f"{left} {right}\n" for left, right in pairs))
        features, global_features = self.extract(
            output / "images", [view.name for view in views], output
        )
        matches = self.matcher.main(
            self.match_conf,
            pairs_path,
            features,
            matches=output / "matches.h5",
            num_workers=DATA_LOADER_WORKERS,
        )
        reconstruction = self.triangulator.main(
            output / "sfm",
            output / "reference",
            output / "images",
            pairs_path,
            features,
            matches,
            estimate_two_view_geometries=True,
        )
        if reconstruction.num_points3D() == 0:
            raise ValueError(
                "Triangulation produced zero landmarks; no usable map was built"
            )
        with h5py.File(global_features, "r") as data:
            for view in views:
                descriptor = np.asarray(data[view.name]["global_descriptor"])
                if not np.isfinite(descriptor).all() or np.linalg.norm(descriptor) == 0:
                    raise ValueError("Missing or invalid retrieval descriptors")
        return {
            "num_keyframes": len(views),
            "num_pairs": len(pairs),
            "num_landmarks": reconstruction.num_points3D(),
            "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
        }

    def query(
        self, map_path: Path, frames: Sequence[Frame], query_path: Path | None = None
    ) -> dict[str, object]:
        """Localize in memory; query_path is accepted but never used or created."""
        start_time_sec = time.perf_counter()
        if self._closed:
            raise RuntimeError("HLOC backend is closed")
        if self._map is None:
            self.prepare_map(map_path)
        map_session, models = self._map, self._models
        if map_session is None or models is None:
            raise RuntimeError("Resident HLOC initialization is incomplete")
        if map_session.path != map_path.absolute():
            raise ValueError(
                "Query map differs from active snapshot; call prepare_map explicitly"
            )
        timings = StageTimings(models.synchronize)
        with timings.stage("extraction"):
            for frame in frames:
                if frame.image.shape[:2] != (frame.camera.height, frame.camera.width):
                    raise ValueError(
                        "Query image dimensions differ from camera calibration"
                    )
            predictions = [models.extract(frame.image) for frame in frames]
        with timings.stage("retrieval"):
            candidates = map_session.retrieve(
                [descriptor for _, descriptor in predictions]
            )
        features = [prediction[0] for prediction in predictions]
        with timings.stage("matching"):
            matches = {
                (camera_index, reference): (
                    models.match(
                        features[camera_index], map_session.features(reference)
                    )
                    if len(features[camera_index]["keypoints"])
                    else np.empty((0, 2), dtype=np.int64)
                )
                for camera_index, references in enumerate(candidates.values())
                for reference in references
            }
        best = self._score_resident_clusters(
            map_session, candidates, features, matches, frames, timings
        )
        output: dict[str, object] = {
            **best,
            "cache_statistics": map_session.cache.snapshot(),
        }
        output["stage_durations_sec"] = {
            **timings.durations_sec,
            "end_to_end_duration_sec": time.perf_counter() - start_time_sec,
        }
        return output

    def _score_resident_clusters(
        self,
        map_session: ResidentMap,
        candidates: dict[str, list[str]],
        features: Sequence[Features],
        matches: dict[tuple[int, str], np.ndarray],
        frames: Sequence[Frame],
        timings: StageTimings,
    ) -> dict[str, object]:
        with timings.stage("correspondence"):
            selected = {
                reference
                for references in candidates.values()
                for reference in references
            }
            points = {name: map_session.landmarks[name] for name in sorted(selected)}
            clusters = covisibility_clusters(points)
        best: dict[str, object] = {"success": False, "reason": "no_matches"}
        # Emit a zero pose stage when no cluster has solvable correspondences.
        timings.durations_sec["pose_duration_sec"] = 0.0
        for cluster in clusters:
            with timings.stage("correspondence"):
                correspondences = self._resident_correspondences(
                    map_session, candidates, cluster, features, matches
                )
            if correspondences is None:
                continue
            with timings.stage("pose"):
                result = estimate_pose(
                    [frame.camera for frame in frames], *correspondences
                )
            if result.get("success"):
                if not best.get("success") or cast(
                    int, result["num_inlier_landmarks"]
                ) > cast(int, best["num_inlier_landmarks"]):
                    best = result
            elif not best.get("success") and cast(
                int, result.get("num_inlier_landmarks", -1)
            ) >= cast(int, best.get("num_inlier_landmarks", -1)):
                # Keep the most informative failure (e.g. insufficient_inliers
                # with counts) instead of reporting every failure as no_matches.
                best = result
        return best

    def _resident_correspondences(
        self,
        map_session: ResidentMap,
        candidates: dict[str, list[str]],
        cluster: Sequence[str],
        features: Sequence[Features],
        matches: dict[tuple[int, str], np.ndarray],
    ) -> tuple[FloatArray, FloatArray, list[int], list[int]] | None:
        model = map_session.model
        if model is None:
            raise RuntimeError("Resident map is closed")
        assignments = []
        for camera_index, references in enumerate(candidates.values()):
            for reference in references:
                if reference not in cluster:
                    continue
                for query_index, reference_index in matches[(camera_index, reference)]:
                    point_id = int(
                        map_session.point_ids[reference][int(reference_index)]
                    )
                    if point_id >= 0:
                        assignments.append((camera_index, int(query_index), point_id))
        assignments = unique_matches(assignments)
        if not assignments:
            return None
        points2d = np.asarray(
            [
                features[camera]["keypoints"][keypoint]
                for camera, keypoint, _ in assignments
            ],
            dtype=np.float64,
        )
        points3d = np.asarray(
            [model.points3D[point].xyz for _, _, point in assignments],
            dtype=np.float64,
        )
        return (
            points2d,
            points3d,
            [camera for camera, _, _ in assignments],
            [point for _, _, point in assignments],
        )

    def retrieve(
        self, reference_path: Path, query_path: Path, names: Sequence[str], model
    ) -> dict[str, list[str]]:
        references = sorted(image.name for image in model.images.values())
        with h5py.File(reference_path, "r") as source:
            descriptors = np.stack(
                [np.asarray(source[name]["global_descriptor"]) for name in references]
            )
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        if not np.isfinite(descriptors).all() or np.any(norms == 0):
            raise ValueError("Map retrieval descriptors are invalid")
        descriptors /= norms
        retrieved = {}
        with h5py.File(query_path, "r") as source:
            for name in names:
                descriptor = np.asarray(source[name]["global_descriptor"])
                if not np.isfinite(descriptor).all() or np.linalg.norm(descriptor) == 0:
                    raise ValueError("Query retrieval descriptor is invalid")
                scores = np.einsum(
                    "ij,j->i",
                    descriptors,
                    descriptor / np.linalg.norm(descriptor),
                    optimize=False,
                )
                if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.0001):
                    raise ValueError("Retrieval scores are invalid")
                retrieved[name] = [
                    references[index] for index in np.argsort(-scores)[:20]
                ]
        return retrieved

    def query_cluster(
        self,
        model,
        candidates: dict[str, list[str]],
        cluster: Sequence[str],
        query_features: Path,
        matches: Path,
        frames: Sequence[Frame],
    ) -> dict[str, object]:
        images = {image.name: image for image in model.images.values()}
        assignments = []
        for camera_index, (query_name, references) in enumerate(candidates.items()):
            for reference in references:
                if reference not in cluster:
                    continue
                pairs, _ = self.io.get_matches(matches, query_name, reference)
                for query_index, reference_index in pairs:
                    point = images[reference].points2D[int(reference_index)]
                    if point.has_point3D():
                        assignments.append(
                            (camera_index, int(query_index), point.point3D_id)
                        )
        assignments = unique_matches(assignments)
        if not assignments:
            return {"success": False, "reason": "no_matches"}
        with h5py.File(query_features, "r") as source:
            features = [np.asarray(source[name]["keypoints"]) for name in candidates]
        points2d = np.asarray(
            [features[camera][keypoint] for camera, keypoint, _ in assignments],
            dtype=np.float64,
        )
        points3d = np.asarray(
            [model.points3D[point].xyz for _, _, point in assignments], dtype=np.float64
        )
        return estimate_pose(
            [frame.camera for frame in frames],
            points2d,
            points3d,
            [camera for camera, _, _ in assignments],
            [point for _, _, point in assignments],
        )
