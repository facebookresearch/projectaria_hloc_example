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

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import hloc_backend as hloc
import numpy as np
import pytest
from aria_utils import Frame
from map_utils import HLOC_COMMIT, inverse_transform, transform_from_quaternion


@pytest.fixture
def backend(monkeypatch, tmp_path):
    extractor = SimpleNamespace(
        confs={"superpoint_aachen": {"name": "local"}, "netvlad": {"name": "global"}},
        main=Mock(),
    )
    matcher = SimpleNamespace(
        confs={"superpoint+lightglue": {"name": "match"}}, main=Mock()
    )
    modules = {
        "torch": SimpleNamespace(),
        "hloc": SimpleNamespace(__file__=str(tmp_path / "hloc" / "__init__.py")),
        "hloc.extract_features": extractor,
        "hloc.match_features": matcher,
        "hloc.triangulation": SimpleNamespace(main=Mock()),
        "hloc.utils.io": SimpleNamespace(get_matches=Mock()),
        "pycolmap": SimpleNamespace(__version__="4.2.1", Reconstruction=Mock()),
    }
    original_import = hloc.importlib.import_module
    monkeypatch.setattr(
        hloc.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    monkeypatch.setattr(hloc, "validate_hloc_checkout", Mock())
    return hloc.HlocBackend()


def test_hloc_checkout_pins_commit_and_checks_process(monkeypatch, tmp_path):
    run = Mock(return_value=SimpleNamespace(stdout=HLOC_COMMIT + "\n"))
    monkeypatch.setattr(hloc.subprocess, "run", run)
    hloc.validate_hloc_checkout(tmp_path)
    run.assert_called_once_with(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    run.return_value.stdout = "unreviewed-commit"
    with pytest.raises(ValueError, match=HLOC_COMMIT):
        hloc.validate_hloc_checkout(tmp_path)
    run.side_effect = subprocess.CalledProcessError(1, "git")
    with pytest.raises(subprocess.CalledProcessError):
        hloc.validate_hloc_checkout(tmp_path)


def test_backend_constructor_validates_both_dependency_snapshots(backend, monkeypatch):
    hloc.validate_hloc_checkout.assert_called_once()
    assert backend.feature_conf == {"name": "local"}
    assert backend.match_conf == {"name": "match"}
    backend.colmap.__version__ = "4.0.0"
    with pytest.raises(ValueError, match="pycolmap==4.2.1"):
        hloc.HlocBackend()


def test_hloc_requires_real_checkout_file_before_loading_model_adapters(monkeypatch):
    imported = Mock(return_value=SimpleNamespace(__file__=None))
    validate = Mock()
    monkeypatch.setattr(hloc.importlib, "import_module", imported)
    monkeypatch.setattr(hloc, "validate_hloc_checkout", validate)
    with pytest.raises(ValueError, match="source checkout"):
        hloc.HlocBackend()
    assert [call.args[0] for call in imported.call_args_list] == ["torch", "hloc"]
    validate.assert_not_called()


def test_constructor_failure_preserves_primary_when_cleanup_also_fails(monkeypatch):
    primary_error = ValueError("adapter initialization failed")
    cleanup_error = OSError("resource cleanup failed")
    load = Mock(side_effect=primary_error)
    close = Mock(side_effect=cleanup_error)
    monkeypatch.setattr(hloc.HlocBackend, "_load_adapters", load)
    monkeypatch.setattr(hloc.HlocBackend, "close", close)
    with pytest.raises(ValueError) as raised:
        hloc.HlocBackend()
    assert raised.value is primary_error
    assert raised.value.__cause__ is cleanup_error
    load.assert_called_once()
    close.assert_called_once()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_extract_calls_local_and_global_inference_at_call_site(
    backend, monkeypatch, tmp_path, num_workers
):
    monkeypatch.setattr(hloc, "DATA_LOADER_WORKERS", num_workers)
    backend.extractor.main.side_effect = [
        tmp_path / "features.h5",
        tmp_path / "global.h5",
    ]
    result = backend.extract(tmp_path / "images", ["a.png"], tmp_path)
    assert result == (tmp_path / "features.h5", tmp_path / "global.h5")
    calls = backend.extractor.main.call_args_list
    assert [call.args[0]["name"] for call in calls] == ["local", "global"]
    assert all(
        call.kwargs["image_list"] == ["a.png"]
        and call.kwargs["as_half"] is False
        and call.kwargs["num_workers"] == num_workers
        for call in calls
    )
    assert [call.kwargs["feature_path"].name for call in calls] == [
        "features.h5",
        "global.h5",
    ]


@pytest.fixture
def solver(monkeypatch):
    def generalized(*args):
        assert args[5] == {"max_error": 12.0, "min_inlier_ratio": 0.2}
        assert "ransac" not in args[5]
        return None

    module = SimpleNamespace(
        Camera=Mock(side_effect=lambda **kwargs: kwargs),
        Rigid3d=Mock(side_effect=lambda value: SimpleNamespace(matrix=lambda: value)),
        estimate_and_refine_absolute_pose=Mock(return_value=None),
        estimate_and_refine_generalized_absolute_pose=Mock(side_effect=generalized),
    )
    original_import = hloc.importlib.import_module
    monkeypatch.setattr(
        hloc.importlib,
        "import_module",
        lambda name: module if name == "pycolmap" else original_import(name),
    )
    return module


def test_central_pose_boundary_adds_half_pixel_exactly_once(solver, camera):
    pixels = np.arange(48, dtype=np.float64).reshape(24, 2)
    points = np.arange(72, dtype=np.float64).reshape(24, 3)
    assert hloc.estimate_pose([camera], pixels, points, [0] * 24, list(range(24))) == {
        "success": False,
        "reason": "pose_estimation_failed",
    }
    args = solver.estimate_and_refine_absolute_pose.call_args.args
    np.testing.assert_array_equal(args[0], pixels + 0.5)
    np.testing.assert_array_equal(args[1], points)
    assert args[2]["params"] == [25, 25, 16, 12]
    assert args[3]["ransac"]["max_error"] == 12
    assert args[4] == {"refine_focal_length": False, "refine_extra_params": False}
    np.testing.assert_array_equal(pixels, np.arange(48).reshape(24, 2))
    solver.estimate_and_refine_generalized_absolute_pose.assert_not_called()


def test_generalized_pose_passes_camera_from_rig_not_inverse_direction(solver, camera):
    pixels = np.zeros((24, 2))
    points = np.ones((24, 3))
    assert (
        hloc.estimate_pose(
            [camera, camera], pixels, points, [0, 1] * 12, list(range(24))
        )["reason"]
        == "pose_estimation_failed"
    )
    args = solver.estimate_and_refine_generalized_absolute_pose.call_args.args
    np.testing.assert_array_equal(args[0], pixels + 0.5)
    assert args[2] == [0, 1] * 12
    for extrinsic in args[3]:
        np.testing.assert_array_equal(
            extrinsic.matrix(), inverse_transform(camera.T_device_camera)[:3]
        )
    solver.estimate_and_refine_absolute_pose.assert_not_called()


@pytest.mark.parametrize(
    "case,error",
    [
        ("length_2d", "lengths"),
        ("length_camera", "lengths"),
        ("length_ids", "lengths"),
        ("no_camera", "camera index"),
        ("negative_camera", "camera index"),
        ("large_camera", "camera index"),
        ("shape_2d", "finite"),
        ("shape_3d", "finite"),
        ("nan_2d", "finite"),
        ("inf_3d", "finite"),
    ],
)
def test_pose_input_validation_before_native_import(monkeypatch, camera, case, error):
    cameras, pixels, points, indices, ids = (
        [camera],
        np.zeros((6, 2)),
        np.ones((6, 3)),
        [0] * 6,
        list(range(6)),
    )
    if case == "length_2d":
        pixels = pixels[:-1]
    elif case == "length_camera":
        indices.pop()
    elif case == "length_ids":
        ids.pop()
    elif case == "no_camera":
        cameras = []
    elif case == "negative_camera":
        indices[0] = -1
    elif case == "large_camera":
        indices[0] = 1
    elif case == "shape_2d":
        pixels = np.zeros((6, 3))
    elif case == "shape_3d":
        points = np.zeros((6, 2))
    elif case == "nan_2d":
        pixels[0, 0] = np.nan
    else:
        points[0, 0] = np.inf
    imported = Mock(
        side_effect=AssertionError("invalid data must not load native solver")
    )
    monkeypatch.setattr(hloc.importlib, "import_module", imported)
    with pytest.raises(ValueError, match=error):
        hloc.estimate_pose(cameras, pixels, points, indices, ids)
    imported.assert_not_called()


def test_insufficient_unique_landmarks_do_not_run_solver(solver, camera):
    result = hloc.estimate_pose(
        [camera], np.zeros((30, 2)), np.ones((30, 3)), [0] * 30, [1, 2, 3] * 10
    )
    assert result == {"success": False, "reason": "insufficient_correspondences"}
    solver.Camera.assert_not_called()


@pytest.mark.parametrize("rig", [False, True])
def test_pose_success_inverts_native_pose_into_world_device(camera, rig):
    expected = transform_from_quaternion([1, 0, 0, 1], [2, -1, 3])
    world_sensor = expected if rig else expected @ camera.T_device_camera
    native = SimpleNamespace(matrix=lambda: inverse_transform(world_sensor)[:3])
    data = {
        "inlier_mask": [True] * 40,
        "rig_from_world" if rig else "cam_from_world": native,
    }
    result = hloc.pose_result(
        data, [camera, camera] if rig else [camera], list(range(20)) * 2
    )
    assert result["success"] is True
    assert result["num_inlier_observations"] == 40
    assert result["num_inlier_landmarks"] == 20
    assert result["observation_inlier_ratio"] == 1
    assert result["landmark_inlier_ratio"] == 1
    np.testing.assert_allclose(result["T_world_device"], expected, atol=1e-12)


@pytest.mark.parametrize(
    "ids,mask,observations,landmarks,observation_ratio,landmark_ratio",
    [
        ([1] * 30, [True] * 30, 30, 1, 1.0, 1.0),
        (
            list(range(20)) + list(range(20, 120)),
            [True] * 20 + [False] * 100,
            20,
            20,
            1 / 6,
            1 / 6,
        ),
        (
            list(range(20)) * 5 + list(range(20, 120)),
            [True] * 100 + [False] * 100,
            100,
            20,
            0.5,
            1 / 6,
        ),
    ],
)
def test_pose_rejects_inflated_observation_or_landmark_metrics(
    camera, ids, mask, observations, landmarks, observation_ratio, landmark_ratio
):
    result = hloc.pose_result({"inlier_mask": mask}, [camera], ids)
    assert result["success"] is False
    assert result["reason"] == "insufficient_inliers"
    assert result["num_inlier_observations"] == observations
    assert result["num_inlier_landmarks"] == landmarks
    assert result["observation_inlier_ratio"] == pytest.approx(observation_ratio)
    assert result["landmark_inlier_ratio"] == pytest.approx(landmark_ratio)
    assert "T_world_device" not in result


def test_pose_rejects_malformed_native_mask(camera):
    with pytest.raises(ValueError, match="inlier mask"):
        hloc.pose_result({"inlier_mask": [[True] * 20]}, [camera], list(range(20)))


def test_reference_model_pose_and_pixel_centers(tmp_path, make_view):
    view = make_view("s000/rgb/0.png", xyz=(1, 2, 3), yaw_deg=90)
    path = tmp_path / "reference"
    hloc.write_reference_model(path, [view])
    camera_values = (path / "cameras.txt").read_text().split()
    assert camera_values[:4] == ["1", "PINHOLE", "32", "24"]
    assert [float(value) for value in camera_values[4:]] == [25, 25, 16, 12]
    image_values = (path / "images.txt").read_text().split()
    assert image_values[-2:] == ["1", view.name]
    pose = transform_from_quaternion(
        [float(value) for value in image_values[1:5]],
        [float(value) for value in image_values[5:8]],
    )
    np.testing.assert_allclose(pose, inverse_transform(view.T_world_camera), atol=1e-12)
    assert (path / "points3D.txt").read_text() == ""


@pytest.mark.parametrize(
    "descriptor,count,error",
    [
        ([1.0, 2.0], 30, None),
        ([0.0, 0.0], 30, "retrieval"),
        ([float("nan"), 1.0], 30, "retrieval"),
        ([1.0, 2.0], 0, "zero landmarks"),
    ],
)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_build_triangulation_statistics_and_bad_maps(
    backend,
    monkeypatch,
    tmp_path,
    make_view,
    write_h5,
    descriptor,
    count,
    error,
    num_workers,
):
    monkeypatch.setattr(hloc, "DATA_LOADER_WORKERS", num_workers)
    views = [make_view("a.png"), make_view("b.png", xyz=(0.2, 0, 0))]
    global_path = write_h5(
        tmp_path / "global.h5", {view.name: descriptor for view in views}
    )
    monkeypatch.setattr(
        backend, "extract", Mock(return_value=(tmp_path / "features.h5", global_path))
    )
    backend.matcher.main.return_value = tmp_path / "matches.h5"
    reconstruction = Mock()
    reconstruction.num_points3D.return_value = count
    reconstruction.compute_mean_reprojection_error.return_value = 0.75
    backend.triangulator.main.return_value = reconstruction
    if error:
        with pytest.raises(ValueError, match=error):
            backend.build(tmp_path, views, [("a.png", "b.png")])
    else:
        assert backend.build(tmp_path, views, [("a.png", "b.png")]) == {
            "num_keyframes": 2,
            "num_pairs": 1,
            "num_landmarks": 30,
            "mean_reprojection_error_px": 0.75,
        }
    assert (tmp_path / "pairs.txt").read_text() == "a.png b.png\n"
    backend.matcher.main.assert_called_once_with(
        backend.match_conf,
        tmp_path / "pairs.txt",
        tmp_path / "features.h5",
        matches=tmp_path / "matches.h5",
        num_workers=num_workers,
    )
    assert (
        backend.triangulator.main.call_args.kwargs["estimate_two_view_geometries"]
        is True
    )


def test_cold_retrieval_normalizes_descriptors_and_caps_at_twenty(
    backend, tmp_path, write_h5
):
    model = SimpleNamespace(
        images={i: SimpleNamespace(name=f"r{i:02}.png") for i in reversed(range(23))}
    )
    references = {f"r{i:02}.png": [10, i] for i in range(23)}
    reference_path = write_h5(tmp_path / "ref.h5", references)
    query_path = write_h5(tmp_path / "query.h5", {"q0.png": [2, 0], "q1.png": [0, 3]})
    result = backend.retrieve(reference_path, query_path, ["q0.png", "q1.png"], model)
    assert result["q0.png"] == [f"r{i:02}.png" for i in range(20)]
    assert result["q1.png"] == [f"r{i:02}.png" for i in reversed(range(3, 23))]


@pytest.mark.parametrize(
    "side,descriptor",
    [
        ("reference", [0, 0]),
        ("reference", [float("nan"), 1]),
        ("query", [0, 0]),
        ("query", [1, float("inf")]),
    ],
)
def test_retrieval_rejects_invalid_h5_descriptors(
    backend, tmp_path, write_h5, side, descriptor
):
    reference_path = write_h5(
        tmp_path / "ref.h5", {"a": descriptor if side == "reference" else [1, 0]}
    )
    query_path = write_h5(
        tmp_path / "query.h5", {"q": descriptor if side == "query" else [1, 0]}
    )
    model = SimpleNamespace(images={1: SimpleNamespace(name="a")})
    with pytest.raises(ValueError, match="descriptor"):
        backend.retrieve(reference_path, query_path, ["q"], model)


def point(point_id):
    return SimpleNamespace(point3D_id=point_id, has_point3D=lambda: point_id >= 0)


def test_query_cluster_discards_conflicts_without_deduplicating_other_cameras(
    backend, monkeypatch, tmp_path, write_h5, camera
):
    model = SimpleNamespace(
        images={
            1: SimpleNamespace(name="a", points2D=[point(10), point(11), point(-1)]),
            2: SimpleNamespace(name="b", points2D=[point(10), point(12), point(13)]),
        },
        points3D={
            i: SimpleNamespace(xyz=np.array([i, 1, 2])) for i in (10, 11, 12, 13)
        },
    )
    pairs = {
        ("q0", "a"): [[0, 0], [1, 1], [2, 2]],
        ("q0", "b"): [[0, 0], [1, 1], [3, 2]],
        ("q1", "a"): [[0, 0]],
    }
    backend.io.get_matches.side_effect = lambda path, query, reference: (
        np.array(pairs[(query, reference)]),
        None,
    )
    features = write_h5(
        tmp_path / "features.h5",
        {"q0": [[0, 1], [2, 3], [4, 5], [6, 7]], "q1": [[8, 9]]},
        dataset="keypoints",
    )
    solve = Mock(
        return_value={"success": False, "reason": "insufficient_correspondences"}
    )
    monkeypatch.setattr(hloc, "estimate_pose", solve)
    frames = [
        Frame(camera.label, i, 1_000_000, np.zeros((2, 2), dtype=np.uint8), camera)
        for i in range(2)
    ]
    result = backend.query_cluster(
        model,
        {"q0": ["a", "b", "not-in-cluster"], "q1": ["a"]},
        ["a", "b"],
        features,
        tmp_path / "matches.h5",
        frames,
    )
    assert result["reason"] == "insufficient_correspondences"
    args = solve.call_args.args
    np.testing.assert_array_equal(args[1], [[0, 1], [6, 7], [8, 9]])
    np.testing.assert_array_equal(args[2], [[10, 1, 2], [13, 1, 2], [10, 1, 2]])
    assert args[3:] == ([0, 0, 1], [10, 13, 10])
    assert backend.io.get_matches.call_count == 3
    indexed = SimpleNamespace(
        model=model,
        point_ids={
            name: np.array([item.point3D_id for item in image.points2D])
            for name, image in ((image.name, image) for image in model.images.values())
        },
    )
    in_memory_features = [
        {"keypoints": np.array([[0, 1], [2, 3], [4, 5], [6, 7]])},
        {"keypoints": np.array([[8, 9]])},
    ]
    direct_matches = {
        (int(query[1:]), reference): np.asarray(value)
        for (query, reference), value in pairs.items()
    }
    monkeypatch.setattr(
        hloc.h5py,
        "File",
        Mock(
            side_effect=AssertionError(
                "resident correspondence assembly must not open HDF5"
            )
        ),
    )
    assembled = backend._resident_correspondences(
        indexed,
        {"q0": ["a", "b", "not-in-cluster"], "q1": ["a"]},
        ["a", "b"],
        in_memory_features,
        direct_matches,
    )
    np.testing.assert_array_equal(assembled[0], args[1])
    np.testing.assert_array_equal(assembled[1], args[2])
    assert assembled[2:] == args[3:]


def test_query_cluster_without_landmarks_does_not_open_features(
    backend, tmp_path, camera
):
    model = SimpleNamespace(images={1: SimpleNamespace(name="a", points2D=[point(-1)])})
    backend.io.get_matches.return_value = (np.array([[0, 0]]), None)
    result = backend.query_cluster(
        model,
        {"q0": ["a"]},
        ["a"],
        tmp_path / "absent.h5",
        tmp_path / "absent-matches.h5",
        [],
    )
    assert result == {"success": False, "reason": "no_matches"}


@pytest.fixture
def resident_backend(backend, monkeypatch, tmp_path, write_h5, camera):
    map_path = tmp_path / "map"
    map_path.mkdir()
    write_h5(map_path / "global.h5", {"a": [1, 0], "b": [0.8, 0.2], "c": [0, 1]})
    with h5py.File(map_path / "features.h5", "w") as source:
        for name in ("a", "b", "c"):
            group = source.create_group(name)
            group["keypoints"] = np.zeros((1, 2), dtype=np.float32)
            group["descriptors"] = np.ones((256, 1), dtype=np.float32)
            group["image_size"] = [camera.width, camera.height]
    model = SimpleNamespace(
        images={
            i: SimpleNamespace(name=name, points2D=[point(i)])
            for i, name in enumerate(("a", "b", "c"), 1)
        },
        points3D={i: SimpleNamespace(xyz=np.array([i, 1, 2])) for i in (1, 2, 3)},
        num_points3D=lambda: 3,
    )
    backend.colmap.Reconstruction.return_value = model
    models = Mock()
    models.extract.return_value = (
        {
            "keypoints": np.zeros((1, 2), dtype=np.float32),
            "descriptors": np.ones((256, 1), dtype=np.float32),
            "image_size": np.array([camera.width, camera.height]),
        },
        np.array([1.0, 0.0]),
    )
    models.match.return_value = np.array([[0, 0]])
    constructor = Mock(return_value=models)
    monkeypatch.setattr(hloc, "ResidentModels", constructor)
    frames = [
        Frame(
            camera.label,
            0,
            1_000_000,
            np.zeros((camera.height, camera.width), dtype=np.uint8),
            camera,
        )
    ]
    return backend, map_path, models, constructor, frames


@pytest.mark.parametrize("successful", [True, False])
def test_query_scores_independent_covisibility_clusters_without_ground_truth(
    resident_backend, monkeypatch, tmp_path, successful
):
    backend, map_path, models, _, frames = resident_backend
    solve = Mock(
        side_effect=[
            {"success": successful, "num_inlier_landmarks": 20},
            {"success": successful, "num_inlier_landmarks": 25},
            {"success": successful, "num_inlier_landmarks": 21},
        ]
    )
    monkeypatch.setattr(hloc, "estimate_pose", solve)
    unused_query_path = tmp_path / "never-created"
    result = backend.query(map_path, frames, unused_query_path)
    assert result["success"] is successful
    if successful:
        assert result["num_inlier_landmarks"] == 25
    else:
        # The most informative failure (most inlier landmarks) is reported.
        assert result["num_inlier_landmarks"] == 25
    assert [call.args[4] for call in solve.call_args_list] == [[1], [2], [3]]
    assert not unused_query_path.exists()
    backend.extractor.main.assert_not_called()
    backend.matcher.main.assert_not_called()
    assert models.match.call_count == 3
    assert set(result["stage_durations_sec"]) == {
        "extraction_duration_sec",
        "retrieval_duration_sec",
        "matching_duration_sec",
        "correspondence_duration_sec",
        "pose_duration_sec",
        "end_to_end_duration_sec",
    }
    assert all(
        np.isfinite(value) and value >= 0
        for value in result["stage_durations_sec"].values()
    )
    assert result["stage_durations_sec"]["end_to_end_duration_sec"] >= sum(
        value
        for key, value in result["stage_durations_sec"].items()
        if key != "end_to_end_duration_sec"
    )
    backend.close()


def test_prepare_and_repeated_queries_load_map_normalize_and_models_once(
    resident_backend, monkeypatch
):
    backend, map_path, models, constructor, frames = resident_backend
    monkeypatch.setattr(hloc, "estimate_pose", Mock(return_value={"success": False}))
    initialization = backend.prepare_map(map_path)
    session = backend._map
    descriptor_identity = session.descriptors
    np.testing.assert_allclose(np.linalg.norm(descriptor_identity, axis=1), 1)
    assert not descriptor_identity.flags["W"]
    assert all(
        key.endswith("_duration_sec") and np.isfinite(value) and value >= 0
        for key, value in initialization.items()
    )
    assert backend.prepare_map(map_path) == {}
    first = backend.query(map_path, frames)
    second = backend.query(map_path, frames)
    assert backend._map is session and session.descriptors is descriptor_identity
    backend.colmap.Reconstruction.assert_called_once_with(map_path / "sfm")
    constructor.assert_called_once_with(backend.extractor, backend.matcher)
    assert models.extract.call_count == 2
    assert first["cache_statistics"]["cache_misses"] == 3
    assert second["cache_statistics"]["cache_hits"] == 3
    assert first["cache_statistics"]["cache_hits"] == 0
    assert session.feature_file.id.valid
    backend.close()
    assert not session.feature_file
    assert session.cache.snapshot()["cache_bytes"] == 0
    assert session.model is None
    models.close.assert_called_once()
    backend.close()
    models.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        backend.query(map_path, frames)
    with pytest.raises(RuntimeError, match="closed"):
        backend.prepare_map(map_path)
    with pytest.raises(RuntimeError, match="closed"):
        backend.__enter__()


def test_query_cross_map_rejection_never_reloads_and_explicit_prepare_reinitializes(
    resident_backend, tmp_path
):
    backend, map_path, models, constructor, frames = resident_backend
    backend.prepare_map(map_path)
    old_session = backend._map
    other = tmp_path / "other"
    other.symlink_to(map_path, target_is_directory=True)
    with pytest.raises(ValueError, match="active snapshot"):
        backend.query(other, frames)
    assert backend._map is old_session
    assert backend.colmap.Reconstruction.call_count == 1
    backend.prepare_map(other)
    assert old_session.model is None and old_session.feature_file is None
    assert backend._map.path == other
    assert constructor.call_count == 2
    assert models.close.call_count == 1
    backend.close()


def test_prepare_failure_closes_partial_map_and_is_terminal(
    resident_backend, monkeypatch
):
    backend, map_path, _, constructor, _ = resident_backend
    sessions = []
    original_map = hloc.ResidentMap

    def tracked_map(*args):
        session = original_map(*args)
        sessions.append((session, session.feature_file))
        return session

    monkeypatch.setattr(hloc, "ResidentMap", tracked_map)
    constructor.side_effect = RuntimeError("model forward initialization failure")
    with pytest.raises(RuntimeError, match="initialization failure"):
        backend.prepare_map(map_path)
    assert len(sessions) == 1 and not sessions[0][1].id.valid
    assert sessions[0][0].model is None and sessions[0][0].cache.bytes == 0
    assert backend._map is None and backend._models is None
    assert backend._closed
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.prepare_map(map_path)


def test_prepare_failure_preserves_primary_when_partial_map_cleanup_fails(
    resident_backend, monkeypatch
):
    backend, map_path, _, constructor, _ = resident_backend
    primary_error = RuntimeError("model initialization failed")
    cleanup_error = OSError("map handle cleanup failed")
    constructor.side_effect = primary_error
    original_close = hloc.ResidentMap.close
    closed_resources = []

    def failing_close(session):
        feature_file = session.feature_file
        original_close(session)
        closed_resources.append((session, feature_file))
        raise cleanup_error

    monkeypatch.setattr(hloc.ResidentMap, "close", failing_close)
    with pytest.raises(RuntimeError) as raised:
        backend.prepare_map(map_path)
    assert raised.value is primary_error
    assert raised.value.__cause__ is cleanup_error
    assert len(closed_resources) == 1
    session, feature_file = closed_resources[0]
    assert not feature_file.id.valid
    assert session.model is None and session.cache.bytes == 0
    assert backend._closed and backend._map is None and backend._models is None
    backend.close()
    assert len(closed_resources) == 1


def test_context_manager_closes_even_when_query_raises(resident_backend):
    backend, map_path, models, _, frames = resident_backend
    models.extract.side_effect = RuntimeError("neural inference failed")
    with pytest.raises(RuntimeError, match="neural inference failed"):
        with backend as active:
            active.prepare_map(map_path)
            session = active._map
            active.query(map_path, frames)
    assert session.model is None and session.feature_file is None
    models.close.assert_called_once()
    backend.matcher.main.assert_not_called()


def test_query_rejects_empty_reconstruction_before_matching(backend, tmp_path):
    backend.colmap.Reconstruction.return_value = SimpleNamespace(num_points3D=lambda: 0)
    with pytest.raises(ValueError, match="no reconstructed landmarks"):
        backend.query(tmp_path, [])
    backend.matcher.main.assert_not_called()
    assert backend._closed


def test_empty_query_and_no_matches_have_zero_pose_stage(resident_backend, monkeypatch):
    backend, map_path, models, _, frames = resident_backend
    result = backend.query(map_path, [])
    assert result["reason"] == "no_matches"
    assert result["stage_durations_sec"]["pose_duration_sec"] == 0.0
    models.extract.assert_not_called()
    models.match.assert_not_called()
    models.match.return_value = np.empty((0, 2), dtype=np.int64)
    solve = Mock(side_effect=AssertionError("empty matches must not estimate pose"))
    monkeypatch.setattr(hloc, "estimate_pose", solve)
    result = backend.query(map_path, frames)
    assert result["reason"] == "no_matches"
    assert result["stage_durations_sec"]["pose_duration_sec"] == 0.0
    solve.assert_not_called()
    backend.close()


@pytest.mark.parametrize("warm_cache", [False, True])
def test_zero_query_keypoints_do_not_read_map_features_or_change_cache(
    resident_backend, monkeypatch, warm_cache
):
    backend, map_path, models, _, frames = resident_backend
    backend.prepare_map(map_path)
    session = backend._map
    if warm_cache:
        session.features("a")
    before = session.cache.snapshot()
    original_features, descriptor = models.extract.return_value
    empty_features = {
        **original_features,
        "keypoints": np.empty((0, 2), dtype=np.float32),
        "descriptors": np.empty((256, 0), dtype=np.float32),
    }
    models.extract.return_value = (empty_features, descriptor)
    read_features = Mock(wraps=session.features)
    monkeypatch.setattr(session, "features", read_features)
    monkeypatch.setattr(
        hloc.h5py,
        "File",
        Mock(side_effect=AssertionError("empty query must not open map feature files")),
    )
    solve = Mock(side_effect=AssertionError("empty query must not estimate pose"))
    monkeypatch.setattr(hloc, "estimate_pose", solve)
    result = backend.query(map_path, frames)
    assert result["reason"] == "no_matches"
    assert result["stage_durations_sec"]["pose_duration_sec"] == 0.0
    assert result["cache_statistics"] == before == session.cache.snapshot()
    read_features.assert_not_called()
    models.match.assert_not_called()
    solve.assert_not_called()
    backend.colmap.Reconstruction.assert_called_once_with(map_path / "sfm")
    backend.close()


def test_empty_camera_query_does_not_skip_other_camera_matches(
    resident_backend, monkeypatch
):
    backend, map_path, models, _, frames = resident_backend
    original_features, descriptor = models.extract.return_value
    empty_features = {
        **original_features,
        "keypoints": np.empty((0, 2), dtype=np.float32),
        "descriptors": np.empty((256, 0), dtype=np.float32),
    }
    models.extract.side_effect = [
        (empty_features, descriptor),
        (original_features, descriptor),
    ]
    solve = Mock(return_value={"success": False, "reason": "insufficient_inliers"})
    monkeypatch.setattr(hloc, "estimate_pose", solve)
    result = backend.query(map_path, [frames[0], frames[0]])
    assert result["reason"] == "insufficient_inliers"
    assert models.match.call_count == 3
    assert result["cache_statistics"]["cache_misses"] == 3
    assert result["cache_statistics"]["cache_hits"] == 0
    assert all(call.args[3] == [1] for call in solve.call_args_list)
    assert solve.call_count == 3
    backend.close()


def test_query_validates_camera_dimensions_before_neural_inference(resident_backend):
    backend, map_path, models, _, frames = resident_backend
    frames[0].image = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="camera calibration"):
        backend.query(map_path, frames)
    models.extract.assert_not_called()
    backend.close()


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5])
def test_cache_bound_rejected_before_neural_import(monkeypatch, max_bytes):
    imported = Mock()
    monkeypatch.setattr(hloc.importlib, "import_module", imported)
    with pytest.raises(ValueError, match="positive integer"):
        hloc.HlocBackend(cache_max_bytes=max_bytes)
    imported.assert_not_called()


def test_unpatched_hloc_is_reported_with_the_patch_command(monkeypatch, tmp_path):
    def unpatched_main(conf, image_dir, export_dir=None, as_half=True):
        raise AssertionError("must not run")

    modules = {
        "torch": SimpleNamespace(),
        "hloc": SimpleNamespace(__file__=str(tmp_path / "hloc" / "__init__.py")),
        "hloc.extract_features": SimpleNamespace(confs={}, main=unpatched_main),
        "hloc.match_features": SimpleNamespace(confs={}, main=Mock()),
        "hloc.triangulation": SimpleNamespace(main=Mock()),
        "hloc.utils.io": SimpleNamespace(),
        "pycolmap": SimpleNamespace(__version__="4.2.1"),
    }
    original_import = hloc.importlib.import_module
    monkeypatch.setattr(
        hloc.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    monkeypatch.setattr(hloc, "validate_hloc_checkout", Mock())
    with pytest.raises(ValueError, match="hloc-workers.patch"):
        hloc.HlocBackend()


@pytest.mark.parametrize("mode", ["fast", None, ""])
def test_invalid_rig_execution_rejected_before_neural_import(monkeypatch, mode):
    imported = Mock()
    monkeypatch.setattr(hloc.importlib, "import_module", imported)
    with pytest.raises(ValueError, match="rig_execution"):
        hloc.HlocBackend(rig_execution=mode)
    imported.assert_not_called()


def sequential_reversed(tasks, workers):
    """Out-of-order completion: the last camera finishes first."""
    assert 2 <= workers <= 4 and workers <= len(tasks)
    results = [None] * len(tasks)
    for index in reversed(range(len(tasks))):
        results[index] = tasks[index]()
    return results


def threaded(tasks, workers):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda task: task(), tasks))


@pytest.fixture
def rig_query(resident_backend, camera):
    backend, map_path, models, _, _ = resident_backend
    with h5py.File(map_path / "features.h5", "a") as source:
        for tag, name in enumerate(("a", "b", "c")):
            source[name]["tag"] = np.array([tag], dtype=np.float32)
    frames = [
        Frame(
            camera.label,
            index,
            1_000_000,
            np.zeros((camera.height, camera.width), dtype=np.uint8),
            camera,
        )
        for index in range(2)
    ]
    local, _ = models.extract.return_value
    extracted = {
        id(frame.image): (
            {**local, "tag": np.array([index], dtype=np.float32)},
            np.array([1.0, 0.0]) if index == 0 else np.array([0.0, 1.0]),
        )
        for index, frame in enumerate(frames)
    }
    models.extract.side_effect = lambda image: extracted[id(image)]
    produced = {}

    def match(query, reference):
        pair = np.array([[0, 0]])
        produced[id(pair)] = (int(query["tag"][0]), int(reference["tag"][0]))
        return pair

    models.match.side_effect = match
    models.device = "cuda"
    models.run_parallel.side_effect = sequential_reversed
    return backend, map_path, models, frames, produced


def record_scoring(monkeypatch):
    recorded = []
    original = hloc.HlocBackend._score_resident_clusters

    def spy(self, map_session, candidates, features, matches, frames, timings):
        recorded.append((candidates, features, matches))
        return original(
            self, map_session, candidates, features, matches, frames, timings
        )

    monkeypatch.setattr(hloc.HlocBackend, "_score_resident_clusters", spy)
    return recorded


def test_parallel_rig_query_equals_the_serial_reference_path(rig_query, monkeypatch):
    backend, map_path, models, frames, produced = rig_query
    serial_backend = hloc.HlocBackend(rig_execution="serial")
    solve = Mock(return_value={"success": False, "reason": "insufficient_inliers"})
    monkeypatch.setattr(hloc, "estimate_pose", solve)
    recorded = record_scoring(monkeypatch)
    parallel = backend.query(map_path, frames)
    serial = serial_backend.query(map_path, frames)
    (
        (parallel_candidates, _, parallel_matches),
        (serial_candidates, _, serial_matches),
    ) = recorded
    assert (
        parallel_candidates
        == serial_candidates
        == {
            "q0.png": ["a", "b", "c"],
            "q1.png": ["c", "b", "a"],
        }
    )
    expected_keys = [(0, "a"), (0, "b"), (0, "c"), (1, "c"), (1, "b"), (1, "a")]
    assert list(parallel_matches) == list(serial_matches) == expected_keys
    for matches in (parallel_matches, serial_matches):
        for (camera_index, reference), pair in matches.items():
            assert produced[id(pair)] == (camera_index, "abc".index(reference))
    assert parallel["cache_statistics"] == serial["cache_statistics"]
    assert parallel["cache_statistics"]["cache_misses"] == 3
    assert parallel["cache_statistics"]["cache_hits"] == 3
    assert {
        key: value
        for key, value in parallel.items()
        if key not in ("rig_execution", "stage_durations_sec")
    } == {
        key: value
        for key, value in serial.items()
        if key not in ("rig_execution", "stage_durations_sec")
    }
    parallel_calls, serial_calls = solve.call_args_list[:3], solve.call_args_list[3:]
    assert len(serial_calls) == 3
    for left, right in zip(parallel_calls, serial_calls):
        assert left.args[0] == right.args[0]
        for left_array, right_array in zip(left.args[1:], right.args[1:]):
            np.testing.assert_array_equal(left_array, right_array)
    session = backend._map
    reference_bytes = sum(
        array.nbytes for name in "abc" for array in session.features(name).values()
    )
    assert parallel["rig_execution"] == {
        "requested": "auto",
        "mode": "parallel",
        "workers": 2,
        "cameras": 2,
        "device": "cuda",
        "reason": "auto",
        "reference_prefetch_bytes": reference_bytes,
    }
    assert serial["rig_execution"]["mode"] == "serial"
    assert serial["rig_execution"]["reason"] == "requested"
    assert [call.args[1] for call in models.run_parallel.call_args_list] == [2, 2]
    assert all(len(call.args[0]) == 2 for call in models.run_parallel.call_args_list)
    assert set(parallel["stage_durations_sec"]) == set(serial["stage_durations_sec"])
    backend.close()
    serial_backend.close()


@pytest.mark.parametrize("empty_camera", [None, 0])
def test_parallel_matching_reads_map_features_only_on_the_coordinator_in_serial_order(
    rig_query, monkeypatch, empty_camera
):
    backend, map_path, models, frames, produced = rig_query
    models.run_parallel.side_effect = threaded
    if empty_camera is not None:
        image = frames[empty_camera].image
        local, descriptor = models.extract.side_effect(image)
        empty = {
            **local,
            "keypoints": np.empty((0, 2), dtype=np.float32),
            "descriptors": np.empty((256, 0), dtype=np.float32),
        }
        original_extract = models.extract.side_effect
        models.extract.side_effect = lambda frame_image: (
            (empty, descriptor)
            if frame_image is image
            else original_extract(frame_image)
        )
    backend.prepare_map(map_path)
    session = backend._map
    coordinator = threading.get_ident()
    reads = []
    original_features = session.features

    def read(name):
        reads.append((name, threading.get_ident() == coordinator))
        return original_features(name)

    monkeypatch.setattr(session, "features", read)
    monkeypatch.setattr(hloc, "estimate_pose", Mock(return_value={"success": False}))
    recorded = record_scoring(monkeypatch)
    result = backend.query(map_path, frames)
    expected = (
        ["a", "b", "c", "c", "b", "a"] if empty_camera is None else ["c", "b", "a"]
    )
    assert reads == [(name, True) for name in expected]
    matched = [
        produced[id(pair)] for pair in recorded[0][2].values() if id(pair) in produced
    ]
    assert sorted({camera for camera, _ in matched}) == (
        [0, 1] if empty_camera is None else [1]
    )
    if empty_camera is not None:
        assert all(
            recorded[0][2][(empty_camera, name)].shape == (0, 2) for name in "abc"
        )
    assert result["rig_execution"]["reference_prefetch_bytes"] > 0
    backend.close()


@pytest.mark.parametrize(
    "requested,device,cameras,reason",
    [
        ("parallel", "cpu", 2, "requires_cuda"),
        ("auto", "cuda", 1, "single_camera"),
        ("serial", "cuda", 2, "requested"),
    ],
)
def test_reference_serial_path_runs_when_parallel_is_unavailable_or_not_requested(
    rig_query, monkeypatch, requested, device, cameras, reason
):
    _, map_path, models, frames, _ = rig_query
    backend = hloc.HlocBackend(rig_execution=requested)
    models.device = device
    monkeypatch.setattr(hloc, "estimate_pose", Mock(return_value={"success": False}))
    result = backend.query(map_path, frames[:cameras])
    models.run_parallel.assert_not_called()
    assert result["rig_execution"] == {
        "requested": requested,
        "mode": "serial",
        "workers": 1,
        "cameras": cameras,
        "device": device,
        "reason": reason,
        "reference_prefetch_bytes": 0,
    }
    assert models.match.call_count == 3 * cameras
    backend.close()


@pytest.mark.parametrize("stage", ["extraction", "matching"])
def test_parallel_worker_failure_propagates_and_the_next_query_runs(
    rig_query, monkeypatch, stage
):
    backend, map_path, models, frames, _ = rig_query
    monkeypatch.setattr(hloc, "estimate_pose", Mock(return_value={"success": False}))
    failure = RuntimeError(f"{stage} worker failed")
    effects = iter(
        ([failure] if stage == "extraction" else [sequential_reversed, failure])
        + [sequential_reversed, sequential_reversed]
    )

    def run(tasks, workers):
        effect = next(effects)
        if isinstance(effect, BaseException):
            raise effect
        return effect(tasks, workers)

    models.run_parallel.side_effect = run
    with pytest.raises(RuntimeError, match=f"{stage} worker failed"):
        backend.query(map_path, frames)
    result = backend.query(map_path, frames)
    assert result["rig_execution"]["mode"] == "parallel"
    assert not backend._closed
    backend.close()


def test_parallel_worker_error_survives_a_failing_stage_sync(rig_query, monkeypatch):
    backend, map_path, models, frames, _ = rig_query
    backend.prepare_map(map_path)
    failure = RuntimeError("camera worker failed")
    models.run_parallel.side_effect = failure
    syncs = iter([None, RuntimeError("stage sync failed")])

    def synchronize():
        effect = next(syncs)
        if effect is not None:
            raise effect

    models.synchronize.side_effect = synchronize
    with pytest.raises(RuntimeError, match="camera worker failed") as raised:
        backend.query(map_path, frames)
    assert raised.value is failure
    assert any("stage sync failed" in note for note in failure.__notes__)
    models.match.assert_not_called()
    backend.close()
