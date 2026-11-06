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

"""Real native COLMAP geometry; no HLOC inference, PAT, device, or weights."""

from __future__ import annotations

import hloc_backend as hloc
import numpy as np
import pycolmap  # @manual=//third-party/colmap:pycolmap
import pytest
from map_utils import Camera, inverse_transform, transform_from_quaternion


def synthetic_scene(num_cameras: int, outliers: bool):
    rng = np.random.default_rng(1729)
    device_points = rng.uniform([-1.4, -0.9, 4.0], [1.4, 0.9, 8.0], size=(60, 3))
    world_device = transform_from_quaternion([0.97, 0.05, -0.1, 0.17], [1.2, -0.7, 2.3])
    world_points = device_points @ world_device[:3, :3].T + world_device[:3, 3]
    cameras, pixels, points, camera_indices, landmark_ids = [], [], [], [], []
    retained_ids = set()
    for index in range(num_cameras):
        angle = np.radians((index - 1.5) * 8)
        extrinsic = transform_from_quaternion(
            [np.cos(angle / 2), 0, np.sin(angle / 2), 0],
            [(index - 1.5) * 0.12, 0.03 * index, 0.02],
        )
        camera = Camera(f"camera-{index}", 640, 480, 430, 319.5, 239.5, extrinsic)
        cameras.append(camera)
        camera_world = inverse_transform(world_device @ extrinsic)
        camera_points = world_points @ camera_world[:3, :3].T + camera_world[:3, 3]
        assert np.all(camera_points[:, 2] > 0)
        image_points = camera_points[:, :2] / camera_points[
            :, 2, None
        ] * camera.focal + [camera.cx, camera.cy]
        rejected = (
            (np.arange(60) + index) % 5 == 0 if outliers else np.zeros(60, dtype=bool)
        )
        image_points[rejected] += [180, -150]
        retained_ids.update(np.flatnonzero(~rejected).tolist())
        pixels.extend(image_points)
        points.extend(world_points)
        camera_indices.extend([index] * 60)
        landmark_ids.extend(range(60))
    return (
        cameras,
        np.asarray(pixels),
        np.asarray(points),
        camera_indices,
        landmark_ids,
        world_device,
        retained_ids,
    )


@pytest.mark.parametrize("num_cameras", [1, 4], ids=["central-pnp", "four-camera-gpnp"])
@pytest.mark.parametrize("outliers", [False, True], ids=["exact", "ransac-outliers"])
def test_native_pose_recovers_world_device_and_distinct_landmarks(
    num_cameras, outliers
):
    cameras, pixels, points, indices, ids, expected, retained_ids = synthetic_scene(
        num_cameras, outliers
    )
    native_camera = hloc.colmap_camera(cameras[0])
    assert isinstance(native_camera, pycolmap.Camera)
    np.testing.assert_array_equal(native_camera.params, [430, 430, 320, 240])
    result = hloc.estimate_pose(cameras, pixels, points, indices, ids)
    assert result["success"] is True, result
    np.testing.assert_allclose(result["T_world_device"], expected, atol=1e-4, rtol=0)
    observations = num_cameras * (48 if outliers else 60)
    assert result["num_inlier_observations"] == observations
    assert result["num_inlier_landmarks"] == len(retained_ids)
    assert result["observation_inlier_ratio"] == pytest.approx(0.8 if outliers else 1)
    assert result["landmark_inlier_ratio"] == pytest.approx(len(retained_ids) / 60)
