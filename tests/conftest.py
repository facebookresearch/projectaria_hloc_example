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

import csv
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
from map_utils import Camera, View


@pytest.fixture
def camera() -> Camera:
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.05]
    return Camera("camera-rgb", 32, 24, 25.0, 15.5, 11.5, transform)


@pytest.fixture
def make_view(camera: Camera) -> Callable[..., View]:
    def make(name="a.png", xyz=(0.0, 0.0, 0.0), yaw_deg=0.0, sequence=0, index=0):
        angle = np.radians(yaw_deg)
        transform = np.eye(4)
        transform[:3, :3] = [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
        transform[:3, 3] = xyz
        return View(
            name, sequence, index, 1_000_000 + index * 1_000_000, camera, transform
        )

    return make


@pytest.fixture
def trajectory_file(tmp_path: Path) -> Callable[..., Path]:
    def write(
        name="trajectory.csv",
        timestamps_us=(1000, 3000, 7000),
        graphs=None,
        translations=None,
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            ["tracking_timestamp_us", "graph_uid"]
            + [f"q{axis}_world_device" for axis in "wxyz"]
            + [f"t{axis}_world_device" for axis in "xyz"]
        )
        with path.open("w", newline="") as source:
            writer = csv.DictWriter(source, fieldnames=fields)
            writer.writeheader()
            for index, timestamp_us in enumerate(timestamps_us):
                xyz = (
                    translations[index]
                    if translations is not None
                    else (float(index), 0, 0)
                )
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            [
                                timestamp_us,
                                graphs[index] if graphs is not None else "graph-A",
                                1,
                                0,
                                0,
                                0,
                                *xyz,
                            ],
                        )
                    )
                )
        return path

    return write


@pytest.fixture
def write_h5() -> Callable[..., Path]:
    def write(path, values, dataset="global_descriptor"):
        with h5py.File(path, "w") as data:
            for name, value in values.items():
                data.create_group(name).create_dataset(
                    dataset, data=np.asarray(value, dtype=np.float64)
                )
        return path

    return write
