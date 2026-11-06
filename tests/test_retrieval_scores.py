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

from types import SimpleNamespace

import h5py
import hloc_backend
import numpy as np
import pytest


@pytest.mark.parametrize("score", [np.nan, np.inf, 1.01])
def test_invalid_cosine_scores_are_not_ranked(tmp_path, monkeypatch, score):
    reference = tmp_path / "reference.h5"
    query = tmp_path / "query.h5"
    for path, name in [(reference, "r.png"), (query, "q.png")]:
        with h5py.File(path, "w") as data:
            data.create_group(name).create_dataset("global_descriptor", data=[1.0, 0.0])
    backend = hloc_backend.HlocBackend.__new__(hloc_backend.HlocBackend)
    model = SimpleNamespace(images={1: SimpleNamespace(name="r.png")})
    monkeypatch.setattr(
        hloc_backend.np, "einsum", lambda *args, **kwargs: np.asarray([score])
    )
    with pytest.raises(ValueError, match="Retrieval scores"):
        backend.retrieve(reference, query, ["q.png"], model)
