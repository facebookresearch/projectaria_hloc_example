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

import hashlib
import io
import json
import urllib.error
import zipfile
from pathlib import Path

import download_sample as sample
import pytest


def make_archive(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)
    return path


class Response(io.BytesIO):
    def __init__(self, content: bytes, url: str = "https://cdn.example/sample") -> None:
        super().__init__(content)
        self.url = url

    def geturl(self) -> str:
        return self.url


@pytest.mark.parametrize("bad_digest", [False, True])
def test_stream_download_checks_bytes_and_digest(tmp_path, monkeypatch, bad_digest):
    content = b"public sample" * 100
    calls = []

    def open_url(url, timeout):
        calls.append((url, timeout))
        return Response(content)

    monkeypatch.setattr(sample.urllib.request, "urlopen", open_url)
    target = tmp_path / "sample"
    digest = "0" * 64 if bad_digest else hashlib.sha256(content).hexdigest()
    if bad_digest:
        with pytest.raises(ValueError, match="checksum"):
            sample.download_file("https://official.example/sample", target, digest)
    else:
        sample.download_file("https://official.example/sample", target, digest)
        assert target.read_bytes() == content
    assert calls == [("https://official.example/sample", 60)]


@pytest.mark.parametrize(
    "url,limit", [("http://cdn.example/sample", 100), ("https://cdn.example/sample", 1)]
)
def test_download_rejects_insecure_redirect_or_size(tmp_path, monkeypatch, url, limit):
    monkeypatch.setattr(
        sample.urllib.request, "urlopen", lambda *a, **k: Response(b"data", url)
    )
    monkeypatch.setattr(sample, "MAX_DOWNLOAD_BYTES", limit)
    with pytest.raises(ValueError):
        sample.download_file(
            "https://official.example/sample", tmp_path / "sample", "0" * 64
        )


def test_extracts_only_fixed_workflow_inputs(tmp_path):
    archive = make_archive(
        tmp_path / "sample.zip",
        [
            (sample.MPS_MEMBERS[0], b"trajectory"),
            (sample.MPS_MEMBERS[1], b"calibration"),
            ("../../escape", b"not extracted"),
            ("slam/semidense_points.csv", b"not extracted"),
        ],
    )
    output = tmp_path / "mps"
    sample.extract_mps(archive, output)
    assert (output / sample.MPS_MEMBERS[0]).read_bytes() == b"trajectory"
    assert (output / sample.MPS_MEMBERS[1]).read_bytes() == b"calibration"
    assert not (output / "slam/semidense_points.csv").exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("content,limit", [(b"", 100), (b"oversize", 2)])
def test_rejects_invalid_member_size(tmp_path, monkeypatch, content, limit):
    archive = make_archive(tmp_path / "sample.zip", [(sample.MPS_MEMBERS[0], content)])
    monkeypatch.setattr(sample, "MAX_MPS_FILE_BYTES", limit)
    with pytest.raises(ValueError, match="size"):
        sample.extract_mps(archive, tmp_path / "mps")


def test_rejects_missing_member(tmp_path):
    archive = make_archive(tmp_path / "sample.zip", [])
    with pytest.raises(ValueError, match="exactly one"):
        sample.extract_mps(archive, tmp_path / "mps")


def test_download_sample_publishes_verified_layout_and_removes_archive(
    tmp_path, monkeypatch
):
    source = make_archive(
        tmp_path / "source.zip", [(name, b"data") for name in sample.MPS_MEMBERS]
    )
    requests = []

    def download(url, target, digest):
        requests.append((url, digest))
        target.write_bytes(source.read_bytes() if url.endswith(".zip") else b"vrs")

    monkeypatch.setattr(sample, "download_file", download)
    output = tmp_path / "data/public_sample"
    result = sample.download_sample(output)
    assert result == {"vrs": str(output / "scan.vrs"), "mps": str(output / "scan_mps")}
    assert (output / "scan.vrs").read_bytes() == b"vrs"
    assert (output / "scan_mps" / sample.MPS_MEMBERS[1]).read_bytes() == b"data"
    assert requests == [
        (sample.SAMPLE_BASE_URL + sample.VRS_FILENAME, sample.VRS_SHA256),
        (sample.SAMPLE_BASE_URL + sample.MPS_FILENAME, sample.MPS_SHA256),
    ]
    assert list(output.parent.iterdir()) == [output]


def test_download_failure_leaves_no_sample_or_temporary_files(tmp_path, monkeypatch):
    def fail(*args):
        raise urllib.error.URLError("unavailable")

    monkeypatch.setattr(sample, "download_file", fail)
    with pytest.raises(urllib.error.URLError):
        sample.download_sample(tmp_path / "sample")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("symlink", [False, True])
def test_never_overwrites_existing_or_dangling_symlink(tmp_path, symlink):
    output = tmp_path / "sample"
    if symlink:
        output.symlink_to(tmp_path / "absent")
    else:
        output.mkdir()
    with pytest.raises(FileExistsError):
        sample.download_sample(output)


def test_cli_outputs_paths_and_reports_network_failure(tmp_path, monkeypatch, capsys):
    output = tmp_path / "sample"
    monkeypatch.setattr(
        sample,
        "download_sample",
        lambda path: {"vrs": str(path / "scan.vrs"), "mps": str(path / "scan_mps")},
    )
    sample.main(["--output-dir", str(output)])
    assert json.loads(capsys.readouterr().out)["vrs"] == str(output / "scan.vrs")

    def fail(path):
        raise urllib.error.URLError("network")

    monkeypatch.setattr(sample, "download_sample", fail)
    with pytest.raises(SystemExit) as caught:
        sample.main(["--output-dir", str(output)])
    assert caught.value.code == 2
    assert "network" in capsys.readouterr().err
