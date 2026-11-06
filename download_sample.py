#!/usr/bin/env fbpython
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

"""Download the public Aria Gen 2 VRS/MPS sample used by Depth From Stereo."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

SAMPLE_BASE_URL: str = (
    "https://www.projectaria.com/async/sample/download/?bucket=core&filename="
)
VRS_FILENAME: str = "aria_gen2_sample_data_1.vrs"
MPS_FILENAME: str = "aria_gen2_sample_data_1_mps_output_dec_2025.zip"
VRS_SHA256: str = "52e78d73dfc12b1d551292e0598b2babc7851c83297ebaab318686fb372f5c7f"
MPS_SHA256: str = "17a78565047a569469d9dd2a0d8b1648740f28cfbb7ea6d3699eae5f1c07631d"
MAX_DOWNLOAD_BYTES: int = 1_073_741_824
MAX_MPS_FILE_BYTES: int = 67_108_864
MPS_MEMBERS: tuple[str, ...] = (
    "slam/closed_loop_trajectory.csv",
    "slam/online_calibration.jsonl",
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download_file(url: str, destination: Path, expected_sha256: str) -> None:
    """Follow the official HTTPS redirect and publish only checksum-verified bytes."""
    with urllib.request.urlopen(url, timeout=60) as response:
        if not response.geturl().startswith("https://"):
            raise ValueError("Sample download redirected away from HTTPS")
        with destination.open("xb") as target:
            total_bytes = 0
            while chunk := response.read(1_048_576):
                total_bytes += len(chunk)
                if total_bytes > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Sample download exceeds the 1 GiB file limit")
                target.write(chunk)
    if sha256(destination) != expected_sha256:
        raise ValueError(
            "Sample checksum mismatch; the official asset may have changed"
        )


def extract_mps(archive_path: Path, output: Path) -> None:
    """Extract only the two fixed workflow inputs, not arbitrary ZIP member paths."""
    with zipfile.ZipFile(archive_path) as archive:
        for name in MPS_MEMBERS:
            matches = [item for item in archive.infolist() if item.filename == name]
            if len(matches) != 1 or matches[0].is_dir():
                raise ValueError(f"MPS archive must contain exactly one {name}")
            info = matches[0]
            if info.file_size <= 0 or info.file_size > MAX_MPS_FILE_BYTES:
                raise ValueError(f"Invalid size for MPS file {name}")
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as destination:
                total_bytes = 0
                while chunk := source.read(1_048_576):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_MPS_FILE_BYTES:
                        raise ValueError(
                            f"MPS file {name} exceeds its extraction limit"
                        )
                    destination.write(chunk)


def download_sample(output: Path) -> dict[str, str]:
    """Stage verified inputs before publishing them in a new sample directory."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Sample directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="aria-sample-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        staging = root / "sample"
        staging.mkdir()
        download_file(SAMPLE_BASE_URL + VRS_FILENAME, staging / "scan.vrs", VRS_SHA256)
        archive = root / "mps.zip"
        download_file(SAMPLE_BASE_URL + MPS_FILENAME, archive, MPS_SHA256)
        extract_mps(archive, staging / "scan_mps")
        # Reserve the name without replacing another user's existing directory.
        output.mkdir()
        for item in staging.iterdir():
            item.rename(output / item.name)
    return {"vrs": str(output / "scan.vrs"), "mps": str(output / "scan_mps")}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/public_sample"))
    args = parser.parse_args(argv)
    try:
        result = download_sample(args.output_dir)
    except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
