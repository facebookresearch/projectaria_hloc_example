# Indoor mapping with LaMAria (Aria Gen 1)

Build a native SLAM-camera map and an RGB map from **`R_03_easy`**, then localize frames that were not selected as mapping keyframes. This short office/lab recording is a mapping demonstration, **not a separate-recording accuracy benchmark**. Keep the [Pilot walkthrough](../README.md#pilot-quickstart-and-vio) for a separate query recording and recorded VIO replay.

First [install the pinned HLOC environment](../README.md#install-the-pinned-environment) and run this guide from the repository root. Keep new maps and query outputs outside tracked source.

## 1. Download the original RGB-capable VRS

Read the [official LaMAria download page](https://lamaria.ethz.ch/slam_datasets) and its source terms. Use the raw VRS; ASL/ROS exports do not include the RGB stream needed here.

```bash
mkdir -p data/lamaria
curl --fail --location \
  https://cvg-data.inf.ethz.ch/lamaria/raw_data/training/R_03_easy.vrs \
  --output data/lamaria/R_03_easy.vrs
```

The original file is **1,135,832,862 bytes**. Verify its SHA-256 before processing:

```python
import hashlib
from pathlib import Path

vrs = Path('data/lamaria/R_03_easy.vrs')
assert vrs.stat().st_size == 1135832862
with vrs.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == (
        '52962a4934d2b94d77ec276d7cf5579065e598c8e8448e2fd9d87e2bbfdb6229'
    )
```

`R_03_easy` has one RGB camera and two synchronized SLAM cameras. The original RGB images are 1408×1408; both SLAM streams are 640×480. The adapter uses native decoded-size calibration and each camera's signed clock offset.

**Trajectory source:** this example uses one genuine single-recording MPS SLAM device trajectory. RGB, SLAM-camera and IMU streams belong to that one VRS; camera poses are composed from the device trajectory and online extrinsics. Multiple sensor streams do not mean Multi-SLAM or multiple aligned recordings. The Pilot walkthrough remains the separate multi-recording map/query example.

## 2. Obtain genuine MPS trajectory and calibration

Use a separate Python 3.12 environment for the public MPS client. The commands below use POSIX shell paths (macOS/Linux). For a managed Project Aria account, follow its normal browser authorization flow; replace the account placeholder, never supply a password in the command:

```bash
python -m venv ../aria-mps-r03
../aria-mps-r03/bin/python -m pip install \
  projectaria-mps==1.4.0 projectaria-tools==2.3.0
../aria-mps-r03/bin/aria_mps single --input data/lamaria/R_03_easy.vrs \
  --features SLAM --username YOUR_PROJECT_ARIA_ACCOUNT --no-ui --no-save-token
```

Run this from the repository root. The separate environment preserves the HLOC pins. The client prints the official verification URL and code for a managed account; authorize it yourself. Other account types should use the [public client's documented login flow](own_recordings.md#3-process-with-mps), not a credential workaround. `--no-save-token` is a headless option and applies to a fresh login, not a previously saved session.

Wait for successful processing and verify the actual outputs. `summary.json` must report SLAM `SUCCESS`, with no errors. A successful client process exit alone does not prove that SLAM completed. The required input layout is:

```text
data/lamaria/
  R_03_easy.vrs
  mps_R_03_easy_vrs/slam/
    summary.json
    closed_loop_trajectory.csv
    online_calibration.jsonl
```

Use this recording's matching **real MPS** files. LaMAria's public lower-rate pseudo-ground-truth text is not this export; interpolating it and repeating factory calibration does not produce genuine online MPS calibration. Do not rename substitutes or copy another sequence's graph identifier.

```bash
python check_inputs.py \
  --vrs data/lamaria/R_03_easy.vrs \
  --mps data/lamaria/mps_R_03_easy_vrs --camera-set all
```

The checker decodes the RGB and both Gen 1 SLAM streams and reports calibration/pose coverage. Mapping frames need online-calibration and MPS pose coverage. Queries need usable calibration and synchronized images, but can localize without an MPS pose reference; those results remain unscored. `--camera-set slam` checks only the two SLAM streams and `--camera-set rgb` only RGB.

## 3. Build the SLAM and RGB maps

Build both maps from the same VRS and MPS into fresh directories; do not overwrite prior evidence. Upstream HLOC model weights download on first inference.

```bash
python build_map.py \
  --vrs data/lamaria/R_03_easy.vrs \
  --mps data/lamaria/mps_R_03_easy_vrs \
  --camera-set slam --output-dir maps/lamaria_R03_slam

python build_map.py \
  --vrs data/lamaria/R_03_easy.vrs \
  --mps data/lamaria/mps_R_03_easy_vrs \
  --camera-set rgb --output-dir maps/lamaria_R03_rgb
```

`--camera-set slam` (the builder default) uses both native Gen 1 tracking cameras, `camera-slam-left` and `camera-slam-right` (all four on Gen 2); `--camera-set rgb` uses `camera-rgb`. Both maps use the same MPS poses and world frame and share its `graph_uid`, but they are different maps: check each map's `map.json` `camera_set` and record the SHA-256 of its `map.json` and `sfm/points3D.bin` with every result.

## 4. Query each map with its camera type

```bash
python query_map.py \
  --map maps/lamaria_R03_slam \
  --vrs data/lamaria/R_03_easy.vrs \
  --mps data/lamaria/mps_R_03_easy_vrs \
  --mode slam --stride 64 --num-queries 50 \
  --output queries/lamaria_R03_slam_map_slam.json

python query_map.py \
  --map maps/lamaria_R03_rgb \
  --vrs data/lamaria/R_03_easy.vrs \
  --mps data/lamaria/mps_R_03_easy_vrs \
  --mode rgb --stride 31 --num-queries 1000 \
  --output queries/lamaria_R03_rgb_map_rgb.json
```

`slam` solves the device pose from the complete native rig with gPnP: both Gen 1 tracking cameras or all four Gen 2 cameras. Incomplete or mixed-generation rigs are rejected, and the original synchronization and pose gates are unchanged. On CUDA the rig's cameras are extracted and matched concurrently (`--rig-execution auto`, the default); `--rig-execution serial` keeps the serial reference path. `rgb` uses one RGB image with PnP. Either mode can also query the other map, because both maps share the MPS frame.

Exact mapping keyframes are skipped by VRS fingerprint plus camera/frame identity; report the actual query count, not the requested limit. Query MPS poses are reference diagnostics only, never retrieval or pose priors. This is still the **same recording and jointly estimated MPS frame**, not independent ground truth or a blind relocalization benchmark.

## Measured runs

The native SLAM-camera map has **2,232 views** (1,116 per camera) and **117,790 points** with 1.348 px mean native reprojection error; the RGB map has **930 views** and **95,253 points** (1.390 px). Both come from the same original VRS and MPS.

| Query | Map | Accepted | ≤10 cm & ≤5° / covered | Translation median / max | Warm query median (serial) |
|---|---|---:|---:|---:|---:|
| Two SLAM cameras | SLAM map | 50 / 50 | 48 / 48 | 1.9 / 6.2 mm | 2.38 s |
| Two SLAM cameras | RGB map | 50 / 50 | 48 / 48 | 4.0 / 22.0 mm | 1.98 s |
| RGB | RGB map | 50 / 50 | 48 / 48 | 2.5 / 9.8 mm | 1.17 s |
| RGB | SLAM map | 49 / 50 | 47 / 48 | 5.0 / 17.6 mm | 1.70 s |

All four map/query cohorts use the same 50 anchors, chosen before any query from 460 eligible of 1,583 RGB captures: a capture is excluded when any of its RGB or SLAM frames lies within 20 ms of any keyframe of either map, in any camera. The stride commands above skip only exact keyframes and do not reproduce this set.

For native two-camera queries, both maps placed all 48 reference-covered queries within 10 cm and 5°; the SLAM map had smaller same-recording residuals, and its warm end-to-end query time was about 20% higher (backend-only medians 1.67 vs 1.65 s). For RGB queries, the RGB map accepted all 50 and was faster; the SLAM map rejected one query for insufficient inliers. Each cohort's two unscored accepted queries are the recording-boundary anchors outside MPS reference coverage. Compare timings only within one query type: in this serial run a two-camera query extracted and matched two 512×512 images one after the other against 20 references each (40 matches), while an RGB query is one 1024×768 image with 20 matches. The cohorts ran one after another on the same GPU; warm end-to-end times include decoding and the harness's keyframe and membership checks, and initialization is separate. These counts describe this one recording; Gen 2 and other RGB devices are untested. See [the matched-anchor protocol, results, timings and map fingerprints](../benchmarks/lamaria_r03_dual_map.json).

Parallel rig execution was measured separately on the same 50 anchors and both maps, with a decode-plus-backend timer that excludes the harness's checks: the two-camera median fell from 1.441 to 1.134 s on the RGB map and from 1.486 to 1.185 s on the SLAM map, with unchanged poses and gates; single-camera RGB queries run serially either way. See [the parallel-query results](../README.md#parallel-rig-queries) and [their benchmark](../benchmarks/lamaria_r03_parallel_query.json).

The earlier RGB-only run is kept as a dated baseline. Its RGB map had **930 keyframes, 95,203 landmarks** and 1.391 px mean native reprojection error. Its stride-31 RGB batch retained **17 queries** and its two-camera batch, run against that RGB map, **50 queries**; all were accepted, the 16 and 49 reference-covered queries were within 10 cm and 5°, and the two unscored accepted fixes remain in the results. See [its recorded protocol, timings and hashes](../benchmarks/lamaria_r03.json).

These are same-recording checks, not physical ground-truth accuracy or evidence that unobserved places will be rejected. Do not call unscored accepted fixes correct based on gate acceptance alone. Timings are end-to-end measurements of this implementation, not a controlled per-image comparison.

## What the demonstration does not claim

- A passed pose gate does not guarantee a correct location, especially outside mapped areas. Retain failed and inaccurate accepted queries.
- Same-recording reference agreement is not cross-recording accuracy.
- `R_03_easy` has no recorded VIO stream. Do not feed its closed-loop MPS trajectory into `replay_demo.py` as VIO; use Pilot for recorded VIO replay.
