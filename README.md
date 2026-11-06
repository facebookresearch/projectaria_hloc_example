# Aria Gen 2 HLOC Reference Example

Build a metric HLOC map from Aria Gen 2 VRS/MPS data, localize RGB or four-camera
SLAM queries, and propagate visual corrections at the incoming VIO rate in the
map frame. Rerun displays the map, mapped pose, RGB and tracking diagnostics.

**The default example maps Pilot `play_0` + `play_2` and tests the independent
third recording, `play_1`.** The testing recording never enters the map, features
or retrieval index. Excluding keyframes from a mapping recording is not
independent recording validation.

## Scope and Inputs

- `build_map.py`: rectify keyframes and triangulate public HLOC features/matches
  using existing MPS camera poses.
- `query_map.py`: cold NetVLAD retrieval, SuperPoint/LightGlue matching and
  central PnP or joint four-camera gPnP through COLMAP.
- `replay_demo.py`: propagate recorded VIO using cached visual fixes and measured
  query delays; output JSONL and Rerun recording.
- `live_demo.py`: an asynchronous RGB worker with capture-time VIO alignment,
  map-frame pose publication and optional Rerun viewer.
- `hloc_example.ipynb`: the same independent Pilot mapping/testing walkthrough.

This is a reference example, not a core Aria Research Kit component. No hosted
service, internal localization algorithm, device Location write, agent,
digital-twin reconstruction or other-device port is included.

Python 3.12 and local disk space are required. Offline Pilot tests need no glasses
or new MPS processing request. Each recording needs `video.vrs` and MPS
`closed_loop_trajectory.csv` / `online_calibration.jsonl`. Use actual nonempty
`graph_uid` values, not scene names, to establish a shared spatial frame.

## Quick Start: Pilot Mapping / Testing

### 1. Install the pinned environment

After the repository is published:

```bash
git clone --recurse-submodules https://github.com/facebookresearch/projectaria_hloc_example.git
cd projectaria_hloc_example
conda env create -f environment.yml
conda activate aria_hloc
python -m pip install --no-deps -e ./Hierarchical-Localization
git -C Hierarchical-Localization apply ../patches/hloc-workers.patch
```

HLOC is fixed at `c13273bd0ecc2917a35910fd843712a1c6243193`, LightGlue at
`edb2b838efb2ecfe3f88097c5fad9887d95aedad`, and pycolmap at `4.2.1`.
The HLOC commit and COLMAP version are checked at runtime. Recursive cloning
initializes nested submodules; `--no-deps` preserves `environment.yml` pins.
Upstream libraries download model weights on first use; weights are not bundled.

Apply the worker-option patch once to the exact checkout. It preserves upstream
algorithm defaults while allowing this wrapper to use zero DataLoader workers
on macOS and two on Linux. The macOS setting avoids a reproduced resource-tracker
shutdown hang. The separate Linux container benchmark used zero workers because
its shared-memory filesystem could not support two.

On macOS, if separate torch/pycolmap OpenMP copies cause `OMP: Error #15` or
`#179`, unify them **inside the activated example environment only**:

```python
from pathlib import Path
import sysconfig

site = Path(sysconfig.get_paths()["purelib"])
omp = site / "pycolmap/.dylibs/libomp.dylib"
if not omp.is_symlink():
    omp.rename(omp.with_name("libomp.dylib.original"))
    omp.symlink_to("../../torch/lib/libomp.dylib")
```

Do not use `KMP_DUPLICATE_LIB_OK` or force process exit to hide failures.

### 2. Download all three Pilot recordings

Open the [official Pilot Explorer](https://explorer.projectaria.com/gen2pilot),
follow its access/consent flow, and obtain the URL-list JSON for **`play_0`,
`play_2` and `play_1`**. Use the original publicly downloadable VRS/MPS outputs;
no new joint MPS request or private dataset is needed. Do not commit generated
CDN lists or guess asset URLs. Download only VRS, SLAM trajectories and calibration:

```bash
aria_dataset_downloader -c /path/to/play_0_download_urls.json \
  -o data/pilot -l play_0 -d 0 1 2
aria_dataset_downloader -c /path/to/play_2_download_urls.json \
  -o data/pilot -l play_2 -d 0 1 2
aria_dataset_downloader -c /path/to/play_1_download_urls.json \
  -o data/pilot -l play_1 -d 0 1 2
python check_inputs.py --vrs data/pilot/play_0/video.vrs --mps data/pilot/play_0/mps
python check_inputs.py --vrs data/pilot/play_2/video.vrs --mps data/pilot/play_2/mps
python check_inputs.py --vrs data/pilot/play_1/video.vrs --mps data/pilot/play_1/mps
```

In the pinned downloader, `0`, `1`, `2` mean `main_vrs`,
`mps_slam_trajectories`, `mps_slam_calibration`. Budget for all three downloads,
extracted inputs, images, features and map outputs. No point cloud, depth, ASR or
full Pilot loader dependency is needed. Pilot has its own access and distribution
terms; this example does not rehost it or accept those terms on your behalf.

```text
data/pilot/
  play_0/video.vrs
  play_0/mps/slam/{closed_loop_trajectory.csv,online_calibration.jsonl}
  play_2/video.vrs
  play_2/mps/slam/{closed_loop_trajectory.csv,online_calibration.jsonl}
  play_1/video.vrs
  play_1/mps/slam/{closed_loop_trajectory.csv,online_calibration.jsonl}
```

Before building, check every listed file for all three recordings.
`check_inputs.py` decodes/rectifies all five camera streams and checks MPS pose
coverage without HLOC inference. Its no-argument default is still `play_0`;
check `play_2` and the query recording explicitly. All three public trajectories
must identify the same nonempty `graph_uid`; a scene name is not a frame ID.
The notebook verifies that graph and the three content fingerprints before building.

### 3. Map `play_0` + `play_2` only

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python build_map.py \
  --vrs data/pilot/play_0/video.vrs data/pilot/play_2/video.vrs \
  --mps data/pilot/play_0/mps data/pilot/play_2/mps \
  --camera-set rgb --output-dir maps/pilot_play02_rgb
```

VRS and MPS lists are parallel and ordered: namespace `s000` is `play_0`,
`s001` is `play_2`. Each keeps its own clock and calibration. Mapping inputs must
have distinct full-VRS SHA-256 fingerprints; neither may match `play_1`.
The builder rejects disconnected/empty MPS graphs before inference. This path
uses the original public shared MPS frame, not separately generated reference assets.

The minimal path is **RGB-built**; `--camera-set slam` or `all` are supported
builder variants, not the configuration of this walkthrough. The CLI default
is `slam`, so pass `rgb` explicitly. Leave `--max-keyframes 0` (uncapped);
a keyframe cap produces a different map.

Use a new output directory. The builder writes images, known-pose `reference/`,
COLMAP `sfm/`, features/retrieval/matches HDF5, pairs, statistics and `map.json`.
A failed build may leave partial files but does not write a completed manifest.
If this exact two-input map already exists, skip building and run the notebook's
fingerprint validation: `source_vrs_sha256` must be exactly the two hashes in
`play_0`, `play_2` order, not merely the same unordered set. Do not automatically
rebuild or substitute a map with different inputs.

### 4. Query the independent third recording, `play_1`

```bash
python query_map.py --map maps/pilot_play02_rgb --vrs data/pilot/play_1/video.vrs \
  --mps data/pilot/play_1/mps --mode rgb --num-queries 100 --stride 34 \
  --output queries/pilot_play1_multi_rgb.json
```

Do **not** pass `play_1` to the builder or append its images/features to the
retrieval database. Query MPS supplies its own calibration and optional same-graph
reference comparisons, not retrieval or pose priors. Check that the query output
reports `source_identity_check: "sha256"` and `query_in_mapping_recordings: false`,
and its `query_vrs_sha256` is absent from the map's two ordered source hashes.

A four-camera gPnP query also works against this RGB-built map:

```bash
python query_map.py --map maps/pilot_play02_rgb --vrs data/pilot/play_1/video.vrs \
  --mps data/pilot/play_1/mps --mode slam --num-queries 50 --stride 208 \
  --output queries/pilot_play1_multi_slam.json
```

`--stride` deterministically samples every Nth anchor and never repeats frames.
The RGB command matches the measured 100-query indices for this two-recording
map. The measured 50-query SLAM supplement used endpoint-inclusive uniformly
spaced indices; stride 208 approximates that
coverage, not the exact anchors. Report this command's actual counts/results
rather than assuming it reproduces the measured SLAM score or batch timing.

Recording identity uses full VRS SHA-256, not producer paths, so copying a VRS
does not defeat keyframe exclusion. New maps store `source_vrs_sha256` parallel
to namespaces. Legacy maps without it still load, but warn and report unknown
membership (`null`); they cannot prove independence from paths alone. Even a
non-keyframe from a mapping recording is still a same-recording test.

### Measured two-recording map / third-recording queries

The uncapped `play_0` + `play_2` RGB map has **1,360 keyframes** (666 + 694),
**135,213 landmarks** and **14,870 cross-recording tracks**. Source fingerprints,
namespace/calibration membership, native tracks and unchanged map files were
checked. All 150 fixed `play_1` queries completed; no failed anchors were dropped.
[Measurements and exact protocol](benchmarks/pilot_multitrajectory.json) records
map/source hashes, sampling, denominators, failures and timings.

| Query cameras | Gate passes / all queries | MPS reference-covered queries | ≤0.10 m AND ≤5° / reference-covered | ≤0.25 m AND ≤5° / reference-covered |
|---|---:|---:|---:|---:|
| RGB | 98 / 100 | 99 | 89 / 99 | 93 / 99 |
| Four SLAM | 48 / 50 | 48 | 47 / 48 | 47 / 48 |

Accuracy denominators include reference-covered localization failures. Among
accepted poses, 97 RGB and 47 SLAM queries had a reference. Their separate
maximum translation/rotation disagreements were **0.418 m / 13.23° RGB** and
**0.075 m / 2.29° SLAM**. No accepted reference-covered pose exceeded 1 m or
170° in this run; that is not a false-positive guarantee. Gate acceptance is
not accuracy or safety assurance.

RGB used exactly `0, 34, ..., 3366`, matching the primary CLI sampling above.
The measured SLAM supplement used 50 endpoint-inclusive uniformly spaced
indices, with gaps of 208/209 and last index 10200. The optional stride-208 CLI
approximates that coverage, not those exact anchors. These previously exposed
anchors were frozen before this build/run, not selected from a new blind test.
Original public MPS supplies the shared frame and reference poses; a joint solve
may include the query recording, so this is not independent physical ground truth.

Observed CPU query median/p90 was **6.10 / 7.02 s RGB** and **12.85 / 13.89 s
SLAM**. Map/model preparation (**8.66 s**) and recording/trajectory opening
(**18.80 s**) are separate. First RGB query was 8.37 s; excluding it gives
6.06 / 6.96 s median/p90. One prepared backend served RGB then SLAM; separate
CLI invocations each initialize, so this batch does not reproduce their cold
startup costs. Eight-thread CPU work ran alongside other jobs; these are not a
controlled speed-gain comparison, GPU or real-time guarantee. Peak RSS was about
4.09 GiB; the 256 MiB reference-feature bound is not a whole-process memory limit.

The recorded replay passed separate CLI, JSONL parity and RRD checks (see below).
The notebook remains unexecuted; these results do not establish physical USB
streaming, complete release-asset readiness or public-release approval.

## Resident Relocalization Runtime

`query_map.py` prepares the map and inference models once before its query loop.
The live worker keeps the same resident runtime across frames and VIO session
resets; a VIO reset still clears alignment and rejects obsolete visual fixes.
Map/model initialization is separate from query latency.

The query path passes decoded image arrays and predicted features in memory;
it does not stage query PNGs, HDF5 feature files or match files. The immutable
COLMAP reconstruction, normalized NetVLAD reference descriptors and feature-to-
landmark index are reused. Reference local features use a 256 MiB byte-bounded CPU LRU cache,
not an unbounded GPU cache. This bounds retained feature-array payloads only,
not reconstruction/global descriptors, model weights, query tensors or total RAM.
Changing maps requires an explicit runtime transition;
do not replace map files while a localizer is using them. Runtime resources close
on normal exit and errors.

Query JSON reports `initialization_durations_sec` once, per-query
`stage_durations_sec`, `query_duration_sec`, and
`query_end_to_end_duration_sec`. End-to-end recorded-query duration includes
frame decoding/rectification and keyframe checks, but excludes initialization,
recording hashing, reference-error evaluation and live queue/bracket waiting.
Cold initialization, the first query and subsequent warm queries are distinct;
report their actual timings separately. The multi-recording batch measurements
above are observational CPU results, not a controlled speedup comparison.

This change keeps feature/model snapshots, retrieval count, map geometry and
pose acceptance thresholds fixed. It does not filter weak landmarks, retune
RANSAC, introduce a VIO retrieval prior or eliminate accepted false poses.

## Replay and Visualization: Testing Recording

After the RGB query command, use **`play_1` VIO and its query cache**:

```bash
python replay_demo.py --map maps/pilot_play02_rgb --vrs data/pilot/play_1/video.vrs \
  --mps data/pilot/play_1/mps --fixes queries/pilot_play1_multi_rgb.json \
  --output-dir outputs/pilot_play1_multi_replay
rerun --bind 127.0.0.1 outputs/pilot_play1_multi_replay/view.rrd
```

Rerun lets you orbit the sparse map, scrub the timeline and inspect poses, RGB
and fix age. Outputs are `poses.jsonl`, `view.rrd`, `replay_stats.json`. JSONL
retains every valid VIO sample after alignment; GUI pose logging is capped at
30 Hz and unchanged status logs are coalesced. Cache validation checks both map
frame and the testing VRS fingerprint. The notebook uses the same separate inputs.

Replay uses recorded high-frequency VIO (regular only if absent) and measured
cached query delays. The new `play_1` native replay CLI and RRD verification
completed successfully; a row-wise JSONL/PoseAlignment parity audit passed.
It consumed **271,964 VIO samples** and wrote **265,344 mapped JSONL poses**;
the initial **6,620** samples were unaligned/lost, not uninterrupted tracking.

All **100 RGB cache entries were retained**, but the latest-pending-frame
scheduler produced **56 scheduled arrivals within VIO coverage and 55 accepted fixes**.
The remaining entries include 42 superseded captures, one capture outside VIO
coverage and one result that arrived after VIO ended. Failed anchor 37 was scheduled; failed anchor 75 was
superseded by scheduling, not removed by selecting successful queries.

First mapped output occurred **8.242 s after the first recorded VIO sample**,
not after process start or cold initialization. First-fix capture-to-arrival
was separately **6.205 s**. The **803.03 Hz** mapped-pose rate is measured in
recorded device time, not visual-fix rate, wall-clock throughput or live USB
performance; GUI pose logging remains capped at 30 Hz.

RRD displays all sparse-map positions with uniform colors, not original point
RGB. The RGB panel is a 640-pixel/JPEG-90 preview; a separate full original-color
point export is not the RRD and is not bundled here. These display limits do not
change native map geometry or full-rate JSONL output. Replay does **not** rerun
models, interpolate MPS reference poses into the output or validate physical USB
streaming. Localization outliers can corrupt alignment; this is an example, not
a safety-critical tracker. See [recorded replay evidence](benchmarks/pilot_multitrajectory.json).

## Live Map-Frame VIO

For physical streaming, build a map of the environment you will use. The Pilot
recorded-test workflow does not require glasses; its apartment map will not
localize an unrelated room. After pairing/configuring USB-NCM with the public SDK:

```bash
python live_demo.py --map maps/MY_ENVIRONMENT_RGB --profile profile9 \
  --device-ip DEVICE_USB_IP --bind-address HOST_USB_NCM_IP \
  --output-dir outputs/live
```

High-frequency VIO is the sole default source. `--vio-stream regular` is an
explicit lower-rate alternative, never a merge of independently queued streams.
One RGB query runs with one latest pending frame; heavy HLOC work and output
sinks run off callback threads. Map/model initialization happens once in the
worker and is reported separately. Each query reports localization duration and
worker end-to-end duration (rectification plus inference, excluding queue/bracket
waiting); VIO callbacks remain independent of those stages. A full pose-output
queue is fatal, not silent loss.
`--viewer none` records without a window. Ctrl+C stops and closes outputs.

At corrected capture time, interpolate valid VIO brackets without extrapolation:

```text
T_map_odometry = T_map_device_visual_fix @ inverse(T_odometry_device_at_capture)
T_map_device(t) = T_map_odometry @ T_odometry_device(t)
```

Regular VIO composes `T_odometry_bodyimu @ T_bodyimu_device`; high-frequency VIO
already supplies device poses. JSONL records original device time, session,
`T_map_device`, fix time, `fix_age_ns` and tracking status. New odometry sessions
clear alignment and reject old in-flight fixes. Bad VIO is not published; fixes
older than 120 s are stale. Three seconds without host arrivals marks lost.
Corrections may jump: this is alignment/propagation, not a VIO estimator/smoother.
Physical USB streaming and native desktop qualification remain unverified;
existing recorded-data Rerun web GUI was inspected, not live-glasses validation.

## Coordinate and Map Contracts

`T_A_B` maps points from B into A. Mapping uses
`T_world_camera = T_world_device @ T_device_camera`; returned `T_world_device`
is a metric pose in the map's MPS frame, not GPS.

PAT array-pixel principal points and query keypoints each receive one half-pixel
conversion at the COLMAP boundary. The generalized solver takes camera-from-device
extrinsics and its returned device-from-world pose is inverted. Frame sets span
at most 2 ms in corrected exposure time:
`capture_timestamp_ns - round(time_offset_sec_device_camera * 1e9)`.
Raw time/offset are retained; MPS coverage and 2 ms pose association are bounded,
online calibration must be within 1 s, and no extra upright rotation is applied.
No VIO motion compensation is performed within the short recorded rig frame set.

A pose needs >=20 distinct inlier landmarks and both observation/landmark ratios
>=0.2. Conflicting duplicate assignments are discarded. Calibration comes from
the query recording; MPS diagnostics require the same graph frame. These checks
do not eliminate the measured false poses.

The builder accepts one or more mapping recordings. For multiple recordings,
pass aligned MPS paths in VRS order, or a Multi-SLAM root with
`vrs_to_multi_slam.json`. Mixed/empty `graph_uid` values are rejected.
Each recording retains its clock, calibration and namespace. Never include the
testing VRS in an evaluation map. Same scene name does not establish shared frame.

## Map Assets

Raw recordings, generated maps and weights are not bundled or hosted by this
repo. Full-reference maps containing the query recording cannot establish
independent validation. Maps rebuilt under different MPS poses need matching
geometry and frame metadata; display assets are not localization ground truth
or a reconstruction dependency.

## Unit Tests and Support

```bash
python -m pytest tests
```

Unit tests use synthetic geometry and mock SDK/PAT/Rerun at call sites; they do
not need recordings, glasses, model downloads or online MPS. Native tests use the
pinned public COLMAP API. These are distinct from the Pilot dataset evaluation.

Report setup/example issues here; algorithm-specific reports may go to
[HLOC](https://github.com/cvg/Hierarchical-Localization/issues),
[LightGlue](https://github.com/cvg/LightGlue/issues), or
[Project Aria Tools](https://github.com/facebookresearch/projectaria_tools/issues).
Maintenance covers the released snapshot, not continuous upstream upgrades or
internal pipeline parity. Major extensions require separate work.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [LICENSE](LICENSE). Meta-authored code
is Apache 2.0; downloaded data, dependencies and weights retain their source terms.
