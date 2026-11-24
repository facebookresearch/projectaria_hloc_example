# HLOC presentation media

## LaMAria indoor example

`lamaria-dual-map-comparison.gif` shows the two HLOC maps built from `R_03_easy`
side by side from one shared virtual camera: the native SLAM-camera map (both Gen 1
tracking cameras; 2,232 views, 117,790 points; blue) and the RGB map (930 views,
95,253 points; orange). Both maps use the recording's original MPS poses and world
frame; each map's points are triangulated only from its own images, and no MPS
point is copied. Every point of both maps is drawn in every frame at 1 CSS pixel,
with no cutaway or filtering. Blue and orange are display categories only:
`lamaria-dual-map-native-colors.png` shows the native point colors (gray for the
SLAM images), and `lamaria-dual-map-overlay.png` draws both maps together in one
view. Point count, density or color is not localization accuracy. This is
single-recording MPS SLAM, using the recording's native sensor streams; it is not
Multi-SLAM across independent recordings.

The GIF is 960×501, 60 frames at 200 ms (12 s per loop). The same 60-frame orbit
supplies `lamaria-dual-map-comparison.webm` (1192×622, VP9, 5 fps), and
`lamaria-dual-map-comparison.png` is an exact still capture. The orbit is a virtual
camera, not recorded device motion; GIF colors are quantized approximations.
[Source maps, point payload hashes and per-frame camera checks](lamaria-dual-map-provenance.json)
record the derivation; the measured map/query results are in
[the dual-map benchmark](../benchmarks/lamaria_r03_dual_map.json).

`lamaria-dual-map-results.svg` plots warm end-to-end query times on the same 50
anchors in two separate panels, RGB queries (one 1024×768 image, 20 matches) and
two-SLAM-camera queries (two 512×512 images, 40 matches), each on the SLAM map and
on the RGB map, with accepted and reference-covered counts. Compare maps only within
one panel: the panels are different workloads, and the times are this
implementation's totals, not a per-image or parallel comparison.

### Dated RGB-only baseline media

The files below come from the earlier RGB-only run, whose RGB map had 930 keyframes
and 95,203 landmarks; they are not the dual-map run above.

`lamaria-localization.gif` and `.png` show the actual 17 fixed RGB query anchors
in the indoor `R_03_easy` HLOC map. All queries are retained, including the first
accepted result without MPS reference coverage. Gold markers are sparse HLOC
solves; cyan is the current HLOC fix/camera. There is no pose interpolation,
connecting trajectory, VIO or continuous-tracking claim. Each GIF frame is held
for 900 ms (15.3 s per loop); this is presentation pace, not inference speed.

RGB is a 90° clockwise display derivative of the exact PAT-rectified query image.
Original decoded pixels, estimator pixels, poses and camera calibration remain
unchanged. The map viewport uses all mapping-camera centres plus a 2 m margin;
all 95,203 native landmarks remain in the source map. Marker glyph size and
visibility are presentation choices, not depth or observed geometry.
[Source fingerprints, exact anchors and media hashes](lamaria-media-provenance.json)
record the derivation; source RRD verification is not a browser-viewer test.

`lamaria-indoor.jpg` shows the actual `R_03_easy` RGB HLOC map (930 keyframes,
95,203 native COLMAP landmarks) beside original decoded RGB frames 395, 789 and
1186. The map is framed at the 0.2/99.8-percentile bounds for presentation; no
map points, estimator inputs or query outcomes were changed. The original RGB
pixels retain their camera orientation.

`lamaria-workflow.svg` is an original explanatory diagram, not sensor data or an
experimental result. The raw indoor VRS is processed through public MPS; selected
RGB keyframes build the map and other frames query it. All queries come from the
same recording. There is no recorded VIO in `R_03_easy`.

[LaMAria source](https://lamaria.ethz.ch) — Anusha Krishnan et al.,
*Benchmarking Egocentric Visual-Inertial SLAM at City Scale*, ICCV 2025.
Dataset, documentation and derived media attribution:
[CC BY 4.0](https://github.com/cvg/lamaria/blob/main/LICENSE-CC-BY).
[Dual-map results](../benchmarks/lamaria_r03_dual_map.json) and the
[dated RGB-only results](../benchmarks/lamaria_r03.json) retain counts, coverage,
source fingerprints and hashes; no account details or expiring URLs.

## Preserved Pilot media

The earlier Gen 2 Pilot files remain available for the separate-recording and
VIO walkthrough; their original provenance and qualification are unchanged:

## Own-recording workflow

`own-recording-workflow.svg` is an original explanatory diagram of public Aria
Gen 2 capture, USB VRS export, MPS processing and HLOC localization. Mapping and
independent query recordings follow separate paths. It contains no captured
sensor data and is not a hardware test, MPS request or benchmark result.
See [the own-recording guide](../docs/own_recordings.md) for commands and sources.

## Pilot point-cloud comparison

`pilot-pointcloud-comparison.jpg` / `.gif` show the joint-MPS living component
(`clean_0`, `play_0`…`play_3`) from one shared perspective camera, with the same
roof cutaway (`z > 0.65 m`) for the **MPS Semidense Point Cloud** and the
**HLOC Point Cloud**.

- MPS Semidense Point Cloud: official dual-confidence display mask, 3,871,690 / 23,758,300 records.
- HLOC Point Cloud: stable landmarks (>=3 distinct views, native mean reprojection<=2px)
  plus a display-only density mask requiring eight other stable neighbours
  within 15cm: 127,842 / 303,741 landmarks. Counts are before view cutaway.

Requested point size is 0.5 CSS pixels, MPS opacity 0.18 and HLOC opacity 0.65.
Eight real WebGL views are presented forward and backward in the GIF (14 frames,
600 ms each, 8.4 s); no synthetic geometry is interpolated. Both files are crops of
the original captured point-cloud viewports with new labels; full native input
buffers are unchanged. The separate `pilot-pointcloud-comparison-provenance.json`
records the sources, filters, camera, crops and captured-view hashes.

This full-reference asset preview is NOT the map behind the held-out
`play_0 + play_2 -> play_1` accuracy/latency results below. The previews
are not bundled runtime maps or a public dataset release.

## Public two-input example media

- `pilot-relocalization.gif`: an 18-second overview of the full 330.4-second
  mapped-pose interval, played at 18.4x speed. The 72 display frames use actual
  recorded `play_1` RGB and mapped device poses from the measured-delay replay
  against the `play_0` + `play_2` map. Presentation speed is not inference speed.
- `pilot-relocalization.webm`: the same overview at 1280x760, VP9, 8 display fps,
  limited-range YUV. It is a rendered preview, not a GUI or live-device capture.
- `pilot-sequences.jpg`: two actual calibrated RGB frames from each mapping
  recording and two from the query recording. These illustrate data appearance,
  not accuracy-selected localization successes.
- `pilot-workflow.svg`: the mapping, localization and capture-time VIO alignment
  workflow. It is an explanatory diagram, not an experimental result.
- `pilot-results.svg`: pose-agreement and latency curves recomputed from all
  100 RGB / 50 four-camera queries. Accuracy uses all MPS-covered queries,
  including failed solves, with a 5-degree rotation threshold. The RGB outcome
  panel retains both failures and large accepted errors. The footer states the
  workloads (rig: four 512×512 images, up to 80 pairs, processed serially; RGB:
  one 1024×768 image, up to 20 pairs). The latency curves are measured
  backend/query durations of this serial implementation, not a per-image or
  parallelized comparison; decode/rectification-inclusive end-to-end durations
  are reported separately in the benchmark.
- `pilot-media-provenance.json`: source hashes, rendering parameters and fixed
  sampling information. No raw VRS, private paths, credentials or model weights.

Inputs are the [public Aria Gen 2 Pilot dataset](https://www.projectaria.com/datasets/gen2pilot/)
using its original published MPS outputs. Mapping uses `play_0` + `play_2`;
query images from `play_1` never enter the map. Dataset terms apply separately
from this repository's code license. Shared-SLAM MPS reference comparisons are
not independent physical ground truth, and the anchors were previously exposed.

The actual map contains 135,213 native landmarks and the replay emits 265,344
mapped poses. Visualization is viewport-framed and the displayed pose trail is
sampled from the full immutable output; no map, query, pose or reference was
changed for the presentation. The renderer performs no neural inference and is
not a runtime dependency. The measured results remain in
`../benchmarks/pilot_multitrajectory.json`.
