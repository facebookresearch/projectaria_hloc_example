# Pilot multi-recording presentation media

The release README uses these small, repository-local media files:

## Own-recording workflow

`own-recording-workflow.svg` is an original explanatory diagram of public Aria
Gen 2 capture, USB VRS export, MPS processing and HLOC localization. Mapping and
independent query recordings follow separate paths. It contains no captured
sensor data and is not a hardware test, MPS request or benchmark result.
See [the own-recording guide](../docs/own_recordings.md) for commands and sources.

## Same-camera asset comparison

`pilot-room-comparison.jpg` / `.gif` show the new joint-MPS living component
(`clean_0`, `play_0`…`play_3`) using one shared perspective camera and the same
roof cutaway (`z > 0.65 m`) across MPS, HLOC and Twin layers.

- MPS: official dual-confidence display mask, 3,871,690 / 23,758,300 records.
- HLOC: stable landmarks (>=3 distinct views, native mean reprojection<=2px)
  plus a display-only density mask requiring eight other stable neighbours
  within 15cm: 127,842 / 303,741 landmarks. Counts are before view cutaway.
- Twin: the existing observed `living_v1` model, rigidly aligned to the new
  frame, not a reconstruction from all new inputs or a localization truth.

Requested point size is 0.5 CSS pixels, MPS opacity 0.18 and HLOC opacity 0.65.
Eight real WebGL views are presented forward and backward in the GIF; no
synthetic geometry is interpolated. Full native input buffers are unchanged.
The separate `pilot-room-comparison-provenance.json` records the sources,
filters, camera and captured-view hashes.

This full-reference asset preview is NOT the map behind the held-out
`play_0 + play_2 -> play_1` accuracy/latency results below. The seven new
canonical Twins are a separate ongoing reconstruction task. The previews
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
  panel retains both failures and large accepted errors.
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
