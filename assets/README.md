# Historical single-recording smoke assets (not the default Pilot test)

These previews were made from the old single-recording kitchen sample. They are
retained for setup provenance only: they are not independent mapping/testing
validation or Pilot media, and no longer headline the release README/notebook.


`sample_replay.png`, `.gif` and `.webm` are rendered presentation previews of
actual public-sample outputs, not Rerun GUI screenshots or live-glasses captures.

Source: official `aria_gen2_sample_data_1.vrs` / matching December 2025 MPS output,
with the SHA-256 fingerprints in `../benchmarks/public_sample.json`. Mapping used
the RGB reference map (66 keyframes, 9,494 sparse landmarks). Recorded RGB and
high-frequency VIO were fed at their original time spacing through the actual
asynchronous `LiveLocalizer`, accepting two visual fixes and publishing 6,192
map-frame poses. No cached solves or MPS pose interpolation were used in that
worker run. The visual shows the final ~8 seconds after initial localization.

The sparse point cloud, pose trail and device axes come from that map and JSONL;
RGB is the nearest corrected-timestamp recorded exposure. A fixed oblique view
and diagnostic labels make a portable preview. The video is VP9/WebM, limited
range, 30 fps; the lightweight README GIF is reduced resolution/frame rate.
The preview renderer is not a runtime dependency or another example framework.

Use the README replay command to generate an interactive Rerun `.rrd` locally.
A rendered preview does not establish interactive GUI or physical USB streaming
qualification. No generated room geometry, agent reconstruction, digital twin,
private home recordings or fabricated localization outputs are used.
