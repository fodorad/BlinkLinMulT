---
title: Blink detection demo
emoji: 👁️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
short_description: Eye-state recognition and blink detection on video
python_version: "3.12"
---

# Blink detection demo

Runs the four published [BlinkLinMulT](https://github.com/fodorad/BlinkLinMulT)
models on a video: the largest face is tracked, **both eyes are scored
separately**, and the per-frame eye-state curve is turned into blink events by a
rule you choose.

## The models

| id | what it is | runtime |
|---|---|---|
| `blinkcnn` | current frame-wise model, ConvNeXt-Femto | PyTorch |
| `densenet121-union` | 1.x frame-wise, DenseNet121 | ONNX |
| `blinklint-union` | 1.x sequence, DenseNet121 + LinT | ONNX |
| `blinklinmult-union` | 1.x two-stream, crops + 160-d iris descriptors | ONNX |

The three `*-union` models are from
[*BlinkLinMulT: Transformer-Based Eye Blink Detection*](https://www.mdpi.com/2313-433X/9/10/196)
(J. Imaging, 2023), frozen as ONNX graphs. They were trained on **15-frame
windows**, so a longer clip is scored by sliding that window and averaging the
overlap.

## The pipeline, in seven stages

Each stage completes before the next and reports its own wall-clock time, so a
slow run is attributable rather than an opaque wait.

1. **Face detection and tracking** — the largest face, kept consistent across frames
2. **Head pose** — 6DRepNet, one face per frame
3. **Landmarks and eye localisation** — FaceMesh 478 points, two eye boxes
4. **Eye selection** — left, right, both or neither
5. **Inference** — per-frame eye state, each eye scored independently
6. **Labelling** — the eye-state curve becomes blink events
7. **Visualisation** — the plot and the annotated video

## Two things worth knowing

**An eye turned away from the camera is suppressed, not guessed at.** Past
`|yaw| > 45°` one eye is hidden by the head itself, so its box turns **red**, its
readout reads `--`, and its curve **breaks**. A gap means "not looked at", which
is a different claim from "open".

**Eye state and blink events are separate decisions.** The model gives a
continuous closeness signal; a blink is a rule laid on top of it. That rule is
yours to choose:

* **Fitted hysteresis** — the model's registered operating point. For `blinkcnn`
  that pair was swept on *RN* validation data, so on any other footage it is a
  transfer, not a measurement. The 1.x models carry a plain `0.5`, which was
  never fitted at all — that work reported eye state, not events.
* **Threshold** — a single cut you supply.
* **Custom range** — your own hysteresis pair. A run must peak above the high
  threshold to count, and extends while above the low one, which recovers the
  shallow onset and offset a single cut clips away.

## The example

TalkingFace's first 10 seconds carry three annotated blinks (frames 168, 225,
274). Its ground truth is plotted in **green** beside the prediction, so the
model can be read against what was annotated rather than judged by eye.

## Limitations

* Analyses a **segment** — 10 seconds by default, adjustable, capped at 10.
* Needs a visible face; a clip without one stops at stage 1 and says so.
* Trained on public blink corpora skewed toward frontal, indoor webcam footage.
  Expect degradation on strong profile views, heavy occlusion, or infrared.

## Citation

```bibtex
@article{fodor2023blinklinmult,
  title   = {BlinkLinMulT: Transformer-Based Eye Blink Detection},
  author  = {Fodor, {\'A}d{\'a}m and Fenech, Kristian and L{\H{o}}rincz, Andr{\'a}s},
  journal = {Journal of Imaging},
  volume  = {9},
  number  = {10},
  pages   = {196},
  year    = {2023},
  doi     = {10.3390/jimaging9100196}
}
```
