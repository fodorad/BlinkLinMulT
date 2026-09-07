<div align="center">

<img src="https://raw.githubusercontent.com/fodorad/BlinkLinMulT/main/docs/assets/logo.svg" alt="BlinkLinMulT" width="360"/>

<br/>

**Transformer-based Eye Blink Detection and Eye State Recognition**

[![GitHub Release](https://img.shields.io/github/v/release/fodorad/BlinkLinMulT?color=purple)](https://github.com/fodorad/BlinkLinMulT/releases)
[![PyPI](https://img.shields.io/pypi/v/blinklinmult?color=purple)](https://pypi.org/project/blinklinmult/)
[![CI](https://github.com/fodorad/BlinkLinMulT/workflows/CI/badge.svg)](https://github.com/fodorad/BlinkLinMulT/actions)
[![Coverage](https://codecov.io/gh/fodorad/BlinkLinMulT/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/BlinkLinMulT)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=githubpages)](https://fodorad.github.io/BlinkLinMulT/)
<br/>
[![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX%20-supported-6b6b6b?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

BlinkLinMulT builds on [LinMulT](https://github.com/fodorad/linmult). It is trained across seven public benchmark corpora for two tasks that the literature usually separates: **eye state recognition** (is this eye closed?) and **blink presence detection** (does this window contain a blink?).

* paper: **BlinkLinMulT: Transformer-Based Eye Blink Detection** ([pdf](https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf), [website](https://www.mdpi.com/2313-433X/9/10/196))
* documentation: **[fodorad.github.io/BlinkLinMulT](https://fodorad.github.io/BlinkLinMulT/)** — guides and the full API reference

**Four models ship from this repository** — the three from the 2023 paper, frozen as ONNX graphs, plus `blinkcnn`, a new frame-wise model. All four are reachable through one interface, compared under one protocol, benchmarked for latency, and runnable live from a webcam.

The network predicts **one thing**: a continuous eye-state score in `[0, 1]` per frame, per eye. Everything else is derived from that signal.

```
eye crops  →  ESR score (T,)  →  event extraction  →  blink events
                                                   →  blink parameters
                                                   →  blink patterns
```

This matters because the corpora do not agree on what they annotate. Some label per-frame closure, others label blink *intervals*. A model trained on one and scored against the other measures a convention gap rather than its own quality: a **perfect** closure detector scores 0.033 at `iou50` on TalkingFace, because a closure covers only 19–34% of the event it belongs to. Predicting the score and deriving events puts every corpus on one footing.

# What's in this repository

- **Public inference API** — `pip install blinklinmult`, then `BlinkDetector.from_pretrained("blinkcnn")`. Weights download from the Hugging Face Hub on first use; one class serves all four models, ONNX and PyTorch alike.
- **Statistical model comparison** — not a table of point estimates. A recording-clustered paired bootstrap with confidence intervals and Holm-corrected p-values across all six model pairs, because "model A scored higher" is not the same claim as "model A is better". See [Model comparison](#model-comparison).
- **ONNX optimization** — the v2 model exported behind a numerical-parity gate, **4.9× faster** (20.19 → 4.15 ms/frame) with accuracy identical to 4.8e-07. See [ONNX optimization](#onnx-optimization).
- **A real-time pipeline** — face detection, eye localisation, head pose and blink scoring at **42.8 fps on CPU**, reached by profiling rather than guessing. See [Real-time pipeline](#real-time-pipeline).
- **Runnable demos** — a live webcam window with per-eye plots, an offline video runner, a Gradio app, a showcase notebook, and a Docker REST service. See [Demos](#demos).
- **Training** — three architectures behind one config key, MLflow-tracked, over a unified masked stream so a corpus that lacks a target contributes an all-`False` mask instead of being dropped.

# Setup

## Install from PyPI for inference

```bash
pip install blinklinmult
```

```python
from blinklinmult import BlinkDetector

detector = BlinkDetector.from_pretrained("blinkcnn")
scores = detector.score(crops)  # (T, 3, 64, 64) in [0,1] -> (T,) closure
events = detector.detect(crops)  # -> [(start, end), ...] blink intervals
```

The core install is **numpy and pyyaml only**. Add the runtime you want:

```bash
pip install blinklinmult[onnx]     # ~100 MB: onnxruntime, no torch
pip install blinklinmult[torch]    # to run or fine-tune the blinkcnn checkpoint
pip install blinklinmult[train]    # the full training stack
```

`blinklinmult[onnx]` runs all four models — the three 1.x graphs and
`blinkcnn-onnx` — **without importing torch at all**, which is 469 MB the
training stack costs and an inference deployment does not need. A test in
`tests/test_packaging.py` fails if that regresses.

## Install for development

```bash
git clone https://github.com/fodorad/BlinkLinMulT && cd BlinkLinMulT
make dev                  # editable install with every extra
make check                # ruff + ty + 1767 tests + docs   (~40 s)
make check-full           # the above, plus the real Lightning training runs
```

`make check` is the gate CI runs. It skips the ~49 s of end-to-end Lightning
runs so a pre-commit check stays fast; `make check-full` includes them.

# Pretrained models

| id | what it is | runtime | window |
|---|---|---|---|
| `blinkcnn` | v2 frame-wise eye state, ConvNeXt-Femto | PyTorch | per-frame |
| `blinkcnn-onnx` | the same weights, frozen | ONNX | per-frame |
| `densenet121-union` | 1.x frame-wise baseline | ONNX | per-frame |
| `blinklint-union` | 1.x sequence model, DenseNet121 + LinT | ONNX | 15 frames |
| `blinklinmult-union` | 1.x two-stream, crops + 160-d iris descriptors | ONNX | 15 frames |

Weights resolve from [`fodorad/blink_detection`](https://huggingface.co/fodorad/blink_detection) into `~/.cache/torch/hub/checkpoints/blinklinmult/` on first use. `blinkcnn` ships as both a checkpoint (trainable, fine-tunable) and a graph (deployable); the rest are inference-only.

# Model comparison

Average precision on the RN30 test split — **35 held-out recordings**, 54 814 valid frames, 1.04% positive. Intervals are 95%, from a paired bootstrap that resamples **recordings**, not frames.

| model | average precision | 95% CI |
|---|---:|---|
| **`blinkcnn`** | **0.7893** | [0.700, 0.861] |
| `blinklint-union` | 0.4338 | [0.340, 0.540] |
| `blinklinmult-union` | 0.3815 | [0.204, 0.585] |
| `densenet121-union` | 0.3569 | [0.213, 0.617] |

All six pairwise differences, Holm-corrected across the family:

| comparison | Δ AP | 95% CI | p (Holm) | |
|---|---:|---|---:|---|
| densenet121 − **blinkcnn** | −0.4323 | [−0.568, −0.190] | 0.0012 | significant |
| blinklinmult − **blinkcnn** | −0.4077 | [−0.557, −0.236] | 0.0012 | significant |
| blinklint − **blinkcnn** | −0.3555 | [−0.448, −0.258] | 0.0012 | significant |
| densenet121 − blinklint | −0.0768 | [−0.260, +0.200] | 1.000 | — |
| blinklint − blinklinmult | +0.0522 | [−0.134, +0.242] | 1.000 | — |
| densenet121 − blinklinmult | −0.0246 | [−0.237, +0.227] | 1.000 | — |

**`blinkcnn` beats all three baselines by 36–43 points.** The p-values sit at the bootstrap's resolution floor: across 50 000 resamples it led in every one.

**The three baselines are statistically indistinguishable from each other.** Their point estimates differ by up to 7.7 points, but every interval overlaps — so this page does not rank them.

```bash
make score-models      # score the four models on a corpus
make compare-shipped   # the statistics -> results/comparison/shipped-rn30.json
```

## ⚠️ What this comparison is, and is not

**The 1.x models were not retrained.** These are the published 2023 weights, run under v2 conditions. Three things differ from the paper, and all three favour the v2 model:

1. **Features are computed with different tools.** The 1.x work used **TDDFA_V2** for landmarks and head pose; v2 uses **FaceMesh** (478 landmarks) and **6DRepNet**. `blinklinmult-union` consumes a 160-d descriptor built from those, so it is being fed features *analogous to* — not identical to — what it trained on.
2. **They see data they never trained on**, including MPEblink (921 clips, eval-only here).
3. **Operating points differ in origin.** `blinkcnn`'s threshold pair was swept on validation; the 1.x entries shipped with no fitted operating point at all. That is why the headline metric is average precision, which needs no threshold.

So this is a like-for-like comparison of **the models as they are deployable today**, not a claim that the v2 architecture beats the 1.x architecture. A fair architectural comparison would retrain all four on identical features and splits.

## Why recordings, not frames

RN30's test split holds 7 312 windows — drawn from 35 recordings. Windows from one recording share a subject, a camera and a blink rate; they are close to duplicates. Resampling them independently claims evidence the corpus does not contain.

The same comparison, computed both ways:

| resampling unit | 95% CI | verdict |
|---|---|---|
| per frame | [−0.024, −0.009] | "significant" |
| per recording | [−0.068, +0.027] | not significant |

The naive interval is **5.9× too narrow and reverses the conclusion**. A corpus is only as large as its number of independent subjects. Full protocol — including the choice of Holm over Benjamini-Hochberg, and why no interval is reported below ten clusters — in [`docs/comparison.md`](docs/comparison.md).

# ONNX optimization

`blinkcnn` is a ConvNeXt-Femto with 5.0 M parameters — *smaller* than the DenseNet121 behind `densenet121-union` — yet it ran **4.9× slower**. The cause was not the architecture: the 1.x models ship as frozen ONNX graphs and `blinkcnn` ran eager PyTorch.

15-frame windows, 4 threads, warm, on a 10-core machine:

| model | runtime | ms/frame | throughput | size | cold start |
|---|---|---:|---:|---:|---:|
| `densenet121-union` | onnx | 3.95 | 253 fps | 28.9 MB | 623 ms |
| `blinklint-union` | onnx | 4.18 | 239 fps | 27.6 MB | 723 ms |
| `blinklinmult-union` | onnx | 4.46 | 224 fps | 30.1 MB | 987 ms |
| `blinkcnn` | **pytorch** | **20.19** | 50 fps | 19.1 MB | **2481 ms** |
| `blinkcnn-onnx` | onnx | **4.15** | **241 fps** | 19.1 MB | **560 ms** |

Exporting the same weights closed the gap entirely, with accuracy identical to **4.8e-07**. Note the cold-start inversion: `blinkcnn` was the **smallest artifact on disk and the slowest to load**, because rebuilding a PyTorch module from a checkpoint costs far more than memory-mapping a graph. As ONNX it becomes the fastest — the number that matters for a service that scales to zero.

## The export is gated on parity, not on running

A graph that runs proves nothing. The export checks the graph against its checkpoint at three window lengths (1, 15, 30) across three input distributions — uniform `[0,1]`, Gaussian, constant — plus an end-to-end check through `BlinkDetector` that catches a wrong normalisation in the registry entry, which no graph-level check can see. Worst observed deviation: **2.6e-06**.

One check exists for a failure parity **cannot** catch. `EyeEncoder` standardises its input *inside* the network, from buffers registered `persistent=False`. If that transform were ever lost, the checkpoint and the graph would both lose it and agree perfectly — every probe green, the model quietly degraded. So the gate also asserts that a black and a white image produce clearly different logits: **1.280** for the shipped graph against **0.117** for a build with the transform removed.

This was found by deliberately breaking the export and discovering the gate did **not** reject it. A gate never watched failing is not known to work.

```bash
make export-blinkcnn-onnx   # export + 9 parity probes
make benchmark-runtime      # latency, size, cold start
```

# Real-time pipeline

A model-only throughput figure answers the wrong question for a live workload. Reaching real time meant three rounds of profiling, and **the right target changed each time**.

| stage | detect | pose | model | total | detect% | model% | result |
|---|---:|---:|---:|---:|---:|---:|---|
| 1. initial pipeline | 50.5 | 32.3 | 11.4 | 94.3 ms | 54% | 12% | 0.29× realtime |
| 2. geometric head pose | 50.5 | 0.05 | 11.4 | 62.0 ms | 81% | 18% | 0.53× realtime |
| 3. detector 640→256 + keypoints | 12.6 | 0.05 | 10.6 | **23.4 ms** | 54% | **45%** | **42.8 fps** |

Read forwards, this is Amdahl's law as a working method rather than a slogan:

- At step 1 the blink model was **12%** of the budget, so the 4.9× ONNX win moved end-to-end by **under 2%**. Real for the crops-in service, nearly invisible live.
- **Head pose** cost 26.7 ms via 6DRepNet. The detector already returns five facial keypoints, so roll and pitch can be read from their geometry at **0.05 ms** — 500× cheaper, correlating **+0.94** with 6DRepNet on both axes.
- **The detector was misconfigured, not slow.** It ran at ultralytics' default `imgsz=640`, upscaling a 576×720 frame. At 256 it costs 8.3 ms instead of 40.3, with the face still found on 60/60 frames, box IoU 0.984, and the blink decision unchanged on every frame.
- It also **already returned both eye centres** in the same forward pass, so the separate FaceMesh call was recomputing information already in hand. Eye span is estimated from inter-eye distance, measured at **0.412 ± 0.016** across 160 eyes.

After all three, the model's share rose from 12% to **45%** — it is only *now* the thing worth optimising further. That inversion is the point.

## Both model families reach usable speed, by different routes

| model | detect | score | total | fps | readout lag |
|---|---:|---:|---:|---:|---:|
| `blinkcnn-onnx` | 12.6 ms | 10.6 ms | 23.4 ms | **42.8** | none |
| `densenet121-union` | 12.7 ms | 10.3 ms | 23.0 ms | **43.4** | none |
| `blinklint-union` | 19.6 ms | 0.0 ms | 19.7 ms | **50.8** | **~233 ms** |

**Frame-wise models are genuinely real time** — scored inline, zero lag, a wink spikes the plot on the same frame.

**The sequence model is near real time.** One 15-frame call over both eyes costs 77.1 ms — 2.3× the entire 33.3 ms budget — so scoring every frame is not expensive but *impossible*. Moving it to a background worker decouples it: the capture loop becomes the **fastest of the three at 50.8 fps**, and the honest cost is a **~233 ms** readout delay. The video never stutters; the blink reading trails a quarter second, which the demo displays rather than hides.

That works because both blocking calls release the GIL — measured, a busy main loop kept **102%** of its idle iterations with an onnxruntime worker running, and **95%** with a capture thread. The worker takes the newest window whenever it is free, so the stride adapts to the machine instead of being tuned for one.

```bash
make benchmark-streaming    # the full ladder -> results/comparison/streaming.json
```

# Demos

## Live webcam

A single window: the frame on top, and beneath it two subplots at half width each, showing the last ten seconds of per-eye closure score with the operating point drawn as a rule. Wink one eye and only that side spikes — which is also the check that the left/right labelling is not inverted.

```bash
make webcam                                  # blinkcnn-onnx, mirrored, 30 fps
make webcam ARGS="--model blinklint-union"   # the sequence model, lag shown on screen
```

macOS asks for camera permission on the first run; the demo waits and says so rather than appearing hung.

## Offline video

The same view over a file — no camera, no permissions, and its output is checkable against the bundled clip's five annotated blinks (frames 170, 171, 227, 275, 276).

```bash
make video-demo                                    # the bundled clip
make video-demo ARGS="--record demo.mp4 --fast"    # write a recording
```

**Start here** if you are trying the repo for the first time: it is the command most likely to work first try.

## Gradio app

Upload a video, pick a model and an event-extraction rule, get an annotated video and per-eye plots back. Seven observable stages, each timed.

```bash
make app        # http://127.0.0.1:7860
```

Also deployed as a Hugging Face Space at [`fodorad/blink_detection`](https://huggingface.co/spaces/fodorad/blink_detection), which runs the released package rather than this checkout:

```bash
make push-space      # upload app.py, README.md and requirements.txt
```

## Notebook

All four models on one RN30 window, with the eye-closeness plot and thresholding:

```
notebooks/demo/blink_detection_rn30.ipynb
```

# Data

Seven corpora, each processed independently into `data/processed/<name>/` and collected into a single HDF5 file embedding the git SHA and the config that built it.

| corpus | kind | train / valid / test | annotates | role |
|---|---|---|---|---|
| CEW | stills | 3 388 / 726 / 732 | eye state | ESR only |
| MRL-Eye | stills | 59 520 / 12 680 / 12 698 | eye state | ESR only |
| RN15 | video, 15 fps | 2 198 / 2 708 / 4 664 | eye state + events | train + events |
| RN30 | video, 30 fps | 4 102 / 4 194 / 7 312 | eye state + events | train + events |
| TalkingFace | video, 25 fps | 0 / 0 / 524 | eye state + events | eval only |
| HUST-LEBW | clips, 30 fps | 762 / 134 / 450 | events (clip-wise) | train, ESR masked |
| MPEblink | video, 25 fps | — | events | eval only |

Counts are samples, one per eye per window. A corpus is routed by what it annotates, not by preference: `eye_state` is the trained target, so a corpus without it can only supervise the event head. Splits are **subject-disjoint** — verified, zero `video_id` overlap between train, valid and test.

```bash
make pull-all          # fetch the prebuilt HDF5 artifacts from Hugging Face
make preprocess-all    # or rebuild them from the raw corpora
```

See [`docs/data.md`](docs/data.md) for licences and the build recipe.

# Training

Three architectures behind one config key — `cnn` (per-frame, no sequence model), `lint` (single-stream LinMulT), `linmult` (cross-modal fusion over crops and handcrafted eye features).

```bash
make train-cnn         # the v2 frame-wise model
make train-joint       # both heads over every corpus
```

Both heads train on one masked stream: a corpus that lacks a target contributes an all-`False` mask, and both the loss and the metrics skip those positions. See [`docs/training.md`](docs/training.md).

# Deployments

## REST service

```bash
make serve                    # http://127.0.0.1:8080
make docker-build             # fodorad/blink_detection
make docker-run               # runs with --network none: weights are baked in
```

`GET /manifest` reports every model with its weight SHA-256, so a result traces to an artifact rather than to a version string. `POST /score` echoes back the extraction rule it applied, so a blink count is reproducible.

Two images: a crops-in REST service (~385 MB resident) and a full-pipeline image with the video endpoints and the Gradio demo (~1 GB, flat regardless of clip length).

# Repository layout

```
blinklinmult/        the installable package
  detector.py        one interface over all five models
  registry.py        what each model expects: normalisation, window, runtime
  stream.py          the background scorer for windowed models
  live.py            the live view: per-eye plots and frame overlay
  pipeline.py        raw video -> blink events, in seven observable stages
  paper/             the 1.x models, frozen as ONNX graphs
  data/ train/       schema, h5 writer, losses, metrics, the Lightning CLI
  preprocess/        one module per corpus, plus shared geometry and features
  bench/ compare/    latency measurement and the statistical layer

demos/               everything you run to SEE it work -- see demos/README.md
  video.py webcam.py gradio/ serve/ notebooks/

experiments/         everything that produced a NUMBER in the results
  the arms, the evaluations, the report generators

tools/               everything that BUILDS an artifact
  the ONNX exports, scoring, comparison, benchmarks, the embedding cache
```

The split is by **what a thing is for**, and inside the package by **what it
pulls in**: `detector` and `registry` need numpy alone, `live` and `pipeline`
add opencv, `preprocess` adds the extraction stack, `train` adds torch. That is
what makes `pip install blinklinmult[onnx]` a ~100 MB install rather than 600.

# Documentation

Full API reference (Sphinx + autoapi) and the guides:
**[fodorad.github.io/BlinkLinMulT](https://fodorad.github.io/BlinkLinMulT/)**

| path | covers |
| --- | --- |
| [`docs/comparison.md`](docs/comparison.md) | the statistical protocol, systems benchmarks, streaming design |
| [`docs/data.md`](docs/data.md) | the corpora, licences, and the HDF5 artifact |
| [`docs/training.md`](docs/training.md) | run configurations and the loss/metric design |
| [`docs/inference.md`](docs/inference.md) | the public API |

```bash
make docs           # build, warnings-as-errors
make docs-serve     # live-reload at http://127.0.0.1:8000
```

# Related projects

## exordium

Collection of preprocessing functions and deep learning methods. This repository contains revised codes for fine landmark detection (including face, eye region, iris and pupil landmarks), head pose estimation, and eye feature calculation.

* code: https://github.com/fodorad/exordium

## (2022) LinMulT

General-purpose Multimodal Transformer with Linear Complexity Attention Mechanism. This base model is further modified and trained for various tasks and datasets. `LinT` is its single-stream form, used by `blinklint-union`.

* code: https://github.com/fodorad/LinMulT

## (2022) PersonalityLinMulT

LinMulT trained for Big Five personality trait and sentiment estimation.

* paper: Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures ([pdf](https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf), [website](https://proceedings.mlr.press/v173/fodor22a.html))
* code: https://github.com/fodorad/PersonalityLinMulT

# Citation - BibTex

If you found our research helpful or influential please consider citing:

## (2023) BlinkLinMulT for blink presence detection and eye state recognition

```
@Article{fodor2023blinklinmult,
  title = {BlinkLinMulT: Transformer-Based Eye Blink Detection},
  author = {Fodor, Ádám and Fenech, Kristian and Lőrincz, András},
  journal = {Journal of Imaging},
  volume = {9},
  year = {2023},
  number = {10},
  article-number = {196},
  url = {https://www.mdpi.com/2313-433X/9/10/196},
  PubMedID = {37888303},
  ISSN = {2313-433X},
  DOI = {10.3390/jimaging9100196}
}
```

## (2022) LinMulT for personality trait and sentiment estimation

```
@InProceedings{pmlr-v173-fodor22a,
  title = {Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures},
  author = {Fodor, {\'A}d{\'a}m and Saboundji, Rachid R. and Jacques Junior, Julio C. S. and Escalera, Sergio and Gallardo-Pujol, David and L{\H{o}}rincz, Andr{\'a}s},
  booktitle = {Understanding Social Behavior in Dyadic and Small Group Interactions},
  pages = {218--241},
  year = {2022},
  editor = {Palmero, Cristina and Jacques Junior, Julio C. S. and Clapés, Albert and Guyon, Isabelle and Tu, Wei-Wei and Moeslund, Thomas B. and Escalera, Sergio},
  volume = {173},
  series = {Proceedings of Machine Learning Research},
  month = {16 Oct},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf},
  url = {https://proceedings.mlr.press/v173/fodor22a.html}
}
```

# Contact

* Ádám Fodor (fodorad201@gmail.com) — [adamfodor.com](https://adamfodor.com)
