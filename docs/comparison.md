# Comparing models

A benchmark table of point estimates cannot answer the question that decides
which model ships: *is this difference real, or is it noise?* This page is the
protocol that answers it, and the answer here is often "this corpus cannot
tell" — which is a finding, not a failure.

## The unit of analysis is the recording

RN30's test split holds **7312 windows drawn from 35 recordings**. Windows cut
from one recording share a subject, a camera, an illuminant, an eyelid shape and
a blink rate; they are close to duplicates. Treating them as 7312 independent
observations claims far more evidence than the corpus contains.

The cost of getting this wrong, measured on this repo's own training arms —
`fw-focal` against `fw-bce`, difference in average precision on RN30:

| resampling unit | 95% interval | verdict |
|---|---|---|
| per frame (**wrong**) | `[-0.024, -0.009]` | "significant" |
| per recording (**right**) | `[-0.068, +0.027]` | not significant |

Same data, same statistic. The naive interval is **5.9× too narrow** and
reverses the conclusion. Everything else on this page follows from taking the
right unit.

Both eyes of one subject also land in the same cluster: they blink together, so
resampling them apart would be the same error one level down.

## The protocol

**Paired.** Every model scores the same recordings, so one draw of recordings is
evaluated by all models at once. A resample heavy on hard recordings drags every
model down together, leaving the *difference* stable where the absolute scores
are not — which is what makes 20–35 clusters usable at all.

**Threshold-free first.** Average precision leads because the task is heavily
imbalanced: on RN30's blink-presence target 3.98% of valid frames are positive,
and on the closure target it is 1.08%. At that rate a model answering "open"
every time scores 98.9% accuracy, and any F1 is a statement about the threshold
as much as the model. AP integrates over every threshold.

**Corrected.** Four models make six pairwise comparisons, and six chances at a
5% bar is a ~26% chance of one false positive. Holm–Bonferroni controls the
family-wise rate without assuming the tests are independent — and they are not,
since all six share models. Benjamini–Hochberg would be the right choice at 600
comparisons, where a few false leads are cheap; at six, each rejection becomes a
claim a reader takes as true.

**No interval below 10 clusters.** TalkingFace is one recording: every resample
draws that same recording, the spread is exactly zero, and the "interval" would
have zero width — reading as a *precise* estimate when it is the opposite. Those
corpora get a point estimate and an explicit refusal.

**Fitting never touches test.** `fit.py` says of itself that it "does not
enforce this; it cannot see where the signals came from." The driver can, and
refuses when the fit and test paths coincide.

```bash
make compare-models COMPARE_CORPUS=rn30
```

## The result on RN30

```
rn30: 35 recordings, 79 634 valid frames, 3.98% positive

  fw-focal-augment    0.5754 [+0.5053, +0.6551]
  fw-bce              0.5630 [+0.4841, +0.6547]
  fw-bce-augment      0.5572 [+0.4788, +0.6507]
  fw-focal            0.5464 [+0.4598, +0.6367]

  0 of 6 pairs separate after correction.
```

The best and worst arms differ by **2.9 AP points** and the corpus cannot
distinguish them. Before correction the closest pair sits at p = 0.10; after
Holm, 0.62.

This is the honest reading: with 35 recordings the data resolves differences of
roughly 0.1 AP and no finer. Publishing a rank order here would be publishing
noise. The constructive version of the finding is that separating these arms
needs roughly **four times the recordings**, not more windows from the same
subjects — √n scaling, and the corpus is subject-limited, not frame-limited.

## Systems metrics

Accuracy decides whether a model is worth deploying; these decide whether it
can be. Measured at 15-frame windows, 4 threads, warm, on a 10-core machine:

| model | runtime | p50 | ms/frame | fps | size | cold start |
|---|---|---:|---:|---:|---:|---:|
| `densenet121-union` | onnx | 59.2 ms | 3.95 | 253 | 28.9 MB | 623 ms |
| `blinklint-union` | onnx | 62.7 ms | 4.18 | 239 | 27.6 MB | 723 ms |
| `blinklinmult-union` | onnx | 66.8 ms | 4.46 | 224 | 30.1 MB | 987 ms |
| **`blinkcnn`** | **pytorch** | **302.8 ms** | **20.19** | **50** | 19.1 MB | **2481 ms** |
| **`blinkcnn-onnx`** | **onnx** | **62.2 ms** | **4.15** | **241** | 19.1 MB | **560 ms** |

```bash
make benchmark-runtime
```

### The 4.9× was the runtime, not the architecture

`blinkcnn` is a ConvNeXt-Femto with 5.0 M parameters — *smaller* than the
DenseNet121 behind `densenet121-union` — yet it ran **4.9× slower**. The cause
was that the 1.x models ship as frozen ONNX graphs while `blinkcnn` ran eager
PyTorch. Exporting the same weights closed the gap entirely: 20.19 → 4.15
ms/frame, landing level with the models it had been losing to.

Note the cold-start inversion: `blinkcnn` is the **smallest artifact on disk**
(19.1 MB) and was the **slowest to load** (2481 ms), because rebuilding a
PyTorch module from a checkpoint costs far more than memory-mapping a graph. As
ONNX it becomes the fastest to load, at 560 ms. For a service that scales to
zero, that is the number that matters.

### The export is gated on parity, not on running

A graph that runs proves nothing. `make export-blinkcnn-onnx` checks the graph
against its checkpoint at three window lengths (1, 15, 30) across three input
distributions — uniform `[0,1]`, Gaussian, and constant — plus an end-to-end
check through `BlinkDetector` that catches a wrong normalisation in the registry
entry, which no graph-level check can see. Observed worst deviation: 2.6e-06.

One check exists for a failure parity **cannot** catch. `EyeEncoder` standardises
its input inside the network, from buffers registered `persistent=False`. If that
transform were ever lost, the checkpoint and the graph would both lose it and
agree perfectly — every parity probe green, the model quietly degraded. So the
gate also asserts that a black and a white image produce clearly different
logits: 1.280 for the shipped graph against 0.117 for a build with the transform
removed. This was verified by deliberately breaking the export and confirming
the gate rejects it.

## Reproducing

Every result carries the seed, the replicate count, the cluster count, the git
SHA and the machine. Two runs at one seed are byte-identical apart from the
timestamp.

```bash
make compare-models      # results/comparison/<corpus>.json
make benchmark-runtime   # results/comparison/runtime.json
```

`benchmark-runtime` is deliberately **not** part of `make check`: a timing taken
on a shared CI runner measures the runner, and would make the gate flaky. The
harness flags its own results as contended when one-minute load exceeds half the
core count — during development, load 7.0 on 10 cores inflated every latency by
about 1.7×, which a saturation-level threshold would have called idle.

## Streaming: can it keep up with a camera?

The throughput figures above are honest for a service that receives eye crops.
They are the wrong number for a live camera, because the model is not the
expensive part.

Measured on the **reduced** pipeline a real-time deployment would run — YOLO11
face detection, FaceMesh landmarks, eye crop, blink model, with no head pose and
no tracking — replaying the bundled 30 fps clip:

| model | detect | crop | score | total | achieved | dropped |
|---|---:|---:|---:|---:|---:|---:|
| `blinkcnn-onnx` | 50.5 ms | 0.1 ms | 11.4 ms | 62.3 ms | 16.0 fps | 80/80 |
| `densenet121-union` | 50.5 ms | 0.1 ms | 10.0 ms | 61.4 ms | 16.3 fps | 80/80 |

The budget at 30 fps is 33.3 ms per frame. The pipeline uses 62, so it runs at
**0.53× realtime and drops every frame**.

**Face detection is 81% of the budget; the blink model is 18%.** Swapping
between the two models changes end-to-end cost by about 1 ms per frame — under
2%. This is Amdahl's law on a real system: the 4.9× won on the model earlier is
a genuine and useful result *for the crops-in service*, and it barely moves the
streaming case, because the model was never the bottleneck there.

That is the point of measuring both. A benchmark reporting only "241 fps" would
be true of the model and would misdescribe the system by a factor of fifteen.

```bash
make benchmark-streaming
```

### Making it real time

The prediction above was right — detection was the target — and the fix needed
no new model. Three changes, each measured on CPU:

| change | detection | end-to-end | fps |
|---|---:|---:|---:|
| baseline: `imgsz=640` + FaceMesh + 6DRepNet | 50.5 ms | 62.3 ms | 16.0 |
| detector input at 256 instead of 640 | 8.3 ms | — | — |
| eye centres from the detector's own keypoints (no FaceMesh) | — | — | — |
| geometric head pose instead of 6DRepNet | 0.05 ms | — | — |
| **all three** | **16.4 ms** | **31.4 ms** | **31.9** |

**The pipeline now keeps up with a 30 fps source on CPU**, which is what
`make benchmark-streaming` runs by default.

Three findings behind that:

**The detector was misconfigured, not slow.** `yolo11n-pose_widerface` ran at
ultralytics' default `imgsz=640`, upscaling a 576×720 frame. At 256 it costs
8.3 ms instead of 40.3. The face was still found on 60/60 frames, box IoU held
at 0.984, and the blink decision was unchanged on every frame.

**The landmark model was recomputing what the detector already had.** The
detector is a *pose* model: one forward pass returns the box **and** five
keypoints, including both eye centres. FaceMesh was then run to recover eye
positions that were already available. Eye span is estimated from inter-eye
distance — measured at **0.412 ± 0.016** across 160 eyes, tight because both
quantities are fixed by the same anatomy. On the five annotated *closed* frames
of the bundled clip, crops from either path give the same open/closed verdict.

**Head pose can come from the same keypoints.** 6DRepNet costs 26.7 ms on CPU;
reading roll from the eye line and pitch from the nose's position costs
**0.05 ms** — 500× less, with no second forward pass. Against 6DRepNet over 80
frames, roll correlates at **+0.94** and pitch at **+0.94**.

Head pose is kept rather than dropped because yaw decides self-occlusion — which
eye is visible enough to score.

**One honest limit:** geometric **yaw is unvalidated**. The bundled clip is
near-frontal, 6DRepNet spanning only −3.5° to +2.4°, so there was no rotation to
correlate against. Since yaw is exactly what the occlusion rule uses, treat it as
a usable ordering rather than a calibrated angle, and use `--head-pose 6drepnet`
where the number itself must be right. Settling this needs a clip with genuine
head turns.

The corpus builders are unchanged: they keep FaceMesh and 6DRepNet, because
corpus quality is not the place to trade landmark precision for latency and the
published results must stay reproducible.

### Streaming a sequence model

`blinklint-union` predicts frame-wise but consumes a **15-frame (~0.5 s)
window**, and one call over both eyes costs **77.1 ms** on CPU — against a
33.3 ms budget at 30 fps. Re-running it per frame is not expensive but
impossible: the call alone is 2.3× the whole budget before a face has been
detected. Thread tuning does not rescue it (74.4 ms at four intra-op threads
against 76.9 on auto); the model is compute-bound.

So the two loops are **decoupled rather than budgeted**. Capture, detection and
display run on the calling thread; a single background worker scores the newest
window whenever it is free. That works because onnxruntime releases the GIL
during `run()`:

| | |
|---|---:|
| main-loop iterations in 300 ms, idle | 4,271,920 |
| main-loop iterations in 300 ms, worker running | 4,338,759 (**102%**) |
| face detection alone | 10.3 ms/frame |
| face detection with the worker running | 10.6 ms/frame |

The worker is genuinely concurrent, not time-slicing.

**The stride is adaptive.** Nothing schedules the next window: the worker
finishes one, takes the newest frames, and starts again. A fast machine gets
denser coverage and a slow one degrades gracefully, where a fixed stride must be
re-tuned per machine. On this laptop it settles at every 5th frame.

Measured against scoring every frame on a real RN30 recording:

| stride | calls | correlation vs stride 1 | peak score at closed frames |
|---:|---:|---:|---:|
| 1 | 616 | 1.0000 | 0.995 |
| 2 | 308 | 0.9985 | 0.998 |
| 3 | 206 | 0.9959 | 0.996 |
| 5 | 124 | 0.9838 | 0.998 |
| 15 | 42 | 0.8075 | 0.999 |

**Blinks are caught at full confidence at every stride** — the peak column never
drops. What thins out is the shape of the signal between blinks. The floor is
set by blink duration: measured on the bundled clip, blinks last 7–9 frames, so
a stride near 5 still puts a window inside every one.

Result, all three models under identical capture:

| model | detect | score | total | fps | lag | stride |
|---|---:|---:|---:|---:|---:|---:|
| `blinkcnn-onnx` | 12.6 ms | 10.6 ms | 23.4 ms | 42.8 | — | — |
| `densenet121-union` | 12.7 ms | 10.3 ms | 23.0 ms | 43.4 | — | — |
| `blinklint-union` | 19.6 ms | 0.0 ms | 19.7 ms | **50.8** | **233 ms** | 5.0 |

The windowed model's capture loop is the *fastest* of the three, because
scoring has left the loop entirely. **The honest cost is the 233 ms lag**, which
`StreamScore` reports rather than hides — frames newer than the last completed
window carry the previous value, never an interpolated one, since interpolating
would invent scores the model did not produce.

Verified end to end: streaming the bundled clip through this path detects all
five annotated blinks at scores ≥0.99.

## Trying it live

Two runnable demos share one view: the frame on top, and beneath it a scrolling
plot per eye showing the last ten seconds of closure score with the operating
point drawn as a rule.

```bash
make video-demo    # the bundled clip, no hardware needed
make webcam        # a live camera
```

`make webcam` opens a window; wink one eye and only that side's plot spikes,
which is also the check that the left/right labelling is not inverted.
`make video-demo` runs the same view over a file and reports which frames
crossed the threshold, so its output is checkable against the bundled clip's
five annotated blinks.

### Why a native window rather than a browser

A Gradio round-trip per frame would add a network hop to a loop with about six
milliseconds of slack. The measured display-loop budget, with capture and
windowed scoring on their own threads:

| stage | ms |
|---|---:|
| detect + geometric pose | 12.4 |
| eye crop | 0.1 |
| two plots | 0.3 |
| overlay | 1.0 |
| `imshow` + `waitKey` | 13.9 |
| **total** | **27.7 → 36 fps** |

Two findings shaped this:

**Plotting with matplotlib would cost more than the model.** A redraw of two
small subplots takes **6.1 ms**; the same lines drawn straight into a numpy
canvas with `cv2.polylines` take **0.13 ms** — 47× less, and the plots are the
cheapest thing on the list rather than the second most expensive.

**The camera is opened at its native resolution deliberately.** Asking for a
smaller frame is *slower*: 16.0 ms per read at 1920×1080 against 32.9 ms at
1280×720, because the driver rescales off its native mode. The frame is
downscaled in numpy afterwards, which is what the detector wants anyway.

Capture and windowed scoring both run on their own threads, which works because
`cap.read()` and onnxruntime's `run()` both release the GIL (95% and 102% of a
busy main loop's iterations retained). A windowed model therefore leaves the
display loop entirely — with `blinklint-union` the capture loop is the *fastest*
of the three models, and the honest cost is the 233 ms readout lag, shown on
screen.
