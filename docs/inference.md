# Inference

Four pretrained models behind one interface. Weights download from the
[Hugging Face Hub](https://huggingface.co/fodorad/blink_detection) on first
use and cache under `~/.cache/torch/hub/checkpoints/blinklinmult`.

| id | what it is | runtime | window |
|---|---|---|---|
| `blinkcnn` | current frame-wise model, ConvNeXt-Femto | PyTorch | frame-wise |
| `densenet121-union` | 1.x frame-wise, DenseNet121 | ONNX | frame-wise |
| `blinklint-union` | 1.x sequence, DenseNet121 + LinT | ONNX | 15 frames |
| `blinklinmult-union` | 1.x two-stream, crops + 160-d descriptors | ONNX | 15 frames |

The three `*-union` models are from the
[1.x paper](https://www.mdpi.com/2313-433X/9/10/196), frozen as ONNX graphs so
they run without torch and cannot be broken by a dependency upgrade.

## Scoring eye crops

```python
from blinklinmult import BlinkDetector

model = BlinkDetector.from_pretrained("blinkcnn")
signal = model.score(crops)  # (T,) eye closeness, 0 open .. 1 closed
blinks = model.detect(crops)  # [(start, end), ...] inclusive frame indices
```

Crops are `(T, 3, 64, 64)` float32 in `[0, 1]`, channel-first. `T=1` is a single
frame; any larger `T` is video.

**Pass raw `[0, 1]` values.** The 1.x models were trained on ImageNet-normalised
crops and `blinkcnn` on plain `/255`; `BlinkDetector` applies each model's own
constants from {py:data}`blinklinmult.registry.MODELS`. Pre-normalising means the
model gets it twice, which costs accuracy and raises nothing.

## Two decisions, kept separate

**Eye state** is what the model produces: a continuous per-frame closeness curve.
**Blink presence** is a rule laid on top of it. They fail independently, so they
are configurable independently.

```python
from blinklinmult.pipeline import Extraction

Extraction(high=0.4)  # a single cut
Extraction(high=0.53, low=0.1325)  # hysteresis
```

A single cut has to be low enough to catch a closure's shallow onset and high
enough to ignore noise, and no one value does both. Hysteresis splits those
jobs: a run must *peak* above `high` to count as a blink, but *extends* while it
stays above `low`. On `blinkcnn` that is worth 0.19 to 0.52 event F1.

**The shipped operating points are not universal.** `blinkcnn`'s pair was swept
on RN validation data, so on other footage it is a transfer rather than a
measurement. The 1.x models carry a plain `0.5`, which was never fitted at all --
that work reported eye state, not events. Fit your own:

```python
from blinklinmult.fit import fit_operating_point

point = fit_operating_point(signals, annotations)  # on validation data
model.calibrate(point)
```

## Whole videos

{py:func}`blinklinmult.pipeline.run` takes raw video through seven stages --
face detection and tracking, head pose, landmarks and eye localisation, eye
selection, inference, labelling, visualisation -- yielding each as it completes.

```python
from blinklinmult.pipeline import run, Stage

for item in run("clip.mp4", model, start=0.0, duration=10.0):
    print(item.line() if isinstance(item, Stage) else item.events)
```

Both eyes are scored **separately**, and an eye rotated away from the camera
(`|yaw| > 45°`) is suppressed rather than guessed at: its signal is `NaN`, not
zero. A gap means "not looked at", which is a different claim from "open".

This path needs the `preprocess` extra, which pulls the exordium extraction
stack.

## The demo

```bash
make app     # http://127.0.0.1:7860
```

Pick a model, upload a clip or use the built-in TalkingFace example, choose an
extraction rule, and read the per-eye plot against the annotation.

## Installing

```bash
pip install blinklinmult[onnx]        # the four models
pip install blinklinmult[preprocess]  # + the video pipeline
pip install blinklinmult[train]       # + training
pip install blinklinmult[demo]        # + the Gradio app
```
