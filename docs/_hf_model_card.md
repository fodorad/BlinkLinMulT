# BlinkLinMulT — pretrained blink and eye-state models

Models for **eye-state recognition** (per-frame closure) and **blink
detection** (temporal events), from the
[BlinkLinMulT](https://github.com/fodorad/BlinkLinMulT) project.

Three are the published 1.x models from
[*BlinkLinMulT: Transformer-Based Eye Blink Detection*](https://www.mdpi.com/2313-433X/9/10/196)
(J. Imaging, 2023), shipped as frozen ONNX graphs. The v2 frame-wise model ships
twice: as a PyTorch checkpoint (`blinkcnn`, trainable and fine-tunable) and as
the same weights frozen into a graph (`blinkcnn-onnx`, deployable).

## Models

| id | runtime | input | output | notes |
|---|---|---|---|---|
| `densenet121-union` | ONNX | `(B, T, 3, 64, 64)` | `(B, T, 1)` | 1.x frame-wise, DenseNet121 |
| `blinklint-union` | ONNX | `(B, T, 3, 64, 64)` | `(B, T, 1)` | 1.x sequence, DenseNet121 + LinT |
| `blinklinmult-union` | ONNX | crops + `(B, T, 160)` | `(B, T, 1)` and `(B, 1)` | 1.x two-stream, the paper's headline model |
| `blinkcnn` | PyTorch | `(B, T, 3, 64, 64)` | `(B, T, 1)` | v2 frame-wise, ConvNeXt-Femto |
| `blinkcnn-onnx` | ONNX | `(B, T, 3, 64, 64)` | `(B, T, 1)` | the same weights, frozen — **4.9x faster** |

Both axes are dynamic: `T=1` is a single frame, any larger `T` is video.

`blinkcnn` and `blinkcnn-onnx` are the *same weights*. The graph was exported
behind a numerical-parity gate — three window lengths, three input
distributions, worst deviation **2.6e-06**, and an end-to-end check through the
public API at **4.8e-07** — so it is a drop-in replacement, not an
approximation. Measured on CPU it runs **20.19 → 4.15 ms/frame** and loads in
560 ms rather than 2481. Prefer `blinkcnn-onnx` unless you are fine-tuning.

The ONNX models need no PyTorch at all: `pip install blinklinmult[onnx]` is a
~100 MB install against ~600 MB for the training stack.

## Usage

```python
from blinklinmult import BlinkDetector

model = BlinkDetector.from_pretrained("blinklint-union")
signal = model.score(crops)  # (T,) closure probability per frame
blinks = model.detect(crops)  # [(start, end), ...] frame indices
```

Weights download and cache on first use. Crops are `(T, 3, 64, 64)` float32 in
`[0, 1]`, channel-first — **`BlinkDetector` applies each model's own
normalisation**, so pass raw `[0, 1]` values and do not pre-normalise.

Install: `pip install blinklinmult[onnx]` for the ONNX models, plus `torch` for
`blinkcnn`.

## Normalisation — the one thing to get right

The two generations were trained on **different input conventions**, and applying
the wrong one costs accuracy *without raising an error*:

| generation | normalisation |
|---|---|
| 1.x (`*-union`) | ImageNet, mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)` |
| v2 (`blinkcnn`) | plain `/255` — no shift, no scale |

Each file's `.json` sidecar records its own constants. Use `BlinkDetector` and
this is handled for you.

## Events, and the operating points

Blink intervals come from **hysteresis extraction** over the per-frame signal: a
run must peak above a high threshold to count as a blink, but extends while it
stays above a low one. The high threshold decides *whether*, the low one decides
*where* — a single cut has to be low enough to catch a closure's shallow onset
and high enough to ignore noise, and no one value does both.

| model | threshold | low ratio | fitted? |
|---|---:|---:|---|
| `blinkcnn` | 0.53 | 0.25 | **yes**, on validation |
| the three `*-union` models | 0.50 | — | **no** — see below |

**The 1.x thresholds are placeholders, not measurements.** Those models were
released as weights alone, with no operating point, and refitting one would mean
re-running their validation splits under this codebase. So:

* frame-level `score()` output is **directly comparable** across generations;
* event-level counts from `detect()` are **not** — `blinkcnn`'s threshold was
  tuned and theirs was not.

To close that gap, fit on your own validation data:

```python
from blinklinmult.fit import fit_operating_point

point = fit_operating_point(signals, annotations)  # validation only
model.calibrate(point)
```

## Provenance

Each ONNX graph was exported from the original published PyTorch weights and
**verified against them before release** at window lengths 1, 15, 30 and 45 —
including lengths the export never traced, which is what proves the time axis is
genuinely dynamic rather than unrolled. Maximum observed deviation across all
three models and all lengths: **3.4e-05**, i.e. float32 rounding.

Every `.json` sidecar carries the source checkpoint's filename and SHA-256, the
opset, the normalisation, and the parity tolerances.

## Limitations

* **Eye crops in, not faces.** These models score a cropped eye region, not a
  full frame. Crop geometry matters: v2 was trained on a square `2.0 ×` the eye's
  corner-to-corner span. A different crop convention degrades accuracy.
* **`blinklinmult-union` needs the 160-d descriptor stream** (iris landmarks,
  eyelid distances, EAR, head pose) alongside the crops. Producing it requires
  the `[preprocess]` extra.
* **Frozen.** The 1.x models ship as graphs and cannot be fine-tuned. That is
  deliberate: they are published artefacts.
* Trained on public blink corpora that skew toward frontal, indoor webcam
  footage. Expect degradation on strong profile views, heavy occlusion, or
  infrared imagery.

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

## License

MIT, matching the source repository. The corpora these models were trained on
each carry their own licence and are **not** redistributed here.
