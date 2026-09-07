# Migrating from 1.x

Version 2 is a full rewrite. **The 1.x public API is gone** — this is a
deliberate breaking change, not an oversight.

## What was removed

```python
# 1.x — no longer works
from blinklinmult.models import DenseNet121, BlinkLinT, BlinkLinMulT

model = DenseNet121()  # weights auto-downloaded
y = model(torch.rand(32, 3, 64, 64))
```

`blinklinmult.models` is now a *module* rather than a package, and it loads
trained v2 checkpoints -- a different thing entirely. Importing a retired name
from it raises an `AttributeError` naming the replacement rather than failing
opaquely.

## The 1.x models are still available

They were not deleted; they were **frozen as ONNX graphs** and load by id:

```python
from blinklinmult import BlinkDetector

model = BlinkDetector.from_pretrained("densenet121-union")
signal = model.score(crops)  # (T,) eye closeness
blinks = model.detect(crops)  # [(start, end), ...]
```

| 1.x class | id today |
|---|---|
| `DenseNet121` | `densenet121-union` |
| `BlinkLinT` | `blinklint-union` |
| `BlinkLinMulT` | `blinklinmult-union` |

Freezing them to ONNX means they need neither torch nor `linmult`, so a future
dependency upgrade cannot break them -- which is what happened when `linmult`
2.x changed its constructor and left the 1.x checkpoints unloadable.

Two differences to know. The graphs expect **ImageNet-normalised** crops where
`blinkcnn` expects plain `/255`, and `BlinkDetector` applies each model's own
constants -- so pass raw `[0, 1]` values. And the two sequence models were
trained on **15-frame windows**; `score_long` slides that window and averages
the overlap.

## The weight auto-download is gone

Constructing a model in 1.x silently fetched a checkpoint over the network from
a GitHub release; a library that reaches the internet on `__init__` cannot be
used offline, cannot be pinned, and gives no way to know which weights a result
came from. Weights now download on an explicit `from_pretrained` call and cache
locally.

The 1.x demo scripts (`webcam_densenet121.py`, `talkingface_inference.py`) and
the 1.x preprocessing scripts are also gone; the latter are replaced by
`blinklinmult.preprocess`.

## What replaces it

Models are built from a config and loaded from an explicit checkpoint:

```python
from blinklinmult.train.module import BlinkLightningModule

module = BlinkLightningModule.load_from_checkpoint("results/joint/checkpoints/best.ckpt")
module.eval()
```

A checkpoint carries its own architecture config, so nothing has to be rebuilt
and passed in — and the run that produced it is identified in MLflow by commit,
config, and dataset hash.

## Config keys that were renamed

1.x model configs are rejected at load with a rename hint rather than silently
ignored:

| 1.x key | 2.x |
|---|---|
| `input_modality_channels` | derived from the backbone and dataset |
| `projected_modality_dim` | `d_model` |
| `number_of_layers` | `cmt_num_layers` |
| `add_projection_fusion` | `add_module_ffn_fusion` |
| `aggregation` | heads are sequence-level; no time reducer |
| `n_heads` | `num_heads` |
| `dropout_qkv` | `dropout_attention` |
| `input_dim` | derived from the dataset's `feature_dim` |
| `output_dim` | derived from the task's targets |
| `weights` | load a checkpoint with `--resume` |

The derived ones are absent by design: they are read from the data, so the model
cannot disagree with what it is fed.

## Behavioural changes worth knowing

These change results, not just APIs.

**Splits are assigned by recording.** 1.x had no split logic at all — it left
the division to whoever ran the training script. v2 assigns splits by hashing
the recording or participant, so windows from one recording cannot straddle the
train/test boundary. Numbers from a 1.x run that split windows randomly are not
comparable with v2's.

**Window edge cases are fixed.** 1.x centred a short blink in its window, then
clamped the start to `0` and the end to the recording length — but the end clamp
set `start = end - win_size` without re-checking `start >= 0`, so a recording
shorter than one window produced a negative start and, through Python's negative
indexing, a window stitched from the *tail* of the recording. Negative windows
were drawn by rejection sampling, which raised on short recordings and looped
forever on ones with no blink-free window. Both are handled explicitly and both
are covered by tests.

**Eye crops are padded, not clipped.** 1.x clipped a crop box to the image
bounds and returned whatever was left, so an eye near the frame edge came out
smaller and was then stretched by a varying amount by the subsequent resize.
v2 pads instead, keeping every crop square and every eye at the same scale.

**Metrics are named for what they are.** 1.x reported accuracy as the headline
number, which on a continuously-sampled recording is above 95% for a model that
never predicts a blink. v2 reports F1, precision, recall, and average precision,
and selects checkpoints on mean F1.

**One model can now train both tasks.** This is the point of the rewrite: with
per-target validity masks, a corpus that annotates only eye state contributes to
that head alone, and one run trains across all six corpora.

## The evaluation set changed

RT-BENE / RT-GENE and ZJU were in the 1.x paper and are not in the v2 benchmark.
HUST-LEBW is included, and TalkingFace is now held out entirely as a
cross-corpus test set rather than being split. Published 1.x numbers are
therefore not directly comparable; `make ablation` reruns the architecture
comparison on the v2 data so the baselines and the full model are measured on
identical corpora and splits.
