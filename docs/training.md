# Training

Every run is three YAML files — data, model, train — and nothing else. Unknown
keys are rejected rather than ignored, so a config always describes what the run
actually did.

## Running

```bash
make train-joint            # both heads, every corpus (the union experiment)
make train-blink-presence   # blink presence, video corpora only
make train-eye-state        # eye state, still-image corpora

make train-cnn              # BlinkCNN
make train-lint             # BlinkLinT
make train-linmult          # BlinkLinMulT

make train-single DB=rn30   # one corpus, trained and evaluated alone
make per-dataset            # every per-corpus arm
make ablation               # the four architectures on identical data
make mlflow-ui              # inspect the runs
```

Any config value can be overridden without editing a file:

```bash
make train-joint ARGS="--set data.strategy_kwargs.temperature=3.0"
make train-joint ARGS="--set train.optimizer.lr=1e-4 --set data.batch_size=16"
```

Or call the CLI directly for full control:

```bash
uv run python -m blinklinmult.train.cli \
    --data config/data/all.yaml \
    --model config/model/blinklinmult.yaml \
    --train config/train/joint.yaml
```

## Configurations

**`config/data/`** — which corpora, and how they are mixed.

| Config | Corpora | Use |
|---|---|---|
| `all.yaml` | TalkingFace, RN15, RN30, HUST-LEBW, MPEblink | the union experiment; the sequence models |
| `stills.yaml` | CEW, MRL-Eye + video corpora as frames | eye state; the frame-wise model |
| `rn.yaml` | RN15 + RN30 | the aggregate RN arm |
| `single_<db>.yaml` | one corpus | the per-dataset arms |
| `smoke.yaml` | RN30, CEW | CI and smoke runs |

**`config/model/`** — the three shipped models plus one baseline:

| Config | Model | Inputs |
|---|---|---|
| `blinkcnn` | BlinkCNN | one eye crop `(C, H, W)` |
| `blinklint` | BlinkLinT | `(T, C, H, W)` + `(T,)` mask |
| `blinklinmult` | BlinkLinMulT | the above **and** `(T, 160)` + its own `(T,)` mask |

The two masks of BlinkLinMulT are independent by design: a usable eye crop can
still defeat the iris landmarker, so a frame can be valid for the image branch
and invalid for the feature branch.

**`config/train/`** — `joint`, `blink_presence`, `eye_state`, `smoke`.

Not every combination is valid, and the invalid ones fail at config load rather
than after an hour of training. `family: cnn` with the blink-presence task is
rejected: a per-frame CNN has no sequence model and cannot express the task.

## The analysis window

`window_seconds: 1.5` is one knob for the whole run; each corpus derives its own
frame count from its rate (30 fps → 45 frames, 15 fps → 22). It matches
MPEblink's 36-frame test clip, so the numbers are readable beside theirs. Every corpus then
spans the same real duration, OmniLoader pads the shorter ones and masks the
padding, and nothing is resampled.

```bash
make train-joint ARGS="--set data.window_seconds=1.0"   # every corpus re-derives
```

A joint run mixing RN15 and RN30 resolves to the longer frame count, so RN15
samples arrive roughly half padding. That is intended — the frames genuinely are
not there.

## Mixing the corpora

The corpora differ in size by three orders of magnitude — MRL-Eye has ~85 000
stills, TalkingFace is one recording. Proportional mixing would make a batch
almost entirely MRL, so the default is `temperature: 2.0`, which samples by
`sqrt(size)`: the large corpora stay dominant without erasing the small ones.
Raise the temperature towards uniform, lower it towards proportional.

```yaml
strategy: temperature      # proportional | temperature | annealed_temperature
strategy_kwargs:           # | fixed | round_robin
  temperature: 2.0
```

Each strategy accepts only its own arguments; passing another's fails at setup
with a message naming the offending key.

## Losses and metrics

**Losses are masked.** Every loss reduces over validly-supervised positions
only. That is not an optimisation — it is what makes joint training correct. The
mean is taken over valid positions, so a batch's loss does not depend on how
many of its samples came from a corpus that annotates the target, and a batch
with no valid position for a head contributes exactly zero to it *with the
gradient path preserved*, so DDP does not deadlock waiting for a rank that
skipped a parameter.

`focal` is the default for blink presence: sampled continuously, the
overwhelming majority of frames are easy negatives whose aggregate gradient
drowns out the few frames where the eye is actually closing. `bce` is the
default for eye state, which is roughly balanced in CEW and MRL.

**Training and evaluation are windowed identically** — a 50%-overlapping sweep
of every split, described in [Data](data.md). That is what makes `focal` the
right default rather than a precaution: roughly 10% of training windows contain
a blink, the same proportion the test split holds, so a threshold calibrated on
validation transfers to test. 1.x balanced the training split to 50% positive,
which made the loss easier to optimise and the operating point meaningless.

**F1 is the primary metric, not accuracy.** A model predicting "no blink"
everywhere scores above 95% accuracy on a continuously-sampled recording, which
says nothing about whether it detects blinks. Precision and recall are reported
alongside because the two failure modes have different costs, and average
precision because it is threshold-free.

**Training is eye-wise; reporting is frame-wise.** Two families of scores are
logged:

```
{split}/{target}/f1                per eye — what the loss sees
{split}/{target}/frame_max/f1      per frame — THE reported number
{split}/{target}/frame_mean/f1     the alternative rule, for comparison
{split}/mean_f1                    mean frame-level F1; drives selection
{split}/mean_eye_f1                mean per-eye F1, as a diagnostic
```

`max` is the primary rule: a blink is a frame-level event that *occurred*, so if
one eye is occluded while the other is plainly closing, `max` recovers it where
`mean` halves the score. The aggregation runs on **validation as well as test**,
so `valid/mean_f1` and `test/mean_f1` are directly comparable and model
selection optimises the number the benchmark reports.

Each run also writes `test_per_dataset.json`: the test split scored separately
per contributing corpus. One aggregate F1 can hide a model that has learned
MRL-Eye's stills and fails on RN30's video entirely.

## Optimization

An ImageNet-pretrained backbone and a randomly-initialised transformer do not
want the same step size — fine-tuning the backbone at the transformer's rate
destroys the pretrained features within a few hundred steps. `backbone_lr`
therefore splits the parameters into two groups, and the one-cycle schedule
keeps each group's own peak rather than flattening them to a single `max_lr`.

```yaml
optimizer:
  lr: 0.001           # transformer and heads
  backbone_lr: 0.0001 # the pretrained CNN
  scheduler: onecycle
```

Set `backbone_lr: null` to use one rate for everything.

## Choosing the backbone

`convnext_femto` is the default. It was chosen by measurement, not by reputation,
and the numbers below are the justification.

**The constraint is the crop size.** Eye patches are 64x64, and every
ImageNet backbone reduces its input by 32, so a 64px crop collapses to a **2x2
feature map** before pooling — four cells to describe an eye. `backbone_wide_stem`
(on by default) removes one stride-2 reduction to recover 4x4. It discards no
pretrained parameter, but the layers downstream then see a feature map at a
sampling rate they were not trained on, so it expects fine-tuning rather than
freezing.

### Measured results

CEW eye state, three seeds per backbone, `config/data/single_cew.yaml` with
`config/train/eye_state.yaml` unchanged. GPU time is one 26-crop batch — a
13-frame window times two eyes — on an otherwise idle Apple MPS device. Measure
it idle or not at all: the same backbone timed 13 ms free and 23 ms while
another run held the GPU, which is enough to reorder the column.

| backbone | year | params | GPU | accuracy | F1 | precision | recall |
|---|---|---|---|---|---|---|---|
| `efficientnet_b0` | 2019 | 4.0M | 35 ms | **96.72 ±1.09** | **96.64 ±1.13** | 96.67 ±1.63 | **96.65 ±2.16** |
| **`convnext_femto`** | 2022 | 4.8M | **14 ms** | 96.63 ±0.97 | 96.49 ±1.06 | **98.12 ±0.15** | 94.92 ±2.08 |
| `densenet121_truncated` | 2016 | 4.8M | 42 ms | 95.27 ±0.08 | 95.11 ±0.07 | 96.19 ±0.72 | 94.06 ±0.66 |
| `densenet121` (the 1.x model) | 2016 | 7.0M | 45 ms | 94.38 ±0.00 | 94.21 ±0.05 | 94.95 ±0.76 | 93.49 ±0.83 |
| `mobileone_s1` | 2023 | 3.5M | 32 ms | 92.23 ±1.23 | 92.34 ±1.13 | 89.22 ±1.94 | 95.69 ±0.29 |
| `shufflenet_v2` | 2018 | 1.3M | 7 ms | 88.81 ±0.49 | 88.89 ±0.51 | 86.36 ±0.35 | 91.57 ±0.72 |
| `mobilenetv4_conv_small` | 2024 | 2.5M | **6 ms** | 87.83 ±0.58 | 88.27 ±0.60 | 83.45 ±0.34 | 93.68 ±1.00 |

**Read the precision column, not the accuracy column.** The fast mobile
backbones match DenseNet's *recall* — `mobilenetv4_conv_small` even beats it,
93.68 against 93.49 — and lose entirely on *precision*, 83.45 against 94.95.
They over-predict "closed". For blink detection that is the expensive
direction, because a false closure becomes a phantom blink. Selecting a backbone
on inference cost alone picks exactly the wrong one.

`efficientnet_b0` and `convnext_femto` are **tied on accuracy**: 0.09pp apart
against ±1 seed spreads. Cost is what separates them, and ConvNeXt is 2.5x
faster with the tightest precision in the table. They differ in character, so
the choice is reversible on purpose — ConvNeXt leans precise (98.1 / 94.9),
EfficientNet balanced (96.7 / 96.7). Prefer EfficientNet if missing a blink ever
costs more than inventing one.

`densenet121_truncated` drops DenseNet's last dense block, which holds 31% of
its parameters and runs on a 2x2 map at this crop size. **It beats the full
network it was cut from** — 95.27 against 94.38, on 4.79M parameters instead of
6.95M — so that block is not merely idle here, it is harmful: 2.16M parameters
fine-tuned on 3.5k images cost more in overfitting than they return. The cut is
free in the sense that matters, since every remaining tensor is bit-identical to
the pretrained checkpoint. It is still not competitive with either leader, and
is kept as evidence rather than as a recommendation.

Note also that 121 is the **smallest pretrained DenseNet in existence** —
torchvision and timm ship 121/161/169/201 and the rest are larger — which is why
a smaller one had to be cut rather than downloaded.

### Caveats

- **Still images only.** CEW is single frames, so this ranking measures
  appearance and nothing temporal. The backbone feeds LinT and LinMulT in the
  models that matter, and a backbone that wins on stills need not win as a
  sequence encoder.
- **The validation split is 626 crops**, and it drives both early stopping and
  checkpoint selection. That is the likeliest source of every ±1 spread here.
- Weight tags are pinned in `TIMM_WEIGHTS`
  (`blinklinmult/train/model.py`) rather than left to timm's defaults, which
  move between releases.

Any backbone in `BACKBONES` can be selected without editing a file:

```bash
make train-eye-state ARGS="--set model.backbone=efficientnet_b0"
```

## What a run produces

Under `results/<task>/`:

- `checkpoints/` — best by F1, best by loss, and last. Full state, so a run
  resumes with `--resume`.
- `metrics_test.json` — the headline test metrics.
- `test_predictions_<target>.csv` — per-position probability, target, and
  validity, so threshold tuning and significance testing never need the model
  re-run.
- `test_per_dataset.json` — per-corpus breakdown.
- `test_precision_recall.png`, `time.json`.

Everything except the checkpoints is logged to MLflow as artifacts, alongside
the flattened config, the library versions, the commit, and **each dataset
file's hash and embedded builder SHA** — two runs over "the same six corpora"
are not comparable if one of them was rebuilt, and nothing else would reveal
that.

## Smoke testing

```bash
make train-smoke   # one batch through every stage, then exit
```

This is what to run after changing anything in the data or model layer; it
exercises the whole chain in seconds.
