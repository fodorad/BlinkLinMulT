# The frame-wise backbone

One ConvNeXt-Femto that answers a single question per eye per frame — *is this
eye closed* — trained on every corpus that annotates closure, then benchmarked
across all seven.

## What each corpus annotates

This is the whole reason the benchmark needs three protocols rather than one.

| corpus | annotation | role |
|---|---|---|
| CEW, MRL-Eye | per-frame closure, still images | train + test |
| RN15, RN30 | per-frame closure **and** blink events | train + test |
| TalkingFace | per-frame closure and blink events | **test only** — one ~200 s recording of one subject, so a within-corpus split would measure memorisation |
| MPEblink | blink **events** only | **test only** — see below |
| HUST-LEBW | one label per **clip** | **test only** |

**MPEblink annotates events, not closure.** Verified at the source: each clip
ships `video.mp4` and `annotation_WFLW.json`, and across 200 clips the union of
per-person keys is `bbox`, `landmark`, `landmark_WFLW`, `blink` — where `blink`
holds `[start, end, category]` intervals, three per clip, not one label per
frame. The authors' own pipeline confirms the direction of travel: their model
emits per-frame `blink_scores` which `scores_to_intervals` thresholds into
intervals. Their `blinks_binary` is a rasterisation of the interval annotation,
exactly what our `rasterise()` produces.

## Closure is a strict subset of the blink event

Measured on every corpus that carries both labels:

| corpus | closed% | blink% | P(blink\|closed) | P(closed\|blink) |
|---|---:|---:|---:|---:|
| RN15 | 0.99 | 5.10 | **100.00** | 19.43 |
| RN30 | 0.61 | 2.16 | **100.00** | 28.37 |
| TalkingFace | 3.11 | 9.04 | **100.00** | 34.44 |

A closed eye is *always* inside an annotated blink — so the closure signal is a
valid blink detector. But a blink spans closing → closed → opening, and only
19–34% of its frames show a closed eye.

## Why the headline criterion is `any`

An oracle experiment settles this. Using **ground-truth closure** as the
prediction against **ground-truth events** on TalkingFace (59 recovered closure
runs against 61 annotated blinks):

| criterion | F1 of a *perfect* closure detector |
|---|---:|
| `any` | **0.9833** |
| `iou20` | 0.7667 |
| `iou50` | 0.0333 |
| `iou75` | 0.0000 |

Median IoU of a closure run against its own event: **0.333**. A 2-frame closure
inside an 8-frame event has IoU 0.25, so `iou50` is unreachable *by
construction*. Reporting it as the headline would present an annotation
convention as a model failure.

Consequently the validation threshold is fitted against `any` for this model
(`cli.build_callbacks`), not the `iou50` the sequence models use — fitting on a
criterion the model cannot satisfy would maximise noise.

## The three protocols

**Stills (CEW, MRL)** — frame accuracy, precision, recall, F1. The model's
output is the annotation; nothing is reconstructed.

**Video (RN15, RN30, TalkingFace, MPEblink)** — per-frame closure
probabilities are reassembled onto each recording's timeline
(`average_overlapping`), thresholded into intervals (`to_intervals`), and
matched one-to-one against the annotated blinks (`match`). Reported at all four
criteria plus Blink-AP.

**HUST-LEBW** — the corpus labels one blink per clip; its per-clip positive
fraction is 1.0 or 0.0, never partial. The clip is therefore the unit, scored
as `max` over its frames (the BPD protocol, `bpd_f1`).

## Training

Loss is **`bce`**, not focal: CEW and MRL are 65% and 96% closed, and
`StillFramesDataset` balances the video corpora to `still_open_to_closed`, so
the loss sees a near-even split and focal's down-weighting buys nothing.

Mixing is **`temperature: 2.0`**. After stills expansion and balancing the
corpora are roughly CEW ~3.4k, MRL ~59.5k, RN15 ~4.4k, RN30 ~3.4k frames — a
~15× spread. Proportional sampling would make a batch almost entirely MRL;
sqrt-weighting cuts the spread to ~4× while keeping MRL dominant, which is
right because it is genuinely the largest source of closed-eye variety.

Selection is on **`valid/mean_f1`** — frame-level F1 on the trained target.
Not event F1: that rests on ~60 events and a threshold fitted on the same
split, so selecting on it would overfit validation.

## Evaluation splits keep every frame

Training thins the open frames (`still_stride`) and balances them against the
closed ones. **Evaluation must not**, and `BlinkDataModule._open` passes
`stride=1` for valid and test accordingly.

This is not a preference. The event protocol reassembles a *continuous*
per-frame signal and thresholds it into intervals; a strided eval split leaves
a scatter of isolated frames, so every retained closed frame becomes its own
one-frame "event" that matches trivially. Measured on a strided eval split
before the fix: median **1** frame covered per recording, median blink length
**1** frame, and all four criteria reporting an identical F1 at 100% recall --
an artefact of the sampling, not a property of the model.

The unstrided test split is ~2.4M frames, of which MPEblink is 1.9M, and the
full test pass takes roughly **3 hours**.

That cost is **I/O, not compute**. Measured: a 64-frame forward is 35.5 ms, so
2.4M frames is only ~22 min of GPU time -- but a 343k-frame pass took 25 min
wall-clock, ~8x its compute. `num_workers` is clamped to 0 because an HDF5
handle cannot cross a process boundary, so every batch is read on the main
thread. Dropping MPEblink from `eval_datasets` gives a ~0.5 h pass for a first
look; the full split is a one-off cost per run.

## Two rounds

**Round 1** trains on CEW + MRL + RN15 + RN30 — consistent closure labels.

**Round 2** adds MPEblink, rasterising its intervals into per-frame labels. This
is a genuinely open question rather than a known improvement: MPEblink brings
63 758 windows of in-the-wild variety, but its positive frames include the
partly-open lid of the closing and opening phases. RN15/RN30 have their own
boundary-convention noise and only a handful of subjects, so which label set is
more consistent is not obvious in advance. The two rounds differ in one flag.

```bash
make train-frame-wise             # round 1
make train-frame-wise-mpeblink    # round 2
make report-frame-wise            # the benchmark table
```

## Reading the numbers honestly

Three caveats belong beside any reported figure:

1. The model is trained on **closure** and scored on **events**. Quote the
   oracle ceiling (0.9833 at `any`) beside the model's number so a reader sees
   what was achievable.
2. The operating point is **fitted on validation** and applied to test — never
   tuned on the test split.
3. MPEblink's Blink-AP is **oracle-instance** (instances come from the
   annotation, so detection is free) and uses the interval **peak** as its
   confidence score, where the authors rank by their detector's instance score.
   Both make our number an upper bound on a directly comparable one.
