# From eye states to blink events

The pipeline learns **one thing** — a continuous per-frame eye-state signal —
and derives everything else from it. Events are post-processing, not a second
model output. This document records what that buys, what it currently costs, and
what the annotations do and do not allow.

## The two stages

1. **Frame-wise.** A ConvNeXt scores each 64x64 eye crop, producing a value in
   `[0, 1]` per frame. Every corpus can supervise this, including the
   still-image ones, because a closed eye is a closed eye with or without a
   temporal axis.
2. **Event extraction.** The per-frame signal is reassembled onto each
   recording's timeline, thresholded into intervals, and matched against the
   annotated blinks under four criteria.

The consequence worth stating plainly: **`iou50` and `iou75` measure the
extractor, not the model.** A perfect closure detector scores 0.033 at `iou50`
on TalkingFace, because a closure covers only 19-34% of the event it belongs to.
`event/any/f1` is the headline; the IoU criteria are diagnostics of the
convention mismatch.

## Thresholds: universal and tuned

Two numbers are reported for every corpus, because neither alone is honest.

| | fitted on | applied to | what it means |
|---|---|---|---|
| **universal** | pooled validation of the *training* corpora | every corpus | the deployable number; the only one available to a corpus with no validation split |
| **tuned** | that corpus's own test FROC curve | that corpus | an oracle upper bound, never a reproducible operating point |

The gap between them is the cost of tuning, made visible. Measured on RN15: the
universal point of 0.01 scores `any/f1 = 0.721`, while the corpus's own optimum
at 0.51 reaches 0.820.

**TalkingFace forces this design.** It ships 0 train and 0 valid windows -- one
~200 second recording of one subject -- so it can only ever borrow a threshold.
Scoring it at a borrowed point while scoring RN15 at a tuned one would put the
two on different scales.

The universal point is fitted on RN15 + RN30 only, and the exclusions are
structural rather than a preference:

| corpus | why it cannot fit the universal threshold |
|---|---|
| CEW, MRL-Eye | annotate no blinks at all -- nothing to fit an *event* threshold on |
| TalkingFace | ships 0 validation windows |
| HUST-LEBW, MPEblink | eval-only; fitting on them would leak evaluation data into the operating point and void the zero-shot claim |

That leaves RN15 + RN30 as the only qualifying set. Round 2 moves MPEblink into
training, at which point it joins the pool and its 20 558 validation windows
dominate the fit.

**Both numbers are always reported; there is no flag.** `make
eval-frame-wise-split` runs `fit-universal-threshold` first (a prerequisite),
which writes one operating point to `results/frame-wise/universal/`, and every
corpus then reads it through `--shared-threshold-dir`. The tuned point is
computed alongside it from each corpus's own test FROC curve.

Without the shared directory each process fits privately, which is what produced
0.01, 0.06, and two silent 0.50 fallbacks in one split run -- four different
scales in a column labelled "universal".

### Boundary hits

A fit that lands on the first or last swept value is not an optimum, it is a
search that ran out of range -- or a flat objective buying recall by firing at
everything. This is logged as a warning, because it was previously silent and
cost 7-10 points of event F1 on RN15 and RN30 without any indication.

The sweep is `linspace(0.01, 0.99, 99)`. The original `linspace(0.05, 0.95, 19)`
was too coarse to tell a fitted optimum from a boundary hit.

## Hysteresis

`to_intervals` accepts an optional `low_threshold`. A run must **peak** above the
high threshold to count as an event, but **extends** while it stays above the low
one.

A single cut has to be low enough to catch the shallow start of a closure and
high enough not to fire on noise, and no single value does both. Two thresholds
separate those jobs: the high one decides *whether* this is a blink, the low one
decides *where* it begins and ends. Since the eye-state signal ramps down through
closing, plateaus while closed, and ramps back up through opening, the ramps
cross the low threshold well before the high one -- so hysteresis recovers the
true onset and offset rather than clipping to the deepest part of the closure.

It also makes an **incomplete blink** -- one crossing low but never high --
explicitly detectable rather than silently truncated.

Off by default (`low_threshold=None` reproduces the single cut exactly).

## Signal regularisers

Two annotation-free constraints, both weighted `0.0` by default:

- **`temporal_smoothness`** penalises frame-to-frame chatter. A blink is
  continuous motion, and a jagged signal makes extraction brittle: every
  spurious crossing of the operating point becomes a spurious event boundary.
- **`duration_prior`** penalises predicted closures far outside ~100-400 ms
  (about 3-12 frames at 30 fps), as a differentiable soft count.

Neither needs a label, so both apply to every corpus -- including those
annotating only blink presence.

## What the annotations allow

| corpus | eye_state | blink_presence | blink_id | role |
|---|---|---|---|---|
| CEW | yes | no | no | frame-wise only |
| MRL-Eye | yes | no | no | frame-wise only |
| RN15 | yes | yes | yes | trainable, fits a threshold |
| RN30 | yes | yes | yes | trainable, fits a threshold |
| TalkingFace | yes | yes | yes | eval-only (0 valid windows) |
| HUST-LEBW | **no** | yes | yes | eval-only, no eye-state head |
| MPEblink | **no** | yes | yes* | eval-only, no eye-state head |

\* MPEblink carried `-1` everywhere until `number_events` was added. Its
annotation ships `[start, end, category]` intervals, so the events were always
individuable -- they were simply never numbered. See below.

Routing a corpus wrongly is loud, not silent: HUST-LEBW and MPEblink raise
`BatchError: Batch has no target 'eye_state'`, and TalkingFace raises
`DataModuleError: No 'valid' split available`.

## MPEblink event numbering

`rasterise` answers "is frame *t* inside a blink". `number_events` answers
"**which** blink", and `blink_ids` carries that per frame while `blink_id`
remains one id per window.

This is what makes the following definable on the corpus that is 80% of the
test split:

- telling a **double blink** from one long closure,
- recognising an **incomplete blink** as its own event,
- supervising an **event count** rather than a frame label.

The per-frame form matters because the window-level scalar cannot distinguish
one long blink from two adjacent ones inside the same window.

## Multi-task learning: what is and is not supportable

Only RN15 + RN30 carry `eye_state`, `blink_id`, and a train split together --
about **83 distinct blink events**. That is a very thin signal for a dedicated
event head, and the predictable failure is that the head memorises those two
corpora's blink-boundary convention rather than learning event structure, which
would void the cross-corpus claim.

The defensible version is **cross-consistency losses on the existing frame-wise
output** (above): constraints on how the signal behaves over time, supervised
where the annotation exists but benefiting every corpus.

A genuine second head becomes reasonable once MPEblink's events are numbered,
which turns 83 supervisable events into thousands. That is the point at which
the multi-task experiment is worth running rather than a memorisation risk.

## Open: deriving the state taxonomy

Open / closing / closed / opening is not annotated by any corpus. It is
derivable as weak supervision -- the sign of the smoothed derivative separates
closing from opening, and the plateau between them is closed -- which would give
incomplete and double blinks as first-class categories and connect to downstream
states such as drowsiness or sustained attention.

This needs its own validation strategy, since there is no ground truth to score
it against. It is deliberately **not** part of the current benchmark.

## Should the continuous signal be derived or learned?

**Learned.** Train with sigmoid + BCE on the binary frame label and let the
continuity emerge; do **not** synthesise a soft target from `blink_id`.

Deriving a graded target from interval position -- a ramp peaking at the
interval centre, say -- would *assert* a closure shape rather than measure one.
It is wrong on two counts: real blinks close faster than they open, so the shape
is not symmetric, and interval boundaries are annotator-defined rather than
physical.

The continuity arrives anyway, for a measurable reason. `P(closed | blink)` is
19-34%, so frames inside an annotated event are labelled inconsistently across
the ramps. A model minimising BCE over those frames converges toward the
empirical probability that a frame like this one is labelled closed -- which is
a graded closure signal grounded in annotation statistics rather than an assumed
curve.

`blink_id` earns its place as a **grouping key**, not a target: it says which
frames belong to one event, which is what lets event counts be supervised, curve
shapes be measured per event, and double blinks be separated.

If the event head later wants phase supervision, derive phase from the
**predicted** signal's derivative (a measurement), never from interval geometry
(an assumption).

## Phase and event shape

`blinklinmult.train.phases` reads phase off the signal's derivative:

| phase | condition |
|---|---|
| `OPEN` | still, low |
| `CLOSING` | falling |
| `CLOSED` | still, high |
| `OPENING` | rising |

The value alone is ambiguous -- 0.5 is an eye halfway shut *or* halfway open --
but the direction is not. This is why per-timestep context matters: a temporal
model can assign phase to a frame whose value carries none.

`describe_event` measures peak depth, the number of local maxima, and the
fall/rise asymmetry; `classify_event` names the result `complete`, `incomplete`,
`double`, `long`, or `brief`.

### Measured: no threshold pair separates these

On a synthetic signal containing one complete blink, one incomplete closure
(peak 0.45), one double blink, and one long closure:

| extractor | events found | classified |
|---|---:|---|
| high 0.50, no hysteresis | 4 | complete, complete, complete, double |
| high 0.50, low 0.20 | 4 | complete, complete, complete, double |
| high 0.35, low 0.15 | 5 | complete, **double**, complete, complete, double |

At 0.50 the incomplete blink is invisible -- its peak never reaches the
threshold. Lowering the threshold to 0.35 finds it, but the *complete* blink is
then misread as a double, because plateau noise creates spurious local maxima.

**No threshold pair gets both right.** That is the concrete argument for a
learned head over the sequence: it reads the whole shape rather than counting
crossings, so it can separate a shallow closure from a noisy plateau in a way no
extraction rule expresses.

Note this is a limitation of the *extractor*, not of the model -- the signal
contained all four events correctly.

## Threshold selection

The operating point is chosen from a **smoothed** validation F1 curve rather
than a raw `argmax`. Validation yields few events after rasterising -- a few
dozen on RN15 -- so the raw maximum lands wherever the noise peaks: it selected
0.01 and 0.06 while the same corpora's test optima were 0.51 and 0.65, costing
10 and 20 points of event F1.

Smoothing prefers a broad maximum to a tall narrow one, which is the point more
likely to survive the move from validation to test. It cannot manufacture a good
threshold where the signal has none.

## Reproducing the analysis

Every finished run writes `test_signals.npz`: the reassembled per-recording
signal, ground truth, and coverage mask. Extractor comparisons, phase
derivation, and curve-shape statistics all run from that file in seconds,
without re-running inference over the split.
