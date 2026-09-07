# Data

Six corpora, each processed independently into a uniform tree and then collected
into one HDF5 file. This page covers obtaining the raw data, rebuilding the
processed trees, and the format the training loop actually reads.

## The corpora

| Corpus | Content | Blink presence | Eye state | fps | `T` at 1.5 s |
|---|---|:---:|:---:|---|---|
| **TalkingFace** | 1 recording, 1 subject | ✅ | ✅ | 25 | 38 |
| **RN15** | Researcher's Night, 15 fps half | ✅ | ✅ | 15 | 22 |
| **RN30** | Researcher's Night, 30 fps half | ✅ | ✅ | 30 | 45 |
| **CEW** | 2 423 face stills, 100×100 | — | ✅ | — | 1 |
| **MRL-Eye** | ~85 000 infrared eye crops, 37 subjects | — | ✅ | — | 1 |
| **HUST-LEBW** | short clips from 20 films, unconstrained | ✅ | — | 30 | 13 |
| **MPEblink 2.0** | 921 untrimmed film clips, up to 24 people | ✅ | — | 25 | 38 |

RT-BENE / RT-GENE and ZJU were used in the 1.x paper and are **not** part of the
v2 benchmark.

Each corpus is declared once, in `config/data/<name>.yaml`, and that declaration
drives the builder, the OmniLoader schema, and the training loop alike — so a
corpus cannot be written under one interpretation and read under another.

### One sample is one eye

A sample carries a window of one eye — its crops, its descriptors, its labels —
with an `eye_side` of `left`, `right`, or `unknown`. That follows the
annotation: the `.tag` format marks closure per eye (`LE_FC`, `RE_FC`), so
eye-wise samples train on the actual ground truth rather than on a label
manufactured by collapsing two.

It also makes the awkward corpora honest. MRL-Eye supplies one eye and does not
say which; CEW labels the whole face. Neither has to be forced into a two-eye
slot that does not exist.

**Evaluation is frame-wise.** The benchmark reports one decision per frame, so
the two eye-wise predictions of a frame are recombined by `max` — a blink is a
frame-level event that occurred, and if one eye is occluded while the other is
plainly closing, `max` recovers it where `mean` would halve the score. `mean` is
logged alongside so the choice is measured. The same aggregation runs on
validation, so model selection optimises the reported number.

### The analysis window is declared in seconds

`window_seconds: 1.5` is the knob; each corpus derives its own frame count from
its rate — 45 frames at 30 fps, 22 at 15 fps, 38 at 25 fps. A fixed frame count
would make a 15 fps window span twice the wall-clock time of a 30 fps one, so
the model would have to learn two different notions of how fast a blink is.

In a joint run OmniLoader pads the shorter corpora to the longest frame count
and masks the padding. **Nothing is resampled**: interpolating a 15 fps clip up
to 30 fps fabricates frames that were never captured, and a blink is only a
handful of frames long. The 15 fps material genuinely carries less temporal
resolution; masking says so, resampling would hide it.

This is also what lets one model handle in-the-wild video at whatever rate it
arrives — 24, 25, 30 fps all map onto the same duration.

### Why the corpora differ, and why that matters

The two tasks are not both annotated everywhere. Training a single model across
all of them is only correct because every target carries a **validity mask**: a CEW
sample arrives with `blink_presence` masked out, and the loss and the metrics
both skip those positions. Without that, the blink head would be trained to
predict a placeholder on every still image in the corpus — a run that looks
perfectly healthy and has quietly learned the wrong thing.

Two corpora carry conventions worth knowing:

- **MRL-Eye encodes `0 = closed, 1 = open`**, the inverse of this project's
  "is the eye closed" label. The filename decoder negates it, and a test pins
  that behaviour, because getting it backwards would invert the largest corpus
  in the benchmark.
- **MRL-Eye supplies one eye per image** and does not say which, so its samples
  carry `eye_side: unknown` rather than a guess. A per-side analysis can exclude
  it on that basis.

## Obtaining the raw data

Place each corpus under `data/raw/<name>/` in its original layout; the raw data
is **not** redistributed here, and each corpus has its own licence and access
procedure. Request access from the original authors.

```
data/raw/
  TalkingFace/talking/talking.avi, .tag, .txt
  RN/{train,val,test}/rn{15,30}/<id>/<rec>.avi, .tag, .txt   # -> rn15, rn30
  CEW/dataset_B_FacialImages/{ClosedFace,OpenFace}/, EyeCoordinatesInfo_*.txt
  MRL-Eye/mrlEyes_2018_01/s0001/…
  HUST-LEBW/{train,test}/{left,right}/{blink,noblink}/<clip>/
  mpeblink2.0/{train,val,test}/<video>/video.mp4, annotation_WFLW.json
```

The three video corpora share the `.tag` annotation format — a `#start`/`#end`
block of colon-separated per-frame records, parsed by
`blinklinmult.preprocess.annotation`.

## Rebuilding

**One command per dataset, one file out.**

```bash
make preprocess            # install the raw-data stack (opencv, exordium)
make preprocess-rn30  # data/raw -> data/processed/rn30/rn30.h5
make preprocess-all        # every corpus
```

A run decodes the video, reads the `.tag` annotation, locates both eyes from the
annotated corners, extracts the 160-d descriptors, cuts the windows, and writes
the HDF5 — in a single pass, with nothing staged in between.

The frames are decoded **once** per recording. Sliding evaluation windows
overlap, so extracting per window would run the face detector repeatedly over
the same pixels, and that call is essentially the whole runtime.

It is slow (~30 ms/frame on CPU) and **not resumable**: a corpus is either built
or absent. Two flags help:

```bash
make preprocess-rn30 ARGS="--device 0"       # extract on a GPU
make preprocess-rn30 ARGS="--no-features"    # image-only, much faster
```

An image-only corpus still trains BlinkLinT and BlinkCNN; only the
cross-modal BlinkLinMulT needs the descriptors.

**CEW and MRL-Eye carry no descriptors at all.** They train the frame-wise
model, which reads eye crops alone, so extracting 160 dimensions for them would
cost hours to produce a tensor nothing ever reads.

MRL-Eye is large enough to be worth limiting during a first pass:

```bash
make preprocess-mrl ARGS="--limit-per-subject 200"
```

### Splits are assigned by recording, never by window

Windows cut from one recording overlap and share a subject; splitting them at
random puts near-copies of the same frames on both sides of the train/test
boundary and inflates every score. Splits are therefore assigned by hashing the
**group** — the recording or the participant — so a group lands wholly in one
split, reproducibly and without storing an assignment file.

Two corpora override this:

- **RN** ships its own train/val/test division by participant. It is honoured
  rather than re-derived, so published RN numbers stay comparable.
- **TalkingFace** is a single recording of a single subject and is held out
  **entirely as a test set**. A train/test split within one subject's continuous
  footage measures memorisation, not generalisation; TalkingFace exists in the
  benchmark to answer "does a model trained elsewhere work here?".

### One sampling rule, for every split

Every split — training included — is a continuous **50%-overlapping sweep** of
the recording. That is the condition the model is deployed under, and it is the
protocol the literature evaluates with, so training and evaluation are the same
process rather than two.

1.x sampled training differently: one window *centred* on each blink, plus a
proportional number of blink-free ones. It was removed for two measured
reasons.

- **The prior did not match.** Balanced sampling fit the model at **50%
  positive** while the swept evaluation splits sat near **10%** — measured at
  50.0 / 17.5 / 9.9 on RN15 and 50.0 / 14.4 / 9.4 on RN30. A decision
  threshold calibrated on one is wrong on the other.
- **Every training blink sat at the window's centre.** The model never saw a
  closure cut by a window edge, which is most of what a sweep produces.

The natural class prior now reaches the loss, which is what `focal` — the
default for blink presence — exists to handle. A sweep also has no RNG in it, so
two builds of one corpus agree exactly.

The cost is that a window is no longer guaranteed to hold at most one blink:
two blinks half a second apart genuinely land together (9 of 713 on
TalkingFace). `first_blink_id` records the earlier event rather than refusing
the sample.

## The built format

One HDF5 file per corpus, in the layout `omniloader.HDF5Dataset` reads:

```
/                                    attrs: schema_version, dataset, created_utc,
                                            git_sha, builder_config_yaml,
                                            time_dim, window_seconds, image_size, fps
/<subset>/<key>/eye_image            (T, C, H, W) float16
/<subset>/<key>/eye_feature          (T, F)      float16   [if provided]
/<subset>/<key>/eye_feature_mask     (T,)        bool      [if provided]
/<subset>/<key>/blink_presence       (T,)        float32   [if annotated]
/<subset>/<key>/eye_state            (T,)        float32   [if annotated]
/<subset>/<key>/{key,dataset,video_id,eye_side,frame_group}  utf-8
```

The sample key is `<video_id>|<frame_group>|<eye_side>`. The two eyes of one
window share a `video_id` and `frame_group` and differ only in the side, which
is how the frame-level evaluation finds the pair.

Four decisions worth stating:

**Images are stored as images.** OmniLoader ≥ 1.1 lets a spec declare a
structured trailing `shape=(C, H, W)`, so the crops are stored, padded, masked,
mixed and batched in native form — the array written is the array the model
receives. Earlier versions only understood a flat `feature_dim`, which forced a
flatten-on-write / reshape-on-read round trip and a schema that described a
12288-wide feature rather than an image.

**Images are `float16`.** The eye crops dominate the file — a 15×3×64×64 window
is 184k values against 15 labels — and fp16 halves both the file and the read
bandwidth at a precision far finer than 8-bit pixel data carries anyway.

**Features carry their own mask.** A frame whose eye crop is perfectly usable
can still defeat the iris landmarker — blur, partial occlusion, a low-confidence
fit. `eye_image` and `eye_feature` therefore have independent validity masks and
no code assumes they agree.

**Metadata uses OmniLoader's names.** `key`, `dataset`, and `subset` are exactly
what its unifier copies verbatim into the batch; a sample id stored under any
other name would be dropped, and every prediction would lose its provenance.

**One file per corpus, not one for all.** OmniLoader's job *is* to mix disjoint
datasets, and it takes one dataset object per corpus. Separate files mean a
corpus can be rebuilt, re-published, or dropped from a run without touching the
others.

## Publishing and fetching

Each built file is published to a public Hugging Face dataset repo, so a run
reproduces without re-running the multi-hour preprocessing:

```bash
make push-all      # publish (maintainer)
make pull-all      # fetch (anyone)
```

Only derived eye crops and labels are published — never the source corpora,
whose licences do not permit re-hosting. Confirm a corpus's licence before
adding it to the push targets.

The dataset card is generated from `docs/dataset.md` (`make push-hf-card`), so
the Hugging Face page and this repository cannot drift apart.

Each file embeds its git SHA and the config that produced it, so a pulled file
traces back to the code that made it. `make preprocess-all` remains the
from-scratch rebuild.

Hugging Face is used instead of DVC, which cannot push to an HF remote and whose
payoff is a paid cloud remote this project does not have.
