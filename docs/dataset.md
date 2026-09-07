# BlinkLinMulT v2 datasets

Derived eye crops and blink/eye-state labels for six public blink corpora,
brought to one format for joint multi-task training. Built by
[BlinkLinMulT](https://github.com/fodorad/BlinkLinMulT); see that repository for
the code, the rebuild recipe, and the training configurations.

**Licensing.** Only *derived* features are published here — never the source
corpora, whose licences do not permit re-hosting. Each source corpus retains its
own licence and access procedure; obtain the raw data from its original authors
before using these derivatives in a way that requires it.

Six corpora, each processed independently into a uniform tree and then collected
into one HDF5 file. This page covers obtaining the raw data, rebuilding the
processed trees, and the format the training loop actually reads.

## The corpora

| Corpus | Content | Blink presence | Eye state | Native `T` |
|---|---|:---:|:---:|---|
| **RN30** | 8 recordings, 4 subjects, 30 fps | ✅ | ✅ | 15 |
| **TalkingFace** | 1 recording, 1 subject, 30 fps | ✅ | ✅ | 15 |
| **RN** (Researcher's Night) | recordings at 15 and 30 fps | ✅ | ✅ | 15 |
| **CEW** | 2 423 face stills, 100×100 | — | ✅ | 1 |
| **MRL-Eye** | ~85 000 infrared eye crops, 37 subjects | — | ✅ | 1 |
| **HUST-LEBW** | short clips from 20 films, unconstrained | ✅ | — | 15 |

RT-BENE / RT-GENE and ZJU were used in the 1.x paper and are **not** part of the
v2 benchmark.

Each corpus is declared once, in `config/data/<name>.yaml`, and that declaration
drives the builder, the OmniLoader schema, and the training loop alike — so a
corpus cannot be written under one interpretation and read under another.

### Why the corpora differ, and why that matters

The two tasks are not both annotated everywhere. Training a single model across
all six is only correct because every target carries a **validity mask**: a CEW
sample arrives with `blink_presence` masked out, and the loss and the metrics
both skip those positions. Without that, the blink head would be trained to
predict a placeholder on every still image in the corpus — a run that looks
perfectly healthy and has quietly learned the wrong thing.

Two corpora carry conventions worth knowing:

- **MRL-Eye encodes `0 = closed, 1 = open`**, the inverse of this project's
  "is the eye closed" label. The filename decoder negates it, and a test pins
  that behaviour, because getting it backwards would invert the largest corpus
  in the benchmark.
- **MRL-Eye supplies one eye per image** and does not say which. The sample is
  written with that crop in both eye slots — honest about the corpus asserting
  one label for one visible eye.

## Obtaining the raw data

Place each corpus under `data/raw/<name>/` in its original layout; the raw data
is **not** redistributed here, and each corpus has its own licence and access
procedure. Request access from the original authors.

```
data/raw/
  RN30/<id>/<rec>.avi, .tag, .txt
  TalkingFace/talking/talking.avi, .tag, .txt
  RN/{train,val,test}/rn{15,30}/<id>/<rec>.avi, .tag, .txt
  CEW/dataset_B_FacialImages/{ClosedFace,OpenFace}/, EyeCoordinatesInfo_*.txt
  MRL-Eye/mrlEyes_2018_01/s0001/…
  HUST-LEBW/{train,test}/{left,right}/{blink,noblink}/<clip>/
```

The three video corpora share the `.tag` annotation format — a `#start`/`#end`
block of colon-separated per-frame records, parsed by
`blinklinmult.preprocess.annotation`.

## Rebuilding from the raw corpora

These files are the *output* of a two-stage pipeline (preprocess, then build).
The recipe lives in the repository so it stays in step with the code:

```bash
git clone https://github.com/fodorad/BlinkLinMulT && cd BlinkLinMulT
make preprocess       # install the raw-data stack
make preprocess-all   # data/raw -> one .h5 per corpus
```

Every file embeds the git SHA and the builder config that produced it, so a file
downloaded from here traces back to the code that made it.

## The built format

One HDF5 file per corpus, in the layout `omniloader.HDF5Dataset` reads:

```
/                                    attrs: schema_version, dataset, created_utc,
                                            git_sha, builder_config_yaml,
                                            time_dim, image_size, fps
/<subset>/<key>/eye_image            (T, 2*3*H*W) float16
/<subset>/<key>/eye_feature          (T, F)       float16   [if provided]
/<subset>/<key>/blink_presence       (T,)         float32   [if annotated]
/<subset>/<key>/eye_state            (T, 2)       float32   [if annotated]
/<subset>/<key>/{key,dataset,video_id}            utf-8
```

Three decisions worth stating:

**Images are stored flattened.** OmniLoader describes values structurally as
vectors or sequences; the natural `(T, 2, 3, H, W)` is neither. Flattening the
image axes into the feature axis lets its padding, cropping, masking, and mixing
machinery treat the crops like any other sequence feature instead of needing a
special case. The image axes are restored at the model boundary.

**Images are `float16`.** The eye crops dominate the file — a 15×2×3×64×64
window is 368k values against 15 labels — and fp16 halves both the file and the
read bandwidth at a precision far finer than 8-bit pixel data carries anyway.

**Metadata uses OmniLoader's names.** `key`, `dataset`, and `subset` are exactly
what its unifier copies verbatim into the batch; a sample id stored under any
other name would be dropped, and every prediction would lose its provenance.

**One file per corpus, not one for all.** OmniLoader's job *is* to mix disjoint
datasets, and it takes one dataset object per corpus. Separate files mean a
corpus can be rebuilt, re-published, or dropped from a run without touching the
others.

## Using these files

```bash
pip install blinklinmult
```

```python
from omniloader import HDF5Dataset

dataset = HDF5Dataset("rn30.h5", subset="train")
sample = dataset[0]  # eye_image, blink_presence, eye_state, key, dataset
```

In a full training run they are loaded, unified, and mixed by
`blinklinmult.data.datamodule.BlinkDataModule`; see the repository's training
guide.
