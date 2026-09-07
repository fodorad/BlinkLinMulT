.PHONY: docker-build docker-build-demo docker-run docker-run-demo docker-push serve app export-paper-onnx export-blinkcnn push-model push-model-card push-space push-corpus pull-corpus push-dataset-card help install dev install-docs train preprocess fix lint type-check test test-full docs docs-serve docs-deploy check check-full check-ci clean \
        export-blinkcnn-onnx compare-models compare-shipped score-models benchmark-runtime benchmark-streaming \
        webcam video-demo \
        preprocess-talkingface preprocess-rn15 preprocess-rn30 \
        preprocess-cew preprocess-mrl preprocess-hust-lebw preprocess-mpeblink preprocess-all \
        push-talkingface push-rn15 push-rn30 push-cew push-mrl push-hust-lebw push-mpeblink push-all \
        pull-talkingface pull-rn15 pull-rn30 pull-cew pull-mrl pull-hust-lebw pull-mpeblink pull-all push-hf-card \
        train-joint train-blink-presence train-eye-state train-cnn train-lint \
        train-linmult train-smoke train-single per-dataset \
        train-encoder train-lint-rn15 \
        train-fw-bce train-fw-focal train-fw-bce-augment \
        train-fw-focal-augment train-frame-wise-mpeblink eval-frame-wise \
        eval-frame-wise-split fit-universal-threshold report-frame-wise \
        video-lint-scratch video-lint-frozen video-lint-ft \
        video-mult-scratch video-mult-frozen video-mult-ft video-arms \
        eval-video report-video video-lint-unfreeze video-mult-unfreeze \
        build-cache show-cache clean-cache video-lint-frozen-cached \
        ablation ablation-stage0 ablation-stage1 ablation-stage2 ablation-stage3 \
        ablation-stage4 ablation-lint ablation-report mlflow-ui

help:
	@echo "Dev (modify files):  fix"
	@echo "Checks (read-only):  lint | type-check | test | docs | check"
	@echo "  check   fast gate, skips the ~49s Lightning runs"
	@echo "  check-full / test-full   everything, incl. real training runs"
	@echo "  check-ci                 same steps in a CI-equivalent venv (run before pushing)"
	@echo "Setup:               install | dev | install-docs | train | preprocess"
	@echo "Docs:                docs-serve | docs-deploy"
	@echo "Build a corpus:      preprocess-<db> | preprocess-all"
	@echo "Artifacts (HF):      push-<db> | pull-<db> | push-all | pull-all | push-hf-card"
	@echo "Deploy the demo:     push-space (then factory-reboot to change python_version)"
	@echo "Training:            train-joint | train-blink-presence | train-eye-state"
	@echo "Shipped models:      train-cnn | train-lint | train-linmult |"
	@echo "Quick check:         train-encoder (CEW) -> train-lint-rn15"
	@echo "Experiments:         train-single DB=<db> | per-dataset | ablation | mlflow-ui"
	@echo "Frame-wise arms:     train-fw-bce | train-fw-focal | train-fw-bce-augment | train-fw-focal-augment"
	@echo "Frame-wise eval:     eval-frame-wise-split (default) | eval-frame-wise (single process)"
	@echo "Video arms:          video-arms | video-{lint,mult}-{scratch,frozen,ft}"
	@echo "Video eval:          eval-video [ARMS=\"...\"] | report-video"
	@echo "Two-stage:           video-lint-frozen then video-lint-unfreeze"
	@echo "Embedding cache:     build-cache | show-cache | clean-cache | video-lint-frozen-cached"
	@echo "Cleanup:             clean"
	@echo ""
	@echo "  <db> is one of: talkingface rn15 rn30 cew mrl hust-lebw mpeblink"

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv sync

install-docs:
	uv sync --extra docs

# Everything needed to build the h5 files and train: Lightning, MLflow, HF Hub.
train:
	uv sync --extra train

# The raw-data stack (opencv, exordium). Only needed to rebuild
# data/processed/<db> from data/raw/<db>.
preprocess:
	uv sync --extra preprocess

dev:
	uv sync --all-extras

# ── Dev helpers (modify files) ─────────────────────────────────────────────────

fix:
	uv run ruff format .
	uv run ruff check --fix .

# ── Checks (read-only — mirrors GitHub CI) ─────────────────────────────────────

# The extras the checks need, matching what CI installs. Without `train` the
# suite errors on missing pandas/mlflow rather than failing honestly. The
# `preprocess` extra stays out: it pulls exordium and its multi-GB weights. The
# one piece of it the tests do need is opencv -- tests/preprocess/test_stream.py
# and test_hust_crops.py write a real video and JPEG as fixtures and decode them
# back. A full dev environment gets cv2 transitively, which is exactly why
# `check` alone cannot prove CI will pass; see `check-ci`.
# `onnx` is included so the 1.x graph tests run rather than skip: those models
# ship as ONNX only, so without it their entire inference path is untested.
# `serve` is in the gate because tests/demos/test_serve.py covers the REST
# service; without it those 15 tests skip rather than fail, which is worse.
EXTRAS := --extra dev --extra train --extra onnx --extra serve --extra compare

lint:
	uv run $(EXTRAS) ruff check .
	uv run $(EXTRAS) ruff format --check .

type-check:
	uv run $(EXTRAS) ty check blinklinmult

# The fast gate. Skips the real Lightning training runs, ~49s of the ~85s
# suite. Nothing here touches the network: the whole suite passes under
# HF_HUB_OFFLINE=1, and every model test reads the local graphs in $(ONNX_DIR).
# `python -m tests`, not `unittest discover`. Same tests, different exit path:
# tests/__main__.py sets the discovery root to the repo (so `demos.docker.serve`
# and `tools.*` import rather than silently skipping -- that hid 15 service
# tests once) and exits without C-level finalisation, which aborted roughly one
# run in twelve *after* every test had passed. See that module's docstring.
test:
	uv run $(EXTRAS) coverage run -m tests -v
	uv run $(EXTRAS) coverage report
	uv run $(EXTRAS) coverage html
	uv run $(EXTRAS) coverage xml -o coverage.xml

# Everything, including the real Lightning training runs. What CI runs, and what
# to run before opening a PR that touches the training stack.
test-full:
	RUN_TRAINING_TESTS=1 uv run $(EXTRAS) coverage run -m tests -v
	uv run $(EXTRAS) coverage report
	uv run $(EXTRAS) coverage html
	uv run $(EXTRAS) coverage xml -o coverage.xml

docs:
	uv run $(EXTRAS) --extra docs sphinx-build -b html -W docs/ site/

check: lint type-check test docs

# The full gate: `check` plus the training runs it skips.
check-full: lint type-check test-full docs

# ── CI parity ──────────────────────────────────────────────────────────────────
#
# `check` runs against the developer's environment, which is a strict SUPERSET of
# CI's: a full local install pulls opencv and exordium in transitively, so tests
# and type-checks that depend on them pass here and fail on the runner. That gap
# is structural -- no amount of care with `check` closes it, because the missing
# pieces are absent only on the runner.
#
# `check-ci` closes it by building a throwaway virtualenv with CI's EXACT install
# line, then running CI's exact steps against it, in CI's order. Green here means
# green there, for everything a single machine can decide.
#
# What it deliberately does NOT reproduce, because one machine cannot:
#   * the 3.12/3.13 x ubuntu/macos matrix -- this runs one interpreter, so a
#     version- or platform-specific break still needs the runner to surface it;
#   * the Hub fetch of the published ONNX graphs. CI downloads them into
#     $(ONNX_DIR); this target uses whatever is already there. With the graphs
#     absent the model tests skip, so the run is weaker than CI rather than
#     wrong -- the summary below says so explicitly.
#
# Run it before pushing anything that touches dependencies, imports or CI config.
CI_EXTRAS  := train,dev,docs,onnx,serve,compare
CI_VENV    := .venv-ci
CI_PY      ?= 3.13

check-ci:
	@echo "── Building CI-equivalent environment ($(CI_VENV), python $(CI_PY)) ──"
	@rm -rf $(CI_VENV)
	@# `--seed` installs pip: `uv venv` omits it, but pip-audit shells out to
	@# `python -m pip` to enumerate what is installed, exactly as it does on the
	@# runner, where setup-python provides pip.
	@uv venv $(CI_VENV) --python $(CI_PY) --seed >/dev/null
	@VIRTUAL_ENV=$(CI_VENV) uv pip install --quiet \
	  -e ".[$(CI_EXTRAS)]" opencv-python-headless
	@echo "── Audit dependencies ──"
	@# PIPAPI_PYTHON_LOCATION so pip-audit inspects the CI venv rather than the
	@# interpreter uvx runs itself under, which would audit the wrong tree.
	@# Non-fatal, matching ci.yml's `continue-on-error: true`: the ML stack
	@# regularly carries advisories with no fixed release, and failing on one
	@# would block every push for something no upgrade can resolve. Parity means
	@# matching CI here too -- a local gate stricter than the runner is its own
	@# kind of false alarm.
	@PIPAPI_PYTHON_LOCATION=$(CURDIR)/$(CI_VENV)/bin/python \
	  VIRTUAL_ENV=$(CI_VENV) uvx pip-audit || \
	  echo "  (advisories above are a warning, as in CI)"
	@echo "── Ruff lint ──"
	@VIRTUAL_ENV=$(CI_VENV) uv run --no-project ruff check .
	@echo "── Ruff format ──"
	@VIRTUAL_ENV=$(CI_VENV) uv run --no-project ruff format --check .
	@echo "── Type check (ty) ──"
	@VIRTUAL_ENV=$(CI_VENV) uv run --no-project ty check blinklinmult
	@echo "── Tests (same env flags as CI) ──"
	@RUN_TRAINING_TESTS=1 RUN_PACKAGING_TESTS=1 \
	  VIRTUAL_ENV=$(CI_VENV) uv run --no-project coverage run -m tests -v 2>&1 \
	  | tee $(CI_VENV)/test.log | tail -3
	@VIRTUAL_ENV=$(CI_VENV) uv run --no-project coverage report
	@echo "── Docs (warnings as errors) ──"
	@VIRTUAL_ENV=$(CI_VENV) uv run --no-project sphinx-build -b html -W docs/ site/
	@echo ""
	@echo "── Skipped tests ──"
	@grep -oE "skipped '[^']+'" $(CI_VENV)/test.log | sort | uniq -c \
	  || echo "  none"
	@echo ""
	@echo "✓ CI parity check passed -- the runner should agree."

# ── Docs ───────────────────────────────────────────────────────────────────────

docs-serve:
	uv run $(EXTRAS) --extra docs sphinx-autobuild docs/ site/

docs-deploy:
	@echo "Docs are deployed automatically via GitHub Actions on push to main."

# ── Preprocess: raw corpora -> data/processed/<db>/<db>.h5 ────────────────────
# One command per dataset, one file out. A run decodes the video, reads the
# .tag annotation, crops both eyes, extracts the 160-d descriptors, cuts the
# windows, and writes the h5 -- no staging tree, no extracted-frame directory,
# no second step.
#
# Slow: the face detector and iris landmarker run once per frame (~30ms on CPU).
# Not resumable -- a corpus is either built or absent. Change the config and
# rebuild.
#
# Requires `make preprocess` first. See docs/data.md for obtaining data/raw.
#
# ARGS is forwarded:
#   make preprocess-mrl       ARGS="--limit-per-subject 50"   a tractable subset
#   make preprocess-rn30      ARGS="--device 0"               extract on a GPU
#   make preprocess-rn30      ARGS="--no-features"            image-only, fast

ARGS ?=

preprocess-talkingface:
	uv run python -m blinklinmult.preprocess.talkingface $(ARGS)

# Both rates in one call; pass ARGS="--rate 15" for one.
preprocess-rn15:
	uv run python -m blinklinmult.preprocess.rn --rate 15 $(ARGS)

preprocess-rn30:
	uv run python -m blinklinmult.preprocess.rn --rate 30 $(ARGS)

# Stills: eye patches only. CEW and MRL-Eye train BlinkCNN, which reads
# crops alone, so extracting descriptors for them would cost hours to produce a
# tensor nothing reads.
preprocess-cew:
	uv run python -m blinklinmult.preprocess.cew $(ARGS)

preprocess-mrl:
	uv run python -m blinklinmult.preprocess.mrl $(ARGS)

preprocess-hust-lebw:
	uv run python -m blinklinmult.preprocess.hust_lebw $(ARGS)

# MPEblink 2.0: 921 untrimmed film clips, up to 24 people each. The heaviest
# corpus here -- one video decode per clip, every tracklet's eyes cropped in the
# same pass. ARGS="--limit 5" builds a tractable subset.
# Resumable: each video writes its own shard, so re-running after an
# interruption skips the videos already built rather than starting over. The
# shards are merged into mpeblink.h5 once every video has one.
preprocess-mpeblink:
	uv run python -m blinklinmult.preprocess.mpeblink $(ARGS)

# Ordered cheapest first, so a schema mistake surfaces in minutes rather than
# after the 20-hour corpus. CEW and MRL are pure image decoding; the video
# corpora run the detector and pose stack per frame; MPEblink is 921 untrimmed
# clips and dominates the total.
preprocess-all: preprocess-cew preprocess-mrl \
                preprocess-rn15 preprocess-rn30 \
                preprocess-hust-lebw preprocess-talkingface \
                preprocess-mpeblink

# The four video corpora that predate the 1.5 s window, rebuilt to match
# MPEblink. CEW and MRL-Eye are stills (T=1) and unaffected; MPEblink is
# already current, so re-running it would cost ~13 h for an identical file.
#
# HUST-LEBW keeps its 13 frames -- the corpus ships fixed-length clips and
# there is no further footage to read. See hust_lebw.TIME_DIM.
preprocess-video-corpora: preprocess-rn15 preprocess-rn30 \
                          preprocess-talkingface preprocess-hust-lebw

# ── Artifact hosting (Hugging Face) ────────────────────────────────────────────
# Each corpus's h5 is published to a public HF dataset repo so anyone can pull it
# and train without re-running preprocessing, while `make preprocess-<db>` remains
# the from-scratch rebuild. Each h5 embeds its git SHA + builder config, so a
# pulled file traces back to the code that made it. HF is used instead of DVC
# (which cannot push to an HF remote).
#
# Only derived eye crops and labels are published -- never the source corpora,
# whose licences do not permit re-hosting. Confirm each corpus's licence before
# adding it here.
#
# HF_XET_HIGH_PERFORMANCE replaces the retired HF_HUB_ENABLE_HF_TRANSFER flag.

# The one dataset repo, PRIVATE: the h5 files store real eye-crop imagery from
# seven corpora with differing licences. See the note at the HF section below.
HF_DATASET_REPO ?= fodorad/blink_detection
H5_DIR = data/processed

define push_h5
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_DATASET_REPO) \
		$(H5_DIR)/$(1)/$(1).h5 $(1).h5 --repo-type dataset
endef

define pull_h5
	HF_XET_HIGH_PERFORMANCE=1 uv run hf download $(HF_DATASET_REPO) $(1).h5 \
		--repo-type dataset --local-dir $(H5_DIR)/$(1)
endef

push-talkingface: ; $(call push_h5,talkingface)
push-rn15:        ; $(call push_h5,rn15)
push-rn30:        ; $(call push_h5,rn30)
push-cew:         ; $(call push_h5,cew)
push-mrl:         ; $(call push_h5,mrl)
push-hust-lebw:   ; $(call push_h5,hust_lebw)
push-mpeblink:    ; $(call push_h5,mpeblink)

# mpeblink is 42 GB on its own -- excluded from push-all/pull-all so a routine
# sync is minutes, not hours. Run `make push-mpeblink` deliberately.
push-all: push-talkingface push-rn15 push-rn30 push-cew push-mrl push-hust-lebw

pull-talkingface: ; $(call pull_h5,talkingface)
pull-rn15:        ; $(call pull_h5,rn15)
pull-rn30:        ; $(call pull_h5,rn30)
pull-cew:         ; $(call pull_h5,cew)
pull-mrl:         ; $(call pull_h5,mrl)
pull-hust-lebw:   ; $(call pull_h5,hust_lebw)
pull-mpeblink:    ; $(call pull_h5,mpeblink)

pull-all: pull-talkingface pull-rn15 pull-rn30 pull-cew pull-mrl pull-hust-lebw

# Publish the HF dataset card from its single source of truth (docs/dataset.md),
# prepending the frontmatter HF renders. docs/dataset.md stays canonical, so the
# GitHub docs page and the HF card cannot drift.
HF_CARD_BUILD ?= .build/hf_dataset_card.md
push-hf-card:
	mkdir -p $(dir $(HF_CARD_BUILD))
	cat docs/_hf_frontmatter.yaml docs/dataset.md > $(HF_CARD_BUILD)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_DATASET_REPO) \
		$(HF_CARD_BUILD) README.md --repo-type dataset

# ── Training ───────────────────────────────────────────────────────────────────
# Requires the h5 files (`make preprocess-all` or `make pull-all`). Pass extra overrides
# via ARGS, e.g.
#   make train-joint ARGS="--set data.strategy_kwargs.temperature=3.0"
#
# Three models ship from this repo:
#   BlinkCNN  per-frame eye state, no sequence model
#   BlinkLinT         eye patches over one analysis window
#   BlinkLinMulT      eye patches + handcrafted descriptors, two masks

DATA_CONFIG ?= config/data/all.yaml
MODEL_CONFIG ?= config/model/blinklinmult.yaml

# Both heads at once over every corpus: the union experiment.
train-joint:
	uv run python -m blinklinmult.train.cli \
		--data $(DATA_CONFIG) --model $(MODEL_CONFIG) \
		--train config/train/joint.yaml $(ARGS)

# Blink presence alone. all.yaml is the video corpora: the still-image corpora
# have no temporal axis and belong to the frame-wise model.
train-blink-presence:
	uv run python -m blinklinmult.train.cli \
		--data config/data/all.yaml --model $(MODEL_CONFIG) \
		--train config/train/blink_presence.yaml $(ARGS)

# Eye state alone, frame-wise over every corpus that labels a single crop.
train-eye-state:
	uv run python -m blinklinmult.train.cli \
		--data config/data/stills.yaml --model config/model/blinkcnn.yaml \
		--train config/train/eye_state.yaml $(ARGS)

# ── The three shipped models ───────────────────────────────────────────────────

# BlinkCNN: eye state from single frames. Trained on the still corpora
# plus the video corpora reduced to frames at load time -- see stills.yaml.
train-cnn:
	uv run python -m blinklinmult.train.cli \
		--data config/data/stills.yaml --model config/model/blinkcnn.yaml \
		--train config/train/eye_state.yaml $(ARGS)

# BlinkLinT: sequence model over eye patches, video corpora only.
train-lint:
	$(MAKE) train-blink-presence MODEL_CONFIG=config/model/blinklint.yaml

# ── The quick sequence check ───────────────────────────────────────────────────
# rn15 is the smaller video corpus -- 6112 training windows of 8 frames against
# RN30's 13016 of 45 -- so it is the fastest honest end-to-end run and the
# right place to iterate before committing to an ablation.
#
#   make train-encoder     frame-wise BlinkCNN on CEW; writes the checkpoint
#   make train-lint-rn15   LinT on rn15, encoder initialised from that
#
# Both group under their own MLflow experiment and results/<experiment>/<run>.

train-encoder:
	uv run python -m blinklinmult.train.cli \
		--data config/data/single_cew.yaml --model config/model/blinkcnn.yaml \
		--train config/train/eye_state.yaml \
		--set train.mlflow.experiment_name=cew-frame-wise \
		--set train.mlflow.run_name=blinkcnn-convnext-s42 $(ARGS)

train-lint-rn15:
	uv run python -m blinklinmult.train.cli \
		--data config/data/single_rn15.yaml \
		--model config/model/blinklint_baseline.yaml \
		--train config/train/blink_presence.yaml \
		--set train.mlflow.experiment_name=rn15-sequence \
		--set train.mlflow.run_name=lint-baseline-s42 $(ARGS)

# BlinkLinMulT: the cross-modal model. Needs the handcrafted descriptors, so the
# corpora must have been built with them (feature_dim set in their declaration).
train-linmult:
	$(MAKE) train-blink-presence MODEL_CONFIG=config/model/blinklinmult.yaml

# ── Per-dataset experiments ────────────────────────────────────────────────────
# Each corpus trained and evaluated on its own, so the union result can be
# compared against per-corpus training. DB names a config/data/single_<DB>.yaml.

DB ?= rn15

train-single:
	uv run python -m blinklinmult.train.cli \
		--data config/data/single_$(DB).yaml --model $(MODEL_CONFIG) \
		--train config/train/joint.yaml \
		--set train.mlflow.run_name=single-$(DB) $(ARGS)

# Every per-dataset arm, then the union, so the comparison is one command.
per-dataset:
	for db in rn15 rn30 hust_lebw mpeblink; do \
		$(MAKE) train-single DB=$$db MODEL_CONFIG=config/model/blinklint.yaml; \
	done
	for db in cew mrl; do \
		$(MAKE) train-single DB=$$db MODEL_CONFIG=config/model/blinkcnn.yaml; \
	done

train-smoke:
	uv run python -m blinklinmult.train.cli \
		--data config/data/smoke.yaml --model config/model/blinklint.yaml \
		--train config/train/smoke.yaml --fast-dev-run $(ARGS)

# The architecture comparison of the paper, rerun on the v2 data: the baselines
# against the full model, all on identical corpora and splits.
ablation:
	$(MAKE) train-cnn
	$(MAKE) train-lint
	$(MAKE)
	$(MAKE) train-linmult

# ── Sequence-model ablation ────────────────────────────────────────────────────
# Sequential greedy on BlinkLinMulT over RN15, not a factorial: each axis
# is measured against a fixed baseline, the winner is frozen, and the next axis
# runs on top of it. 6 configurations x 3 seeds = 18 runs, against 96 for the
# full grid.
#
# Order is by how confounded the result can be, not by effect size. Attention
# first because it changes ZERO parameters -- a pure mechanism swap at identical
# capacity, so it cannot be confused with a capacity effect. Then TCN (+6.4%),
# then d_model (+273%, and the num_heads comparison in disguise: 32->64 moves
# each head from 4d to 8d), then depth last because it interacts most with width.
#
# Run one stage, read the table, then set the next stage's HOLD to the winner:
#
#   make ablation-stage1
#   make ablation-report
#   make ablation-stage2 HOLD="--set model.attention_type=flash"
#
# HOLD carries the frozen choices forward; ARGS is passed through as usual.

ABL_DATA  ?= config/data/single_rn15.yaml
ABL_MODEL ?= config/model/blinklinmult.yaml
ABL_TRAIN ?= config/train/blink_presence.yaml
ABL_SEEDS ?= 42 43 44
HOLD ?=

# $(1) = arm name, $(2) = the --set flags defining this arm
define abl_arm
	for seed in $(ABL_SEEDS); do \
		uv run python -m blinklinmult.train.cli \
			--data $(ABL_DATA) --model $(ABL_MODEL) --train $(ABL_TRAIN) \
			$(HOLD) $(2) --set train.seed=$$seed \
			--set train.mlflow.run_name=abl-$(1)-s$$seed $(ARGS) || exit 1; \
	done
endef

# Stage 0 is the published configuration, and every later stage compares to it.
# Run it once, first; the later stages do not depend on it as a make target
# because that would re-run the baseline every time.
ablation-stage0:
	$(call abl_arm,base,)

ablation-stage1:
	$(call abl_arm,flash,--set model.attention_type=flash)

ablation-stage2:
	$(call abl_arm,tcn,--set model.add_module_tcn=true)

ablation-stage3:
	$(call abl_arm,d64,--set model.d_model=64)

ablation-stage4:
	$(call abl_arm,L3,--set model.cmt_num_layers=3 --set model.branch_sat_num_layers=3)
	$(call abl_arm,L6,--set model.cmt_num_layers=6 --set model.branch_sat_num_layers=6)

# Confirm the chosen configuration transfers to the image-only model. LinT has
# no branches, so branch_sat_num_layers does not apply to it.
ablation-lint:
	$(MAKE) ablation-stage0 ABL_MODEL=config/model/blinklint.yaml

ablation-report:
	uv run python experiments/ablation_report.py $(ARGS)

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# ── Misc ───────────────────────────────────────────────────────────────────────

clean:
	rm -rf .venv coverage_html dist/ .pytest_cache/ site/ tmp/ .build/
	rm -f .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# ── LinT diagnostics on RN30 ──────────────────────────────────────────────────
# Short, structured runs that answer one question each, all on RN30 alone
# (15 608 samples, 4.3% positive) so the loss question is isolated from
# corpus-mixing effects. HUST-LEBW is excluded throughout: at 54.9% positive it
# labels per *clip* rather than per frame, and would pull any global alpha away
# from what the other corpora want.
#
# Every arm starts from the CEW frame-wise encoder, so they differ only in the
# variable named.

LINT_RN30 = --data config/data/single_rn30.yaml \
            --model config/model/blinklint_baseline.yaml \
            --train config/train/lint_blink.yaml

# The LR range test. Run this FIRST -- `lr: 0.001` in the config is a
# placeholder, and the 0.005 PersonalityLinMulT uses was measured on a
# different architecture and task.
lr-find-lint:
	uv run python -m blinklinmult.train.lr_find $(LINT_RN30) $(ARGS)

# The range test narrows the range; it cannot pick the value. It sweeps at a
# CONSTANT lr, while a real run anneals over max_epochs -- so a peak that looks
# fine held constant may be far too high or low once cosine is applied. Worse,
# the automatic suggestion is fooled by a local bump: on RN30 it reported
# 5.25e-06 while the loss went on falling to a minimum near 1e-02, ~200x higher.
#
# This settles it the only honest way: identical short runs that differ ONLY in
# the peak, so their schedules are directly comparable.
lr-compare-lint:
	for lr in 0.001 0.003 0.01; do \
	  uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	    --set train.optimizer.lr=$$lr \
	    --set train.mlflow.experiment_name=lint-lrcompare \
	    --set train.mlflow.run_name=lr-$$lr $(ARGS); \
	done

# Does focal's alpha help or hurt at this imbalance? alpha weights the POSITIVE
# class, so the inherited 0.25 gives blink frames a quarter weight and makes the
# effective imbalance ~42:1 against a measured 14:1. Four arms, 10 epochs each.
# Feasibility settings, not publication ones. Measured cost on this hardware is
# ~15.7 min/epoch on full RN30 (compute-bound: a batch is 32 x 45 x 2 = 2 880
# ConvNeXt forwards, while reading the whole corpus costs 16 s), so the planned
# 8 arms x 10 epochs would have run ~21 h.
#
# `limit_fit_batches` narrows train and validation to a third and leaves the
# TEST split whole -- capping test too would score each arm on a different
# subset and the comparison would mean nothing. T stays 45 and the sweep
# protocol is untouched, so what these arms measure is unchanged; only the
# amount of training behind each number is reduced.
DIAG = --set train.max_epochs=4 \
       --set train.limit_fit_batches=0.33 \
       --set train.early_stopping.patience=20 \
       --set train.optimizer.lr=0.003 \
       --set train.mlflow.experiment_name=lint-diagnose

ablation-loss:
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set train.loss_kwargs.alpha=0.25 --set train.mlflow.run_name=loss-focal-a025 $(ARGS)
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set train.loss_kwargs.alpha=0.5  --set train.mlflow.run_name=loss-focal-a050-v2 $(ARGS)
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set train.loss_kwargs.alpha=0.75 --set train.mlflow.run_name=loss-focal-a075 $(ARGS)
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set train.loss=bce --set train.mlflow.run_name=loss-bce $(ARGS)

# Does PersonalityLinMulT's HPO finding (dropout_attention 0.2 >> 0.0) transfer
# to a CNN-encoder model on a detection task? It was found on precomputed
# features regressing five traits, so this is not a given.
ablation-dropout:
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set model.dropout_attention=0.0 --set train.mlflow.run_name=drop-attn-000 $(ARGS)
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set model.dropout_attention=0.2 --set train.mlflow.run_name=drop-attn-020 $(ARGS)

# Is the CEW frame-wise encoder actually worth transferring, or would ImageNet
# do? Clearing encoder_weights starts the backbone from ImageNet instead.
ablation-encoder:
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set train.mlflow.run_name=enc-cew $(ARGS)
	uv run python -m blinklinmult.train.cli $(LINT_RN30) $(DIAG) \
	  --set model.encoder_weights=null --set train.mlflow.run_name=enc-imagenet $(ARGS)

# Everything above, in order.
diagnose-lint: ablation-loss ablation-dropout ablation-encoder

# ── Frame-wise backbone ───────────────────────────────────────────────────────
# A ConvNeXt-Femto eye-state model trained on every corpus that annotates
# per-frame closure, then benchmarked on all seven: stills scored directly,
# video corpora scored as blink events recovered from the per-frame signal,
# HUST-LEBW scored clip-wise.
#
# ROUND 1 trains on cew+mrl+rn15+rn30. ROUND 2 adds mpeblink, whose annotation
# is blink *intervals* rather than closure -- rasterising them labels the whole
# event, and only 19-34% of an event's frames show a closed eye. Whether its
# 63 758 windows of in-the-wild variety outweigh that is an open question, so
# the two rounds differ in one flag and are compared.

FRAME_WISE = --data config/data/stills_all.yaml \
             --model config/model/blinkcnn.yaml \
             --train config/train/frame_wise.yaml

# Run FIRST: `lr: 0.001` in the config is a placeholder. Read the plot, do not
# take the printed suggestion -- it locks onto local bumps.
lr-find-frame-wise:
	uv run python -m blinklinmult.train.lr_find $(FRAME_WISE) $(ARGS)

# Round 1: closure-labelled corpora only.
# Four arms over two independent axes -- loss and augmentation -- so each pair
# differs in exactly one thing and the comparison measures that thing:
#
#              no augmentation        augmentation
#   BCE        train-fw-bce           train-fw-bce-augment
#   focal      train-fw-focal         train-fw-focal-augment
#
# The loss axis is not just a loss swap. BCE pairs with subsampling
# (`still_open_to_closed=1.0` balances the classes), focal pairs with keeping
# every frame -- down-weighting easy negatives is the whole point of focal, and
# doing both would correct the imbalance twice.
# The bce comparison arm. `frame_wise.yaml` now defaults to focal, which won the
# four-arm comparison on every corpus, so this must set bce explicitly -- without
# it the arm would silently train focal and the comparison would be two identical
# runs.
train-fw-bce:
	uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	  --set train.loss=bce \
	  --set 'train.loss_kwargs={}' \
	  --set train.mlflow.run_name=fw-bce $(ARGS)

train-fw-focal:
	uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	  --set train.loss=focal \
	  --set train.loss_kwargs.gamma=2.0 \
	  --set data.still_open_to_closed=1000.0 \
	  --set train.mlflow.run_name=fw-focal $(ARGS)

train-fw-bce-augment:
	uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	  --set train.loss=bce \
	  --set 'train.loss_kwargs={}' \
	  --set data.augment.strength=1.0 \
	  --set train.mlflow.run_name=fw-bce-augment $(ARGS)

train-fw-focal-augment:
	uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	  --set train.loss=focal \
	  --set train.loss_kwargs.gamma=2.0 \
	  --set data.still_open_to_closed=1000.0 \
	  --set data.augment.strength=1.0 \
	  --set train.mlflow.run_name=fw-focal-augment $(ARGS)

# ── Video-level benchmark ──────────────────────────────────────────────────────
#
# Six arms: {BlinkLinT, BlinkLinMulT} x {scratch, frozen init, fine-tuned init}.
# Each pair differs in exactly one thing, so the table answers three separate
# questions rather than one muddled one:
#
#   1. Does the frame-wise encoder transfer?   scratch vs frozen vs fine-tuned
#   2. Do the 160-d descriptors beat RGB alone? LinT vs LinMulT
#   3. How much does temporal modelling buy?    all six vs the frame-wise arm
#
# The seed checkpoint is **fw-focal-augment**, chosen on the per-dataset event
# table rather than on a pooled mean. It is best on the established webcam
# benchmark -- rn15 0.7972 and rn30 0.7963 event F1, both the highest of the
# four arms -- and best on rn15 frame-level ESR (0.7057).
#
# No arm wins everywhere: fw-bce-augment leads on MPEblink (0.6140) and fw-focal
# on HUST-LEBW (0.8239). RN is the corpus this work is measured against, so it
# breaks the tie. Override with VIDEO_SEED=... to seed from another arm.
VIDEO = --data config/data/video_all.yaml --train config/train/video.yaml
VIDEO_SEED ?= results/frame-wise/fw-focal-augment/checkpoints/best.ckpt

video-lint-scratch:
	uv run python -m blinklinmult.train.cli $(VIDEO) \
	  --model config/model/blinklint.yaml \
	  --set train.mlflow.run_name=vid-lint-scratch $(ARGS)

video-lint-frozen:
	uv run python -m blinklinmult.train.cli $(VIDEO) \
	  --model config/model/blinklint.yaml \
	  --set model.encoder_weights=$(VIDEO_SEED) \
	  --set model.encoder_freeze=true \
	  --set train.mlflow.run_name=vid-lint-frozen $(ARGS)

video-lint-ft:
	uv run python -m blinklinmult.train.cli $(VIDEO) \
	  --model config/model/blinklint.yaml \
	  --set model.encoder_weights=$(VIDEO_SEED) \
	  --set train.mlflow.run_name=vid-lint-ft $(ARGS)

video-mult-scratch:
	uv run python -m blinklinmult.train.cli $(VIDEO) \
	  --model config/model/blinklinmult.yaml \
	  --set train.mlflow.run_name=vid-mult-scratch $(ARGS)

video-mult-frozen:
	uv run python -m blinklinmult.train.cli $(VIDEO) \
	  --model config/model/blinklinmult.yaml \
	  --set model.encoder_weights=$(VIDEO_SEED) \
	  --set model.encoder_freeze=true \
	  --set train.mlflow.run_name=vid-mult-frozen $(ARGS)

video-mult-ft:
	uv run python -m blinklinmult.train.cli $(VIDEO) \
	  --model config/model/blinklinmult.yaml \
	  --set model.encoder_weights=$(VIDEO_SEED) \
	  --set train.mlflow.run_name=vid-mult-ft $(ARGS)

# The cheap pair first: scratch and fine-tuned bound the transfer question, and
# the frozen arms train 0.1% of the parameters so they cost far less.
video-arms: video-lint-scratch video-lint-ft video-lint-frozen \
            video-mult-scratch video-mult-ft video-mult-frozen

# ── Embedding cache ────────────────────────────────────────────────────────────
#
# A frozen encoder returns the same vector for a given crop on every epoch, so
# caching turns the expensive part of a batch into a lookup. Build it once, then
# every frozen arm sharing that seed checkpoint reads it.
#
#   make build-cache          # ~one epoch of forward passes, then reusable
#   make show-cache           # what is cached, and which encoder built it
#   make video-lint-frozen-cached
#
# Only valid with a frozen encoder and augmentation off; config refuses any
# other combination rather than silently training on stale vectors.
CACHE_ROOT ?= cache/embeddings

build-cache:
	uv run python tools/build_embedding_cache.py \
	  --encoder $(VIDEO_SEED) --root $(CACHE_ROOT) $(ARGS)

show-cache:
	uv run python tools/show_embedding_cache.py --root $(CACHE_ROOT)

clean-cache:
	rm -rf $(CACHE_ROOT)

# **Its own batch and worker settings**, because the cache changes which
# resource is scarce. `video_all.yaml` uses batch 8 / 0 workers: batch 8 because
# one live batch put 20 GB of *wired* memory in the MPS pool, and 0 workers
# because the GPU was the bottleneck so parallel readers bought nothing.
#
# With embeddings precomputed, neither holds. The pool drops to ~1.2 GB (the CNN
# activations are gone) and the loader becomes the constraint, so workers now
# pay off dramatically. Measured, windows/second:
#
#     ======  =======  =========  ==========
#     batch   workers  windows/s  epoch
#     ======  =======  =========  ==========
#     8       0             167   7.1 min
#     64      0             233   5.1 min
#     64      4           1 030   1.1 min
#     **64**  **6**   **1 507**   **0.8 min**
#     64      8             860   1.4 min
#     128     6           1 018   1.2 min
#     256     6             691   1.7 min
#     ======  =======  =========  ==========
#
# A clean interior peak at 64/6, and larger batches are *worse* -- past 64 the
# per-batch transfer outweighs the fewer Python round-trips.
CACHED_BATCH ?= 64
CACHED_WORKERS ?= 6

video-lint-frozen-cached:
	uv run python -m blinklinmult.train.cli $(VIDEO) \
	  --model config/model/blinklint.yaml \
	  --set model.encoder_weights=$(VIDEO_SEED) \
	  --set model.encoder_freeze=true \
	  --set data.augment=null \
	  --set data.embedding_cache=$(CACHE_ROOT) \
	  --set data.batch_size=$(CACHED_BATCH) \
	  --set data.num_workers=$(CACHED_WORKERS) \
	  --set train.mlflow.run_name=vid-lint-frozen-cached $(ARGS)

# Stage 2 of the two-stage recipe: continue a frozen run end-to-end at a low
# learning rate. Stage 1 (`video-lint-frozen`) trains the transformer against a
# fixed encoder at 7.7x the speed and without risking the pretrained features;
# stage 2 refines everything jointly from that checkpoint.
#
# `--unfreeze-from` loads the whole model, unlike model.encoder_weights (encoder
# only) and unlike --resume (which would restore the frozen state too).
UNFREEZE_FROM ?= results/blink-video/vid-lint-frozen/checkpoints/best.ckpt
UNFREEZE_MULT_FROM ?= results/blink-video/vid-mult-frozen/checkpoints/best.ckpt

video-lint-unfreeze:
	uv run python -m blinklinmult.train.cli \
	  --data config/data/video_all.yaml \
	  --train config/train/video_unfreeze.yaml \
	  --model config/model/blinklint.yaml \
	  --unfreeze-from $(UNFREEZE_FROM) \
	  --set train.mlflow.run_name=vid-lint-unfreeze $(ARGS)

video-mult-unfreeze:
	uv run python -m blinklinmult.train.cli \
	  --data config/data/video_all.yaml \
	  --train config/train/video_unfreeze.yaml \
	  --model config/model/blinklinmult.yaml \
	  --unfreeze-from $(UNFREEZE_MULT_FROM) \
	  --set train.mlflow.run_name=vid-mult-unfreeze $(ARGS)

# Score trained arms per corpus, one process each. ARMS selects a subset:
#   make eval-video ARMS="vid-lint-ft vid-mult-ft"
#
# Per-corpus rather than one pass, for two reasons: the full test split SIGKILLs
# a 34 GB host (MPEblink alone is 1.9M frames behind a 40 GB h5), and rn15/rn30
# share recording ids so a pooled dump cannot be split by corpus afterwards.
ARMS ?=
eval-video:
	bash experiments/eval_video.sh $(ARMS)

# The benchmark table: per dataset, never pooled.
report-video:
	uv run python experiments/video_report.py $(ARGS)

# Score a trained frame-wise checkpoint on the test split without retraining.
# `--eval-only` loads the checkpoint, fits the event operating point on
# validation, and runs the test split -- the passes a finished run would have
# performed, so a run interrupted before its test pass can still produce its
# benchmark table. Reports on the full eval set by default (stills_all.yaml
# declares talkingface, mpeblink and hust_lebw); override with ARGS.
# The run keeps the training run's name so its artifacts land in the same
# directory `report-frame-wise` reads by default.
#
#   make eval-frame-wise
#   make report-frame-wise
CKPT ?= results/frame-wise/frame-wise-r1/checkpoints/best.ckpt
eval-frame-wise:
	uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	  --eval-only $(CKPT) \
	  --set train.mlflow.run_name=frame-wise-r1 $(ARGS)

# Fit the ONE universal operating point, on the training corpora only.
#
# Only RN15 and RN30 qualify, and the exclusions are structural rather than a
# preference:
#
#   cew, mrl      annotate no blinks at all -- nothing to fit an event
#                 threshold on
#   talkingface   ships 0 validation windows
#   hust_lebw,    eval-only; fitting on them would leak evaluation data into
#   mpeblink      the operating point and void the zero-shot claim
#
# The fitted value lands in $(THRESHOLD_DIR)/event_threshold.json, which every
# corpus in the split then reads. Without this each process fits privately and
# the corpora end up on different scales -- measured at 0.01, 0.06, and two
# silent 0.50 fallbacks in a single run.
#
# Round 2 moves MPEblink into training, at which point it joins this pool and
# its 20 558 validation windows dominate the fit.
THRESHOLD_DIR ?= results/frame-wise/universal
UNIVERSAL_CORPORA ?= rn15,rn30
fit-universal-threshold:
	@mkdir -p $(THRESHOLD_DIR)
	uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	  --eval-only $(CKPT) \
	  --shared-threshold-dir $(THRESHOLD_DIR) \
	  --set data.datasets=[$(UNIVERSAL_CORPORA)] \
	  --set data.eval_datasets=[] \
	  --set train.limit_test_batches=1 \
	  --set train.mlflow.run_name=universal-threshold $(ARGS)
	@echo "Universal threshold: $$(cat $(THRESHOLD_DIR)/event_threshold.json)"

# The same evaluation, one corpus at a time.
#
# Prefer this on a memory-constrained host. The single-pass `eval-frame-wise`
# holds the whole test split -- 2.4M frames across seven corpora, of which
# MPEblink alone is 1.9M (80%) backed by a 40 GB h5 -- and streaming that much
# through the page cache on a 34 GB machine, with swap already near its ceiling,
# gets the process SIGKILLed by the OS rather than failing cleanly.
#
# Each corpus is scored in its own process, so the memory floor is the largest
# single corpus rather than their sum, and a corpus that dies takes only its own
# pass with it.
#
# **Corpora are routed by what they annotate, not forced into `datasets`.**
# `eye_state` is the trained target, so a corpus that does not carry it can only
# ever be an eval corpus:
#
#   cew, mrl                 eye_state only     -> frame-wise scores only
#   rn15, rn30               both               -> trainable, fits a threshold
#   talkingface              both, 0 valid      -> eval-only, needs a threshold
#   hust_lebw, mpeblink      blink_presence     -> eval-only, no eye_state head
#
# Routing them wrongly is not a silent mistake: hust_lebw and mpeblink raise
# `BatchError: Batch has no target 'eye_state'`, and talkingface raises
# `DataModuleError: No 'valid' split available`.
#
# **Two thresholds are reported for every corpus, with no flag to set.** The
# universal one is fitted once by `fit-universal-threshold` (a prerequisite, so
# it runs automatically) and shared through $(THRESHOLD_DIR); the tuned one is
# each corpus's own optimum, read off its test FROC curve and labelled `tuned/`.
# The gap between them is the cost of tuning, made visible rather than hidden in
# whichever was chosen.
#
#   make eval-frame-wise-split
#   make report-frame-wise
TRAIN_CORPORA ?= cew mrl rn15 rn30
# hust_lebw is deliberately absent: it annotates blinks clip-wise with no
# per-frame closure label, so it can neither train the eye-state head nor be
# scored against it, and `config/data/stills_all.yaml` already excludes it. A
# number reported for it here would measure the closure-versus-event convention
# gap rather than the model. It belongs to the event-level benchmark.
EVAL_ONLY_CORPORA ?= talkingface mpeblink
# An eval-only corpus still needs a `datasets` entry, because the run needs a
# valid split to fit its threshold on and a trained target to build the head
# from. CEW is the carrier: it is the smallest corpus (732 test frames, 123 MB)
# so it costs almost nothing in memory, and it annotates only `eye_state`, so it
# contributes no blink events that could contaminate the event report. Its rows
# appear separately in `test_per_dataset.json` and are ignored by the reporter.
CARRIER ?= cew
eval-frame-wise-split: fit-universal-threshold
	@for corpus in $(TRAIN_CORPORA); do \
	  echo "==> evaluating $$corpus (train-corpus route)"; \
	  uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	    --eval-only $(CKPT) \
	    --reuse-threshold \
	    --resumable \
	    --shared-threshold-dir $(THRESHOLD_DIR) \
	    --set data.datasets=[$$corpus] \
	    --set data.eval_datasets=[] \
	    --set train.mlflow.run_name=frame-wise-r1-$$corpus $(ARGS) \
	    || echo "!!! $$corpus failed; continuing"; \
	done
	@for corpus in $(EVAL_ONLY_CORPORA); do \
	  echo "==> evaluating $$corpus (eval-only route)"; \
	  uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	    --eval-only $(CKPT) \
	    --reuse-threshold \
	    --resumable \
	    --shared-threshold-dir $(THRESHOLD_DIR) \
	    --set data.datasets=[$(CARRIER)] \
	    --set data.eval_datasets=[$$corpus] \
	    --set train.mlflow.run_name=frame-wise-r1-$$corpus $(ARGS) \
	    || echo "!!! $$corpus failed; continuing"; \
	done

# Round 2: the same run with mpeblink added to *training*.
#
# An open question rather than an improvement. MPEblink annotates blink events,
# not closure, so rasterising its intervals marks the whole event -- closing,
# closed, opening -- while only 19-34% of those frames show a closed eye.
# Training on that teaches a half-open lid as "closed". Whether its 63 758
# windows of in-the-wild variety outweigh the noisier label is what this
# measures.
train-frame-wise-mpeblink:
	uv run python -m blinklinmult.train.cli $(FRAME_WISE) \
	  --set data.datasets=[cew,mrl,rn15,rn30,mpeblink] \
	  --set data.eval_datasets=[talkingface] \
	  --set train.mlflow.run_name=frame-wise-r2-mpeblink $(ARGS)

# The benchmark table: per-corpus frame, event, and clip-wise scores.
report-frame-wise:
	uv run python experiments/frame_wise_report.py $(ARGS)

# ── Hugging Face: published models and corpora ────────────────────────────────
# Three repos, all named `blink_detection` -- HF namespaces by type, so a model,
# a dataset and a Space may share one name:
#   fodorad/blink_detection  model    public   the four shipped models + sidecars
#   fodorad/blink_detection  dataset  PRIVATE  the built h5 corpora
#   fodorad/blink_detection  space    public   the Gradio demo
#
# The dataset repo is **private on purpose**: the h5 files store real eye-crop
# imagery derived from seven corpora with differing licences, and docs/data.md
# states that raw data is not redistributed here. Flipping a corpus public is a
# per-corpus licence decision, never a blanket one.
#
# HF_XET_HIGH_PERFORMANCE replaces the retired HF_HUB_ENABLE_HF_TRANSFER flag.
#
# There was once a second dataset variable here, because there were two dataset
# repos -- a public one and a private one. `?=` does not override an earlier
# assignment, so reusing the name silently pushed eye-crop imagery to the
# *public* repo. There is now exactly one dataset repo and it is private, so the
# hazard is gone: HF_DATASET_REPO (line ~183) is the single name for corpora.
HF_MODEL_REPO    ?= fodorad/blink_detection
HF_SPACE_REPO    ?= fodorad/blink_detection
ONNX_DIR         ?= artifacts/onnx

# Freeze the 1.x models into ONNX. Needs the 1.x PyTorch definitions, which have
# been removed -- see the script's docstring for recovering them. Kept so the
# published graphs' provenance is reproducible rather than folklore.
export-paper-onnx:
	uv run --extra export python tools/export_paper_onnx.py --out $(ONNX_DIR)

# Strip a training checkpoint to its inference payload (57 MB -> 19 MB) and write
# its sidecar. Named PUBLISH_CKPT, not CKPT: an earlier target already defines
# CKPT for a different arm, and `?=` would not override it.
PUBLISH_CKPT ?= results/frame-wise/fw-focal-augment/checkpoints/best.ckpt
export-blinkcnn:
	uv run $(EXTRAS) python tools/export_blinkcnn.py \
		--checkpoint $(PUBLISH_CKPT) --out $(ONNX_DIR)

# Publish one model file plus its sidecar. MODEL_ID is the filename stem.
push-model:
	@test -n "$(MODEL_ID)" || (echo "MODEL_ID is required, e.g. MODEL_ID=blinkcnn.pt" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(ONNX_DIR)/$(MODEL_ID) $(MODEL_ID) --repo-type model
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		$(ONNX_DIR)/$(MODEL_ID).json $(MODEL_ID).json --repo-type model

# Deploy the Gradio demo. The Space is the one published surface with no CI
# behind it: nothing builds it on a push, and nothing notices when it breaks. It
# had no deploy target at all, so its files were uploaded by hand and drifted --
# the live app.py still read the example clip from `data/raw/`, a corpus
# directory that does not exist on a Space, while the repo had long since moved
# to the bundled `blinklinmult.assets` copy. This target makes the repo the
# source of truth so that cannot happen twice.
#
# Uploaded one file at a time rather than as a directory: the Space keeps
# `app.py` and `README.md` at its root, while they live under `demos/gradio/`
# here, so each needs its destination named.
#
# A rebuild after this does NOT pick up a changed `python_version` on its own --
# Hugging Face reuses the cached base image. To change the interpreter, follow
# this with a factory reboot, which discards those layers:
#
#   uv run python -c "from huggingface_hub import HfApi; \
#     HfApi().restart_space('$(HF_SPACE_REPO)', factory_reboot=True)"
push-space:
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_SPACE_REPO) \
		demos/gradio/app.py app.py --repo-type space
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_SPACE_REPO) \
		demos/gradio/README.md README.md --repo-type space
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_SPACE_REPO) \
		demos/gradio/requirements.txt requirements.txt --repo-type space

# The model card, built from its canonical source so the GitHub docs and the Hub
# page cannot drift.
push-model-card:
	mkdir -p .build
	cat docs/_hf_model_frontmatter.yaml docs/_hf_model_card.md > .build/hf_model_card.md
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_MODEL_REPO) \
		.build/hf_model_card.md README.md --repo-type model

# One corpus at a time, smallest first -- mpeblink alone is 42 GB, so validating
# the path on cew (171 MB) before committing to it is worth the extra command.
push-corpus:
	@test -n "$(CORPUS)" || (echo "CORPUS is required, e.g. CORPUS=cew" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_DATASET_REPO) \
		data/processed/$(CORPUS)/$(CORPUS).h5 $(CORPUS).h5 --repo-type dataset

pull-corpus:
	@test -n "$(CORPUS)" || (echo "CORPUS is required, e.g. CORPUS=cew" && exit 1)
	HF_XET_HIGH_PERFORMANCE=1 uv run hf download $(HF_DATASET_REPO) \
		$(CORPUS).h5 --repo-type dataset --local-dir data/processed/$(CORPUS)

push-dataset-card:
	mkdir -p .build
	cat docs/_hf_frontmatter.yaml docs/dataset.md > .build/hf_dataset_card.md
	HF_XET_HIGH_PERFORMANCE=1 uv run hf upload $(HF_DATASET_REPO) \
		.build/hf_dataset_card.md README.md --repo-type dataset

# ── Live demos ────────────────────────────────────────────────────────────────
# A single window: the frame on top, a scrolling score plot per eye beneath.
# `webcam` needs a camera (macOS will ask for permission on first run); `video`
# runs the same view over a file and needs no hardware.
DEMO_VIDEO ?= blinklinmult/assets/talkingface_10s.mp4
DEMO_MODEL ?= blinkcnn-onnx

webcam:
	uv run --extra preprocess $(EXTRAS) python demos/webcam.py \
		--model $(DEMO_MODEL) --weights-dir $(ONNX_DIR) $(ARGS)

video-demo:
	uv run --extra preprocess $(EXTRAS) python demos/video.py \
		--video $(DEMO_VIDEO) --model $(DEMO_MODEL) --weights-dir $(ONNX_DIR) $(ARGS)

# ── Model comparison ──────────────────────────────────────────────────────────
# Statistical comparison of the shipped models: average precision with
# recording-clustered bootstrap intervals, Holm-corrected across every pair.
#
# **The resampling unit is the recording, not the window.** RN30 holds 7312 test
# windows drawn from 35 recordings; resampling the windows shrank the interval
# 5.9x on this repo's own data and turned an inconclusive difference into a
# "significant" one. See docs/comparison.md.
COMPARE_DIR    ?= results/comparison
COMPARE_CORPUS ?= rn30
COMPARE_ARMS   ?= fw-bce fw-bce-augment fw-focal fw-focal-augment

# Freeze blinkcnn into an ONNX graph. Gated on numerical parity against the
# checkpoint, at three window lengths and three input distributions.
export-blinkcnn-onnx:
	uv run --extra export $(EXTRAS) python tools/export_blinkcnn_onnx.py \
		--checkpoint $(ONNX_DIR)/blinkcnn.pt --out $(ONNX_DIR)

# Latency, memory, artifact size and cold start for every model with local
# weights. NOT part of `check`: a timing on a shared CI runner measures the
# runner, and would make the gate flaky.
benchmark-runtime:
	@mkdir -p $(COMPARE_DIR)
	uv run $(EXTRAS) --extra compare python tools/benchmark_runtime.py \
		--weights-dir $(ONNX_DIR) --out $(COMPARE_DIR)/runtime.json $(ARGS)

# End-to-end streaming: detect -> landmark -> crop -> score, replayed at the
# source frame rate. Reports frames dropped against the deadline, not just a
# mean throughput, and the per-stage budget that explains it.
benchmark-streaming:
	@mkdir -p $(COMPARE_DIR)
	uv run --extra preprocess $(EXTRAS) python tools/benchmark_streaming.py \
		--weights-dir $(ONNX_DIR) --out $(COMPARE_DIR)/streaming.json $(ARGS)

# Score the four shipped models on a corpus, so they can be compared with each
# other rather than with training arms. Writes signal archives in the same shape
# the training callbacks write, so compare-models consumes them unchanged.
SHIPPED_DIR ?= results/shipped
score-models:
	uv run $(EXTRAS) python tools/score_models.py \
		--corpus $(COMPARE_CORPUS) --weights-dir $(ONNX_DIR) --out-dir $(SHIPPED_DIR) $(ARGS)

# The four shipped models against each other. This is the comparison that
# answers "which model should I use", as opposed to compare-models, which
# compares training arms.
SHIPPED_MODELS ?= densenet121-union blinklint-union blinklinmult-union blinkcnn
compare-shipped:
	@mkdir -p $(COMPARE_DIR)
	uv run $(EXTRAS) --extra compare python tools/compare_models.py \
		$(foreach m,$(SHIPPED_MODELS),--signals $(SHIPPED_DIR)/$(m)-$(COMPARE_CORPUS)/test_signals.npz=$(m)) \
		--corpus $(COMPARE_CORPUS) --target eye_state \
		--out $(COMPARE_DIR)/shipped-$(COMPARE_CORPUS).json $(ARGS)

compare-models:
	@mkdir -p $(COMPARE_DIR)
	uv run $(EXTRAS) --extra compare python tools/compare_models.py \
		$(foreach a,$(COMPARE_ARMS),--signals results/frame-wise/$(a)-hyst-$(COMPARE_CORPUS)/test_signals.npz=$(a)) \
		--corpus $(COMPARE_CORPUS) --out $(COMPARE_DIR)/$(COMPARE_CORPUS).json $(ARGS)

# ── Gradio demo ───────────────────────────────────────────────────────────────
# Blink detection on an uploaded video, locally. Needs three extras: `demo` for
# gradio, `preprocess` for the face/eye/pose extraction the pipeline runs on raw
# video, and `onnx` for the three 1.x models. Opens on http://127.0.0.1:7860.
app:
	uv run --extra demo --extra preprocess --extra onnx python demos/gradio/app.py

# ── Docker ────────────────────────────────────────────────────────────────────
# Two images, because they are different products. The REST image scores eye
# crops and needs onnxruntime alone (~385 MB resident); the demo image runs the
# whole video pipeline and carries exordium, YOLO11, FaceMesh and opencv (~1 GB).
# Merging them would make the deployable artifact carry a face detector it never
# calls.
#
# Both bake the model weights in, so `docker run --network none` works: an image
# that downloads weights on first use can silently score with different weights
# than the ones it was tested against.
DOCKER_IMAGE ?= fodorad/blink_detection
DOCKER_TAG   ?= $(shell grep -m1 '^version' pyproject.toml | cut -d'"' -f2)

docker-build:
	docker build -f demos/docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) -t $(DOCKER_IMAGE):latest .

docker-build-demo:
	docker build -f demos/docker/Dockerfile.demo \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG)-full -t $(DOCKER_IMAGE):full .

# --network none is the point, not a precaution: it proves the weights are in the
# image rather than fetched on first use.
docker-run:
	docker run --rm --network none -p 8080:8080 $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-demo:
	docker run --rm -p 7860:7860 $(DOCKER_IMAGE):$(DOCKER_TAG)-full python demos/gradio/app.py

docker-push:
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):latest

# The REST service locally, without Docker.
serve:
	uv run --extra serve --extra onnx --extra preprocess \
		uvicorn demos.docker.serve:app --host 127.0.0.1 --port 8080
