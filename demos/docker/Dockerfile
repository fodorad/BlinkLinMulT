# The REST service: eye crops in, eye-state curves and blink events out.
#
# **No torch.** The three 1.x models are frozen ONNX graphs and `blinkcnn` loads
# through onnxruntime too, so a crops-in service needs numpy and onnxruntime and
# nothing else. That is the whole reason this image is separate from
# `Dockerfile.demo`: adding the video pipeline pulls exordium, YOLO11, FaceMesh,
# 6DRepNet and opencv, and the deployable artifact should not carry a face
# detector it never calls.
#
# Measured: the crops-only path holds ~385 MB resident against ~1 GB for the full
# pipeline, and starts in a fraction of the time.
FROM python:3.12-slim

WORKDIR /app

# 3.12, not 3.13, matching the Hugging Face Space so one Python is tested across
# both deployment targets rather than a version only the container ever sees.

# git: hatch-vcs reads VCS metadata at build time (see the throwaway repo below).
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# uv, not plain pip: `uv pip install --override` is how demos/docker/overrides.txt
# reaches the build. pyproject.toml's [tool.uv].override-dependencies is a
# uv-sync-only mechanism that plain pip cannot read at all, so without this the
# timm pin that breaks exordium's import chain would silently return.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Only what the build needs, so a code change does not invalidate the dependency
# layer.
COPY pyproject.toml README.md demos/docker/overrides.txt ./
COPY blinklinmult ./blinklinmult

# hatch-vcs derives the version from git tags, and this build context has no
# .git. A throwaway repo with one tag gives it real (if meaningless) metadata;
# the served app never reads __version__.
RUN git init -q && git config user.email a@a.a && git config user.name a \
    && git add -A && git commit -q -m init && git tag v0.0.0
RUN uv pip install --system --no-cache --override overrides.txt ".[serve,onnx]"

# **Weights baked in, not fetched at runtime.** `docker run --network none` must
# work: an image that downloads weights on first use is not reproducible, and can
# silently score with different weights than the ones it was tested against.
# HF_HOME is set so the cache lands where the runtime user can read it.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from pathlib import Path; \
from huggingface_hub import hf_hub_download; \
from blinklinmult import WEIGHTS_DIR; \
from blinklinmult.registry import MODELS; \
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True); \
[Path(WEIGHTS_DIR, s.filename).write_bytes(Path(hf_hub_download('fodorad/blink_detection', s.filename)).read_bytes()) for s in MODELS.values()]; \
print('baked', len(MODELS), 'models')"

COPY demos/docker/serve.py ./demos/docker/serve.py

ENV PORT=8080
ENV MODEL_ID=blinkcnn
EXPOSE 8080

CMD ["sh", "-c", "uvicorn demos.docker.serve:app --host 0.0.0.0 --port ${PORT}"]
