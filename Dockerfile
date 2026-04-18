# ---------------------------------------------------------------------------
# Stage 1: builder — install Python deps via uv into an isolated venv
# ---------------------------------------------------------------------------
# python:3.13-slim has Python 3.13 built-in.
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build-time system deps: libsndfile1 (soundfile wheel), build tools for C extensions,
# git + git-lfs (modelscope downloads model weights via git-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from the official pinned image
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv

WORKDIR /build

# Copy only dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock ./

# Create venv and install all production deps into /opt/venv
# UV_PROJECT_ENVIRONMENT directs uv sync to install into /opt/venv instead of .venv
RUN uv venv --python python3.13 /opt/venv \
    && UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-dev --no-install-project

# Copy source and install the project package itself (non-editable, no deps)
# Done here in builder so we use the correct Python interpreter path
COPY src/ ./src/
COPY pyproject.toml ./
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-deps --python /opt/venv/bin/python .

# ---------------------------------------------------------------------------
# Stage 2: runtime — CUDA + cuDNN9 image, no build tools
# ---------------------------------------------------------------------------
# ubuntu22.04 + CUDA 12.6 + cuDNN 9
# 575.57.08 (CUDA 12.9). onnxruntime-gpu 1.20.2 requires CUDA 12.x + cuDNN 9.
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime system deps only:
#   python3.13        — interpreter (via deadsnakes, available on jammy/22.04)
#   python3.13-venv   — needed so the venv is usable
#   libsndfile1       — soundfile audio backend
#   libgomp1          — OpenMP (faiss, onnxruntime)
#   ffmpeg            — audio decoding used by modelscope/torchaudio
#   git + git-lfs     — modelscope downloads model weights via git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gpg-agent \
    curl \
    && curl -fsSL https://keyserver.ubuntu.com/pks/lookup?op=get\&search=0xF23C5A6CF475977595C89F51BA6932366A755776 \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
    > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    git \
    git-lfs \
    && git lfs install \
    && apt-get purge -y --auto-remove gpg-agent curl \
    && rm -rf /var/lib/apt/lists/*

# Copy fully-built venv from builder (all wheels + project package installed)
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy configs (data/ and models/ are bind-mounted at runtime)
COPY configs/ ./configs/

# Directories that will be bind-mounted; create them so paths always exist
RUN mkdir -p data models experiments

# Patch the venv's python symlink to point to the runtime interpreter
# (builder used /usr/local/bin/python3.13, runtime has /usr/bin/python3.13)
RUN ln -sf /usr/bin/python3.13 /opt/venv/bin/python3.13 \
    && ln -sf /usr/bin/python3.13 /opt/venv/bin/python3 \
    && ln -sf /usr/bin/python3.13 /opt/venv/bin/python

# Default: print help. Override command via `docker compose run`.
CMD ["speakerid-infer", "--help"]
