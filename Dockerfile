# ---------------------------------------------------------------------------
# Stage 1: builder — install Python deps via uv into an isolated venv
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: Python 3.13 (deadsnakes PPA) + libsndfile1 (soundfile backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /build

# Copy only dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock ./

# Create venv and install production deps (frozen = deterministic)
RUN uv venv --python python3.13 /opt/venv \
    && uv sync --frozen --no-dev --no-install-project --python /opt/venv/bin/python

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean image without build tools
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime system deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed venv from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy source and configs (data/ and models/ are mounted at runtime)
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml ./

# Install the package itself in editable-like mode (no deps, already in venv)
RUN pip install --no-deps -e .

# Directories that will be bind-mounted; create them so paths always exist
RUN mkdir -p data models experiments

# Default entrypoint — override via `docker run ... speakerid-infer ...`
ENTRYPOINT []
CMD ["speakerid-infer", "--help"]
