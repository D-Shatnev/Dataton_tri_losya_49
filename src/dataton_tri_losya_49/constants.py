"""
Project-wide constants.

The goal is to keep defaults consistent across:
  - config parsing
  - CLI defaults
  - dataset/audio loaders
"""

DEFAULT_FILEPATH_COL: str = "filepath"
DEFAULT_SPEAKER_ID_COL: str = "speaker_id"

# Deterministic evaluation chunk duration (seconds)
DEFAULT_CHUNK_SECONDS: float = 6.0

# Target sample rate used across the pipeline
DEFAULT_TARGET_SR: int = 16_000

# encoders defaults
DEFAULT_ONNX_EMBEDDINGS_OUTPUT_NAME: str = "embeddings"
DEFAULT_SPEECHBRAIN_EMBEDDINGS_OUTPUT_NAME: str = "embeddings"

# Default encoders Runtime providers priority list used by CLI (can be overridden).
# Kept as a tuple to prevent accidental mutation.
DEFAULT_ONNX_PROVIDERS: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)
DEFAULT_SPEECHBRAIN_PROVIDERS: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)

# Common CLI defaults
DEFAULT_TOPK: int = 10
DEFAULT_BATCH_SIZE: int = 1

# Number of samples used to probe embedding dimensionality when model input length is dynamic.
# Kept consistent with pipeline defaults.
DEFAULT_ONNX_DIM_PROBE_NUM_SAMPLES: int = int(DEFAULT_CHUNK_SECONDS * DEFAULT_TARGET_SR)
DEFAULT_SPEECHBRAIN_DIM_PROBE_NUM_SAMPLES: int = int(DEFAULT_CHUNK_SECONDS * DEFAULT_TARGET_SR)
