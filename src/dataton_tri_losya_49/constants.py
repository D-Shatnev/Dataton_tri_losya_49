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

# ONNX encoder defaults
DEFAULT_ONNX_EMBEDDINGS_OUTPUT_NAME: str = "embeddings"

# Number of samples used to probe embedding dimensionality when model input length is dynamic.
# Kept consistent with pipeline defaults.
DEFAULT_ONNX_DIM_PROBE_NUM_SAMPLES: int = int(DEFAULT_CHUNK_SECONDS * DEFAULT_TARGET_SR)
