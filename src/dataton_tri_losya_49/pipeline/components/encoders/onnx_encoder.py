"""
ONNX-based encoder.

This module provides OnnxEncoder, a small wrapper around
onnxruntime.InferenceSession.

The wrapper exists to keep a stable encoder contract for the pipeline and make
encoder swapping in experiments trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataton_tri_losya_49.constants import (
    DEFAULT_ONNX_DIM_PROBE_NUM_SAMPLES,
    DEFAULT_ONNX_EMBEDDINGS_OUTPUT_NAME,
)


@dataclass
class OnnxEncoder:
    """
    Computes embeddings using an ONNX model.

    Args:
        model_path: Path to an ONNX model file.
        providers: ONNX Runtime providers list. Order matters: the first
            available provider is used.
        output_name: Name of the output tensor that contains embeddings.

    Attributes:
        dim: Embedding dimensionality (D).

    Notes:
        This encoder expects a 2D batch input with shape (B, T) and dtype
        float32. Padding to the same T within a batch is handled by the pipeline.
    """

    model_path: Path
    providers: list[str]
    output_name: str = DEFAULT_ONNX_EMBEDDINGS_OUTPUT_NAME
    dim_probe_num_samples: int | None = None

    def __post_init__(self) -> None:
        """Initializes the ONNX session and infers embedding dimensionality."""
        import onnxruntime as ort  # noqa: PLC0415

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self._sess = ort.InferenceSession(str(self.model_path), providers=self.providers)
        inputs = self._sess.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected 1 ONNX input, got {len(inputs)}")
        self._input_name = inputs[0].name

        if self.output_name not in {o.name for o in self._sess.get_outputs()}:
            available = [o.name for o in self._sess.get_outputs()]
            raise ValueError(f"Unknown output_name={self.output_name}. Available outputs: {available}")

        probe_len = self._infer_probe_num_samples(inputs[0].shape)

        # Infer embedding dim with a tiny dummy run
        dummy = np.zeros((1, probe_len), dtype=np.float32)
        out = self._sess.run([self.output_name], {self._input_name: dummy})
        emb = out[0]
        if emb.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape: {emb.shape}")
        self._dim = int(emb.shape[1])

    def _infer_probe_num_samples(self, input_shape: list[object]) -> int:
        if self.dim_probe_num_samples is not None:
            if int(self.dim_probe_num_samples) <= 0:
                raise ValueError("dim_probe_num_samples must be > 0")
            return int(self.dim_probe_num_samples)

        if len(input_shape) >= 2:
            t = input_shape[1]
            if isinstance(t, int) and t > 0:
                return int(t)

        return int(DEFAULT_ONNX_DIM_PROBE_NUM_SAMPLES)

    @property
    def dim(self) -> int:
        """Embedding dimensionality (D)."""
        return self._dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Computes embeddings for a batch of waveforms.

        Args:
            batch_waveforms: Waveforms with shape (B, T), float32.

        Returns:
            Embeddings with shape (B, D), float32.
        """
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B,T], got shape {x.shape}")
        out = self._sess.run([self.output_name], {self._input_name: x})
        emb = np.asarray(out[0], dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != x.shape[0]:
            raise ValueError(f"Unexpected output shape {emb.shape} for input batch {x.shape}")
        return emb
