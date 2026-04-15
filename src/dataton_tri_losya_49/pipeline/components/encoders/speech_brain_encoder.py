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

import torch
import numpy as np
from speechbrain.pretrained import SpeakerRecognition

from dataton_tri_losya_49.constants import (
    DEFAULT_SPEECHBRAIN_DIM_PROBE_NUM_SAMPLES,
    DEFAULT_SPEECHBRAIN_EMBEDDINGS_OUTPUT_NAME,
)

"""
class SpeechBrainEmbedder(SpeakerEmbedder):
    def __init__(self, device="cpu"):
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
    
    def extract_embedding(self, file_path: str) -> np.ndarray:
        signal, fs = sf.read(file_path, dtype='float32')
        signal = torch.from_numpy(signal).unsqueeze(0)
        
        embedding = self.model.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()
"""


@dataclass
class SpeechBrainEncoder:
    """
    Computes embeddings using an SpeechBrain model.

    Args:
        save_dir: directory to load pretrained speechbrain model in.
        providers: Runtime providers list. Order matters: the first
            available provider is used.
        output_name: Name of the output tensor that contains embeddings.

    Attributes:
        dim: Embedding dimensionality (D).

    Notes:
        This encoder expects a 2D batch input with shape (B, T) and dtype
        float32. Padding to the same T within a batch is handled by the pipeline.
    """

    save_dir: Path
    providers: list[str]
    output_name: str = DEFAULT_SPEECHBRAIN_EMBEDDINGS_OUTPUT_NAME
    dim_probe_num_samples: int | None = None

    def __post_init__(self) -> None:
        self._model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self.save_dir,
            run_opts={"device": self.providers[0]},
        )

        probe_len = int(DEFAULT_SPEECHBRAIN_DIM_PROBE_NUM_SAMPLES)

        dummy = torch.zeros(1, probe_len)
        with torch.no_grad():
            emb = self._model.encode_batch(dummy)

        self._dim = emb.shape[-1]

    @property
    def dim(self) -> int:
        """Embedding dimensionality (D)."""
        return self._dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Computes embeddings for a batch of waveforms.

        Args:
            batch_waveforms: Waveforms with shape (B, T), float32, 16kHz.

        Returns:
            Embeddings with shape (B, D), float32.
        """
        x = torch.from_numpy(np.asarray(batch_waveforms, dtype=np.float32))
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B,T], got shape {x.shape}")

        with torch.no_grad():
            emb = self._model.encode_batch(x)

        emb = emb.cpu().numpy()[0]
        if emb.ndim != 2 or emb.shape[0] != x.shape[0]:
            raise ValueError(f"Unexpected output shape {emb.shape} for input batch {x.shape}")
        return emb
