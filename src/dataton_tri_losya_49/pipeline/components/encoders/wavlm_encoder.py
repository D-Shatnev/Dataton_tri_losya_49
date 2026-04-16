"""
WavLM-based encoder for speaker verification.

This module provides WavLMEncoder, a wrapper around microsoft/wavlm-base-plus-sv
from Hugging Face transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

from dataton_tri_losya_49.constants import (
    DEFAULT_DIM_PROBE_NUM_SAMPLES,
    DEFAULT_EMBEDDINGS_OUTPUT_NAME,
    DEFAULT_TARGET_SR,
)


@dataclass
class WavLMEncoder:
    """
    Computes embeddings using Microsoft's WavLM-Base-Plus-SV model.

    Args:
        save_dir: directory to cache pretrained model in.
        providers: Runtime providers list. The first provider specifies the 
            torch device (e.g., 'cpu', 'cuda', 'cuda:0').
        output_name: Name of the output tensor that contains embeddings
            (kept for API compatibility).
        dim_probe_num_samples: Number of samples to use for probing embedding 
            dimension. If None, uses DEFAULT_DIM_PROBE_NUM_SAMPLES.

    Attributes:
        dim: Embedding dimensionality (D) - for WavLM this is 512.
    """

    save_dir: Path
    model_name: str
    providers: list[str]
    output_name: str = DEFAULT_EMBEDDINGS_OUTPUT_NAME
    dim_probe_num_samples: int | None = None

    def __post_init__(self) -> None:
        self._device = torch.device(self.providers[0])
        self._model = WavLMForXVector.from_pretrained(
            self.model_name,
            cache_dir=str(self.save_dir),
        ).to(self._device)
        self._model.eval()
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            cache_dir=str(self.save_dir),
        )
        probe_len = self.dim_probe_num_samples or int(DEFAULT_DIM_PROBE_NUM_SAMPLES)
        with torch.no_grad():
            dummy_waveform = torch.zeros(1, probe_len, dtype=torch.float32, device=self._device)
            dummy_numpy = dummy_waveform.cpu().numpy()
            inputs = self._feature_extractor(
                dummy_numpy,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            emb = self._model(**inputs).embeddings
        self._dim = emb.shape[-1]

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        waveforms_list = list(batch_waveforms)
        inputs = self._feature_extractor(
            waveforms_list,
            sampling_rate=DEFAULT_TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self._model(**inputs).embeddings
        return embeddings.cpu().numpy().astype(np.float32)