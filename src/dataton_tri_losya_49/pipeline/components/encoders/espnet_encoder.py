"""
ESPnet-based speaker embedding encoder.

Wraps espnet2.bin.spk_inference.Speech2Embedding to conform to the
dataton_tri_losya_49.pipeline.interfaces.Encoder protocol.

The model is loaded once at construction time (from a HuggingFace model tag
or from local checkpoint files). Embedding dimensionality is inferred via a
short probe forward pass.

Batch processing is implemented as a sequential loop over waveforms because
Speech2Embedding processes one utterance at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EspnetEncoder:
    """
    Computes speaker embeddings using an ESPnet2 SPK model.

    Exactly one of model_tag or model_path must be provided.

    Args:
        model_tag: HuggingFace / ESPnet model zoo tag, e.g.
            "espnet/voxcelebs12_ecapa_wavlm_joint".
            Used when loading a pretrained hub model.
        model_path: Path to a local model checkpoint (.pth file).
            Requires train_config to also be set.
        train_config: Path to the training config YAML that accompanies
            model_path. Ignored when model_tag is set.
        device: PyTorch device string passed to Speech2Embedding
            (e.g. "cuda", "cpu", "cuda:0").

    Attributes:
        dim: Embedding dimensionality inferred from a probe forward pass.

    Raises:
        ValueError: If neither model_tag nor model_path is provided, or if
            model_path is set without train_config.
    """

    model_tag: str | None = None
    model_path: Path | None = None
    train_config: Path | None = None
    device: str = "cuda"
    _model: object = field(init=False, repr=False)
    _dim: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load Speech2Embedding and infer embedding dimensionality."""
        if self.model_tag is None and self.model_path is None:
            raise ValueError("EspnetEncoder: provide model_tag or model_path")
        if self.model_path is not None and self.train_config is None:
            raise ValueError("EspnetEncoder: train_config is required when model_path is set")

        from espnet2.bin.spk_inference import Speech2Embedding  # type: ignore[import]

        if self.model_tag is not None:
            self._model = Speech2Embedding.from_pretrained(
                model_tag=self.model_tag,
                device=self.device,
            )
        else:
            self._model = Speech2Embedding(
                model_file=str(self.model_path),
                train_config=str(self.train_config),
                device=self.device,
            )

        # Probe forward pass to determine embedding dimensionality.
        # 16500 samples = ~1 s at 16 kHz — matches ESPnet demo snippet.
        probe = np.zeros(16500, dtype=np.float32)
        raw = self._model(probe)
        # Speech2Embedding may return a CUDA tensor — move to CPU before numpy conversion.
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        emb = np.asarray(raw, dtype=np.float32)
        if emb.ndim == 1:
            self._dim = int(emb.shape[0])
        elif emb.ndim == 2:
            self._dim = int(emb.shape[-1])
        else:
            raise ValueError(f"Unexpected probe embedding shape: {emb.shape}")

    @property
    def dim(self) -> int:
        """Embedding dimensionality D.

        Returns:
            Size of the output embedding vector (e.g. 192 for
            voxcelebs12_ecapa_wavlm_joint).
        """
        return self._dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """Embed a batch of waveforms.

        Iterates over the batch sequentially because Speech2Embedding
        processes one utterance at a time.

        Args:
            batch_waveforms: float32 array shaped [B, T].

        Returns:
            float32 embeddings array shaped [B, D].

        Raises:
            ValueError: If batch_waveforms is not 2-D.
        """
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B, T], got shape {x.shape}")

        embeddings = []
        for waveform in x:
            raw = self._model(waveform)
            if hasattr(raw, "cpu"):
                raw = raw.cpu()
            emb = np.asarray(raw, dtype=np.float32).reshape(-1)
            embeddings.append(emb)

        result = np.stack(embeddings, axis=0)
        if result.shape != (x.shape[0], self._dim):
            raise ValueError(f"Unexpected output shape {result.shape} for input batch {x.shape}")
        return result
