"""
ReDimNet-based encoder.

This module provides ReDimNetEncoder, a wrapper around the ReDimNet model
loaded via torch.hub from the IDRnD/ReDimNet repository.

Reference:
    Yakovlev et al., "Reshape Dimensions Network for Speaker Recognition",
    Interspeech 2024. https://arxiv.org/pdf/2407.18223
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dataton_tri_losya_49.constants import DEFAULT_CHUNK_SECONDS, DEFAULT_TARGET_SR


@dataclass
class ReDimNetEncoder:
    """
    Computes speaker embeddings using a pretrained ReDimNet model.

    The model is loaded once during __post_init__ via torch.hub and kept
    in eval mode for the lifetime of the encoder. Loading happens before
    the inference timer starts in the experiment runner, so model download
    time is not counted as inference time.

    Args:
        hub_repo: GitHub repo slug passed to torch.hub.load.
        model_name: ReDimNet size identifier (e.g. "b0" … "b6", "M", "L").
        train_type: Training regime identifier (e.g. "ptn", "ft_lm", "ft_mix").
        dataset: Dataset the model was pretrained on (e.g. "vox2", "vb2+vox2+cnc").
        device: Compute device. "auto" selects CUDA when available, else CPU.
        embedding_dim: Expected output embedding dimensionality.
            All published ReDimNet checkpoints output 192-dimensional vectors.
        force_reload: If True, bypass torch.hub cache and re-download weights.

    Attributes:
        dim: Embedding dimensionality D.
    """

    hub_repo: str = "IDRnD/ReDimNet"
    model_name: str = "b6"
    train_type: str = "ft_lm"
    dataset: str = "vox2"
    device: str = "auto"
    embedding_dim: int = 192
    force_reload: bool = False
    checkpoint_path: str = ""
    _model: torch.nn.Module = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _dtype: torch.dtype = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load pretrained model from torch.hub and prepare for inference."""
        device_str = self._resolve_device(self.device)
        self._device = torch.device(device_str)
        self._dtype = torch.float16 if device_str == "cuda" else torch.float32

        torch.hub.set_dir("/app/models/torch_hub")
        self._model = torch.hub.load(
            self.hub_repo,
            "ReDimNet",
            model_name=self.model_name,
            train_type=self.train_type,
            dataset=self.dataset,
            force_reload=self.force_reload,
            trust_repo=True,
        )

        if self.checkpoint_path:
            sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(sd, strict=True)

        self._model = self._model.to(self._device)
        self._model.eval()
        self._model = torch.compile(self._model, backend="inductor", mode="reduce-overhead")
        self._warmup()

    def _warmup(self) -> None:
        """Run dummy forward passes to trigger JIT compilation.

        Inductor compiles the model on the first forward pass.  Running a few
        dummy passes here ensures compilation happens during initialisation
        (before the inference timer starts) rather than on the first real batch.
        """
        num_samples = int(DEFAULT_CHUNK_SECONDS * DEFAULT_TARGET_SR)
        dummy = torch.zeros(1, num_samples, dtype=self._dtype, device=self._device)
        with torch.no_grad():
            for _ in range(3):
                self._model(dummy)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve "auto" to "cuda" or "cpu" based on availability.

        Args:
            device: Device string or "auto".

        Returns:
            Resolved device string: "cuda" or "cpu".
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def dim(self) -> int:
        """Embedding dimensionality D.

        Returns:
            The size of the output embedding vector.
        """
        return self.embedding_dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Compute speaker embeddings for a batch of waveforms.

        Args:
            batch_waveforms: float32 array shaped [B, T] at 16 kHz.

        Returns:
            float32 embeddings array shaped [B, D].

        Raises:
            ValueError: If input is not a 2-D array.
        """
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B, T], got shape {x.shape}")

        tensor = torch.from_numpy(x).to(self._device)

        with torch.no_grad(), torch.autocast(device_type=self._device.type, dtype=self._dtype):
            embeddings = self._model(tensor)

        return embeddings.cpu().float().numpy()
