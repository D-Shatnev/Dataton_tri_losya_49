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


@dataclass
class ReDimNetEncoder:
    """
    Computes speaker embeddings using a pretrained ReDimNet model.

    The model is loaded once during __post_init__ via torch.hub and kept
    in eval mode for the lifetime of the encoder. Loading happens before
    the inference timer starts in the experiment runner, so model download
    time is not counted as inference time.

    On CUDA the model is compiled with ``torch.compile(mode="reduce-overhead")``,
    which uses CUDA Graphs to minimise CPU-side kernel-launch overhead.  A
    warm-up pass with a dummy tensor of the expected shape is performed
    immediately after compilation so that the first real inference call does
    not pay the JIT-compilation cost.

    Args:
        hub_repo: GitHub repo slug passed to torch.hub.load.
        model_name: ReDimNet size identifier (e.g. "b0" … "b6", "M", "L").
        train_type: Training regime identifier (e.g. "ptn", "ft_lm", "ft_mix").
        dataset: Dataset the model was pretrained on (e.g. "vox2", "vb2+vox2+cnc").
        device: Compute device. "auto" selects CUDA when available, else CPU.
        embedding_dim: Expected output embedding dimensionality.
            All published ReDimNet checkpoints output 192-dimensional vectors.
        force_reload: If True, bypass torch.hub cache and re-download weights.
        chunk_samples: Number of samples per chunk used for warm-up.  Must match
            the actual chunk length passed to :meth:`embed` so that the compiled
            CUDA Graph is valid for all subsequent calls.  Defaults to 64000
            (4 s × 16 kHz — the value used by ChunkingEncoder with the standard
            config).

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
    chunk_samples: int = 64000
    _model: torch.nn.Module = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _dtype: torch.dtype = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load pretrained model from torch.hub, compile, and warm up."""
        device_str = self._resolve_device(self.device)
        self._device = torch.device(device_str)
        self._dtype = torch.float16 if device_str == "cuda" else torch.float32

        self._model = torch.hub.load(
            self.hub_repo,
            "ReDimNet",
            model_name=self.model_name,
            train_type=self.train_type,
            dataset=self.dataset,
            force_reload=self.force_reload,
            trust_repo=True,
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        if device_str == "cuda":
            # "reduce-overhead" instructs inductor to use CUDA Graphs, which
            # eliminates CPU kernel-launch overhead on every forward pass.
            # Requires fixed input shape — guaranteed by ChunkingEncoder which
            # always calls embed() with [1, chunk_samples].
            self._model = torch.compile(self._model, mode="reduce-overhead")
            self._warmup()

    def _warmup(self) -> None:
        """Run a few dummy forward passes to trigger JIT compilation.

        torch.compile with mode="reduce-overhead" records a CUDA Graph on the
        first call and replays it on subsequent calls.  Running warm-up here
        (at model-load time, before the inference timer starts) ensures that
        the first real inference call does not pay the compilation cost.
        """
        dummy = torch.zeros(1, self.chunk_samples, dtype=self._dtype, device=self._device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self._dtype):
            for _ in range(3):
                _ = self._model(dummy)
        torch.cuda.synchronize()

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
