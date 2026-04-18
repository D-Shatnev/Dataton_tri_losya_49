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

    On CUDA the model is compiled with ``torch.compile(dynamic=True)`` so
    that the compiled kernel handles variable batch sizes.  This is required
    because :class:`ChunkingEncoder` passes all chunks of a waveform as a
    single ``[N, chunk_samples]`` batch, and N varies per file.

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
    # Upper bound on audio duration used by the runner to pre-compute a fixed
    # padded_total so torch.compile always sees the same tensor shape.
    # Set to 0.0 to fall back to auto-detect from the first batch.
    max_audio_duration_s: float = 0.0
    _model: torch.nn.Module = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _dtype: torch.dtype = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load pretrained model from torch.hub and compile for inference."""
        device_str = self._resolve_device(self.device)
        self._device = torch.device(device_str)
        self._dtype = torch.float16 if device_str == "cuda" else torch.float32

        if device_str == "cuda":
            # TF32 gives ~10% speedup on Ampere+ GPUs for matmul/conv with
            # negligible accuracy loss (19-bit mantissa vs 23-bit float32).
            torch.set_float32_matmul_precision("high")
            # cuDNN auto-selects the fastest conv algorithm for each input shape.
            torch.backends.cudnn.benchmark = True

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
            # dynamic=True: generates a single compiled kernel that handles any
            # batch size N without recompilation.  We intentionally avoid
            # mode="reduce-overhead" here because that mode enables CUDA Graphs
            # which pre-allocate memory for the maximum observed tensor shape and
            # cause OOM when batches are large (e.g. 32 files × 59 chunks each).
            self._model = torch.compile(self._model, dynamic=True)

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
