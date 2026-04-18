"""
CAM++ speaker encoder via ModelScope pipeline.

This module provides CamPlusPlusEncoder — a wrapper around the ModelScope
speaker verification pipeline for the CAM++ model family.

The encoder follows the same Encoder protocol as OnnxEncoder:
    - Input:  batch of waveforms shaped [B, T], float32, 16 kHz.
    - Output: embeddings shaped [B, D], float32.

Each waveform in the batch is processed independently through the ModelScope
pipeline (which internally handles feature extraction and the neural network).
The resulting embeddings are stacked into a single [B, D] array.

API notes (SpeakerVerificationPipeline):
    - Accepts a **list** of np.ndarray (1-D float32) or file paths.
    - Returns embeddings when called with ``output_emb=True``:
      ``result['embs']`` is a numpy array shaped [N, D].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on torch availability.

    Args:
        device: Device string. 'auto' triggers runtime detection.

    Returns:
        Resolved device string ('cuda' or 'cpu').
    """
    if device != "auto":
        return device
    try:
        import torch  # pylint: disable=import-outside-toplevel

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class CamPlusPlusEncoder:
    """Speaker embeddings via the ModelScope CAM++ speaker verification pipeline.

    The ModelScope pipeline is initialised lazily in ``__post_init__`` so that
    import-time errors surface early (missing modelscope / torch packages).

    The pipeline expects a **list** of 1-D float32 numpy arrays (16 kHz mono).
    Embeddings are extracted by calling the pipeline with ``output_emb=True``,
    which returns ``result['embs']`` as a numpy array shaped [N, D].

    Args:
        model_id: ModelScope model identifier.
        device: Compute device. ``"auto"`` selects CUDA when available.

    Attributes:
        dim: Embedding dimensionality (D), inferred on first init.

    Notes:
        The pipeline processes each waveform independently at the model level.
        For large datasets prefer a bigger ``batch_size`` at the runner level
        to amortise Python overhead.
    """

    model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    device: str = "auto"
    _pipeline: object = field(init=False, repr=False, compare=False)
    _dim: int = field(init=False, repr=False, compare=False)

    # ModelScope pipeline sample rate expectation
    _SAMPLE_RATE: int = field(default=16_000, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialise the ModelScope pipeline and probe embedding dimensionality."""
        from modelscope.pipelines import pipeline  # pylint: disable=import-outside-toplevel
        from modelscope.utils.constant import Tasks  # pylint: disable=import-outside-toplevel

        resolved_device = _resolve_device(self.device)
        self._pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=self.model_id,
            device=resolved_device,
        )

        # Probe dimensionality with a 1-second silent waveform.
        # Pipeline expects a list of 1-D float32 arrays.
        probe = np.zeros(self._SAMPLE_RATE, dtype=np.float32)
        result = self._pipeline([probe], output_emb=True)
        embs = np.asarray(result["embs"], dtype=np.float32)
        # embs shape: [N, D]
        self._dim = int(embs.shape[1])

    @property
    def dim(self) -> int:
        """Embedding dimensionality D.

        Returns:
            Size of the output embedding vector.
        """
        return self._dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """Compute embeddings for a batch of waveforms.

        Each waveform is passed as a list to the ModelScope pipeline with
        ``output_emb=True``. Results are returned as a [B, D] array.

        Args:
            batch_waveforms: float32 array shaped [B, T].

        Returns:
            float32 embeddings array shaped [B, D].

        Raises:
            ValueError: If input is not a 2-D array.
        """
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B, T], got shape {x.shape}")

        # Pipeline accepts a list of 1-D arrays.
        wav_list = [x[i] for i in range(x.shape[0])]
        result = self._pipeline(wav_list, output_emb=True)
        embs = np.asarray(result["embs"], dtype=np.float32)
        return embs
