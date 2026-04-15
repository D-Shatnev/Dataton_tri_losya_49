"""
NVIDIA TitaNet-Large encoder.

Wraps ``nemo.collections.asr.models.EncDecSpeakerLabelModel`` and exposes the
standard :class:`~dataton_tri_losya_49.pipeline.interfaces.Encoder` contract.

The implementation bypasses NeMo's file-based ``get_embedding()`` API and calls
the model's internal preprocessor and encoder directly, so no audio is written
to disk during inference. This is critical for performance on HDD-backed storage.

Internal data flow::

    batch_waveforms [B, T]  (float32, 16 kHz)
        → torch.Tensor [B, T] + lengths [B]
        → model.preprocessor  → mel-spectrogram [B, n_mels, T']
        → model.encoder       → [B, D_enc, T'']
        → AdaptiveAvgPool1d   → [B, D_enc, 1]  (speaker pooling)
        → squeeze + L2-norm   → [B, 192]
        → numpy float32       → [B, 192]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import nemo.collections.asr as nemo_asr
import numpy as np
import torch

from dataton_tri_losya_49.constants import DEFAULT_TARGET_SR

_TITANET_LARGE_DIM = 192


@dataclass
class TitaNetEncoder:
    """
    Speaker embedding encoder backed by NVIDIA TitaNet-Large.

    Loads the model from a HuggingFace model ID or a local ``.nemo`` checkpoint
    and runs inference entirely in memory — no temporary files are created.

    Args:
        model_name: HuggingFace model ID (e.g.
            ``"nvidia/speakerverification_en_titanet_large"``) or path to a
            local ``.nemo`` file.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
            If ``None``, CUDA is used when available, otherwise CPU.
        sample_rate: Expected input sample rate in Hz. Must match the rate
            produced by the waveform loader (default 16 000).

    Attributes:
        dim: Embedding dimensionality (192 for TitaNet-Large).

    Raises:
        RuntimeError: If the model cannot be loaded from ``model_name``.
    """

    model_name: str
    device: str | None = None
    sample_rate: int = DEFAULT_TARGET_SR
    _model: nemo_asr.models.EncDecSpeakerLabelModel = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the NeMo model and move it to the target device."""
        resolved_device = self.device if self.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved_device)

        self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self._device)

    @property
    def dim(self) -> int:
        """Embedding dimensionality D (192 for TitaNet-Large).

        Returns:
            192
        """
        return _TITANET_LARGE_DIM

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Compute speaker embeddings for a batch of waveforms.

        Calls the model's preprocessor and encoder directly without writing
        any audio to disk.

        Args:
            batch_waveforms: float32 array shaped [B, T] at ``self.sample_rate`` Hz.

        Returns:
            float32 embeddings shaped [B, 192].

        Raises:
            ValueError: If ``batch_waveforms`` is not 2-D.
        """
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B, T], got shape {x.shape}")

        batch_size, time_steps = x.shape
        audio = torch.from_numpy(x).to(self._device)
        lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=self._device)

        with torch.no_grad():
            processed, proc_lengths = self._model.preprocessor(input_signal=audio, length=lengths)
            encoded, _enc_lengths = self._model.encoder(audio_signal=processed, length=proc_lengths)
            # encoded: [B, D_enc, T''] — apply temporal mean pooling
            pooled = encoded.mean(dim=2)  # [B, D_enc]
            # L2-normalise to unit sphere (matches NeMo's speaker verification convention)
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return normed.cpu().numpy().astype(np.float32)
