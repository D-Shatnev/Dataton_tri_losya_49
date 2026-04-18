"""
FunASR VAD-aware waveform loader.

This module provides :class:`FunASRVadWaveformLoader` — a drop-in replacement for
any :class:`~dataton_tri_losya_49.pipeline.interfaces.WaveformLoader` implementation
that runs FunASR FSMN-VAD on each audio file before returning the waveform.

Processing steps:
    1. Load raw waveform (float32) via the injected ``base_loader``.
    2. Convert to int16 and pass to FunASR ``AutoModel.generate()`` as a numpy array.
    3. Concatenate all detected speech segments into a single waveform.
    4. If no speech is detected, fall back to the original waveform and log a warning.

The speech probability threshold is hardcoded to 0.7 (``SPEECH_THRESHOLD`` class
variable) and the model is hardcoded to ``iic/speech_fsmn_vad_zh-cn-16k-common-pytorch``
(``MODEL_ID`` class variable).

The loader accumulates total VAD wall-clock time in ``vad_time_s`` so that
:func:`~dataton_tri_losya_49.pipeline.runner.run_experiment` can report it
separately from encoder time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from dataton_tri_losya_49.constants import DEFAULT_TARGET_SR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunASRVadWaveformLoader:
    """
    Load audio via a base loader, strip silence via FunASR FSMN-VAD, return speech-only waveform.

    Implements the same interface as any
    :class:`~dataton_tri_losya_49.pipeline.interfaces.WaveformLoader`
    (``load(path) -> np.ndarray``).

    The audio file is read **once** by ``base_loader`` and the resulting waveform
    is passed directly to FunASR VAD as an int16 numpy array — no second disk read occurs.

    Speech segments detected by VAD are concatenated into a single waveform.
    If VAD finds no speech the original waveform is returned unchanged and a
    warning is emitted.

    Total VAD inference time is accumulated in :attr:`vad_time_s` and can be
    read by the runner after the full dataset pass.

    Class variables:
        MODEL_ID: FunASR model identifier (ModelScope / HuggingFace hub).
        SPEECH_THRESHOLD: Hardcoded speech probability threshold (0.7).

    Args:
        base_loader: Pre-constructed waveform loader used to read audio files.
            Any object implementing ``load(path: Path) -> np.ndarray`` is accepted.
        target_sr: Sample rate of the waveforms produced by ``base_loader``.
            Must match ``base_loader.target_sr``. Used to convert VAD timestamps
            (milliseconds) to sample indices.
        use_gpu: Whether to run VAD inference on GPU.
    """

    MODEL_ID: ClassVar[str] = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    SPEECH_THRESHOLD: ClassVar[float] = 0.7

    base_loader: Any  # WaveformLoader protocol; Any avoids circular import
    target_sr: int = DEFAULT_TARGET_SR
    use_gpu: bool = False

    # Private fields — initialised in __post_init__, not part of the public API.
    _model: Any = field(init=False, repr=False)
    _vad_time_s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise FunASR VAD model and timing accumulator."""
        from funasr import AutoModel  # pylint: disable=import-outside-toplevel

        device = "cuda" if self.use_gpu else "cpu"
        model = AutoModel(
            model=self.MODEL_ID,
            device=device,
            disable_update=True,
        )
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_vad_time_s", 0.0)

    @property
    def vad_time_s(self) -> float:
        """Accumulated VAD wall-clock time in seconds across all :meth:`load` calls."""
        return self._vad_time_s

    def load(self, path: Path) -> np.ndarray:
        """
        Load audio and return speech-only waveform at target_sr.

        The file is read once via the base loader. The float32 waveform is
        converted to int16 and passed to FunASR FSMN-VAD. Detected speech
        segments (returned as ``[begin_ms, end_ms]`` pairs) are sliced from
        the original float32 waveform and concatenated.

        Args:
            path: Path to an audio file readable by the base loader.

        Returns:
            1-D mono float32 waveform containing only detected speech segments
            concatenated together. Falls back to the full waveform if VAD
            detects no speech.
        """
        wav = self.base_loader.load(path)

        # FunASR AutoModel.generate() accepts int16 numpy array.
        wav_int16 = (wav * 32767.0).clip(-32768, 32767).astype(np.int16)

        t0 = time.perf_counter()
        results = self._model.generate(
            input=wav_int16,
            input_len=np.array([len(wav_int16)]),
            disable_pbar=True,
        )
        elapsed = time.perf_counter() - t0
        object.__setattr__(self, "_vad_time_s", self._vad_time_s + elapsed)

        # FunASR returns a list of dicts; first element contains "value" key
        # with a list of [begin_ms, end_ms] segment pairs.
        segments_ms: list[list[int]] = []
        if results and isinstance(results, list) and results[0]:
            raw = results[0]
            if isinstance(raw, dict):
                segments_ms = raw.get("value", []) or []

        if not segments_ms:
            logger.warning("FunASR VAD: no speech detected in %s, using original waveform", path)
            return wav

        ms_to_samples = self.target_sr / 1000.0
        segments = [
            wav[int(begin_ms * ms_to_samples) : int(end_ms * ms_to_samples)]
            for begin_ms, end_ms in segments_ms
        ]
        # Filter out empty slices that could arise from rounding edge cases.
        segments = [s for s in segments if s.size > 0]

        if not segments:
            logger.warning(
                "FunASR VAD: all extracted segments are empty for %s, using original waveform",
                path,
            )
            return wav

        return np.concatenate(segments).astype(np.float32, copy=False)
