"""
VAD-aware waveform loader.

This module provides :class:`VadWaveformLoader` — a drop-in replacement for
:class:`~dataton_tri_losya_49.pipeline.components.loaders.soundfile.SoundFileWaveformLoader`
that runs FireRedVAD on each audio file before returning the waveform.

Processing steps:
    1. Load raw waveform (float32) via :class:`SoundFileWaveformLoader`.
    2. Convert to int16 and pass directly to :class:`fireredvad.FireRedVad` as
       ``(wav_int16, sample_rate)`` tuple — avoids a second disk read.
    3. Concatenate all detected speech segments into a single waveform.
    4. If no speech is detected, fall back to the original waveform and log a warning.

The loader accumulates total VAD wall-clock time in ``vad_time_s`` so that
:func:`~dataton_tri_losya_49.pipeline.runner.run_experiment` can report it
separately from encoder time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fireredvad import FireRedVad, FireRedVadConfig

from dataton_tri_losya_49.constants import DEFAULT_TARGET_SR
from dataton_tri_losya_49.pipeline.components.loaders.soundfile import SoundFileWaveformLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VadWaveformLoader:
    """
    Load audio, strip silence via FireRedVAD, return speech-only waveform.

    Implements the same interface as
    :class:`~dataton_tri_losya_49.pipeline.components.loaders.soundfile.SoundFileWaveformLoader`
    (``load(path) -> np.ndarray``).

    The audio file is read **once** by :class:`SoundFileWaveformLoader` and the
    resulting waveform is passed directly to FireRedVAD as an ``(int16, sr)``
    tuple — no second disk read occurs.

    Speech segments detected by VAD are concatenated into a single waveform.
    If VAD finds no speech the original waveform is returned unchanged and a
    warning is emitted.

    Total VAD inference time is accumulated in :attr:`vad_time_s` and can be
    read by the runner after the full dataset pass.

    Args:
        target_sr: Target sample rate in Hz. Audio is resampled if necessary.
        clip: If True, clip waveform values to [-1, 1].
        vad_model_dir: Path to the FireRedVAD model directory
            (e.g. ``models/FireRedVAD/VAD``).
        vad_use_gpu: Whether to run VAD inference on GPU.
        vad_speech_threshold: Probability threshold for speech detection.
        vad_min_speech_frame: Minimum number of consecutive frames to be
            considered speech.
    """

    target_sr: int = DEFAULT_TARGET_SR
    clip: bool = False
    vad_model_dir: Path = Path("models/FireRedVAD/VAD")
    vad_use_gpu: bool = False
    vad_speech_threshold: float = 0.4
    vad_min_speech_frame: int = 20

    # Private fields — initialised in __post_init__, not part of the public API.
    _base_loader: SoundFileWaveformLoader = field(init=False, repr=False)
    _vad: FireRedVad = field(init=False, repr=False)
    _vad_time_s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise base loader, VAD model and timing accumulator."""
        base_loader = SoundFileWaveformLoader(target_sr=self.target_sr, clip=self.clip)
        object.__setattr__(self, "_base_loader", base_loader)

        vad_config = FireRedVadConfig(
            use_gpu=self.vad_use_gpu,
            speech_threshold=self.vad_speech_threshold,
            min_speech_frame=self.vad_min_speech_frame,
        )
        vad = FireRedVad.from_pretrained(str(self.vad_model_dir), vad_config)
        object.__setattr__(self, "_vad", vad)

        object.__setattr__(self, "_vad_time_s", 0.0)

    @property
    def vad_time_s(self) -> float:
        """Accumulated VAD wall-clock time in seconds across all :meth:`load` calls."""
        return self._vad_time_s

    def load(self, path: Path) -> np.ndarray:
        """
        Load audio and return speech-only waveform at target_sr.

        The file is read once via the base loader. The float32 waveform is
        converted to int16 and passed to FireRedVAD as ``(wav_int16, sr)``
        to avoid a second disk read.

        Args:
            path: Path to an audio file readable by soundfile.

        Returns:
            1-D mono float32 waveform containing only detected speech segments
            concatenated together. Falls back to the full waveform if VAD
            detects no speech.
        """
        wav = self._base_loader.load(path)

        # FireRedVAD's AudioFeat.extract() accepts (wav_np, sample_rate) tuple
        # where wav_np must be int16 (matches its sf.read(..., dtype="int16") path).
        wav_int16 = (wav * 32767.0).clip(-32768, 32767).astype(np.int16)

        t0 = time.perf_counter()
        result, _ = self._vad.detect((wav_int16, self.target_sr))
        elapsed = time.perf_counter() - t0
        object.__setattr__(self, "_vad_time_s", self._vad_time_s + elapsed)

        timestamps: list[tuple[float, float]] = result.get("timestamps", [])

        if not timestamps:
            logger.warning("VAD: no speech detected in %s, using original waveform", path)
            return wav

        segments = [
            wav[int(start * self.target_sr) : int(end * self.target_sr)]
            for start, end in timestamps
        ]
        # Filter out empty slices that could arise from rounding edge cases.
        segments = [s for s in segments if s.size > 0]

        if not segments:
            logger.warning(
                "VAD: all extracted segments are empty for %s, using original waveform", path
            )
            return wav

        return np.concatenate(segments).astype(np.float32, copy=False)
