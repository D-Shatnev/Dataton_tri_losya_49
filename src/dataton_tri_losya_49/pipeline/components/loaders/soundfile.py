"""
Waveform loader based on *soundfile*.

This module defines :class:~dataton_tri_losya_49.pipeline.components.loaders.soundfile.SoundFileWaveformLoader.

The loader is intentionally small and side-effect free:

- Loads audio from disk using soundfile.
- Converts to mono.
- Resamples to a configurable target sampling rate.
- Sanitizes NaNs/Infs and clips to [-1, 1].

It is used by higher-level dataset loaders that handle CSV metadata and chunking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from dataton_tri_losya_49.constants import DEFAULT_TARGET_SR


@dataclass(frozen=True)
class SoundFileWaveformLoader:
    """Load audio with soundfile and resample to target SR (mono float32)."""

    target_sr: int = DEFAULT_TARGET_SR

    def load(self, path: Path) -> np.ndarray:
        """
        Load an audio file and return a mono waveform at target_sr.

        Args:
            path: Path to an audio file readable by soundfile.

        Returns:
            np.ndarray: 1-D mono waveform of shape (T,) with dtype float32.

        Raises:
            ValueError: If the audio array has an unexpected number of dimensions.
        """
        audio, sr = sf.read(path, dtype="float32", always_2d=False)

        # to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            raise ValueError(f"Unexpected audio ndim={audio.ndim} for file: {path}")

        # resample if needed
        if int(sr) != int(self.target_sr):
            gcd = int(np.gcd(int(sr), int(self.target_sr)))
            up = int(self.target_sr) // gcd
            down = int(sr) // gcd
            audio = resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32, copy=False)
