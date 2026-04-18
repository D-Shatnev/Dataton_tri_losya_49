"""
Waveform loader based on soundfile.

This module contains SoundFileWaveformLoader.

The loader is intentionally small and side-effect free:

- Loads audio from disk.
- Converts to mono.
- Resamples to a configurable target sampling rate.
- Sanitizes NaNs/Infs.
- Optionally clips values to [-1, 1].

It is used by higher-level dataset loaders that handle CSV metadata and chunking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from dataton_tri_losya_49.constants import DEFAULT_TARGET_SR


@dataclass(frozen=True)
class SoundFileWaveformLoader:
    """
    Load audio with soundfile and resample to target SR (mono float32).

    Args:
        target_sr: Sampling rate to resample to.
        clip: If True, clip waveform values to [-1, 1]. Default is False to
            match the baseline behaviour (no explicit clipping).

    Note:
        The dataclass is frozen to prevent accidental mutation of configuration
        during a run (for reproducibility).
    """

    target_sr: int = DEFAULT_TARGET_SR
    clip: bool = False

    def load(self, path: Path) -> np.ndarray:
        """Load an audio file and return a mono waveform at target_sr.

        Args:
            path: Path to an audio file readable by soundfile.

        Returns:
            1-D mono waveform of shape (T,) with dtype float32.

        Notes:
            - Multi-channel audio is downmixed to mono by mean.
            - If the file sample rate differs from target_sr, audio is resampled
              using scipy.signal.resample_poly.
            - NaNs/Infs are replaced with zeros.
            - If clip=True, values are clipped to [-1, 1].

        Raises:
            ValueError: If the audio array has an unexpected number of dimensions.
        """
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            container = av.open(str(path))
            stream = container.streams.audio[0]
            sr = stream.sample_rate
            frames = [f.to_ndarray() for f in container.decode(stream)]
            container.close()
            raw = np.concatenate(frames, axis=-1).astype(np.float32)
            audio = (raw[0] if raw.ndim == 2 else raw) / 32768.0

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
        if self.clip:
            audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32, copy=False)
