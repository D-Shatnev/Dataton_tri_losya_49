"""Shared pipeline utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_path(p: Path) -> Path:
    """
    Resolve a path to absolute.

    Args:
        p: Input path.

    Returns:
        ``p`` if it is already absolute, otherwise ``Path(p).resolve()``.
    """
    return p if p.is_absolute() else Path(p).resolve()


def chunk_waveform(wav: np.ndarray, chunk_samples: int, hop_samples: int) -> np.ndarray:
    """
    Split a 1-D waveform into overlapping fixed-size chunks.

    Uses a sliding window with step ``hop_samples``. The last chunk (and any
    chunk shorter than ``chunk_samples``) is repeat-padded to exactly
    ``chunk_samples`` samples.  If the waveform is shorter than
    ``chunk_samples``, a single repeat-padded chunk is returned.

    Args:
        wav: 1-D float32 waveform of shape (T,).
        chunk_samples: Desired chunk length in samples.
        hop_samples: Step between consecutive chunk start positions in samples.

    Returns:
        float32 array of shape (N, chunk_samples) where N >= 1.

    Raises:
        ValueError: If chunk_samples or hop_samples are not positive.
    """
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be > 0")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be > 0")

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(wav.shape[0])

    if n == 0:
        return np.zeros((1, chunk_samples), dtype=np.float32)

    if n <= chunk_samples:
        return _repeat_pad(wav, chunk_samples)[np.newaxis]

    chunks: list[np.ndarray] = []
    start = 0
    while start < n:
        end = start + chunk_samples
        chunk = wav[start:end]
        if chunk.shape[0] < chunk_samples:
            chunk = _repeat_pad(chunk, chunk_samples)
        chunks.append(chunk)
        start += hop_samples

    return np.stack(chunks, axis=0)


def _repeat_pad(wav: np.ndarray, target: int) -> np.ndarray:
    """
    Repeat-pad a 1-D waveform to exactly ``target`` samples.

    Args:
        wav: 1-D float32 waveform, length <= target.
        target: Desired output length in samples.

    Returns:
        float32 array of shape (target,).
    """
    n = int(wav.shape[0])
    if n == 0:
        return np.zeros(target, dtype=np.float32)
    reps = target // n + 1
    return np.tile(wav, reps)[:target].astype(np.float32, copy=False)
