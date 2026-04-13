"""
CSV-backed dataset loader for speaker recognition pipelines.

This module provides :class:~dataton_tri_losya_49.pipeline.components.loaders.csv_audio_dataset.CsvAudioDatasetLoader:

- Reads metadata from a CSV file (relative audio paths, optional speaker ids).
- Resolves paths against a configurable root directory.
- Loads waveforms via :class:~dataton_tri_losya_49.pipeline.components.loaders.soundfile.SoundFileWaveformLoader.
- Normalizes duration deterministically using :func:crop_or_pad_repeat_start.

Deterministic chunking is important for reproducible evaluation:

- If the audio is longer than the target duration, we crop from the start.
- If the audio is shorter, we repeat (tile) it to reach the target length.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dataton_tri_losya_49.constants import DEFAULT_CHUNK_SECONDS, DEFAULT_FILEPATH_COL, DEFAULT_SPEAKER_ID_COL
from dataton_tri_losya_49.pipeline.components.loaders.soundfile import SoundFileWaveformLoader


def crop_or_pad_repeat_start(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    """Normalize waveform length to exactly num_samples.

    This function follows the baseline idea of deterministic chunking:
    - if waveform is longer than num_samples, it is cropped from the start
    - if waveform is shorter, it is repeated (tiled) until enough samples are available
    - if waveform has zero length, it returns zeros

    Parameters
    ----------
    waveform:
        Mono waveform, shape (T,)
    num_samples:
        Desired number of samples in output

    Returns
    -------
    np.ndarray
        Waveform of shape (num_samples,) and dtype float32
    """

    target = int(num_samples)
    if target <= 0:
        raise ValueError("num_samples must be > 0")

    w = np.asarray(waveform, dtype=np.float32).reshape(-1)
    n = int(w.shape[0])

    if n >= target:
        return w[:target].astype(np.float32, copy=False)

    if n == 0:
        return np.zeros(target, dtype=np.float32)

    reps = target // n + 1
    out = np.tile(w, reps)[:target]
    return out.astype(np.float32, copy=False)


@dataclass(frozen=True)
class CsvAudioDatasetLoader:
    """
    Load dataset from CSV and audio files.

    The loader reads a CSV table with relative file paths and optional speaker ids,
    resolves paths against root directory, loads audio as float32 waveforms at
    target sample rate and applies deterministic length normalization.

    This component is designed for reproducible evaluation and inference.

    Parameters
    ----------
    csv_path:
        Path to a CSV file
    root:
        Root directory used to resolve relative file paths from CSV
    filepath_col:
        Name of the CSV column containing audio file paths
    speaker_id_col:
        Name of the CSV column containing speaker ids. If missing, labels are None
    chunk_seconds:
        Fixed chunk length in seconds. If audio is longer, it is cropped from start.
        If shorter, it is repeated
    loader:
        Audio loader implementation. Defaults to SoundFileWaveformLoader
    """

    csv_path: Path
    root: Path
    filepath_col: str = DEFAULT_FILEPATH_COL
    speaker_id_col: str = DEFAULT_SPEAKER_ID_COL
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    loader: SoundFileWaveformLoader = field(default_factory=SoundFileWaveformLoader)

    _df: pd.DataFrame = field(init=False, repr=False)
    _filepaths: list[str] = field(init=False, repr=False)
    _labels: np.ndarray | None = field(init=False, repr=False)
    _num_samples: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs, read CSV metadata and pre-compute derived fields."""
        if float(self.chunk_seconds) <= 0:
            raise ValueError("chunk_seconds must be > 0")

        csv_path = self.csv_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if self.filepath_col not in df.columns:
            cols = ", ".join(str(c) for c in df.columns)
            raise ValueError(f"CSV {csv_path} must contain column {self.filepath_col}. Got: [{cols}]")

        filepaths = df[self.filepath_col].astype(str).tolist()
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_filepaths", filepaths)
        object.__setattr__(self, "_labels", self._make_labels(df))

        num_samples = int(self.chunk_seconds * self.loader.target_sr)
        object.__setattr__(self, "_num_samples", num_samples)

    @property
    def df(self) -> pd.DataFrame:
        """Underlying pandas DataFrame read from CSV (kept for debugging/analysis)."""
        return self._df

    @property
    def filepaths(self) -> Sequence[str]:
        """Relative audio file paths from the CSV, in the original row order."""
        return self._filepaths

    @property
    def labels(self) -> np.ndarray | None:
        """Integer speaker labels aligned with filepaths or None if unavailable."""
        return self._labels

    @property
    def num_samples(self) -> int:
        """Fixed chunk length in samples used by :meth:iter_waveforms."""
        return self._num_samples

    def iter_waveforms(self) -> Iterator[np.ndarray]:
        """Iterate over normalized waveforms (mono, resampled, cropped/padded)."""
        for fp in self._filepaths:
            wav = self.loader.load(self.root / fp)
            yield crop_or_pad_repeat_start(wav, self._num_samples)

    def _make_labels(self, df: pd.DataFrame) -> np.ndarray | None:
        """Build deterministic integer labels from speaker_id_col if present."""
        if self.speaker_id_col not in df.columns:
            return None

        speakers = sorted(df[self.speaker_id_col].astype(str).unique())
        spk2id = {s: i for i, s in enumerate(speakers)}
        return df[self.speaker_id_col].astype(str).map(spk2id).to_numpy(dtype=np.int64)
