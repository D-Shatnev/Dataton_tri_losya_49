"""
CSV-backed dataset loader that yields raw waveforms without length normalization.

This module provides :class:`RawCsvAudioDatasetLoader` — a variant of
:class:`~dataton_tri_losya_49.pipeline.components.loaders.CsvAudioDatasetLoader`
that skips the ``crop_or_pad_repeat_start`` step.

Use this loader when downstream processing (e.g. :class:`ChunkingEncoder`)
needs the full, untruncated waveform to perform its own segmentation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dataton_tri_losya_49.constants import DEFAULT_FILEPATH_COL, DEFAULT_SPEAKER_ID_COL
from dataton_tri_losya_49.pipeline.components.loaders.soundfile import SoundFileWaveformLoader


@dataclass(frozen=True)
class RawCsvAudioDatasetLoader:
    """
    Load dataset from CSV and audio files without length normalization.

    Reads a CSV table with relative file paths and optional speaker ids,
    resolves paths against a root directory, loads audio as float32 waveforms
    at target sample rate, and yields them as-is (no cropping or padding).

    This loader is intended for use with :class:`ChunkingEncoder`, which
    performs its own segmentation of the full waveform.

    Args:
        csv_path: Path to a CSV file.
        root: Root directory used to resolve relative file paths from CSV.
        filepath_col: Name of the CSV column containing audio file paths.
        speaker_id_col: Name of the CSV column containing speaker ids.
            If missing, labels are None.
        loader: Audio loader implementation. Defaults to SoundFileWaveformLoader.
    """

    csv_path: Path
    root: Path
    filepath_col: str = DEFAULT_FILEPATH_COL
    speaker_id_col: str = DEFAULT_SPEAKER_ID_COL
    loader: SoundFileWaveformLoader = field(default_factory=SoundFileWaveformLoader)

    _df: pd.DataFrame = field(init=False, repr=False)
    _filepaths: list[str] = field(init=False, repr=False)
    _labels: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs, read CSV metadata and pre-compute derived fields."""
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

    def iter_waveforms(self) -> Iterator[np.ndarray]:
        """Iterate over raw waveforms (mono, resampled, no length normalization)."""
        for fp in self._filepaths:
            yield self.loader.load(self.root / fp)

    def _make_labels(self, df: pd.DataFrame) -> np.ndarray | None:
        """Build deterministic integer labels from speaker_id_col if present."""
        if self.speaker_id_col not in df.columns:
            return None

        speakers = sorted(df[self.speaker_id_col].astype(str).unique())
        spk2id = {s: i for i, s in enumerate(speakers)}
        return df[self.speaker_id_col].astype(str).map(spk2id).to_numpy(dtype=np.int64)
