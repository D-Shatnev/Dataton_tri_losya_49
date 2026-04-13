"""Pipeline interfaces (contracts)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    """
    Embeddings extractor.

    Encoder is a component that converts audio waveforms into fixed-size vectors
    (embeddings).

    Contract:
        - Input: batch of waveforms shaped [B, T] (float32), 16 kHz.
        - Output: embeddings shaped [B, D] (float32).

    Where:
        * B — batch size.
        * T — number of samples per item inside the batch.
        * D — embedding dimensionality (must be stable for a given encoder).

    The encoder implementation may internally normalize audio, run feature
    extraction, a neural network, etc.
    """

    @property
    def dim(self) -> int:
        """
        Embedding dimensionality D.

        Returns:
            The size of the output embedding vector.
        """
        raise NotImplementedError

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Embed a batch of waveforms.

        Args:
            batch_waveforms: float32 array shaped [B, T].

        Returns:
            float32 embeddings array shaped [B, D].
        """
        raise NotImplementedError


class WaveformLoader(Protocol):
    """
    Loads audio from disk and returns a mono float32 waveform.

    Contract:
        - Input: file path.
        - Output: 1-D float32 waveform shaped [T].

    Note:
        Sample rate conversion is **allowed** here (and commonly implemented by
        concrete loaders) so that the rest of the pipeline can assume 16 kHz.
    """

    def load(self, path: Path) -> np.ndarray:
        """
        Load a single audio file.

        Args:
            path: Audio file path.

        Returns:
            1-D mono waveform (float32).
        """
        raise NotImplementedError


class DatasetLoader(Protocol):
    """
    Dataset loader for speaker retrieval experiments.

    The loader is responsible for:
    - reading a CSV with file paths (and optionally speaker ids)
    - resolving relative paths against a root directory
    - loading audio as mono float32 waveforms at 16 kHz
    - applying deterministic length normalization (crop from start, repeat if shorter)

    Contract:
        - filepaths are in the same order as waveforms yielded by
          :meth:iter_waveforms.
        - waveforms yielded by :meth:iter_waveforms are 1-D float32 arrays.
        - labels are integer speaker ids aligned with filepaths.

    Notes:
        labels могут отсутствовать (None).
    """

    @property
    def filepaths(self) -> Sequence[str]:
        """Relative file paths (as read from the input CSV), aligned with data."""
        raise NotImplementedError

    @property
    def labels(self) -> np.ndarray | None:
        """
        Optional integer labels aligned with filepaths.

        Returns:
            int64 array shaped [N] or None if labels are not available.
        """
        raise NotImplementedError

    def iter_waveforms(self) -> Iterator[np.ndarray]:
        """
        Iterate over normalized waveforms.

        Yields:
            1-D mono waveform arrays (float32).
        """
        raise NotImplementedError


class Indexer(Protocol):
    """
    Builds nearest-neighbors indices for embeddings.

    Indexer implements *retrieval* step: for every embedding vector it returns
    indices of its closest neighbors in the same set.
    """

    def neighbors(self, embeddings: np.ndarray, topk: int) -> np.ndarray:
        """
        Compute top-k neighbors for each item.

        Args:
            embeddings: float32 array shaped [N, D].
            topk: Number of neighbors to return per item.

        Returns:
            int64 array shaped [N, topk] with neighbor indices.

        Notes:
            Implementations should avoid returning self-index (i -> i) in the
            first topk positions. If backend returns self-index, it can be
            moved to the end or filtered out.
        """
        raise NotImplementedError


class Evaluator(Protocol):
    """
    Evaluates retrieval quality.

    Evaluator compares predicted neighbor indices against ground-truth labels and
    computes retrieval metrics (e.g. Precision@K).
    """

    def evaluate(self, neighbors: np.ndarray, labels: np.ndarray, ks: list[int]) -> dict:
        """
        Compute metrics for given neighbor predictions.

        Args:
            neighbors: int64 array shaped [N, topk].
            labels: int64 array shaped [N] with speaker ids.
            ks: List of cutoffs (e.g. [1, 5, 10]) at which metrics are computed.

        Returns:
            Dictionary of metric values (JSON-serializable).
        """
        raise NotImplementedError
