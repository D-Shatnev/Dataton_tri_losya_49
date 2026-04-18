"""
Chunking encoder with Simple Average Pooling.

This module provides :class:`ChunkingEncoder` — a decorator around any
:class:`~dataton_tri_losya_49.pipeline.interfaces.Encoder` that:

1. Splits each input waveform into overlapping fixed-size chunks.
2. Embeds each chunk individually (one encoder call per chunk).
3. Averages the resulting embeddings (Simple Average Pooling).

Calling the inner encoder with a fixed shape ``[1, chunk_samples]`` per chunk
ensures compatibility with ``torch.compile(backend="cudagraphs")``, which
requires stable tensor shapes across calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dataton_tri_losya_49.pipeline.utils import chunk_waveform


@dataclass
class ChunkingEncoder:
    """
    Wrap any Encoder to embed full waveforms via overlapping chunks + average pooling.

    Each waveform is split into overlapping chunks of ``chunk_duration_s`` seconds
    with a step of ``(chunk_duration_s - chunk_overlap_s)`` seconds.  Every chunk
    is embedded individually with a ``[1, chunk_samples]`` call to the inner
    encoder (fixed shape — safe for cudagraphs).  The resulting chunk embeddings
    are averaged (Simple Average Pooling) to produce one embedding per waveform.

    Args:
        encoder: Any object implementing the Encoder protocol
            (must have ``dim`` property and ``embed`` method).
        chunk_duration_s: Chunk length in seconds. Default: 4.0.
        chunk_overlap_s: Overlap between consecutive chunks in seconds.
            Must be strictly less than ``chunk_duration_s``. Default: 2.0.
        sample_rate: Audio sample rate in Hz used to convert seconds to samples.
            Must match the sample rate of waveforms passed to :meth:`embed`.
            Default: 16000.
    """

    encoder: Any
    chunk_duration_s: float = 4.0
    chunk_overlap_s: float = 2.0
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        """Validate chunking parameters."""
        if self.chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be > 0")
        if self.chunk_overlap_s < 0:
            raise ValueError("chunk_overlap_s must be >= 0")
        if self.chunk_overlap_s >= self.chunk_duration_s:
            raise ValueError("chunk_overlap_s must be < chunk_duration_s")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

    @property
    def dim(self) -> int:
        """Embedding dimensionality D (delegated to inner encoder).

        Returns:
            The size of the output embedding vector.
        """
        return self.encoder.dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Embed a batch of waveforms via chunking and Simple Average Pooling.

        Each waveform in the batch is processed independently:
        1. Split into overlapping chunks of ``chunk_samples`` samples.
        2. Each chunk is embedded with a single ``[1, chunk_samples]`` call
           to the inner encoder (fixed shape for cudagraphs stability).
        3. Chunk embeddings are averaged element-wise (Simple Average Pooling).

        Args:
            batch_waveforms: float32 array shaped [B, T] at ``sample_rate`` Hz.
                When called from the standard pipeline with ``batch_size=1``,
                B=1 and T equals the true waveform length (no zero-padding).

        Returns:
            float32 embeddings array shaped [B, D].

        Raises:
            ValueError: If input is not a 2-D array.
        """
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B, T], got shape {x.shape}")

        chunk_samples = int(self.chunk_duration_s * self.sample_rate)
        hop_samples = int((self.chunk_duration_s - self.chunk_overlap_s) * self.sample_rate)

        results: list[np.ndarray] = []
        for wav in x:
            chunks = chunk_waveform(wav, chunk_samples, hop_samples)  # [N, chunk_samples]
            chunk_embs = self._embed_chunks(chunks)                    # [N, D]
            results.append(chunk_embs.mean(axis=0))                    # [D]

        return np.stack(results, axis=0)  # [B, D]

    def _embed_chunks(self, chunks: np.ndarray) -> np.ndarray:
        """
        Embed each chunk individually with a fixed [1, chunk_samples] call.

        Calling the inner encoder one chunk at a time keeps the input shape
        constant across all files and all chunks, which is required for
        ``torch.compile(backend="cudagraphs")`` to reuse the recorded graph
        without recompilation.

        Args:
            chunks: float32 array shaped [N, chunk_samples].

        Returns:
            float32 embeddings shaped [N, D].
        """
        embs: list[np.ndarray] = []
        for chunk in chunks:
            emb = self.encoder.embed(chunk[np.newaxis])  # [1, chunk_samples] -> [1, D]
            embs.append(emb[0])                          # [D]
        return np.stack(embs, axis=0)                    # [N, D]
