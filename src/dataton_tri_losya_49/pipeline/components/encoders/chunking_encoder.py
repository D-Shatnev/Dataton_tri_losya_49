"""
Chunking encoder with Simple Average Pooling.

This module provides :class:`ChunkingEncoder` — a decorator around any
:class:`~dataton_tri_losya_49.pipeline.interfaces.Encoder` that:

1. Splits each input waveform into overlapping fixed-size chunks.
2. Embeds all chunks in a single batched encoder call (up to ``max_chunk_batch``
   chunks at a time).
3. Averages the resulting embeddings (Simple Average Pooling).

Batching all chunks of a waveform into a single ``[N, chunk_samples]`` call
reduces the number of GPU kernel launches from N to 1 per waveform, which is
the dominant speedup for long audio files.
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
    with a step of ``(chunk_duration_s - chunk_overlap_s)`` seconds.  All chunks
    are embedded in a single batched call ``[N, chunk_samples]`` to the inner
    encoder (or in sub-batches of ``max_chunk_batch`` if N is large).  The
    resulting chunk embeddings are averaged (Simple Average Pooling) to produce
    one embedding per waveform.

    Args:
        encoder: Any object implementing the Encoder protocol
            (must have ``dim`` property and ``embed`` method).
        chunk_duration_s: Chunk length in seconds. Default: 4.0.
        chunk_overlap_s: Overlap between consecutive chunks in seconds.
            Must be strictly less than ``chunk_duration_s``. Default: 2.0.
        sample_rate: Audio sample rate in Hz used to convert seconds to samples.
            Must match the sample rate of waveforms passed to :meth:`embed`.
            Default: 16000.
        max_chunk_batch: Maximum number of chunks to pass to the inner encoder
            in a single call.  Limits peak VRAM usage for very long audio files.
            Set to 0 to disable the limit (all chunks in one call).  Default: 32.
    """

    encoder: Any
    chunk_duration_s: float = 4.0
    chunk_overlap_s: float = 2.0
    sample_rate: int = 16000
    max_chunk_batch: int = 32

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
        if self.max_chunk_batch < 0:
            raise ValueError("max_chunk_batch must be >= 0")

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
        2. All chunks are embedded in a single batched ``[N, chunk_samples]``
           call to the inner encoder (or in sub-batches of ``max_chunk_batch``).
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
        Embed all chunks in a single batched encoder call.

        Passes all N chunks as a single ``[N, chunk_samples]`` array to the
        inner encoder, reducing GPU kernel launches from N to 1.  When
        ``max_chunk_batch > 0`` and N exceeds it, chunks are split into
        sub-batches to cap peak VRAM usage.

        Args:
            chunks: float32 array shaped [N, chunk_samples].

        Returns:
            float32 embeddings shaped [N, D].
        """
        n = chunks.shape[0]
        cap = self.max_chunk_batch if self.max_chunk_batch > 0 else n

        if n <= cap:
            return self.encoder.embed(chunks)  # [N, chunk_samples] -> [N, D]

        # Split into sub-batches to avoid OOM on very long files.
        parts: list[np.ndarray] = []
        for start in range(0, n, cap):
            parts.append(self.encoder.embed(chunks[start : start + cap]))
        return np.concatenate(parts, axis=0)  # [N, D]
