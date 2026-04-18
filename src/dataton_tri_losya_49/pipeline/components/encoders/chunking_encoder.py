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

        All waveforms in the batch are chunked and their chunks are concatenated
        into a single ``[total_chunks, chunk_samples]`` array for a single
        batched encoder call.  Results are then split back per waveform and
        averaged (Simple Average Pooling).  This cross-file batching keeps the
        GPU saturated even when individual files are short.

        Args:
            batch_waveforms: float32 array shaped [B, T] at ``sample_rate`` Hz.

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

        # Chunk all waveforms and record per-file chunk counts.
        all_chunks: list[np.ndarray] = []
        chunk_counts: list[int] = []
        for wav in x:
            chunks = chunk_waveform(wav, chunk_samples, hop_samples)  # [N_i, chunk_samples]
            all_chunks.append(chunks)
            chunk_counts.append(chunks.shape[0])

        # Single GPU call for all chunks across all waveforms in the batch.
        combined = np.concatenate(all_chunks, axis=0)  # [sum(N_i), chunk_samples]
        all_embs = self._embed_chunks(combined)         # [sum(N_i), D]

        # Split back per waveform and average (Simple Average Pooling).
        results: list[np.ndarray] = []
        offset = 0
        for n in chunk_counts:
            results.append(all_embs[offset : offset + n].mean(axis=0))  # [D]
            offset += n

        return np.stack(results, axis=0)  # [B, D]

    def _embed_chunks(self, chunks: np.ndarray) -> np.ndarray:
        """
        Embed all chunks in fixed-size sub-batches.

        When ``max_chunk_batch > 0``, the input is split into sub-batches of
        exactly ``max_chunk_batch`` rows.  Each sub-batch is zero-padded to
        ``max_chunk_batch`` so that ``torch.compile`` always receives a tensor
        of the same shape and never triggers recompilation.  Only the real
        (non-padding) rows are kept in the output.

        When ``max_chunk_batch == 0``, all chunks are passed in a single call
        (original behaviour, no padding).

        Args:
            chunks: float32 array shaped [N, chunk_samples].

        Returns:
            float32 embeddings shaped [N, D].
        """
        n = chunks.shape[0]
        cap = self.max_chunk_batch if self.max_chunk_batch > 0 else n

        if n <= cap:
            if n == cap:
                # Exact fit — no padding needed, stable shape for torch.compile.
                return self.encoder.embed(chunks)
            # Pad to cap so torch.compile always sees [cap, chunk_samples].
            pad = np.zeros((cap - n, chunks.shape[1]), dtype=np.float32)
            padded = np.concatenate([chunks, pad], axis=0)
            return self.encoder.embed(padded)[:n]

        # Split into fixed-size sub-batches; last sub-batch is padded to cap.
        parts: list[np.ndarray] = []
        for start in range(0, n, cap):
            sub = chunks[start : start + cap]
            real = sub.shape[0]
            if real < cap:
                pad = np.zeros((cap - real, chunks.shape[1]), dtype=np.float32)
                sub = np.concatenate([sub, pad], axis=0)
            parts.append(self.encoder.embed(sub)[:real])
        return np.concatenate(parts, axis=0)  # [N, D]
