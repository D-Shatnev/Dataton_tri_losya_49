"""Shared pipeline utilities."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np

_T = TypeVar("_T")

_SENTINEL = object()


def prefetch_iter(
    source: Iterator[_T],
    load_fn: Callable[[_T], np.ndarray],
    prefetch: int = 2,
) -> Iterator[np.ndarray]:
    """Wrap a source iterator with background prefetching via a thread pool.

    Submits ``prefetch`` items ahead of consumption to a background thread so
    that disk I/O overlaps with GPU inference.  The ordering of results is
    preserved — items are yielded in the same order as the source iterator.

    Args:
        source: Iterator that yields items consumed by ``load_fn`` (e.g. file paths).
        load_fn: Callable that converts one source item to a waveform array.
            Called in a background thread; must be thread-safe (soundfile is).
        prefetch: Number of items to load ahead of the consumer.  Must be >= 1.
            A value of 2 means one item is being loaded while the previous is
            being processed.

    Yields:
        Loaded waveform arrays in source order.

    Raises:
        ValueError: If prefetch < 1.
        Exception: Re-raises any exception raised inside the background thread.
    """
    if prefetch < 1:
        raise ValueError("prefetch must be >= 1")

    result_queue: queue.Queue = queue.Queue(maxsize=prefetch)

    def _producer() -> None:
        try:
            for item in source:
                result_queue.put(load_fn(item))
        except Exception as exc:  # noqa: BLE001
            result_queue.put(exc)
        finally:
            result_queue.put(_SENTINEL)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        value = result_queue.get()
        if value is _SENTINEL:
            break
        if isinstance(value, Exception):
            raise value
        yield value

    thread.join()


def prefetch_chunk_batches(
    waveforms: Iterator[np.ndarray],
    chunk_samples: int,
    hop_samples: int,
    batch_size: int,
    prefetch: int = 4,
    padded_total: int = 0,
) -> Iterator[tuple[np.ndarray, list[int]]]:
    """Pre-chunk waveforms into GPU-ready batches in a background thread.

    Reads waveforms from ``waveforms``, slices each into overlapping chunks,
    and accumulates chunks until ``batch_size`` waveforms have been processed.
    The resulting ``[total_chunks, chunk_samples]`` array is placed into a
    queue so the GPU thread can consume it immediately without waiting for I/O
    or CPU chunking of the next batch.

    This decouples CPU work (I/O + chunking) from GPU work (encoder forward
    pass), keeping the GPU saturated even when individual files are short.

    The combined chunk array is zero-padded to a fixed size so that
    ``torch.compile`` always sees the same tensor shape and never triggers
    recompilation mid-run.  The ``chunk_counts`` list tells the consumer how
    many rows are real so it can ignore the padding rows.

    When ``padded_total > 0`` the fixed size is known upfront (computed from
    ``max_audio_duration_s`` in the caller) and is used from the very first
    batch, guaranteeing a single compilation.  When ``padded_total == 0`` the
    size is inferred from the first full batch (legacy behaviour).

    If a batch produces more chunks than ``padded_total`` (e.g. an unexpectedly
    long file), the array is truncated to ``padded_total`` rows.  The
    ``chunk_counts`` list is adjusted accordingly so the consumer never reads
    beyond the real data.

    Args:
        waveforms: Iterator of 1-D float32 waveform arrays (one per file).
        chunk_samples: Chunk length in samples.
        hop_samples: Hop length in samples between consecutive chunks.
        batch_size: Number of waveforms to accumulate per GPU batch.
        prefetch: Number of pre-chunked batches to buffer ahead.  Must be >= 1.
        padded_total: Fixed number of rows in every output chunk array.
            When > 0 this value is used from the first batch so the GPU always
            sees the same shape.  When 0 the size is fixed on the first full
            batch (legacy behaviour).

    Yields:
        Tuples of ``(chunks, chunk_counts)`` where:
        - ``chunks`` is a float32 array shaped ``[padded_total, chunk_samples]``
          padded to a fixed size so the GPU always sees the same shape.
        - ``chunk_counts`` is a list of ints recording how many chunks each
          waveform contributed (padding rows are ignored by the consumer).

    Raises:
        ValueError: If prefetch < 1 or batch_size < 1.
        Exception: Re-raises any exception raised inside the background thread.
    """
    if prefetch < 1:
        raise ValueError("prefetch must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    result_queue: queue.Queue = queue.Queue(maxsize=prefetch)

    def _producer() -> None:
        # Use the caller-supplied fixed size when available; otherwise infer
        # from the first full batch (legacy path).
        fixed_total: int = padded_total

        try:
            batch_chunks: list[np.ndarray] = []
            batch_counts: list[int] = []

            for wav in waveforms:
                chunks = chunk_waveform(wav, chunk_samples, hop_samples)
                batch_chunks.append(chunks)
                batch_counts.append(chunks.shape[0])

                if len(batch_counts) == batch_size:
                    combined = np.concatenate(batch_chunks, axis=0)
                    real_total = combined.shape[0]

                    # Fix padded size on first full batch when not pre-supplied.
                    if fixed_total == 0:
                        fixed_total = real_total

                    combined = _pad_or_trim(combined, fixed_total, chunk_samples, batch_counts)

                    result_queue.put((combined, batch_counts))
                    batch_chunks = []
                    batch_counts = []

            # Flush remaining waveforms — pad/trim to the same fixed size.
            if batch_counts:
                combined = np.concatenate(batch_chunks, axis=0)
                real_total = combined.shape[0]
                if fixed_total == 0:
                    fixed_total = real_total
                combined = _pad_or_trim(combined, fixed_total, chunk_samples, batch_counts)
                result_queue.put((combined, batch_counts))

        except Exception as exc:  # noqa: BLE001
            result_queue.put(exc)
        finally:
            result_queue.put(_SENTINEL)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        value = result_queue.get()
        if value is _SENTINEL:
            break
        if isinstance(value, Exception):
            raise value
        yield value

    thread.join()


def _pad_or_trim(
    combined: np.ndarray,
    fixed_total: int,
    chunk_samples: int,
    chunk_counts: list[int],
) -> np.ndarray:
    """Pad or trim ``combined`` to exactly ``fixed_total`` rows.

    When ``combined`` is shorter than ``fixed_total``, zero-rows are appended.
    When it is longer, it is truncated and ``chunk_counts`` is adjusted in-place
    so the consumer never reads beyond the real data.

    Args:
        combined: float32 array shaped ``[real_total, chunk_samples]``.
        fixed_total: Target number of rows.
        chunk_samples: Number of samples per chunk (used for zero-padding).
        chunk_counts: Per-waveform chunk counts; mutated in-place when trimming.

    Returns:
        float32 array shaped ``[fixed_total, chunk_samples]``.
    """
    real_total = combined.shape[0]

    if real_total < fixed_total:
        pad = np.zeros((fixed_total - real_total, chunk_samples), dtype=np.float32)
        return np.concatenate([combined, pad], axis=0)

    if real_total > fixed_total:
        # Trim excess rows and shrink chunk_counts from the end so the consumer
        # only reads rows that are actually present in the truncated array.
        combined = combined[:fixed_total]
        remaining = fixed_total
        for i in range(len(chunk_counts) - 1, -1, -1):
            if remaining <= 0:
                chunk_counts[i] = 0
            elif chunk_counts[i] > remaining:
                chunk_counts[i] = remaining
                remaining = 0
            else:
                remaining -= chunk_counts[i]

    return combined


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
