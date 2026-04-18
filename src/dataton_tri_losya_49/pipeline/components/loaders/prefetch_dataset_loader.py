"""
Prefetch decorator for DatasetLoader.

This module provides :class:`PrefetchDatasetLoader` — a transparent decorator
over any :class:`~dataton_tri_losya_49.pipeline.interfaces.DatasetLoader` that
loads waveforms ahead of time in background threads.

Architecture
------------
The inner loader exposes ``filepaths`` (a list of relative paths) and a
``loader`` attribute that implements
:class:`~dataton_tri_losya_49.pipeline.interfaces.WaveformLoader`.
``PrefetchDatasetLoader`` submits ``loader.load(root / filepath)`` calls to a
:class:`~concurrent.futures.ThreadPoolExecutor` and maintains a sliding window
of ``prefetch_factor`` in-flight :class:`~concurrent.futures.Future` objects.
Results are yielded in the original order, so the downstream batching logic in
:func:`~dataton_tri_losya_49.pipeline.runner.extract_embeddings` is unaffected.

When ``prefetch_factor`` is 0 the decorator is a no-op and simply delegates
``iter_waveforms`` to the inner loader.

Thread safety
-------------
Each call to ``iter_waveforms`` creates its own executor and future window,
so concurrent iteration is safe as long as the inner loader's ``load``
method is thread-safe (which is true for both
:class:`~dataton_tri_losya_49.pipeline.components.loaders.soundfile.SoundFileWaveformLoader`
and
:class:`~dataton_tri_losya_49.pipeline.components.loaders.vad_waveform_loader.VadWaveformLoader`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataton_tri_losya_49.constants import DEFAULT_PREFETCH_FACTOR
from dataton_tri_losya_49.pipeline.interfaces import DatasetLoader


@dataclass(frozen=True)
class PrefetchDatasetLoader:
    """
    Transparent decorator that prefetches waveforms in background threads.

    Wraps any :class:`~dataton_tri_losya_49.pipeline.interfaces.DatasetLoader`
    that exposes a ``loader`` attribute (waveform loader) and a ``root``
    attribute (base directory).  ``iter_waveforms`` submits ``loader.load``
    calls to a thread pool ahead of the consumer using a sliding window of
    futures.

    Args:
        inner: The wrapped dataset loader.  Must expose ``loader`` (a
            :class:`~dataton_tri_losya_49.pipeline.interfaces.WaveformLoader`)
            and ``root`` (a :class:`~pathlib.Path`) as attributes, in addition
            to the standard ``DatasetLoader`` protocol.
        prefetch_factor: Number of waveforms to load ahead of the consumer.
            Must be >= 0.  When 0, prefetching is disabled and
            ``iter_waveforms`` delegates directly to
            ``inner.iter_waveforms()``.

    Attributes:
        filepaths: Delegated to ``inner.filepaths``.
        labels: Delegated to ``inner.labels``.
    """

    inner: DatasetLoader
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR

    @property
    def filepaths(self) -> Sequence[str]:
        """Relative audio file paths from the inner loader."""
        return self.inner.filepaths

    @property
    def labels(self) -> np.ndarray | None:
        """Integer speaker labels from the inner loader, or None."""
        return self.inner.labels

    def iter_waveforms(self) -> Iterator[np.ndarray]:
        """
        Yield waveforms with background prefetch.

        Uses a sliding window of ``prefetch_factor`` futures backed by a
        :class:`~concurrent.futures.ThreadPoolExecutor`.  Each future calls
        ``inner.loader.load(inner.root / filepath)`` in a worker thread.
        Results are yielded in the original order.  Any exception raised
        inside a worker thread is re-raised in the calling thread when the
        corresponding future is consumed.

        When ``prefetch_factor`` is 0, delegates directly to
        ``inner.iter_waveforms()`` without spawning any threads.

        Yields:
            1-D mono float32 waveforms in dataset order.

        Raises:
            AttributeError: If ``inner`` does not expose ``loader`` or ``root``
                attributes required for parallel loading.
        """
        if self.prefetch_factor == 0:
            yield from self.inner.iter_waveforms()
            return

        loader = self.inner.loader  # type: ignore[attr-defined]
        root: Path = self.inner.root  # type: ignore[attr-defined]
        num_samples: int = self.inner.num_samples  # type: ignore[attr-defined]
        filepaths = list(self.filepaths)

        from dataton_tri_losya_49.pipeline.components.loaders.csv_audio_dataset import (
            crop_or_pad_repeat_start,
        )

        def _load(fp: str) -> np.ndarray:
            wav = loader.load(root / fp)
            return crop_or_pad_repeat_start(wav, num_samples)

        window: deque[Future[np.ndarray]] = deque()
        paths_iter = iter(filepaths)

        with ThreadPoolExecutor(max_workers=self.prefetch_factor) as pool:
            # Pre-fill the sliding window.
            for fp in _take(paths_iter, self.prefetch_factor):
                window.append(pool.submit(_load, fp))

            # Slide: yield oldest result, submit next path.
            for fp in paths_iter:
                future = window.popleft()
                yield future.result()
                window.append(pool.submit(_load, fp))

            # Drain remaining futures.
            while window:
                yield window.popleft().result()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _take(it: Iterator[str], n: int) -> list[str]:
    """Consume up to n items from an iterator and return them as a list.

    Args:
        it: Source iterator of file path strings.
        n: Maximum number of items to consume.

    Returns:
        List of up to n path strings.
    """
    result: list[str] = []
    for _ in range(n):
        try:
            result.append(next(it))
        except StopIteration:
            break
    return result
