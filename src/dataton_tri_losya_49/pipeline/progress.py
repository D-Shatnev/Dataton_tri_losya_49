"""
Progress tracking utility for long-running pipeline steps.

Provides :class:`ProgressTracker` — a generic iterable wrapper that measures
throughput and prints ETA to stderr after each item.

Usage example::

    tracker = ProgressTracker(dataset.iter_waveforms(), total=len(filepaths), desc="Encoding")
    embeddings = extract_embeddings(tracker, encoder, batch_size)
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

_SMOOTHING_WINDOW = 50  # number of recent items used for rolling throughput


class ProgressTracker(Iterable[T]):
    """
    Wraps an iterable and prints throughput and ETA to stderr.

    Measures wall-clock time per item and maintains a rolling window average
    to produce stable ETA estimates even when processing speed varies.

    Args:
        iterable: Source iterable to wrap.
        total: Total number of items expected. Used to compute ETA and percentage.
        desc: Short label printed before the progress line (e.g. "Encoding").
        unit: Name of a single item used in throughput display (e.g. "files").
        print_every: Print a progress line every this many items. Default 1.

    Example::

        tracker = ProgressTracker(waveforms, total=5000, desc="Encoding", unit="files")
        for wav in tracker:
            ...
    """

    def __init__(
        self,
        iterable: Iterable[T],
        total: int,
        desc: str = "",
        unit: str = "items",
        print_every: int = 1,
    ) -> None:
        self._iterable = iterable
        self._total = total
        self._desc = desc
        self._unit = unit
        self._print_every = max(1, print_every)

    def __iter__(self) -> Iterator[T]:
        """Iterate over wrapped iterable, printing progress after each item."""
        processed = 0
        start_time = time.monotonic()
        recent_times: deque[float] = deque(maxlen=_SMOOTHING_WINDOW)
        last_item_time = start_time

        for item in self._iterable:
            yield item

            now = time.monotonic()
            recent_times.append(now - last_item_time)
            last_item_time = now
            processed += 1

            if processed % self._print_every == 0 or processed == self._total:
                elapsed = now - start_time
                overall_rate = processed / elapsed if elapsed > 0 else 0.0
                rolling_rate = 1.0 / (sum(recent_times) / len(recent_times)) if recent_times else overall_rate

                remaining = self._total - processed
                eta_sec = remaining / rolling_rate if rolling_rate > 0 else 0.0
                pct = 100.0 * processed / self._total if self._total > 0 else 0.0

                label = f"{self._desc}: " if self._desc else ""
                line = (
                    f"\r{label}"
                    f"{processed:>{len(str(self._total))}}/{self._total} "
                    f"[{pct:5.1f}%]  "
                    f"{rolling_rate:6.1f} {self._unit}/s  "
                    f"{'Total ' + _fmt_time(elapsed) if processed == self._total else 'ETA ' + _fmt_time(eta_sec)}"
                )
                sys.stderr.write(line)
                sys.stderr.flush()

        sys.stderr.write("\n")
        sys.stderr.flush()


def _fmt_time(seconds: float) -> str:
    """
    Format seconds as MM:SS or HH:MM:SS.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable time string.
    """
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"
