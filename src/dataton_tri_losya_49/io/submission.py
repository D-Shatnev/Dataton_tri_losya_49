"""
Submission file writing helpers.

The submission format is a CSV with columns:
  - filepath: input audio filepath
  - neighbours: comma-separated neighbor indices
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def write_submission_csv(path: Path, filepaths: Iterable[str], neighbors: np.ndarray) -> None:
    """
    Write a submission CSV file.

    Args:
        path: Output .csv path.
        filepaths: Iterable of audio filepaths in the same order as rows in neighbors.
        neighbors: Neighbor indices array of shape (N, K). Values are written as
            comma-separated integers.

    Notes:
        This function does not perform any filtering (e.g. self-index removal).
        The pipeline is expected to provide already post-processed neighbors.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    neigh = np.asarray(neighbors, dtype=np.int64)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "neighbours"])
        for fp, row in zip(filepaths, neigh, strict=True):
            w.writerow([fp, ",".join(str(int(x)) for x in row.tolist())])
