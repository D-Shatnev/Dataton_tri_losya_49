"""
Embeddings serialization helpers.

This module contains small utility functions for persisting embedding matrices
produced by the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_embeddings_npz(path: Path, filepaths: list[str], embeddings: np.ndarray) -> None:
    """
    Save embeddings into a compressed NPZ file.

    The file contains two arrays:
      - filepaths: object array of strings (N,)
      - embeddings: float32 array (N, D)

    Args:
        path: Target .npz file path.
        filepaths: List of audio file paths corresponding to embeddings.
        embeddings: Embedding matrix of shape (N, D).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        filepaths=np.asarray(filepaths, dtype=object),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
