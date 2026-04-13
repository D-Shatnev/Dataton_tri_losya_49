"""
FAISS-based indexer for cosine similarity nearest-neighbor search.

This module provides FaissInnerProductIndexer - a thin wrapper around
faiss.IndexFlatIP that performs L2-normalization before indexing so that
inner product equals cosine similarity.

The indexer is intentionally stateless: every call to neighbors builds a
fresh FAISS index, which keeps the API simple and experiments reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row of a 2-D array to unit L2 norm.

    Args:
        x: Input array of shape (N, D), dtype float32.
        eps: Small value added to the denominator to avoid division by zero.

    Returns:
        Array of the same shape as x with each row divided by its L2 norm,
        clipped from below by eps.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


@dataclass(frozen=True)
class FaissInnerProductIndexer:
    """
    Cosine-similarity KNN indexer backed by faiss.IndexFlatIP.

    Embeddings are L2-normalized before being added to the index so that
    inner product search is equivalent to cosine similarity search.

    The self-index (a query being its own nearest neighbor) is detected and
    moved to the end of each result row, then excluded from the returned
    top-k neighbors.

    Note:
        The dataclass is frozen to prevent accidental mutation of configuration
        during a run (for reproducibility).
    """

    def neighbors(self, embeddings: np.ndarray, topk: int) -> np.ndarray:
        """
        Build a FAISS index and return the top-k nearest neighbors for each embedding.

        Args:
            embeddings: Float32 array of shape (N, D) with N embeddings of
                dimensionality D.
            topk: Number of nearest neighbors to return per query (excluding
                the query itself).

        Returns:
            Int64 array of shape (N, topk) with neighbor indices.  The
            self-index is never present in the result.

        Raises:
            ValueError: If embeddings is not a 2-D array or contains fewer
                than 2 rows.
            RuntimeError: If a self-index is still present after filtering
                (should not happen under normal usage).
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2D [N,D], got {emb.shape}")

        n = int(emb.shape[0])
        if n <= 1:
            raise ValueError("Need at least 2 embeddings to build neighbors")

        emb = _l2_normalize_rows(emb).astype(np.float32, copy=False)

        k_search = int(topk) + 1
        index = faiss.IndexFlatIP(int(emb.shape[1]))
        index.add(emb)
        _, indices = index.search(emb, k_search)

        indices = np.asarray(indices, dtype=np.int64)
        row_ids = np.arange(n, dtype=np.int64)[:, None]

        # Move self-index to the end if present
        order = np.argsort(indices == row_ids, axis=1, kind="stable")
        indices = np.take_along_axis(indices, order, axis=1)

        neigh = indices[:, : int(topk)]
        if np.any(neigh == row_ids):
            raise RuntimeError("Self-index detected in neighbors after filtering")
        return neigh
