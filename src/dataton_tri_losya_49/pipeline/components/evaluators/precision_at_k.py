"""
Precision@K metrics for a retrieval task.

This module provides utilities to compute Precision@K from a matrix of nearest-neighbor
indices.

Baseline-compatible behavior:

- input indices may contain a self-index (the query item index) in any column;
- if present, the self-index is moved to the end of each row and excluded from the first K;
- if only K columns are provided and a self-index is found among them, the evaluation becomes
  ambiguous (we cannot recover K non-self neighbors), so an error is raised.
"""

from __future__ import annotations

import numpy as np


def _move_self_to_end(indices: np.ndarray) -> np.ndarray:
    """
    Return indices with potential self-index moved to the end of each row.

    For each row i, if the value i appears anywhere in that row, it is shifted
    to the last position using a stable argsort. All other values retain their
    relative order. Rows without a self-index are left unchanged.

    Args:
        indices: Integer array of shape [N, M] with neighbor ids per query.

    Returns:
        Array of the same shape and dtype as indices, with self-indices
        moved to the last column of each row.

    Raises:
        ValueError: If indices is not a 2D array.
    """

    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError("indices must be a 2D array")

    n = int(idx.shape[0])
    row_ids = np.arange(n, dtype=idx.dtype)[:, None]

    # False < True => stable argsort moves self-indices to the end if present
    order = np.argsort(idx == row_ids, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def precision_at_k_from_indices(indices: np.ndarray, labels: np.ndarray, ks: list[int]) -> dict[str, float]:
    """
    Compute Precision@K from neighbor indices.

    This function evaluates retrieval quality given precomputed neighbor indices.

    Notes on self-index (baseline-style):
      - indices may include the query item itself (self-index) in any column.
      - we move self-indices to the end of each row (stable, preserving other order)
        and then take the first k_max = max(ks) columns.
      - in typical setups (e.g. FAISS search over the same set) you should provide at
        least k_max + 1 columns so that after dropping self you still have k_max
        non-self neighbors.

    Args:
        indices: int array of shape [N, M] with neighbor ids per query.
        labels: array of shape [N] with ground-truth class / speaker id.
        ks: list of K values to compute (e.g. [1, 5, 10]).

    Raises:
        ValueError:
            - if indices is not 2D
            - if labels length does not match indices rows
            - if ks is empty
            - if max(ks) is greater than the number of available columns
            - if a self-index is still present within the first k_max columns
              (ambiguous case: cannot recover K non-self neighbors)

    Returns:
        Mapping {"precision@K": value} for each requested K.
    """

    neigh = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels)

    if neigh.ndim != 2:
        raise ValueError("indices must be 2D array [N, M]")
    n, m = neigh.shape
    if y.shape[0] != n:
        raise ValueError(f"labels length {y.shape[0]} != indices rows {n}")
    if len(ks) == 0:
        raise ValueError("ks must be non-empty")

    ks = [int(k) for k in ks]
    k_max = max(ks)
    if k_max > m:
        raise ValueError(f"indices has only {m} columns but max(ks)={k_max}")

    # Filter self indices (baseline-style)
    neigh = _move_self_to_end(neigh)
    neigh = neigh[:, :k_max]

    row_ids = np.arange(n, dtype=np.int64)[:, None]
    if np.any(neigh == row_ids):
        # This happens when caller passed only k_max columns and self-index was present.
        raise ValueError(
            "Self-index detected within the first k_max columns. "
            "Provide at least (k_max + 1) indices per row (e.g. from topk+1 search), "
            "or ensure self-index is filtered out before evaluation."
        )

    matches = y[neigh] == y[:, None]
    pref = np.cumsum(matches, axis=1, dtype=np.int32)
    res: dict[str, float] = {}
    for k in ks:
        hits = pref[:, k - 1]
        res[f"precision@{k}"] = float((hits / k).mean())
    return res


class PrecisionAtKEvaluator:
    """
    Evaluator wrapper for Precision@K.

    This class provides a small OOP-style adapter around precision_at_k_from_indices
    so it can be plugged into the pipeline components API.
    """

    def evaluate(self, neighbors: np.ndarray, labels: np.ndarray, ks: list[int]) -> dict[str, float]:
        """
        Compute Precision@K metrics.

        Args:
            neighbors: Neighbor indices array of shape [N, M].
            labels: Labels array of shape [N].
            ks: List of K values to compute.

        Returns:
            Mapping {"precision@K": value} for each requested K.
        """
        return precision_at_k_from_indices(indices=neighbors, labels=labels, ks=ks)
