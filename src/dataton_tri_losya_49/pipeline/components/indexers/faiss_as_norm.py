"""
FAISS-based indexer with Adaptive Score Normalization (AS-Norm).

AS-Norm calibrates cosine similarity scores by normalizing each raw score
relative to a background cohort of imposters. This reduces per-speaker
score bias and improves retrieval ranking.

Algorithm:
    1. L2-normalize all embeddings.
    2. Sample a random cohort of ``cohort_size`` embeddings (seed=``cohort_seed``).
    3. Compute score matrix [N, cohort_size] = emb @ cohort.T.
    4. For each item i, take the top-``top_n`` cohort scores -> mu_i, sigma_i.
       These statistics are used symmetrically for both query and candidate roles.
    5. Use FAISS to retrieve top-``faiss_candidates`` neighbors per query
       (by raw cosine similarity). This avoids materializing the full [N, N] matrix.
    6. For each retrieved candidate j, compute the AS-Norm score:
       s_norm(i, j) = 0.5 * ((s(i,j) - mu_i)/sigma_i + (s(i,j) - mu_j)/sigma_j)
    7. Re-rank candidates by normalized score and return top-k.

Memory footprint (RAM only -- all computation is CPU/numpy):
    - Cohort score matrix [N, cohort_size] float32: N x cohort_size x 4 bytes.
    - FAISS candidate scores [N, faiss_candidates] float32: N x faiss_candidates x 4 bytes.
    - At N=134 698, cohort_size=1000, faiss_candidates=100: ~515 MB + ~51 MB ~= 570 MB total.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row of a 2-D array to unit L2 norm.

    Args:
        x: Input array of shape (N, D), dtype float32.
        eps: Small value added to the denominator to avoid division by zero.

    Returns:
        Array of the same shape as x with each row divided by its L2 norm.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _cohort_stats(scores: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row mean and std of the top-n scores.

    Args:
        scores: Float32 array of shape (N, C) -- scores against cohort.
        top_n: Number of top scores per row used to compute statistics.

    Returns:
        Tuple (mu, sigma) each of shape (N,).
        sigma is clipped from below at 1e-12 to avoid division by zero.
    """
    effective_n = min(top_n, scores.shape[1])
    top_scores = np.sort(scores, axis=1)[:, -effective_n:]
    mu = top_scores.mean(axis=1)
    sigma = np.clip(top_scores.std(axis=1), 1e-12, None)
    return mu, sigma


@dataclass(frozen=True)
class FaissASNormIndexer:
    """KNN indexer with Adaptive Score Normalization (AS-Norm).

    Uses a random subset of the dataset as the imposter cohort to compute
    per-item normalization statistics (mu, sigma). FAISS retrieves a candidate
    set per query; AS-Norm re-ranks those candidates by normalized score.

    This avoids materializing the full [N, N] score matrix: peak RAM is
    O(N x cohort_size + N x faiss_candidates) instead of O(N^2).

    All computation runs on CPU using numpy and faiss-cpu; no GPU memory
    is consumed during the indexing/search phase.

    Attributes:
        cohort_size: Number of randomly sampled embeddings used as the cohort
            for computing normalization statistics. Recommended: 500-2000.
        top_n: Number of top cohort scores used to estimate mu and sigma per item.
            Must be <= cohort_size. Recommended: ~20% of cohort_size.
        cohort_seed: Random seed for cohort sampling. Ensures full reproducibility.
        faiss_candidates: Number of candidates retrieved by FAISS before AS-Norm
            re-ranking. Must be >= topk. Larger values reduce the risk of missing
            a true top-k neighbor after normalization. Recommended: 5-10x topk.
    """

    cohort_size: int = 1000
    top_n: int = 200
    cohort_seed: int = 42
    faiss_candidates_coef: int = 10

    def __post_init__(self) -> None:
        if self.cohort_size <= 0:
            raise ValueError("cohort_size must be > 0")
        if self.top_n <= 0:
            raise ValueError("top_n must be > 0")
        if self.top_n > self.cohort_size:
            raise ValueError(f"top_n ({self.top_n}) must be <= cohort_size ({self.cohort_size})")
        if self.faiss_candidates_coef < 1:
            raise ValueError("faiss_candidates_coef must be >= 1")

    def neighbors(self, embeddings: np.ndarray, topk: int) -> np.ndarray:
        """Compute top-k neighbors using AS-Norm normalized cosine scores.

        Args:
            embeddings: Float32 array of shape (N, D).
            topk: Number of nearest neighbors to return per query (excluding self).

        Returns:
            Int64 array of shape (N, topk) with neighbor indices.
            The self-index is never present in the result.

        Raises:
            ValueError: If embeddings is not 2-D, contains fewer than 2 rows,
                or faiss_candidates < 1.
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2D [N, D], got shape {emb.shape}")

        n = int(emb.shape[0])
        if n <= 1:
            raise ValueError("Need at least 2 embeddings to build neighbors")

        k = int(topk)
        fc = int(topk * self.faiss_candidates_coef)
        if fc < k:
            raise ValueError(f"faiss_candidates ({fc}) must be >= topk ({k})")

        emb = _l2_normalize_rows(emb).astype(np.float32, copy=False)

        # Step 1: compute normalization statistics via random cohort.
        rng = np.random.default_rng(self.cohort_seed)
        effective_cohort = min(self.cohort_size, n)
        cohort_idx = rng.choice(n, size=effective_cohort, replace=False)
        cohort = emb[cohort_idx]  # [C, D]

        # [N, C] cosine scores against cohort members.
        scores_vs_cohort = (emb @ cohort.T).astype(np.float32)  # [N, C]

        # Per-item mu and sigma (same role for query and candidate).
        mu, sigma = _cohort_stats(scores_vs_cohort, self.top_n)  # [N], [N]

        # Step 2: FAISS retrieval of top-faiss_candidates neighbors.
        # +1 to retrieve self-index so we can exclude it cleanly.
        k_faiss = min(fc + 1, n)
        index = faiss.IndexFlatIP(int(emb.shape[1]))
        index.add(emb)
        raw_scores, cand_indices = index.search(emb, k_faiss)
        # raw_scores: [N, k_faiss], cand_indices: [N, k_faiss]

        raw_scores = np.asarray(raw_scores, dtype=np.float32)
        cand_indices = np.asarray(cand_indices, dtype=np.int64)

        # Step 3: AS-Norm re-ranking over FAISS candidates.
        # s_norm(i, j) = 0.5 * ((s - mu_i)/sigma_i + (s - mu_j)/sigma_j)
        mu_cands = mu[cand_indices]        # [N, k_faiss] -- mu of each candidate
        sigma_cands = sigma[cand_indices]  # [N, k_faiss] -- sigma of each candidate

        norm_q = (raw_scores - mu[:, None]) / sigma[:, None]  # [N, k_faiss]
        norm_e = (raw_scores - mu_cands) / sigma_cands        # [N, k_faiss]
        norm_scores = 0.5 * (norm_q + norm_e)                 # [N, k_faiss]

        # Mask self-index (i == j) with -inf so it never wins.
        row_ids = np.arange(n, dtype=np.int64)[:, None]
        norm_scores[cand_indices == row_ids] = -np.inf

        # Sort candidates by descending normalized score.
        sort_order = np.argsort(-norm_scores, axis=1)
        cand_indices = np.take_along_axis(cand_indices, sort_order, axis=1)

        # Return top-k (self-index is at the end due to -inf masking).
        return cand_indices[:, :k]
