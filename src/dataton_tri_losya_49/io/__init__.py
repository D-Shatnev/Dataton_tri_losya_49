"""
I/O helpers for pipeline artifacts.

This package contains small utilities for saving intermediate artifacts and
final outputs (e.g. embeddings dumps and submission files).
"""

from __future__ import annotations

from dataton_tri_losya_49.io.embeddings import save_embeddings_npz
from dataton_tri_losya_49.io.submission import write_submission_csv

__all__ = [
    "save_embeddings_npz",
    "write_submission_csv",
]
