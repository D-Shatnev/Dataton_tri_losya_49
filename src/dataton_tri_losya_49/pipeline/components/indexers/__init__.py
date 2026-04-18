"""
Pipeline indexers.

This package contains implementations of dataton_tri_losya_49.pipeline.interfaces.Indexer.
"""

from __future__ import annotations

from dataton_tri_losya_49.pipeline.components.indexers.faiss_as_norm import FaissASNormIndexer
from dataton_tri_losya_49.pipeline.components.indexers.faiss_ip import FaissInnerProductIndexer

__all__ = [
    "FaissASNormIndexer",
    "FaissInnerProductIndexer",
]
