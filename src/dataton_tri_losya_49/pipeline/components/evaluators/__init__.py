"""
Pipeline evaluators.

This package contains implementations of :class:dataton_tri_losya_49.pipeline.interfaces.Evaluator.

Public API:
    - :class:PrecisionAtKEvaluator
    - :func:precision_at_k_from_indices
"""

from __future__ import annotations

from dataton_tri_losya_49.pipeline.components.evaluators.precision_at_k import (
    PrecisionAtKEvaluator,
    precision_at_k_from_indices,
)

__all__ = [
    "PrecisionAtKEvaluator",
    "precision_at_k_from_indices",
]
