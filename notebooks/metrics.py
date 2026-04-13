"""
This module provides a simple implementation of a challenge-style
precision@K metric computed from two CSV files:

- reference CSV (ground truth neighbours)
- predicted CSV (model neighbours)

The rest of the repository also contains a *pipeline-native* evaluator
(:class:dataton_tri_losya_49.pipeline.components.evaluators.precision_at_k.PrecisionAtKEvaluator)
that works directly with neighbor index matrices and labels.
"""

from csv import reader
from pathlib import Path


def precision_k(reference: str | Path, predict: str | Path) -> dict:
    """
    Compute a CSV-based precision@K score.

    The function reads both CSVs, aligns them by filepath and computes a per-row
    score based on the fraction of predicted neighbours that are present in the
    reference set.

    Args:
        reference: Path to reference CSV (ground truth).
        predict: Path to predicted CSV.

    Returns:
        A dict with:
            - score: mean score across rows.
            - detail: per-file breakdown with "extra", "missing" and row "score".
    """
    y = {}
    with open(reference, encoding="utf-8") as reference_file:
        reference_file.readline()
        for line in reader(reference_file):
            y[line[0]] = set(line[1].split(","))

    result = {}
    total_score = 0
    count = 0

    with open(predict, encoding="utf-8") as predict_file:
        predict_file.readline()
        for line in reader(predict_file):
            predictions = set(line[1].split(","))
            result[line[0]] = {
                "extra": list(predictions - y[line[0]]),
                "missing": list(y[line[0]] - predictions),
                "score": 1 - len(predictions - y[line[0]]) / len(predictions),
            }
            count += 1
            total_score += result[line[0]]["score"]

    return {"score": total_score / count, "detail": result}
