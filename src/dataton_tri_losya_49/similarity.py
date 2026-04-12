import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Computes cosine similarity between two vectors.

    Measures the cosine of the angle between two non-zero vectors,
    which indicates their orientation similarity regardless of magnitude.
    Returns a value between -1 and 1, where 1 means identical direction,
    0 means orthogonal, and -1 means opposite direction.

    Args:
        vec1 (array-like object): First input vector.
        vec2 (array-like object): Second input vector.

    Returns:
        float: Cosine similarity between vec1 and vec2.

    Raises:
        ValueError: If vectors have different lengths or either vector is zero.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Vectors must be non-zero.")

    return dot_product / (norm1 * norm2)
