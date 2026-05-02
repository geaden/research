"""Module contains utility methods."""

import numpy as np


def random_euclidean_ball(R: int, radius: float = 1.0) -> np.ndarray:
    """Return random point in origin-centered l2-ball."""
    x = np.random.randn(R)
    return radius * x / np.linalg.norm(x)
