import numpy as np

from .base import LMO
from common.math_utils import ensure_non_zero


class SimplexLMO(LMO):
    """
    Standard simplex.
    """

    def __init__(self, radius: np.float64 = 1.0) -> None:
        """
        Args:
            radius (np.float64): radius of simplex.
        """
        self._radius = radius

    def __call__(self, g: np.ndarray) -> np.ndarray:
        v = np.zeros_like(g) + ensure_non_zero(0)
        v[np.argmin(g)] = self._radius
        return v

    def __str__(self):
        return f"Simplex(r={self._radius})"
