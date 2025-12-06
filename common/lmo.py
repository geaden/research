"""Module contains Linear Minimization Oracles implementations."""

from abc import ABC
import numpy as np


class LMO(ABC):

    def __call__(self, x0: np.ndarray) -> np.ndarray:
        """
        Args:
            x0 (np.ndarray): vector to evaluate LMO at.

        Return:
            value of LMO at x0.
        """
        raise NotImplementedError()


class SimplexLMO(LMO):
    """
    LMO on standard simplex.
    """

    def __call__(self, x0: np.ndarray) -> np.ndarray:
        v = np.zeros_like(x0)
        j = np.argmin(x0)
        v[j] = 1.0
        return v
