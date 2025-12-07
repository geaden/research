"""Module contains Linear Minimization Oracles implementations."""

from abc import ABC
import numpy as np

from common.math_utils import ensure_non_zero


class LMO(ABC):
    """Linear Minimization Oracle."""

    def __call__(self, g: np.ndarray) -> np.ndarray:
        """
        Args:
            g (np.ndarray): value of |gradient| to evaluate LMO at.

        Return:
            value of LMO at |g|.
        """
        raise NotImplementedError()


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


class MinLinearDirectionL2BallLMO(LMO):
    """
    Linear Minimization Oracle that minimizes linear function in l2-ball.
    s = \arg \min_{||s||<=r} <g, s>
    solution: s = -r * g/||g||
    """

    def __init__(self, radius: float = 1.0) -> None:
        self._r = radius

    def __call__(self, g: np.ndarray) -> np.float64:
        norm_g = np.linalg.norm(g)
        if norm_g == 0:
            return np.zeros_like(g)
        return -self._r * g / norm_g

    def __str__(self):
        return rf"$\ell_2$-ball(r={self._r})"


class ShiftedBallLMO(LMO):
    """
    The nuclear norm ball

    $x \in ||radius - center||_2 <= radius$
    """

    def __init__(self, center: float, radius: float = 1.0):
        self._c = center
        self._r = radius

    def __call__(self, g: np.ndarray) -> np.float64:
        g = np.asarray(g)
        norm_g = np.linalg.norm(g)
        return self._c - self._r * g / ensure_non_zero(norm_g)

    def __str__(self):
        return f"Ball(center={self._c}, r={self._r})"
