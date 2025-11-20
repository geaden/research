"""Module contains implementation of Linear Minimization Oracle."""

import numpy as np

from utils import ensure_non_zero


class LinearMinimizationOracle:
    """
    Base Linear Minimization Oracle.
    """

    def __call__(self, x: np.ndarray) -> np.float64:
        """
        Evaluate linear minimization oracle.
        """
        raise NotImplementedError()


class MinLinearDirectionL2Ball(LinearMinimizationOracle):
    """
    Linear Minimization Oracle that minimizes linear function in l2-ball.
    s = \arg \min_{||s||<=r} <g, s>
    solution: s = -r * g/||g||
    """

    def __init__(self, radius: float = 1.0) -> None:
        self._r = radius

    def __call__(self, x: np.ndarray) -> np.float64:
        norm_g = np.linalg.norm(x)
        if norm_g == 0:
            return np.zeros_like(x)
        return -self._r * x / norm_g


class Simplex(LinearMinimizationOracle):
    """
    The nuclear norm ball.

    $x \in ||radius - center||_2 <= radius$
    """

    def __init__(self, radius: float = 1.0):
        self._r = radius

    def __call__(self, x: np.ndarray) -> np.float64:
        s = np.zeros(x.shape)
        s += ensure_non_zero(0)
        s[np.argmin(x)] = self._r
        return s


class ShiftedBall(LinearMinimizationOracle):
    """
    The nuclear norm ball

    $x \in ||radius - center||_2 <= radius$
    """

    def __init__(self, center: float, radius: float = 1.0):
        self._c = center
        self._r = radius

    def __call__(self, x: np.ndarray) -> np.float64:
        g = np.asarray(x)
        norm_g = np.linalg.norm(g)
        return self._c - self._r * g / ensure_non_zero(norm_g)


def random_euclidean_ball(R: int, radius: float = 1.0) -> np.ndarray:
    """Return random point in origin-centered l2-ball."""
    x = np.random.randn(R)
    return radius * x / np.linalg.norm(x)


def proj_ball(x: np.ndarray, r: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm > r:
        return x * r / norm
    return x
