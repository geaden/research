"""Module contains implementation of Linear Minimization Oracle."""

import numpy as np


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
    """

    def __init__(self, radius: float = 1.0) -> None:
        self._r = radius

    def __call__(self, x: np.ndarray) -> np.float64:
        return min_linear_direction_l2_ball(x, self._r)


def random_euclidean_ball(R: int, radius: float = 1.0) -> np.ndarray:
    """Return random point in origin-centered l2-ball."""
    x = np.random.randn(R)
    return radius * x / np.linalg.norm(x)


def proj_ball(x: np.ndarray, r: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm > r:
        return x * r / norm
    return x


def min_linear_direction_l2_ball(g: np.ndarray, r: float = 1.0) -> np.ndarray:
    """
    s = \arg \min_{||s||<=r} <g, s>
    solution: s = -r * g/||g||
    """
    norm_g = np.linalg.norm(g)
    if norm_g == 0:
        return np.zeros_like(g)
    return -r * g / norm_g
