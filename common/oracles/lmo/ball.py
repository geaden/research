import numpy as np

from . import LMO


def _center_of_ball(g: np.ndarray, center: float | None = None) -> np.ndarray:
    if center is None:
        return np.zeros_like(g)
    return np.array(center)


class L2BallLMO(LMO):
    r"""
    Linear Minimization Oracle that minimizes linear function in l2-ball.
    s = \arg \min_{||s||<=r} <g, s>
    solution: s = -r * g/||g||
    """

    def __init__(self, radius: float = 1.0, center: float | None = None) -> None:
        self._radius = radius
        self._cneter = center

    def __call__(self, g: np.ndarray) -> np.float64:
        center = _center_of_ball(g, self._cneter)
        norm_g = np.linalg.norm(g)
        if norm_g == 0:
            return np.zeros_like(g)
        return center - self._radius * g / norm_g

    def __str__(self) -> str:
        return rf"$\ell_2$-ball(r={self._radius},c={self._center or 0.0})"


class LinfBallLMO(LMO):
    r"""
    Linear Minimization Oracle for the $\ell_\infty$-ball.
    """

    def __init__(self, radius: float, center: float | None = None) -> None:
        self._radius = radius
        self._center = center

    def __call__(self, g: np.ndarray) -> np.float64:
        return _center_of_ball(g, self._center) - self._radius * np.sign(g)

    def __str__(self) -> str:
        return rf"$\ell_\infty$-ball(r={self._radius},c={self._center or 0.0})"
