"""Objective functions for (H0, H1)-smooth optimization problems."""

import numpy as np
from common.objectives import Objective


class QuarticH0H1Problem(Objective):
    r"""Synthetic convex problem satisfying (H0, H1)-smoothness.

    Minimizes:
        f(x) = (H0 / 2) * ||x - x*||^2 + (H1 / 12) * ||x - x*||^4

    where ||\nabla^2 f(x)|| = H0 + H1 * ||x - x*||^2.
    For small ||x - x*|| or bounded domain, ||\nabla^2 f(x)|| is controlled
    affinely by f(x) - f*.
    """

    def __init__(
        self,
        dim: int = 50,
        H0: float = 1e-3,
        H1: float = 4.0,
        shift_scale: float = 1.2,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        self.dim = dim
        self.H0 = H0
        self.H1 = H1
        raw_shift = rng.randn(dim)
        self.x_star = shift_scale * raw_shift / np.linalg.norm(raw_shift)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        diff = x - self.x_star
        norm_sq = float(np.dot(diff, diff))
        return np.float64(0.5 * self.H0 * norm_sq + (self.H1 / 12.0) * (norm_sq ** 2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        diff = x - self.x_star
        norm_sq = float(np.dot(diff, diff))
        return (self.H0 + (self.H1 / 3.0) * norm_sq) * diff

    @property
    def x_optimal(self) -> np.ndarray:
        return self.x_star.copy()
