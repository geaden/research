"""Module contains algorithms implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from common.algorithmx import Result
from common.algorithmx.fw import BaseFrankWolfe, dual_gap, ShortStepSizeStrategy
from common.algorithmx.mixins import MaxIterMixin
from common.objectives import Objective
from common.oracles.lmo import LMO
from common.math_utils import ensure_non_zero


class AbsoluteInexactGradient:
    """
    Absolute inexact gradient implementation.
    """

    def __init__(
        self,
        obj: Objective,
        rng: np.random.Generator,
        delta: float = 1e-4,
    ) -> None:
        """
        Args:
            obj (Objective): objective function
            rng (np.random.Generator): random number generator
            delta (float): noise level (default: 1e-4)
        """
        self._obj = obj
        self._rng = rng
        self._delta = delta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x (np.ndarray): vector

                    Return:
            value of absolute inexact gradient.
        """
        grad = self._obj.grad(x)
        noise = self._rng.normal(size=grad.shape)
        norm = np.linalg.norm(noise)
        noise = noise / ensure_non_zero(norm) * self._delta
        return grad + noise


class BaseInexactFW(BaseFrankWolfe, MaxIterMixin):

    def __init__(
        self,
        obj: Objective,
        inexact_grad: AbsoluteInexactGradient,
        lmo: LMO,
        L: np.float64,
        N: int = 1000,
        delta: float = 1e-4,
    ):
        super().__init__(obj=obj, lmo=lmo, max_iter=N)
        self._inexact_grad = inexact_grad
        self._L = L
        self._delta = delta


class AbsoluteInexactFW(BaseInexactFW):
    """
    Implementation of Frank-Wolfe method with absolute inexact gradient.
    """

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy().astype(np.float64)
        self.track(x)
        step_size = ShortStepSizeStrategy()

        for _ in range(self.max_iter):
            g = self._inexact_grad(x)
            v = self._lmo(g)
            d = v - x

            dg = dual_gap(g, d)
            if dg**2 <= self._delta**2:
                break

            gamma_t = max(step_size(0, g, d, self._L, 0, 0), 0)

            x += gamma_t * d
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())
