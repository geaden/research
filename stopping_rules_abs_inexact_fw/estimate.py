"""Module contains estimates based on different assumptions."""

import numpy as np

from common.objectives import Objective


class Estimate:

    def __init__(
        self,
        L: np.float64,
        D: np.float64,
        delta: np.float64 = 1e-4,
    ) -> None:
        self._L = L
        self._D = D
        self._delta = delta

    def run(self, N: int) -> np.ndarray:
        """
        Run the estimate in accordance with theoretical results.

        Args:
            N (int): iteration numbers

        Returns:
            list[np.ndarray]: list of estimated vectors.
        """
        raise NotImplementedError()


class InductionConvexEstimate(Estimate):
    r"""$f(x_N) - f^* \leq \frac{8LD^2}{N+2}+\frac{3N\Delta^2}{4L}.$"""

    def run(self, N: int) -> np.float64:
        lhs = 8 * self._L * self._D**2 / (N + 2)
        rhs = 3 * N * self._delta**2 / (4 * self._L)
        return lhs + rhs


class GapSumEstimate(Estimate):
    r"""$f(x_N) - f^* \leq \frac{\sqrt{4LD^2(f(x_1)-f^*)}}{\sqrt{N}} + \sqrt{2}D\Delta.$"""

    def __init__(
        self,
        obj: Objective,
        x0: np.ndarray,
        x_opt: np.ndarray,
        L: np.float64,
        D: np.float64,
        delta: float = 1e-4,
    ):
        super().__init__(L, D, delta)
        self._obj = obj
        self._x0 = x0
        self._x_opt = x_opt

    def run(self, N: int) -> np.float64:
        lhs = (
            4
            * self._L
            * self._D**2
            * (self._obj(self._x0) - self._obj(self._x_opt))
            / N
        )
        lhs = np.sqrt(lhs)
        rhs = np.sqrt(2) * self._D * self._delta
        return lhs + rhs
