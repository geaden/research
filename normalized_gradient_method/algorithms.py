"""Module contains algorithms implementations."""

from typing import Optional
import numpy as np
from common.algorithmx import BaseAlgorithm, Result, MaxIterMixin
from common.objectives import Objective
from common.oracles import ComparisonOracle


class NormalizedGradientMethodHoelder(BaseAlgorithm, MaxIterMixin):
    """
    Implementation Normalized gradient method for functions with a Hölder gradient
    """

    def __init__(
        self,
        obj: Objective,
        L_nu: np.float64,
        nu: np.float64,
        epsilon: np.float64,
        delta: np.float64,
        max_iter: Optional[int] = 1000,
    ):
        """
        Args:
            obj (Objective): The objective function.
            L_nu (np.float64): The Hölder constant.
            nu (np.float64): The smoothness parameter (0, 1].
            epsilon (np.float64): The desired precision, used as gamma for the oracle.
            delta (np.float64): The precision for the comparison oracle.
            max_iter (Optional[int]): The number of iterations (N).
        """
        super().__init__(max_iter=max_iter)
        self._obj = obj
        self._L_nu = L_nu
        self._nu = nu
        self._epsilon = epsilon
        self._delta = delta
        self._oracle = ComparisonOracle(
            obj=self._obj,
            gamma=self._epsilon,
            delta=self._delta,
            nu=self._nu,
        )

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy().astype(np.float64)
        self.track(x)
        h = (np.sqrt(self._epsilon) * (1 + self._nu) / (2 * self._L_nu)) ** (
            1 / self._nu
        )

        for k in range(self.max_iter):
            g_k = self._oracle(x, self._L_nu)
            x += h * g_k
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())
