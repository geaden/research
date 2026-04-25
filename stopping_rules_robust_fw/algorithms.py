"""Module contains algorithms implementations."""

from typing import Optional
import numpy as np
from common.algorithms_mixins import MaxIterMixin, InexactMixin
from common.algorithmx import Result
from common.gradient import relative_inexact_gradient
from common.algorithmx.fw import (
    BaseFrankWolfe,
    DiminishingStepSizeStrategy,
)
from common.objectives import Objective
from common.lmo import LMO
from common.math_utils import Interval
from common.oracles import ComparisonOracle, comparison_gde


_diminishing_step_size = DiminishingStepSizeStrategy()


class AdaptiveFrankWolfeRobustRelativeInexactness(
    BaseFrankWolfe,
    MaxIterMixin,
    InexactMixin,
):
    """Adaptive Frank-Wolfe with Robust Relative Inexactness."""

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        L: np.float64,
        alpha: np.float64 = 1e-4,
        max_iter: int = 10000,
    ):
        super().__init__(obj=obj, lmo=lmo, max_iter=max_iter, alpha=alpha)
        self._L = L

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy().astype(np.float64)
        self.track(x)

        M = 0

        for k in range(self.max_iter):
            inexact_grad = relative_inexact_gradient(self._obj, x, self.alpha)
            s_k = self._lmo(inexact_grad)
            d_k = s_k - x
            inexact_gap = -np.dot(inexact_grad, d_k)
            M = max(M, np.linalg.norm(inexact_grad))
            eta = self._eta(inexact_gap, M, d_k)
            theta = self._theta(inexact_gap, M, d_k)
            gamma = self._gamma(eta, theta)
            if gamma not in Interval(lower=0, upper=1):
                gamma = _diminishing_step_size(k)
            x += gamma * d_k
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    def _eta(self, gap: np.ndarray, M: np.float64, d_k: np.ndarray) -> np.float64:
        nominator = gap + self.alpha / (1 - self.alpha) * M * np.linalg.norm(d_k)
        denominator = self._L * np.linalg.norm(d_k) ** 2
        return nominator / denominator

    def _theta(self, gap: np.ndarray, M: np.float64, d_k: np.ndarray) -> np.float64:
        nominator = gap + (1 + self.alpha) * M * np.linalg.norm(d_k)
        denominator = self._L * np.linalg.norm(d_k) ** 2
        return nominator / denominator

    def _gamma(self, eta: np.float64, theta: np.float64) -> np.float64:
        min_step_size = min(eta, theta)
        if min_step_size < 0:
            return max(theta, eta)
        return min_step_size


class AdaptiveFrankWolfeRobustComparison(BaseFrankWolfe, MaxIterMixin):
    """Adaptive Frank-Wolfe with Robust Comparison."""

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        L: np.float64,
        delta: Optional[np.float64] = 1e-4,
        max_iter: int = 10000,
    ):
        super().__init__(obj=obj, lmo=lmo, max_iter=max_iter)
        self._L = L
        self._delta = delta

    def run(self, x0: np.ndarray) -> Result:
        oracle = ComparisonOracle(self._obj)
        x = x0.copy().astype(np.float64)
        self.track(x)

        for k in range(self.max_iter):
            g = comparison_gde(
                oracle=oracle,
                x=x,
                gamma=0.5,  # TODO(geaden): Adjust parameter
                delta=self._delta,
                L=self._L,
            )
            s = self._lmo(g)
            d = s - x
            d_norm2 = float(np.dot(d, d))

            gap_hat = float(np.dot(g, x - s))
            eta = min(gap_hat / (self._L * d_norm2), 1.0)

            if eta >= 0:
                gamma = eta
            else:
                gamma = _diminishing_step_size(k)

            x += gamma * d
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())
