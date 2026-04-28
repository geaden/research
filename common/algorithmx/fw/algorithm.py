import numpy as np

from common.objectives import Objective
from common.lmo import LMO
from common.algorithmx import Result, MaxIterMixin

from .base import BaseFrankWolfe
from .step_size import BaseStepSizeStrategy


class FrankWolfe(BaseFrankWolfe, MaxIterMixin):
    """
    Implementation of |BaseFrankWolfe| algorithm.
    """

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        L: float,
        step_size: BaseStepSizeStrategy,
        max_iter: int = 1000,
        tol=1e-14,
    ) -> None:
        super().__init__(obj=obj, lmo=lmo, max_iter=max_iter)
        assert L >= 0, "Value of L must be non-negative"
        self._step_size = step_size
        self._L = L
        self._tol = tol

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        self.track(x)
        for k in range(self.max_iter):
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x
            alpha_k = self._step_size(
                k
            )  # TODO(geaden): pass arguments for other strategies
            x += alpha_k * d_k
            self.track(x)
        return Result(x0=x0.copy(), x_opt=x.copy())
