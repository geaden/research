import numpy as np
from common.algorithms import BaseAlgorithm, Result
from common.objectives import Objective
from common.lmo import LMO


class FrankWolfe(BaseAlgorithm):
    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        L: float,
        iterations_count: int = 1000,
        tol=1e-14,
    ) -> None:
        super().__init__()
        self._obj = obj
        self._lmo = lmo
        assert L >= 0, "Value of L must be non-negative"
        self._L = L
        self._iterations_count = iterations_count
        self._tol = tol

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        self.track(x)
        for _ in range(self._iterations_count):
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x

            numerator = -np.dot(grad, d_k)
            denominator = self._L * np.linalg.norm(d_k) ** 2

            alpha_k = min(1, numerator / denominator)
            x += alpha_k * d_k

            self.track(x)
        return Result(x0=x0.copy(), x_opt=x.copy())


class FrankWolfeL0L1(BaseAlgorithm):

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        L0: float,
        L1: float,
        iterations_count: int = 1000,
        tol=1e-14,
    ) -> None:
        super().__init__()
        self._obj = obj
        self._lmo = lmo
        assert L0 >= 0, "Value of L_0 must be non-negative"
        self._L0 = L0
        assert L1 >= 0, "Value of L_1 must be non-negative"
        self._L1 = L1
        self._iterations_count = iterations_count
        assert tol > 0, "Tolerance must be positive"
        self._tol = tol

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        dual_gap = 0
        self.track(x)
        for _ in range(self._iterations_count):
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x

            dual_gap = -np.dot(grad, d_k)

            if dual_gap**2 <= self._tol:
                break

            numerator = dual_gap
            denominator = (
                (self._L0 + self._L1 * np.linalg.norm(grad))
                * np.linalg.norm(d_k) ** 2
                * np.e
            )
            alpha_k = min(1, numerator / denominator)
            x += alpha_k * d_k
            self.track(x)
        return Result(x0=x0.copy(), x_opt=x.copy())
