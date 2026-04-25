import numpy as np
from common.algorithmx import BaseAlgorithm, Result
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

            if numerator**2 <= self._tol:
                break

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


class AdaptiveFrankWolfeL0L1(BaseAlgorithm):

    _L_0_max = 1e10
    _L_1_max = 1e10

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        L0: float,
        L1: float,
        rho: int = 2,
        iterations_count: int = 1000,
        tol=1e-14,
    ) -> None:
        super().__init__()
        self._obj = obj
        self._lmo = lmo
        assert rho >= 1, "Value of rho must be greater than or equal to 1"
        self._rho = rho
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
        L0 = self._L0
        L1 = self._L1
        toggle = False
        for _ in range(self._iterations_count):
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x

            dual_gap = -np.dot(grad, d_k)

            if dual_gap**2 <= self._tol:
                break

            conv_factor = self._conv_factor(grad, L0, L1)
            L0 /= self._rho + L0 / conv_factor
            L1 /= self._rho + L1 * np.linalg.norm(grad) / conv_factor

            while True:
                conv_factor = self._conv_factor(grad, L0, L1)
                numerator = dual_gap
                denominator = conv_factor * np.linalg.norm(d_k) ** 2 * np.e
                alpha_k = min(1, numerator / denominator)
                x_next = x + alpha_k * d_k
                if (
                    self._obj(x_next)
                    <= self._obj(x)
                    + alpha_k * np.dot(grad, d_k)
                    + conv_factor * np.e / 2 * alpha_k**2 * np.linalg.norm(d_k) ** 2
                ):
                    break
                elif not toggle:
                    L0 *= self._rho - L0 / conv_factor
                    L0 = min(L0, self._L_0_max)
                    toggle = True
                else:
                    L1 *= self._rho - L1 * np.linalg.norm(grad) / conv_factor
                    L1 = min(L1, self._L_1_max)
                    toggle = False

            x = x_next
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @staticmethod
    def _conv_factor(grad: np.ndarray, L0: float, L1: float) -> float:
        return L0 + L1 * np.linalg.norm(grad)
