"""Module contains algorithms and implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from objectives import Objective
from step import StepSize
from utils import check_alpha, dual_gap, log, ensure_non_zero
from lmo import LinearMinimizationOracle


@dataclass
class Result:
    x0: np.ndarray
    x_opt: np.ndarray
    M: np.float64
    D: np.float64
    is_adaptive: bool


class BaseInexactFW(ABC):
    """
    Representation of abase inexact Frank-Wolfe.
    """

    _verbose = False

    def __init__(
        self,
        obj: Objective,
        step_size: StepSize,
        lmo: LinearMinimizationOracle,
        L: np.float64,
        alpha: float = 0.0,
        delta: float = 10e-4,
        N: int = 1000,
    ) -> None:
        """
        Initializes base method.

        Args:
            obj (Objective): objective function.
            L: L-Lipschitz gradient constant.
            alpha (float): relative inexactness [0,1].
        """
        self._obj = obj
        self._step_size = step_size
        self._lmo = lmo
        self._L = L
        self._alpha = alpha
        check_alpha(self._alpha)
        self._delta = ensure_non_zero(delta)
        self._N = N
        self._history = []

    @abstractmethod
    def run(self, x0: np.ndarray) -> Result:
        """
        Run algorithm and return solution.

        Args:
            x0 (np.ndarray): initial vector

        Returns:
            np.ndarray: solution
        """
        raise NotImplementedError()

    def track(self, x: np.ndarray) -> None:
        """
        Add x to history.

        Args:
            x (np.ndarray): vector to add
        """
        self._history.append(x.copy())

    @property
    def history(self) -> list[np.ndarray]:
        """
        Return history.

        Returns:
            list[np.ndarray]: history
        """
        return self._history

    def theoretical_iterations(self, result: Result) -> int:
        """
        Return theoretical number of iterations.

        Returns:
            int: theoretical number of iterations
        """
        residual = ensure_non_zero(self._obj(result.x0) - self._obj(result.x_opt))
        c = 2 if not result.is_adaptive else 4
        estimate = (
            np.sqrt(self._delta)
            + 2
            * self._alpha
            * (1 + self._alpha)
            / (1 - self._alpha)
            * result.M
            * result.D
        )
        lhs = c * self._L * result.D**2 / self._delta * residual
        rhs = max(
            1 + (np.log2(residual) - np.log2(estimate)) / (np.log2(3) - 1),
            0,
        )
        N = lhs + rhs
        return int(N)

    @property
    def verbose(self) -> bool:
        """
        Return verbose.

        Returns:
            bool: verbose
        """
        return self._verbose

    def dual_gap_alike_param(
        self,
        inexact_relative_grad: np.ndarray,
        d: np.ndarray,
        M: np.float64,
        D: np.float64,
    ) -> np.float64:
        """
        Return dual gap alike parameter for inexact relative gradient.
        """
        gap = dual_gap(inexact_relative_grad, d)
        log(gap, verbose=self._verbose)
        return np.float64(-(gap + self._alpha / (1 - self._alpha) * M * D))

    def relative_inexact_gradient(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Return (1+u)*g, where u is uniform in [-alpha, alpha]."""
        grad = self._obj.grad(x)
        u = np.random.uniform(-self._alpha, self._alpha, size=grad.shape)
        return grad * (1 + u)


class InexactFW(BaseInexactFW):
    """
    Inexact Frank-Wolfe algorithm.
    """

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()

        # Close to zero, but not zero to ensure non division by zero
        M = D = ensure_non_zero(0)

        self.track(x)

        for k in range(self._N):
            inexact_relative_grad = self.relative_inexact_gradient(x)
            s = self._lmo(inexact_relative_grad)
            d = s - x

            M = max(np.linalg.norm(inexact_relative_grad, 2), M)
            D = max(np.linalg.norm(d, 2), D)

            gap = self.dual_gap_alike_param(inexact_relative_grad, d, M, D)

            if gap**2 <= self._delta:
                break

            h = self._step_size(k, inexact_relative_grad, d, self._L, M, D)
            x = x + h * d
            self.track(x)

        return Result(
            x0=x0.copy(),
            x_opt=x.copy(),
            M=M,
            D=D,
            is_adaptive=False,
        )


class AdaptiveInexactFW(BaseInexactFW):
    """
    Adaptive Inexact Frank-Wolfe algorithm.
    """

    def run(self, x0: np.ndarray) -> np.ndarray:
        L = self._L / 2
        x = x0.copy()

        M = D = ensure_non_zero(0)

        self.track(x)

        for k in range(self._N):
            inexact_relative_grad = self.relative_inexact_gradient(x)
            s = self._lmo(inexact_relative_grad)
            d = s - x

            M = max(np.linalg.norm(inexact_relative_grad, 2), M)
            D = max(np.linalg.norm(d, 2), D)

            gap = self.dual_gap_alike_param(inexact_relative_grad, d, M, D)

            if gap**2 <= self._delta:
                break

            adapted = False
            while not adapted:
                theta = self._step_size(k, inexact_relative_grad, d, L, M, D)
                if theta < 1:
                    numinator = gap**2
                    denominator = 2 * L * np.linalg.norm(d) ** 2
                    if (
                        self._obj(x + theta * d)
                        <= self._obj(x) - numinator / denominator
                    ):
                        h = theta
                        x = x + h * d
                        self.track(x)
                        adapted = True
                    else:
                        L *= 2
                else:
                    if (
                        self._obj(x + d)
                        <= self._obj(x) - gap + L / 2 * np.linalg.norm(d) ** 2
                    ):
                        h = 1
                        x = x + h * d
                        self.track(x)
                        adapted = True
                    else:
                        L *= 2

        return Result(
            x0=x0.copy(),
            x_opt=x.copy(),
            M=M,
            D=D,
            is_adaptive=True,
        )
