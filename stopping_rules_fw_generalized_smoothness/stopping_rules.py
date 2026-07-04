"""Module contains stopping rule strategies."""

from abc import ABC, abstractmethod

import numpy as np

from common.objectives import Objective
from common.utils import log


class StoppingRuleStrategy(ABC):
    """Abstract stopping rule strategy."""

    @abstractmethod
    def check(self, **kwargs: dict[str, object]) -> bool:
        """Check stopping rule.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            True if stopping rule is satisfied, False otherwise.
        """
        raise NotImplementedError


class DualGapStoppingRuleStrategy(StoppingRuleStrategy):
    """Dual gap stopping rule strategy."""

    def __init__(self, tol: float = 10e-4) -> None:
        """Dual gap stopping rule strategy.

        Args:
            tol: tolerance.
        """
        self._tol = tol

    def check(self, **kwargs: dict[str, object]) -> bool:
        if "dual_gap" not in kwargs:
            return False
        dual_gap = kwargs.get("dual_gap")
        return dual_gap <= self._tol


class ConvergenceRateStoppingRuleStrategy(StoppingRuleStrategy):
    """Convergence rate stopping rule strategy."""

    def __init__(
        self,
        x0: np.ndarray,
        obj: Objective,
        strong_convexity_const: float,
        tol: float = 10e-4,
    ) -> None:
        """Convergence rate stopping rule strategy.

        Args:
            x0: Initial point.
            obj: Objective function.
            strong_convexity_param: strong convexity constant.
            tol: tolerance.
        """
        self._x0 = x0
        self._obj = obj
        assert strong_convexity_const > 0, "Strong convexity constant must be positive"
        self._lam = strong_convexity_const
        self._tol = tol
        self._T1 = 1
        self._T2 = 1
        self._T3 = 1
        self._min_grad_norm: float = 1e10

    def check(self, **kwargs: dict[str, object]) -> bool:
        log(f"{self}=")
        if "x" not in kwargs:
            return False

        x = kwargs.get("x")
        alpha = kwargs.get("alpha")
        L0 = kwargs.get("L0")
        L1 = kwargs.get("L1")
        grad_norm = np.linalg.norm(self._obj.grad(x))
        self._min_grad_norm = min(grad_norm, self._min_grad_norm)

        T1 = (self._lam * self._tol) / (2 * np.e * L1) * self._T1
        T2 = (self._lam * self._min_grad_norm * self._tol) / (2 * np.e * L0) * self._T2
        T3 = 0.5 * self._tol * self._T3

        stopped = self._obj(self._x0) - self._obj(x) <= T1 + T2 + T3

        # Accumulate info about iterations for the next step.
        if alpha < 1:
            if L0 <= L1 * grad_norm:
                self._T1 += 1
            else:
                self._T2 += 1
        else:
            self._T3 += 1

        return stopped
