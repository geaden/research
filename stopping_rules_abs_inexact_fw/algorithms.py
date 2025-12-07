"""Module contains algorithms implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any
from common.objectives import Objective
from common.lmo import LMO
from common.math_utils import ensure_non_zero


@dataclass
class Result:
    x0: np.ndarray
    x_opt: np.ndarray


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


class BaseFW(ABC):

    def __init__(
        self,
        obj: Objective,
        inexact_grad: AbsoluteInexactGradient,
        lmo: LMO,
        L: np.float64,
        N: int = 1000,
        delta: float = 1e-4,
    ):
        self._obj = obj
        self._inexact_grad = inexact_grad
        self._lmo = lmo
        self._L = L
        self._N = N
        self._delta = delta
        self._history: list[np.ndarray] = []

    @abstractmethod
    def run(self, x0: np.ndarray) -> Result:
        """
        Run algorithm and return solution.

        Args:
            x0 (np.ndarray): initial vector

        Returns:
            result (Result): solution
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


class AbsoluteInexactFW(BaseFW):
    """
    Implementation of Frank-Wolfe method with absolute inexact gradient.
    """

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy().astype(np.float64)
        self.track(x)

        for _ in range(self._N):
            g = self._inexact_grad(x)
            v = self._lmo(g)
            d = v - x

            dual_gap = -np.dot(g, d)
            if dual_gap**2 <= self._delta**2:
                break

            gamma_t = max(min(dual_gap / (2 * self._L * np.linalg.norm(d) ** 2), 1), 0)

            x += gamma_t * d
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())
