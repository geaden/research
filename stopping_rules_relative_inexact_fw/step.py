"""Module contains step size strategies."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from utils import dual_gap, check_alpha


class StepSize(ABC):
    """
    Representation of abstract step size strategy.
    """

    @abstractmethod
    def __call__(
        self,
        k: int,
        inexact_relative_grad: np.ndarray,
        d: np.ndarray,
        L: float,
        M: np.ndarray,
        D: np.ndarray,
    ) -> float:
        """Calculate step size based on the given arguments.

        Args:
            k (int): current iteration number.
            inexact_relative_gradient (np.ndarray): value of inexact relative gradient.
            d (np.ndarray): search direction.
            L: Lipschitz gradient constant.
            M: maximum norm of gradient.
            D: maximum search direction.
        """
        raise NotImplementedError()


class DecayingStepSize(StepSize):
    """
    Decaying step size strategy.
    """

    def __call__(
        self,
        k: int,
        *_: Tuple[Any, ...],
    ) -> np.float64:
        """Return step size at the given iteration |k|."""
        return 2 / (k + 2)


class ShortStepSize(StepSize):
    """
    Known as short step size strategy.

    See https://link.springer.com/article/10.1365/s13291-023-00275-x.
    """

    def __init__(self, alpha: float) -> None:
        """
        Initialize step size strategy.
        Args:
            alpha (float): relative inexactness [0,1].
        """
        self._alpha = check_alpha(alpha)

    def __call__(
        self,
        _: int,
        inexact_relative_grad: np.ndarray,
        d: np.ndarray,
        L: np.float64,
        M: np.float64,
        D: np.float64,
    ) -> np.float64:
        numerator = -(
            dual_gap(inexact_relative_grad, d) + self._alpha / (1 - self._alpha) * M * D
        )
        denominator = 2 * L * np.linalg.norm(d) ** 2
        return min(numerator / denominator, 1)
