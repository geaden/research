"""Module contains step size strategies."""

from abc import ABC, abstractmethod

import numpy as np

from common.algorithmx.fw.base import dual_gap
from common.oracles.relative_inexact_oracle import check_alpha


class BaseStepSizeStrategy(ABC):
    """Base class for step size strategies."""

    @abstractmethod
    def __call__(
        self,
        *args: tuple[object],
        **kwargs: dict[object, object],
    ) -> np.float64:
        raise NotImplementedError


class DiminishingStepSizeStrategy(BaseStepSizeStrategy):
    """Diminishing step size strategy."""

    def __call__(
        self,
        k: int,
        *_: tuple[object],
        **__: dict[object, object],
    ) -> np.float64:
        return 2.0 / (k + 2.0)


class ShortStepSizeStrategy(BaseStepSizeStrategy):
    """
    Known as short step size strategy.

    See https://link.springer.com/article/10.1365/s13291-023-00275-x.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        """
        Initialize step size strategy.
        Args:
            alpha (float): relative inexactness [0,1].
        """
        self._alpha = check_alpha(alpha)

    def __call__(
        self,
        _: int,
        g: np.ndarray,
        d: np.ndarray,
        L: np.float64,
        M: np.float64,
        D: np.float64,
    ) -> np.float64:
        # TODO(geaden): consider more cases of inexacntess.
        numerator = dual_gap(g, d, self._alpha / (1 - self._alpha) * M * D)
        denominator = 2 * L * np.linalg.norm(d) ** 2
        return min(numerator / denominator, 1)
