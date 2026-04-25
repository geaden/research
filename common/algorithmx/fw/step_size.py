"""Module contains step size strategies."""

from abc import ABC, abstractmethod

import numpy as np


class BaseStepSizeStrategy(ABC):
    """Base class for step size strategies."""

    @abstractmethod
    def __call__(
        self, *args: tuple[object], **kwargs: dict[object, object]
    ) -> np.float64:
        raise NotImplementedError


class DiminishingStepSizeStrategy(BaseStepSizeStrategy):
    """Diminishing step size strategy."""

    def __call__(self, k: int) -> np.float64:
        return 2.0 / (k + 2.0)
