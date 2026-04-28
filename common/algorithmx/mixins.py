"""Module contains various mixins."""

import numpy as np


class MaxIterMixin:
    """
    Maximum number of iterations mixin.
    """

    def __init__(self, max_iter: int = 10000, **kwargs: dict[object, object]) -> None:
        """
        Args:
            N (int): maximum number of iterations.
        """
        super().__init__(**kwargs)
        self._max_iter = max_iter

    @property
    def max_iter(self) -> int:
        """
        Returns:
            int: maximum number of iterations.
        """
        return self._max_iter


class InexactMixin:
    """
    Inexact algorithm mixin.
    """

    def __init__(
        self,
        alpha: np.float64 = 1e-4,
        **kwargs: dict[object, object],
    ) -> None:
        """
        Args:
            alpha (float): inexact parameter.
        """
        super().__init__(**kwargs)
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """
        Returns:
            numpy.float64: inexact parameter.
        """
        return self._alpha
