"""Module contains regularization implementations."""

import numpy as np

from common.objectives import Objective
from common.math_utils import significant_figures


class L1Regular(Objective):
    """Special kind of |Objective| implementing L1 regularization."""

    def __init__(self, obj: Objective, lam: float) -> None:
        super().__init__()
        self._obj = obj
        self._lam = lam
        self.__doc__ = rf"{self._obj.__doc__}$ + \lambda \cdot \|x\|_1, \lambda={significant_figures(lam, n=4)}$"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._obj(x) + self._lam * np.linalg.norm(x, 1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        g = self._obj.grad(x)
        return g + self._lam * np.sign(x)


class L2Regular(Objective):
    """Special kind of |Objective| implementing L2 regularization."""

    def __init__(self, obj: Objective, lam: float) -> None:
        super().__init__()
        self._obj = obj
        self._lam = lam
        self.__doc__ = rf"{self._obj.__doc__}$ + \lambda \cdot \dfrac{{1}}{{2}}\|x\|_2^2, \lambda={significant_figures(lam, n=4)}$"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._obj(x) + 0.5 * self._lam * np.dot(x, x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self._obj.grad(x) + self._lam * x
