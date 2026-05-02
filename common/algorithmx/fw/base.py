"""Base implementation of Frank-Wolfe algorithm."""

import numpy as np

from common.algorithmx.base import BaseAlgorithm
from common.objectives import Objective
from common.oracles.lmo import LMO


class BaseFrankWolfe(BaseAlgorithm):
    """Base implementation of Frank-Wolfe algorithm."""

    def __init__(self, obj: Objective, lmo: LMO, **kwargs):
        super().__init__(**kwargs)
        self._obj = obj
        self._lmo = lmo


def dual_gap(
    g: np.ndarray,
    d: np.ndarray,
    inexactness: float | None = None,
) -> np.float64:
    """
    Returns dual gap as dot product of gradient and search direction including inexactness.
    """
    gap = -np.dot(g, d)
    if inexactness is not None:
        gap -= inexactness
    return gap
