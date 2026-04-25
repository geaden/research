"""Base implementation of Frank-Wolfe algorithm."""

from common.algorithmx.base import BaseAlgorithm
from common.objectives import Objective
from common.lmo import LMO


class BaseFrankWolfe(BaseAlgorithm):
    """Base implementation of Frank-Wolfe algorithm."""

    def __init__(self, obj: Objective, lmo: LMO, **kwargs):
        super().__init__(**kwargs)
        self._obj = obj
        self._lmo = lmo
