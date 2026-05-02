import numpy as np

from common.oracles.base import Oracle


class LMO(Oracle):
    """
    Abstract Linear Minimization Oracle.
    """

    def __call__(self, g: np.ndarray) -> np.ndarray:
        return super().__call__(g)
