import numpy as np
from common.objectives import Objective


class GasNetworkObjective(Objective):
    r"""
    Objective function for the gas transmission network problem.
    Minimizes $sum_{i=1}^m(a_i * |x_i|^3)$ subject to $Ax=d$.
    Since we use Frank-Wolfe, the constraint Ax=d is handled by the LMO.
    The objective function for the unconstrained FW variant is just the sum.
    """

    def __init__(self, a: np.ndarray):
        self.a = a

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.a * np.abs(x) ** 3)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.a * 3.0 * np.abs(x) ** 2.0 * np.sign(x)
