import numpy as np
from common.objectives import Objective


def relative_inexact_gradient(
    obj: Objective,
    x: np.ndarray,
    alpha: np.float64,
) -> np.ndarray:
    """Return (1+u)*g, where u is uniform in [-alpha, alpha]."""
    grad = obj.grad(x)
    u = np.random.uniform(-alpha, alpha, size=grad.shape)
    return grad * (1 + u)
