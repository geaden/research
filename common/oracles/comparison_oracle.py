"""Module contains comparison oracle implementation."""

from typing import Optional
import numpy as np
from common.objectives import Objective
from common.math_utils import ensure_non_zero


class ComparisonOracle:
    """
    Comparison oracle implementation.
    """

    def __init__(self, obj: Objective):
        self._obj = obj

    def __call__(self, x: np.ndarray, y: np.ndarray) -> int:
        """Returns -1 if f(y) <= f(x), 1 otherwise."""
        return np.sign(self._obj(y) - self._obj(x))


def directional_preference(
    oracle: ComparisonOracle,
    x: np.ndarray,
    v_unit: np.ndarray,
    Delta: np.float64,
    L: np.float64,
) -> int:
    """
    Returns sign information about scalar product of gradient and |v_unit|.
    """
    step = (2.0 * Delta / L) * v_unit
    return oracle(x + step, x)


def comparison_gde(
    oracle: ComparisonOracle,
    x: np.ndarray,
    gamma: np.ndarray,
    delta: np.float64,
    L: np.float64,
) -> np.ndarray:
    """
    Estimate a unit vector using comparisons.

    Args:
        oracle (ComparisonOracle): comparison oracle.
        x (np.ndarray): point in R^n.
        gamma (np.ndarray): lower bound on gradient norm.
        delta (np.float64): precision

    Returns:
      g: unit vector in R^n.
    """
    n = x.size

    Delta = delta * gamma / (4 * n ** (3 / 2))

    # Step 1. Determine sign patterns.
    signs = np.ones(n, dtype=float)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        dp = directional_preference(oracle, x, e, Delta=Delta, L=L)
        if dp != 1:
            signs[i] *= -1.0

    # Step 2.
    def dp_compare(i: int, j: int) -> int:
        ui = np.zeros(n)
        ui[i] = signs[i]
        uj = np.zeros(n)
        uj[j] = signs[j]
        v = ui - uj
        return directional_preference(
            oracle, x, 1 / np.sqrt(2.0) * v, Delta=Delta / np.sqrt(2.0), L=L
        )

    idx = 0
    for j in range(1, n):
        res = dp_compare(idx, j)
        if res != 1:
            idx = j
    i_star = idx

    # Step 3.
    alpha = np.zeros(n, dtype=float)
    alpha[i_star] = 1.0

    iters = int(np.ceil(np.log2(max(gamma / max(Delta, ensure_non_zero(0)), 2.0)))) + 1
    for i in range(n):
        if i == i_star:
            continue

        lo, hi = 0.0, 1.0
        alpha_i = 0.5
        for _ in range(iters):
            alpha_i = 0.5 * (lo + hi)
            res = directional_preference(
                oracle,
                x,
                (alpha_i * signs[i_star] - signs[i]) / np.sqrt(1 + alpha_i**2),
                Delta=Delta,
                L=L,
            )

            if res == 1:
                lo = alpha_i
            else:
                hi = alpha_i

        alpha[i] = alpha_i

    alpha_signed = alpha * signs
    norm = np.linalg.norm(alpha_signed)
    if norm < 1e-12:
        g_hat = np.random.normal(size=n)
        g_hat /= np.linalg.norm(g_hat) + ensure_non_zero(0)
        return g_hat
    return alpha_signed / norm
