"""Module contains comparison oracle implementation."""

import numpy as np
from .base import Oracle
from common.objectives import Objective
from common.math_utils import ensure_non_zero


class ComparisonOracle(Oracle):
    """
    Comparison oracle implementation.
    """

    def __init__(
        self,
        obj: Objective,
        gamma: np.float64,
        delta: np.float64,
    ) -> None:
        """
        Args:
            obj (Objective): objective function.
            gamma (float): A parameter controlling the precision/scale of the oracle's gradient estimation.
            delta (np.float64): precision
        """
        self._obj = obj
        self._gamma = gamma
        self._delta = delta

    def __call__(self, x: np.ndarray, L: np.float64) -> np.ndarray:
        """
        Estimate a unit vector using comparisons.

        Args:
            x (np.ndarray): point in R^n.

        Returns:
            a unit vector in R^n.
        """
        n = x.size

        Delta = self._delta * self._gamma / (4 * n ** (3 / 2))

        # Step 1. Determine sign patterns.
        signs = np.ones(n, dtype=float)

        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            dp = self._directional_preference(x, e, Delta=Delta, L=L)
            signs[i] *= -1.0 if dp != 1 else 1.0

        # Step 2.
        def dp_compare(i: int, j: int) -> int:
            ui = np.zeros(n)
            ui[i] = signs[i]
            uj = np.zeros(n)
            uj[j] = signs[j]
            v = ui - uj
            return self._directional_preference(
                x, 1 / np.sqrt(2.0) * v, Delta=Delta / np.sqrt(2.0), L=L
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

        iters = (
            int(
                np.ceil(np.log2(max(self._gamma / max(Delta, ensure_non_zero(0)), 2.0)))
            )
            + 1
        )
        for i in range(n):
            if i == i_star:
                continue

            lo, hi = 0.0, 1.0
            alpha_i = 0.5
            for _ in range(iters):
                alpha_i = 0.5 * (lo + hi)
                v = np.zeros(n)
                v[i_star] = alpha_i * signs[i_star]
                v[i] = -signs[i]
                v_unit = v / np.sqrt(1.0 + alpha_i**2)
                res = self._directional_preference(
                    x,
                    v_unit,
                    Delta=Delta,
                    L=L,
                )

                if res != 1:
                    hi = alpha_i
                else:
                    lo = alpha_i

            alpha[i] = alpha_i

        alpha_signed = alpha * signs
        norm = np.linalg.norm(alpha_signed)
        return -alpha_signed / ensure_non_zero(norm)

    def _comparison_oracle(self, x: np.ndarray, y: np.ndarray) -> int:
        """Returns -1 if f(y) <= f(x), 1 otherwise."""
        return np.sign(self._obj(y) - self._obj(x))

    def _directional_preference(
        self,
        x: np.ndarray,
        v_unit: np.ndarray,
        Delta: np.float64,
        L: np.float64,
    ) -> int:
        """Returns sign information about scalar product of gradient and |v_unit|."""
        step = (2.0 * Delta / L) * v_unit
        return self._comparison_oracle(x + step, x)
