"""Stopping rules for (H0, H1)-smooth Frank-Wolfe algorithms."""

import numpy as np


class GapClippedStoppingRule:
    r"""Stopping rule based on the clipped gap \Phi_k = min(G(x^k), f(x^k) - f_low).

    Since G(x^k) >= f(x^k) - f* by convexity and f(x^k) - f_low >= f(x^k) - f*
    for any lower bound f_low <= f*, the quantity \Phi_k satisfies
    f(x^k) - f* <= \Phi_k <= G(x^k). Thus \Phi_k <= \Delta guarantees
    f(x^k) - f* <= \Delta.
    """

    def __init__(self, tol: float, f_low: float = 0.0):
        self.tol = tol
        self.f_low = f_low

    def compute_metrics(
        self, f_val: float, grad: np.ndarray, d: np.ndarray
    ) -> dict[str, float]:
        """Compute dual gap G(x^k), overestimated residual, and clipped gap."""
        G_k = float(-np.dot(grad, d))
        D_k = float(np.linalg.norm(d))
        M_k = float(np.linalg.norm(grad))
        r_check = max(0.0, float(f_val) - self.f_low)
        phi_k = min(G_k, r_check)
        return {
            "G": G_k,
            "r_check": r_check,
            "phi": phi_k,
            "D": D_k,
            "M": M_k,
        }

    def check(self, f_val: float, grad: np.ndarray, d: np.ndarray) -> bool:
        r"""Return True if \Phi_k <= tol."""
        metrics = self.compute_metrics(f_val, grad, d)
        return metrics["phi"] <= self.tol


class PolyakResidualStoppingRule:
    r"""Stopping rule for Polyak-type Frank-Wolfe based on \hat{r}_k = f(x^k) - f_up.

    For an upper estimate f_up >= f*, \hat{r}_k <= \Delta guarantees
    f(x^k) - f* <= \Delta + (f_up - f*).
    """

    def __init__(self, tol: float, f_low: float = 0.0, f_up: float = 0.0):
        assert f_up >= f_low, "Upper bound f_up must be >= lower bound f_low"
        self.tol = tol
        self.f_low = f_low
        self.f_up = f_up

    def compute_metrics(
        self, f_val: float, grad: np.ndarray, d: np.ndarray
    ) -> dict[str, float]:
        """Compute residual estimates and dual gap."""
        G_k = float(-np.dot(grad, d))
        D_k = float(np.linalg.norm(d))
        M_k = float(np.linalg.norm(grad))
        r_check = max(0.0, float(f_val) - self.f_low)
        r_hat = float(f_val) - self.f_up
        return {
            "G": G_k,
            "r_check": r_check,
            "r_hat": r_hat,
            "phi": r_hat,
            "D": D_k,
            "M": M_k,
        }

    def check(self, f_val: float, grad: np.ndarray, d: np.ndarray) -> bool:
        """Return True if f(x^k) - f_up <= tol."""
        metrics = self.compute_metrics(f_val, grad, d)
        return metrics["r_hat"] <= self.tol
