"""Frank-Wolfe algorithms for mixed inexact optimization with (L0, L1)-smoothness."""

import numpy as np
from typing import Optional

from common.algorithmx import BaseAlgorithm, Result
from common.algorithmx.fw import BaseFrankWolfe
from common.objectives import Objective
from common.oracles.lmo import LMO
from common.math_utils import ensure_non_zero

try:
    from .stopping_rules import InexactDualGapStoppingRule
except ImportError:
    from stopping_rules import InexactDualGapStoppingRule


class ProposedMixedInexactFW(BaseFrankWolfe):
    """
    Frank-Wolfe with short adaptive step for mixed inexact oracle and (L0, L1)-smoothness.

    Implements the proposed method with:
    - Adaptive step size: h_k = min(1, G̃_k / (e * L̂_k * D_k²))
    - Where L̂_k = L₀ + L₁ (||ĝ_k|| + δ) / (1 - α) is an upper bound for
      L₀ + L₁||∇f(x^k)|| computed from the inexact gradient only.
    - Stopping rule based on inexact dual gap: G̃(x^k) ≤ Δ

    If ``stop_on_rule`` is False the method keeps iterating after the stopping
    rule has fired; the first iteration at which it fired is stored in
    ``stopped_at`` (``None`` if it never fired).
    """

    def __init__(self,
                 obj: Objective,
                 lmo: LMO,
                 L0: float,
                 L1: float,
                 stopping_rule: InexactDualGapStoppingRule,
                 max_iter: int = 150,
                 tol: float = 1e-14,
                 stop_on_rule: bool = True):
        """
        Initialize the proposed Frank-Wolfe method.

        Args:
            obj: Objective function (should be wrapped with MixedInexactObjectiveWrapper)
            lmo: Linear minimization oracle
            L0: L0-smoothness constant
            L1: L1-smoothness constant
            stopping_rule: Inexact dual gap stopping rule
            max_iter: Maximum number of iterations
            tol: Numerical tolerance
            stop_on_rule: Whether to terminate once the stopping rule fires
        """
        super().__init__(obj=obj, lmo=lmo)
        assert L0 >= 0, "L0 must be non-negative"
        assert L1 >= 0, "L1 must be non-negative"

        self.L0 = L0
        self.L1 = L1
        self.stopping_rule = stopping_rule
        self.max_iter = max_iter
        self.tol = tol
        self.stop_on_rule = stop_on_rule

        # Track additional metrics
        self.gap_history = []
        self.step_history = []
        self.stopped_at: Optional[int] = None

    def _smoothness_bound(self, M_k: float) -> float:
        """Upper bound L̂_k ≥ L₀ + L₁||∇f(x^k)|| computed from ||ĝ_k||."""
        alpha = self.stopping_rule.alpha
        delta = self.stopping_rule.delta
        return self.L0 + self.L1 * (M_k + delta) / (1.0 - alpha)

    def run(self, x0: np.ndarray) -> Result:
        """
        Run the proposed Frank-Wolfe algorithm.

        Args:
            x0: Initial point

        Returns:
            Result with initial and final points
        """
        x = x0.copy()
        self.track(x)
        self.stopped_at = None

        for k in range(self.max_iter):
            # Get inexact gradient
            g_hat = self._obj.grad(x)
            M_k = np.linalg.norm(g_hat)

            # Compute Frank-Wolfe direction
            s_k = self._lmo(g_hat)
            d_k = s_k - x
            D_k = np.linalg.norm(d_k)

            # Compute gap metrics
            gap_info = self.stopping_rule.compute_inexact_gap(g_hat, d_k)
            self.gap_history.append(gap_info)

            # Check stopping criterion
            if self.stopped_at is None and self.stopping_rule.check(g_hat, d_k):
                self.stopped_at = k
                if self.stop_on_rule:
                    break

            # Compute adaptive step size
            L_hat_k = self._smoothness_bound(M_k)
            tilde_G = gap_info['tilde_G']

            if D_k > 1e-12 and tilde_G > 0:
                # h_k = min(1, G̃_k / (e L̂_k D_k²))
                h_raw = tilde_G / (np.e * ensure_non_zero(L_hat_k) * D_k ** 2)
                h_k = min(1.0, h_raw)
            else:
                # G̃_k ≤ 0: the oracle cannot certify a descent direction,
                # the point stays (h_k = 0).
                h_k = 0.0

            self.step_history.append(h_k)

            # Update iterate
            x = x + h_k * d_k
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @property
    def metrics(self) -> dict:
        """Return algorithm metrics."""
        return {
            'gap_history': self.gap_history,
            'step_history': self.step_history,
            'stopped_at': self.stopped_at,
            'iterations': (self.stopped_at if self.stopped_at is not None
                           else len(self.history) - 1),
        }


class ClassicalMixedInexactFW(BaseFrankWolfe):
    """
    Classical Frank-Wolfe with 2/(k+2) step for mixed inexact oracle.

    This is a baseline method for comparison that uses:
    - Classical step size: α_k = 2/(k+2)
    - No stopping rule (runs for max_iter iterations). If a stopping rule is
      supplied, the first iteration at which it fires is recorded in
      ``stopped_at`` but the method keeps running.
    """

    def __init__(self,
                 obj: Objective,
                 lmo: LMO,
                 alpha: float,
                 delta: float,
                 max_iter: int = 150,
                 tol: float = 1e-14,
                 stopping_rule: Optional[InexactDualGapStoppingRule] = None):
        """
        Initialize classical Frank-Wolfe for mixed inexact oracle.

        Args:
            obj: Objective function (should be wrapped with MixedInexactObjectiveWrapper)
            lmo: Linear minimization oracle
            alpha: Relative inexactness (for tracking)
            delta: Absolute inexactness (for tracking)
            max_iter: Maximum number of iterations
            tol: Numerical tolerance
            stopping_rule: Optional stopping rule, used only to mark its firing
        """
        super().__init__(obj=obj, lmo=lmo)
        self.alpha = alpha
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        self.stopping_rule = stopping_rule

        # Track metrics for analysis
        self.gap_history = []
        self.stopped_at: Optional[int] = None

    def run(self, x0: np.ndarray) -> Result:
        """
        Run classical Frank-Wolfe algorithm.

        Args:
            x0: Initial point

        Returns:
            Result with initial and final points
        """
        x = x0.copy()
        self.track(x)
        self.stopped_at = None

        for k in range(self.max_iter):
            # Get inexact gradient
            g_hat = self._obj.grad(x)
            M_k = np.linalg.norm(g_hat)

            # Compute Frank-Wolfe direction
            s_k = self._lmo(g_hat)
            d_k = s_k - x
            D_k = np.linalg.norm(d_k)

            # Compute gaps for tracking
            hat_G_k = -np.dot(g_hat, d_k)
            noise_bound = (self.alpha * M_k + self.delta) / (1.0 - self.alpha)
            tilde_G_k = hat_G_k - noise_bound * D_k

            self.gap_history.append({
                'tilde_G': tilde_G_k,
                'hat_G': hat_G_k,
                'M': M_k,
                'D': D_k
            })

            if (self.stopping_rule is not None and self.stopped_at is None
                    and self.stopping_rule.check(g_hat, d_k)):
                self.stopped_at = k

            # Classical step size
            h_k = 2.0 / (k + 2.0)

            # Update iterate
            x = x + h_k * d_k
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @property
    def metrics(self) -> dict:
        """Return algorithm metrics."""
        return {
            'gap_history': self.gap_history,
            'stopped_at': self.stopped_at,
            'iterations': len(self.history) - 1
        }


class AdaptiveMixedInexactFW(BaseFrankWolfe):
    """
    Adaptive (L0, L1)-Frank-Wolfe for composite inexact gradient oracle.

    Adapts ``AdaptiveFrankWolfeL0L1`` from ``stopping_rules_fw_generalized_smoothness``
    to the composite inexact gradient setting using the inexact dual gap G̃(x^k).
    """

    _L_0_max = 1e10
    _L_1_max = 1e10

    def __init__(self,
                 obj: Objective,
                 lmo: LMO,
                 L0: float,
                 L1: float,
                 stopping_rule: InexactDualGapStoppingRule,
                 rho: float = 2.0,
                 max_iter: int = 150,
                 tol: float = 1e-14,
                 stop_on_rule: bool = True):
        super().__init__(obj=obj, lmo=lmo)
        assert rho >= 1.0, "rho must be >= 1"
        assert L0 >= 0, "L0 must be non-negative"
        assert L1 >= 0, "L1 must be non-negative"
        self.L0 = L0
        self.L1 = L1
        self.rho = rho
        self.stopping_rule = stopping_rule
        self.max_iter = max_iter
        self.tol = tol
        self.stop_on_rule = stop_on_rule

        self.gap_history = []
        self.step_history = []
        self.stopped_at: Optional[int] = None

    def _conv_factor(self, M_k: float, L0: float, L1: float) -> float:
        alpha = self.stopping_rule.alpha
        delta = self.stopping_rule.delta
        return ensure_non_zero(L0 + L1 * (M_k + delta) / (1.0 - alpha))

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        self.track(x)
        self.stopped_at = None
        L0 = ensure_non_zero(self.L0)
        L1 = ensure_non_zero(self.L1)
        flip = False

        for _ in range(self.max_iter):
            g_hat = self._obj.grad(x)
            M_k = np.linalg.norm(g_hat)
            s_k = self._lmo(g_hat)
            d_k = s_k - x
            D_k = np.linalg.norm(d_k)

            gap_info = self.stopping_rule.compute_inexact_gap(g_hat, d_k)
            self.gap_history.append(gap_info)
            tilde_G = gap_info['tilde_G']

            if self.stopped_at is None and self.stopping_rule.check(g_hat, d_k):
                self.stopped_at = len(self.gap_history) - 1
                if self.stop_on_rule:
                    break

            if D_k <= 1e-12 or tilde_G <= 0:
                self.step_history.append(0.0)
                self.track(x)
                continue

            alpha_err = self.stopping_rule.alpha
            delta_err = self.stopping_rule.delta
            grad_bound = (M_k + delta_err) / (1.0 - alpha_err)

            conv_factor = self._conv_factor(M_k, L0, L1)
            L0 /= self.rho + L0 / conv_factor
            L1 /= self.rho + L1 * grad_bound / conv_factor

            for _ in range(60):
                conv_factor = self._conv_factor(M_k, L0, L1)
                h_k = min(1.0, tilde_G / (np.e * conv_factor * D_k ** 2))
                x_next = x + h_k * d_k
                if (
                    self._obj(x_next)
                    <= self._obj(x)
                    - h_k * tilde_G
                    + 0.5 * np.e * conv_factor * (h_k ** 2) * (D_k ** 2)
                ):
                    break
                if not flip:
                    L0 = min(L0 * (self.rho - L0 / conv_factor), self._L_0_max)
                else:
                    L1 = min(
                        L1 * (self.rho - L1 * grad_bound / conv_factor),
                        self._L_1_max,
                    )
                flip = not flip

            self.step_history.append(h_k)
            x = x_next
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @property
    def metrics(self) -> dict:
        return {
            'gap_history': self.gap_history,
            'step_history': self.step_history,
            'stopped_at': self.stopped_at,
            'iterations': (self.stopped_at if self.stopped_at is not None
                           else len(self.history) - 1),
        }

