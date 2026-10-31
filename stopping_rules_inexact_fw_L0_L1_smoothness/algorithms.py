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
    - Adaptive step size: h_k = min(1, G̃_k / (2L̄_k D_k² e))
    - Where L̄_k = (L₀ + L₁||ĝ_k||) / 2
    - Stopping rule based on inexact dual gap: G̃(x^k) ≤ Δ
    """
    
    def __init__(self,
                 obj: Objective,
                 lmo: LMO,
                 L0: float,
                 L1: float,
                 stopping_rule: InexactDualGapStoppingRule,
                 max_iter: int = 150,
                 tol: float = 1e-14):
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
        """
        super().__init__(obj=obj, lmo=lmo)
        assert L0 >= 0, "L0 must be non-negative"
        assert L1 >= 0, "L1 must be non-negative"
        
        self.L0 = L0
        self.L1 = L1
        self.stopping_rule = stopping_rule
        self.max_iter = max_iter
        self.tol = tol
        
        # Track additional metrics
        self.gap_history = []
        self.step_history = []
        
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
        
        stopped_iteration = self.max_iter
        
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
            if self.stopping_rule.check(g_hat, d_k):
                stopped_iteration = k
                break
            
            # Compute adaptive step size
            # L̄_k = (L₀ + L₁||ĝ_k||) / 2
            L_bar_k = (self.L0 + self.L1 * M_k) / 2.0
            
            if D_k > 1e-12 and L_bar_k > 1e-12 and gap_info['tilde_G'] > 0:
                # h_k = min(1, G̃_k / (2L̄_k D_k² e))
                h_raw = gap_info['tilde_G'] / (2.0 * L_bar_k * (D_k ** 2) * np.e)
                h_k = min(1.0, max(0.0, h_raw))
            else:
                # Fallback to classical step
                h_k = 2.0 / (k + 2.0)
            
            self.step_history.append(h_k)
            
            # Update iterate
            x = x + h_k * d_k
            self.track(x)
        
        # Store final iteration info
        self.stopped_at = stopped_iteration
        
        return Result(x0=x0.copy(), x_opt=x.copy())
    
    @property
    def metrics(self) -> dict:
        """Return algorithm metrics."""
        return {
            'gap_history': self.gap_history,
            'step_history': self.step_history,
            'iterations': self.stopped_at if hasattr(self, 'stopped_at') else len(self.history) - 1
        }


class ClassicalMixedInexactFW(BaseFrankWolfe):
    """
    Classical Frank-Wolfe with 2/(k+2) step for mixed inexact oracle.
    
    This is a baseline method for comparison that uses:
    - Classical step size: α_k = 2/(k+2)
    - No stopping rule (runs for max_iter iterations)
    """
    
    def __init__(self,
                 obj: Objective,
                 lmo: LMO,
                 alpha: float,
                 delta: float,
                 max_iter: int = 150,
                 tol: float = 1e-14):
        """
        Initialize classical Frank-Wolfe for mixed inexact oracle.
        
        Args:
            obj: Objective function (should be wrapped with MixedInexactObjectiveWrapper)
            lmo: Linear minimization oracle
            alpha: Relative inexactness (for tracking)
            delta: Absolute inexactness (for tracking)
            max_iter: Maximum number of iterations
            tol: Numerical tolerance
        """
        super().__init__(obj=obj, lmo=lmo)
        self.alpha = alpha
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        
        # Track metrics for analysis
        self.gap_history = []
        
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
            'iterations': len(self.history) - 1
        }