"""Stopping rules for inexact Frank-Wolfe algorithms."""

import numpy as np
from typing import Optional


class InexactDualGapStoppingRule:
    """
    Stopping rule based on inexact dual gap for mixed inexact oracle.
    
    Stops when: G̃(x^k) ≤ Δ
    
    where the inexact dual gap is:
        G̃(x^k) = Ĝ(x^k) - (αM_k + δ)/(1-α) * D_k
        
    and Ĝ(x^k) = -⟨ĝ(x^k), d^k⟩ is the empirical dual gap.
    """
    
    def __init__(self, 
                 tol: float,
                 alpha: float,
                 delta: float):
        """
        Initialize stopping rule.
        
        Args:
            tol: Target precision Δ
            alpha: Relative inexactness parameter
            delta: Absolute inexactness parameter
        """
        assert 0 <= alpha < 1, f"Alpha must be in [0, 1), got {alpha}"
        self.tol = tol
        self.alpha = alpha
        self.delta = delta
        
    def check(self,
              g_hat: np.ndarray,
              d: np.ndarray,
              **kwargs) -> bool:
        """
        Check if stopping criterion is satisfied.
        
        Args:
            g_hat: Inexact gradient ĝ(x^k)
            d: Search direction d^k = s^k - x^k
            
        Returns:
            True if G̃(x^k) ≤ Δ
        """
        # Compute norms
        M_k = np.linalg.norm(g_hat)
        D_k = np.linalg.norm(d)
        
        # Compute empirical dual gap
        hat_G_k = -np.dot(g_hat, d)
        
        # Compute inexact dual gap
        noise_bound = (self.alpha * M_k + self.delta) / (1.0 - self.alpha)
        tilde_G_k = hat_G_k - noise_bound * D_k
        
        return tilde_G_k <= self.tol
    
    def compute_inexact_gap(self,
                            g_hat: np.ndarray,
                            d: np.ndarray) -> dict:
        """
        Compute inexact dual gap and related quantities.
        
        Args:
            g_hat: Inexact gradient
            d: Search direction
            
        Returns:
            Dictionary with gap values and norms
        """
        M_k = np.linalg.norm(g_hat)
        D_k = np.linalg.norm(d)
        
        # Empirical dual gap
        hat_G_k = -np.dot(g_hat, d)
        
        # Inexact dual gap
        noise_bound = (self.alpha * M_k + self.delta) / (1.0 - self.alpha)
        tilde_G_k = hat_G_k - noise_bound * D_k
        
        return {
            'tilde_G': tilde_G_k,
            'hat_G': hat_G_k,
            'M': M_k,
            'D': D_k,
            'noise_bound': noise_bound
        }


class StandardStoppingRule:
    """
    Standard stopping rule for comparison (no inexactness adjustment).
    
    Simply checks if empirical dual gap is small enough.
    """
    
    def __init__(self, tol: float):
        """
        Initialize standard stopping rule.
        
        Args:
            tol: Tolerance for stopping
        """
        self.tol = tol
        
    def check(self,
              g_hat: np.ndarray,
              d: np.ndarray,
              **kwargs) -> bool:
        """
        Check if stopping criterion is satisfied.
        
        Args:
            g_hat: Gradient (possibly inexact)
            d: Search direction
            
        Returns:
            True if dual gap is small enough
        """
        dual_gap = -np.dot(g_hat, d)
        return dual_gap <= self.tol
    
    def compute_gap(self,
                    g_hat: np.ndarray,
                    d: np.ndarray) -> float:
        """Compute standard dual gap."""
        return -np.dot(g_hat, d)