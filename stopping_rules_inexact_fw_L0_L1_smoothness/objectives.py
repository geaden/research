"""Objective functions for (L0, L1)-smooth optimization problems."""

import numpy as np
from common.objectives import Objective


class ShiftedPointProblem(Objective):
    r"""
    Shifted point optimization problem with (L0, L1)-smoothness.
    
    Minimizes:
        f(x) = (L0/2) * ||x - x*||^2 + (L1/3) * ||x - x*||^3
    
    where x* is a shifted optimal point.
    
    The Hessian satisfies:
        ||∇²f(x)|| ≤ L0 + 2*L1*||x - x*|| ≤ L0 + L1*||∇f(x)||
    """
    
    def __init__(self, 
                 dim: int = 50, 
                 L0: float = 1e-4, 
                 L1: float = 2.5, 
                 shift_scale: float = 1.2,
                 seed: int = 42):
        """
        Initialize the shifted point problem.
        
        Args:
            dim: Problem dimension
            L0: L0-smoothness constant
            L1: L1-smoothness constant  
            shift_scale: Scaling factor for shifted optimal point
            seed: Random seed for reproducibility
        """
        rng = np.random.RandomState(seed)
        self.dim = dim
        self.L0 = L0
        self.L1 = L1
        
        # Generate shifted optimal point x*
        raw_shift = rng.randn(dim)
        self.x_star = shift_scale * raw_shift / np.linalg.norm(raw_shift)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate objective function at x.
        
        Args:
            x: Current point
            
        Returns:
            Objective value f(x)
        """
        diff = x - self.x_star
        norm_diff = np.linalg.norm(diff)
        return 0.5 * self.L0 * (norm_diff ** 2) + (self.L1 / 3.0) * (norm_diff ** 3)
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient at x.
        
        Args:
            x: Current point
            
        Returns:
            Gradient ∇f(x)
        """
        diff = x - self.x_star
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-12:
            return np.zeros_like(x)
        return (self.L0 + self.L1 * norm_diff) * diff
    
    @property
    def x_optimal(self) -> np.ndarray:
        """Return the optimal point x*."""
        return self.x_star.copy()
    
    def __doc__(self):
        """Return LaTeX-formatted description."""
        return r"$f(x) = \frac{L_0}{2}\|x - x^*\|^2 + \frac{L_1}{3}\|x - x^*\|^3$"