"""Inexact gradient oracles for optimization problems."""

import numpy as np
from common.objectives import Objective
from common.oracles.base import Oracle
from common.math_utils import ensure_non_zero


class MixedInexactOracle(Oracle):
    """
    Mixed inexact gradient oracle with both relative and absolute noise.
    
    Implements oracle satisfying:
        ||ĝ(x) - ∇f(x)|| ≤ α||ĝ(x)|| + δ
    
    where:
        α ∈ [0, 1) - relative inexactness parameter
        δ ≥ 0 - absolute inexactness parameter
    """
    
    def __init__(self, 
                 obj: Objective,
                 alpha: float = 0.04,
                 delta: float = 1e-3,
                 rng: np.random.Generator = None):
        """
        Initialize mixed inexact oracle.
        
        Args:
            obj: Objective function to approximate
            alpha: Relative inexactness parameter (must be in [0, 1))
            delta: Absolute inexactness parameter (must be non-negative)
            rng: Random number generator for reproducibility
        """
        assert 0 <= alpha < 1, f"Alpha must be in [0, 1), got {alpha}"
        assert delta >= 0, f"Delta must be non-negative, got {delta}"
        
        self.obj = obj
        self.alpha = alpha
        self.delta = delta
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute inexact gradient at x.
        
        Args:
            x: Current point
            
        Returns:
            Inexact gradient ĝ(x) satisfying mixed noise condition
        """
        exact_grad = self.obj.grad(x)
        norm_grad = np.linalg.norm(exact_grad)
        
        # Generate random noise direction
        noise_dir = self.rng.standard_normal(size=x.shape)
        noise_norm = np.linalg.norm(noise_dir)
        
        if noise_norm > 0:
            noise_dir /= noise_norm
            
        # Sample noise magnitude uniformly in valid range
        max_noise = self.alpha * norm_grad + self.delta
        noise_magnitude = self.rng.uniform(0, max_noise)
        
        return exact_grad + noise_dir * noise_magnitude
    
    @property
    def parameters(self) -> dict:
        """Return oracle parameters."""
        return {
            "alpha": self.alpha,
            "delta": self.delta
        }


class MixedInexactObjectiveWrapper(Objective):
    """
    Wrapper that adds mixed inexact oracle to an objective.
    
    This allows using inexact gradients with algorithms expecting Objective interface.
    """
    
    def __init__(self, 
                 obj: Objective,
                 oracle: MixedInexactOracle):
        """
        Wrap objective with inexact oracle.
        
        Args:
            obj: Original objective function
            oracle: Mixed inexact oracle
        """
        self._obj = obj
        self._oracle = oracle
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate objective (exact)."""
        return self._obj(x)
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Return inexact gradient from oracle."""
        return self._oracle(x)
    
    @property
    def exact_grad(self):
        """Access exact gradient if needed."""
        return self._obj.grad
    
    @property
    def oracle_params(self):
        """Return oracle parameters."""
        return self._oracle.parameters