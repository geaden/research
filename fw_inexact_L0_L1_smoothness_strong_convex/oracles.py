"""Inexact gradient oracles for optimization problems."""

import numpy as np
from common.objectives import Objective
from common.oracles.base import Oracle
from common.math_utils import ensure_non_zero


class MixedInexactOracle(Oracle):
    """
    Mixed inexact gradient oracle with both relative and absolute noise.
    
    Implements oracle satisfying:
        ||ĝ(x) - ∇f(x)|| ≤ α||∇f(x)|| + δ,
    which in terms of the available ĝ(x) gives
        ||ĝ(x) - ∇f(x)|| ≤ (α||ĝ(x)|| + δ) / (1 - α).
    
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


class LowPrecisionCompressedOracle(Oracle):
    """
    Deterministic composite oracle: floating point rounding + compression.

    The exact gradient is first rounded to a floating point format with
    ``mantissa_bits`` bits of mantissa (relative error, unit roundoff
    u = 2^{-t}) and then compressed by uniform quantization with step ``q``
    (absolute error at most q/2 per coordinate). Hence

        ||ĝ(x) - ∇f(x)|| ≤ 2^{-t} ||∇f(x)|| + q √n / 2,

    i.e. α = 2^{-t} and δ = q √n / 2.
    """

    def __init__(self, obj: Objective, mantissa_bits: int, q: float, dim: int):
        assert mantissa_bits >= 1, "Mantissa must have at least one bit"
        assert q >= 0, "Quantization step must be non-negative"
        self.obj = obj
        self.mantissa_bits = mantissa_bits
        self.q = q
        self.alpha = 2.0 ** (-mantissa_bits)
        self.delta = q * np.sqrt(dim) / 2.0

    def _round(self, v: np.ndarray) -> np.ndarray:
        m, e = np.frexp(v)
        scale = 2.0 ** self.mantissa_bits
        return np.ldexp(np.round(m * scale) / scale, e)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        g = self._round(self.obj.grad(x))
        if self.q > 0:
            g = self.q * np.round(g / self.q)
        return g

    @property
    def parameters(self) -> dict:
        return {"alpha": self.alpha, "delta": self.delta}


class ShiftedPointOracle(Oracle):
    """
    Oracle that returns the gradient computed at a shifted point (Devolder et al.).

    ĝ(x) = ∇f(x + s), ||s|| = ε. For an (L0, L1)-smooth function

        ||∇f(x + s) - ∇f(x)|| ≤ (L0 + L1||∇f(x)||) ε e^{L1 ε},

    so the shift produces a composite error with
    α = L1 ε e^{L1 ε} and δ = L0 ε e^{L1 ε}.
    """

    def __init__(self,
                 obj: Objective,
                 eps: float,
                 L0: float,
                 L1: float,
                 dim: int,
                 rng: np.random.Generator = None):
        assert eps >= 0, "Shift must be non-negative"
        self.obj = obj
        self.eps = eps
        rng = rng if rng is not None else np.random.default_rng(0)
        u = rng.standard_normal(size=(dim,))
        self.shift = eps * u / ensure_non_zero(np.linalg.norm(u))
        factor = eps * np.exp(L1 * eps)
        self.alpha = L1 * factor
        self.delta = L0 * factor
        assert self.alpha < 0.5, f"Shift is too large: alpha = {self.alpha} >= 0.5"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.obj.grad(x + self.shift)

    @property
    def parameters(self) -> dict:
        return {"alpha": self.alpha, "delta": self.delta}


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