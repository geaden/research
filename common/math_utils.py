"""Module contains math related utility functions."""

import numpy as np


def non_singular_matrix(
    N: int,
    MinL: np.float64,
    MaxL: np.float64,
    MinM: np.float64,
    MaxM: np.float64,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct non singular matrix.
    """
    main_diag = rng.uniform(MinL, MaxL, size=(N,))
    upper_tri = rng.uniform(MinM, MaxM, size=(N, N))
    lower_tri = np.tri(N, k=-1, dtype=np.float64)
    result_matrix = np.diag(main_diag) + lower_tri * upper_tri
    return result_matrix.T.astype(np.float64)


def significant_figures(x: float, n: int = 1) -> float:
    """Return significant |n| digits of |x|."""
    return float(f"{x:.{n}g}")


def ensure_non_zero(x: np.float64) -> np.float64:
    """
    Ensure that |x| is non-zero.
    Close to zero, but not zero to ensure non division by zero occurrence.
    """
    return max(x, 1e-20)
