"""Module contains utility methods."""

import time
import numpy as np


def check_alpha(alpha: float) -> None:
    """
    Check if alpha is in [0, 1].
    """
    assert 0 <= alpha <= 1, f"Incorrect value of alpha={alpha}"


def dual_gap(inexact_relative_grad: np.ndarray, d: np.ndarray) -> np.float64:
    """
    Find dual gap as dot product of inexact relative gradient and search direction.
    """
    return np.dot(inexact_relative_grad, d)


def log(message: str, verbose: bool = False):
    """
    Log message if verbose is |True|.
    """
    if verbose:
        print(f"LOG {time.time()}: {message}")


def significant_figures(x: float) -> float:
    """Return significant figures of |x|."""
    powers = np.floor(np.log10(x))
    return 10**powers * np.floor(x / 10**powers)


def non_singular_matrix(
    N: int,
    MinL: np.float64,
    MaxL: np.float64,
    MinM: np.float64,
    MaxM: np.float64,
) -> np.ndarray:
    """
    Construct non singular matrix.
    """
    main_diag = np.random.uniform(MinL, MaxL, size=(N,))
    upper_tri = np.random.uniform(MinM, MaxM, size=(N, N))
    lower_tri = np.tri(N, k=-1, dtype=np.float64)
    result_matrix = np.diag(main_diag) + lower_tri * upper_tri
    return result_matrix.T


def ensure_non_zero(x: np.float64) -> np.float64:
    """
    Ensure that |x| is non-zero.
    Close to zero, but not zero to ensure non division by zero occurrence.
    """
    return max(x, 1e-12)
