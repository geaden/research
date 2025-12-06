"""Module contains utility functions."""

import numpy as np
import time


def log(message: str, verbose: bool = False):
    """
    Log message if verbose is |True|.
    """
    if verbose:
        print(f"LOG {time.time()}: {message}")


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
