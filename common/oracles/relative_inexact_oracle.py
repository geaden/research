"""Module contains functions and objects related to relative inexact oracle."""

from common.math_utils import Interval


def check_alpha(alpha: float) -> float:
    """
    Check if alpha is in [0, 1].
    """
    assert alpha in Interval(0, 1), f"Incorrect value of alpha={alpha}"
    return alpha
