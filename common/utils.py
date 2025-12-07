"""Module contains utility functions."""

import numpy as np
import time


def log(message: str, verbose: bool = False):
    """
    Log message if verbose is |True|.
    """
    if verbose:
        print(f"LOG {time.time()}: {message}")
