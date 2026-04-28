"""Module contains utility functions."""

import time


def log(message: str, verbose: bool = False):
    """
    Log message if verbose is |True|.
    """
    if verbose:
        print(
            f"LOG [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]: {message}"
        )
