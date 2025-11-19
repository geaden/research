"""Module contains necessary aliases for type hinting and readability."""

from typing import Callable

import numpy as np

VecFn = Callable[[np.ndarray], np.ndarray]
