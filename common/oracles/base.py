"""Module contains"""

from abc import ABC, abstractmethod

import numpy as np


class Oracle(ABC):
    """
    Representation of abstract oracle.
    """

    @abstractmethod
    def __call__(
        self,
        x: np.ndarray,
        *args: tuple[object],
        **kwargs: dict[object, object],
    ) -> np.ndarray:
        """
        Returns desired value based on oracle implemenation.
        """
        raise NotImplementedError()
