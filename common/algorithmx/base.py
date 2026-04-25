from dataclasses import dataclass
from abc import ABC, abstractmethod
from common.objectives import Objective

import numpy as np


@dataclass
class Result:
    x0: np.ndarray
    x_opt: np.ndarray


class BaseAlgorithm(ABC):

    def __init__(self, **kwargs: dict[object, object]) -> None:
        super().__init__(**kwargs)
        self._history: list[np.ndarray] = []

    @abstractmethod
    def run(self, x0: np.ndarray) -> Result:
        """
        Run algorithm and return solution.

        Args:
            x0 (np.ndarray): initial vector

        Returns:
            result (Result): solution
        """
        raise NotImplementedError()

    def track(self, x: np.ndarray) -> None:
        """
        Add x to history.

        Args:
            x (np.ndarray): vector to add
        """
        self._history.append(x.copy())

    @property
    def history(self) -> list[np.ndarray]:
        """
        Return history.

        Returns:
            list[np.ndarray]: history
        """
        return self._history
