"""Module contains objective functions."""

from abc import ABC, abstractmethod

import numpy as np

from aliases import VecFn


class Objective(ABC):
    """
    Representation of abstract objective function.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return objective value at the given vector |x|."""
        raise NotImplementedError()

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Return objective gradient value at the given vector |x|."""
        raise NotImplementedError()


class MSE(Objective):
    """
    Mean squared error loss error as functor.

    f(x) = 1/2 ||Ax - b||^2
    """

    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self._A = A.copy()
        self._b = b.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        diff = self._A @ x - self._b
        return 0.5 * np.dot(diff, diff)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self._A.T @ (self._A @ x - self._b)


class LogisticRegression(Objective):
    """
    Logistic regression loss error as functor.

    f(x) = mean(log(1 + exp(-y_i * <a_i, x>)))
    """

    def __init__(self, A: np.ndarray, y: np.ndarray) -> None:
        self._A = A.copy()
        self._y = y.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = self._y * (self._A @ x)
        return np.mean(np.log1p(np.exp(-z)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        z = self._y * (self._A @ x)
        coeff = -self._y / (np.exp(z) + 1)
        return self._A.T @ coeff / self._A.shape[0]
