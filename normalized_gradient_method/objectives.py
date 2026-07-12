import numpy as np
from common.objectives import Objective


class GasNetworkObjective(Objective):
    r"""
    Objective function for the gas transmission network problem.
    This implements the dual of a problem with a primal objective of the form:
    min_y phi(y) where phi(y) = max_x [ <y, d - Ax> - f(x) ]
    and f(x) = sum_{i=1 to m} alpha_i * |x_i|^p.

    This simplifies to:
    phi(y) = <d, y> + f*(-A^T y), where f* is the convex conjugate of f.
    """

    def __init__(self, alpha: np.ndarray, A: np.ndarray, d: np.ndarray, p: float = 1.3):
        self.alpha = alpha
        self.A = A
        self.d = d
        self.p = p
        self.q = p / (p - 1.0)  # Holder conjugate exponent

    def __call__(self, y: np.ndarray) -> np.ndarray:
        z = -self.A.T @ y
        f_star = (self.p - 1.0) * np.sum(
            self.alpha * (np.abs(z) / (self.p * self.alpha)) ** self.q
        )
        return self.d @ y + f_star

    def grad(self, y: np.ndarray) -> np.ndarray:
        z = -self.A.T @ y
        # This is x_opt(z) = argmax_x(<z,x> - f(x))
        x_at_z = np.sign(z) * (np.abs(z) / (self.p * self.alpha)) ** (
            1.0 / (self.p - 1.0)
        )
        return self.d - self.A @ x_at_z


class TwoLayerNetwork(Objective):
    r"""
    Objective function for a two-layer neural network with random first layer features.
    The objective is to train the second layer weights.
    $f(x) = \frac{1}{2} \| \sigma(X W_1^T + b_1) x - y \|_2^2$
    where $W_1, b_1$ are random and fixed.
    The optimization variable is $x$.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_hidden: int, seed: int):
        self._X = X
        self._y = y
        n_samples, n_features = X.shape

        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_hidden, n_features))
        self.b1 = rng.standard_normal((n_hidden,))

        # Pre-compute hidden layer features using ReLU
        self.A = np.maximum(0, self._X @ self.W1.T + self.b1)

        # Add bias term to hidden features
        self.A = np.c_[self.A, np.ones(n_samples)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x includes weights and bias for the output layer."""
        diff = self.A @ x - self._y
        return 0.5 * np.dot(diff, diff)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """x includes weights and bias for the output layer."""
        return self.A.T @ (self.A @ x - self._y)
