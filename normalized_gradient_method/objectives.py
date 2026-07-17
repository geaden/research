import numpy as np
from typing import Tuple
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


class NonConvexTwoLayerNetworkObjective(Objective):
    """
    Objective function for a two-layer network where all weights are trained.
    This results in a non-convex optimization problem.
    The loss is 0.5 * ||activation(X @ w1.T + b1) @ w2 - y||^2.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_hidden: int):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.n_hidden = n_hidden

    def _unpack_params(
        self, params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w1_size = self.n_hidden * self.n_features
        b1_size = self.n_hidden
        w1 = params[:w1_size].reshape((self.n_hidden, self.n_features))
        b1 = params[w1_size : w1_size + b1_size]
        w2 = params[w1_size + b1_size :]
        return w1, b1, w2

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def __call__(self, params: np.ndarray) -> float:
        w1, b1, w2 = self._unpack_params(params)
        h = self._sigmoid(self.X @ w1.T + b1)
        h_with_bias = np.c_[h, np.ones(self.n_samples)]
        y_pred = h_with_bias @ w2
        return 0.5 * np.sum((y_pred - self.y) ** 2)

    def _sigmoid_grad(self, z: np.ndarray) -> np.ndarray:
        s = self._sigmoid(z)
        return s * (1 - s)

    def grad(self, params: np.ndarray) -> np.ndarray:
        w1, b1, w2 = self._unpack_params(params)

        z1 = self.X @ w1.T + b1
        h = self._sigmoid(z1)
        h_with_bias = np.c_[h, np.ones(self.n_samples)]
        y_pred = h_with_bias @ w2
        error = (y_pred - self.y).reshape(-1, 1)
        grad_w2 = (h_with_bias.T @ error).flatten()
        w2_without_bias = w2[:-1].reshape(1, -1)
        grad_h = error @ w2_without_bias
        grad_z1 = grad_h * self._sigmoid_grad(z1)
        grad_b1 = np.sum(grad_z1, axis=0)
        grad_w1 = grad_z1.T @ self.X
        return np.concatenate([grad_w1.flatten(), grad_b1, grad_w2])
