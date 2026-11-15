import numpy as np
import pytest

from common.objectives import Objective, LogisticRegression
from common.oracles.lmo import LMO, L2BallLMO

_DIMENSION = 10
_RADIUS = 1.0


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)


@pytest.fixture()
def data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    A = rng.uniform(-1.0, 1.0, size=(_DIMENSION, _DIMENSION))
    y = rng.choice([-1, 1], size=(_DIMENSION,))
    return A, y


@pytest.fixture()
def objective(data: tuple[np.ndarray, np.ndarray]) -> Objective:
    A, y = data
    return LogisticRegression(A, y)


@pytest.fixture()
def smoothness(data: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """(L0, L1) constants of the logistic regression."""
    A, _ = data
    L = np.linalg.norm(A, 2) ** 2 / (4 * A.shape[0])
    return 1e-2 * L, float(np.linalg.norm(A, axis=1).max())


@pytest.fixture()
def lmo() -> LMO:
    return L2BallLMO(radius=_RADIUS)


@pytest.fixture()
def x0() -> np.ndarray:
    return np.zeros(_DIMENSION)


@pytest.fixture()
def f_star(objective: Objective, data: tuple[np.ndarray, np.ndarray]) -> float:
    """Reference optimal value on the l2-ball (projected gradient method)."""
    A, _ = data
    step = 4 * A.shape[0] / np.linalg.norm(A, 2) ** 2  # 1 / L
    x = np.zeros(_DIMENSION)
    for _ in range(20_000):
        x = x - step * objective.grad(x)
        nrm = np.linalg.norm(x)
        if nrm > _RADIUS:
            x *= _RADIUS / nrm
    return float(objective(x))
