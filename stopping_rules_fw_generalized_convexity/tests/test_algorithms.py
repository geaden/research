import pytest
import numpy as np
from algorithms import FrankWolfe, FrankWolfeL0L1
from common.lmo import LMO, SimplexLMO
from common.objectives import Objective, MSE


_DIMENSION = 2


@pytest.fixture()
def objective() -> Objective:
    A = [
        [2, 1],
        [1, 2],
    ]
    b = np.ones(_DIMENSION)
    return MSE(A, b)


@pytest.fixture()
def lmo() -> LMO:
    return SimplexLMO()


def test_classic_frank_wolfe(objective: Objective, lmo: LMO):
    """It should run Classic Frank-Wolfe implementation."""
    algorithm = FrankWolfe(obj=objective, lmo=lmo, L=1)

    result = algorithm.run(np.zeros(_DIMENSION))

    assert result
    assert len(algorithm.history) == 1


def test_frank_wolfe_L0L1(objective: Objective, lmo: LMO):
    """It should run L0, L1 Frank-Wolfe implementation."""
    algorithm = FrankWolfeL0L1(obj=objective, lmo=lmo, L0=1e-12, L1=2)

    result = algorithm.run(np.zeros(_DIMENSION))

    assert result
    assert len(algorithm.history) == 1
