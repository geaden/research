import numpy as np
from ..algorithms import (
    AdaptiveFrankWolfeRobustRelativeInexactness,
    AdaptiveFrankWolfeRobustComparison,
)
from common.objectives import MSE
from common.lmo import MinLinearDirectionL2BallLMO

np.random.seed(2026)


def test_adaptive_fw_robust_relative_inexactness():
    algorithm = AdaptiveFrankWolfeRobustRelativeInexactness(
        obj=MSE(A=np.array([[1, 2], [3, 4]]), b=np.array([1, 2])),
        lmo=MinLinearDirectionL2BallLMO(radius=1.0),
        L=1.5,
        alpha=0.5,
    )

    algorithm.run(np.array([1, 2]))

    assert len(algorithm.history) == 10001


def test_adaptive_fw_robust_comparison():
    algorithm = AdaptiveFrankWolfeRobustComparison(
        obj=MSE(A=np.array([[1, 2], [3, 4]]), b=np.array([1, 2])),
        lmo=MinLinearDirectionL2BallLMO(radius=1.0),
        L=1.5,
    )

    algorithm.run(np.array([1, 2]))

    assert len(algorithm.history) == 10001
