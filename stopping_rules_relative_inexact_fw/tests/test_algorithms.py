from ..algorithms import InexactFW
from ..algorithms import AdaptiveInexactFW
from ..step import DecayingStepSize, ShortStepSize
from common.objectives import MSE
from common.lmo import MinLinearDirectionL2BallLMO
import numpy as np


np.random.seed(2025)


def test_non_adaptive():
    """It should return results for non-adaptive variant of the algorithm."""
    A = np.array([[1, 2], [4, 5]])
    b = np.array([1, 2])
    alpha = 0.5
    algorithm = InexactFW(
        obj=MSE(
            A=A,
            b=b,
        ),
        step_size=DecayingStepSize(),
        lmo=MinLinearDirectionL2BallLMO(radius=1.0),
        L=np.linalg.eigvals(A).max(),
        alpha=alpha,
    )
    algorithm._verbose = True

    result = algorithm.run(x0=np.zeros(A.shape[1]))

    algorithm.theoretical_iterations(result)

    assert result


def test_adaptive():
    """It should return results for adaptive variant of the algorithm."""
    alpha = 0.5
    A = np.array([[1, 2], [4, 5]])
    b = np.array([1, 2])
    algorithm = AdaptiveInexactFW(
        obj=MSE(
            A=A,
            b=b,
        ),
        step_size=ShortStepSize(alpha=alpha),
        lmo=MinLinearDirectionL2BallLMO(radius=1.0),
        L=np.linalg.eigvals(A).max(),
        alpha=alpha,
    )
    algorithm._verbose = True

    algorithm.run(x0=np.zeros(A.shape[1]))

    result = algorithm.run(x0=np.zeros(A.shape[1]))

    algorithm.theoretical_iterations(result)

    assert result
