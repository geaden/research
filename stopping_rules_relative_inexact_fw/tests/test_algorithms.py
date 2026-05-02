from ..algorithms import InexactFW
from ..algorithms import AdaptiveInexactFW
from common.algorithmx.fw import DiminishingStepSizeStrategy, ShortStepSizeStrategy
from common.objectives import MSE
from common.oracles.lmo import L2BallLMO
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
        step_size=DiminishingStepSizeStrategy(),
        lmo=L2BallLMO(radius=1.0),
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
        step_size=ShortStepSizeStrategy(alpha=alpha),
        lmo=L2BallLMO(radius=1.0),
        L=np.linalg.eigvals(A).max(),
        alpha=alpha,
    )
    algorithm._verbose = True

    algorithm.run(x0=np.zeros(A.shape[1]))

    result = algorithm.run(x0=np.zeros(A.shape[1]))

    algorithm.theoretical_iterations(result)

    assert result
