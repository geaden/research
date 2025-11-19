from algorithms import InexactFW
from algorithms import AdaptiveInexactFW
from step import DecayingStepSize, ShortStepSize
from objectives import MSE
from lmo import MinLinearDirectionL2Ball
import numpy as np


def test_non_adaptive():
    np.random.seed(2025)
    A = np.array([[1, 2], [4, 5]])
    b = np.array([1, 2])
    alpha = 0.5
    algorithm = InexactFW(
        obj=MSE(
            A=A,
            b=b,
        ),
        step_size=DecayingStepSize(),
        lmo=MinLinearDirectionL2Ball(radius=1.0),
        L=np.linalg.eigvals(A).max(),
        alpha=alpha,
    )
    algorithm._verbose = True

    result = algorithm.run(x0=np.zeros(A.shape[1]))

    algorithm.theoretical_iterations(result)

    assert result


def test_adaptive():
    np.random.seed(2025)
    alpha = 0.5
    A = np.array([[1, 2], [4, 5]])
    b = np.array([1, 2])
    algorithm = AdaptiveInexactFW(
        obj=MSE(
            A=A,
            b=b,
        ),
        step_size=ShortStepSize(alpha=alpha),
        lmo=MinLinearDirectionL2Ball(radius=1.0),
        L=np.linalg.eigvals(A).max(),
        alpha=alpha,
    )
    algorithm._verbose = True

    algorithm.run(x0=np.zeros(A.shape[1]))

    result = algorithm.run(x0=np.zeros(A.shape[1]))

    algorithm.theoretical_iterations(result)

    assert result
