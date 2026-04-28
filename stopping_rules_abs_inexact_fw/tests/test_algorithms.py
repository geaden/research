import numpy as np
from ..algorithms import AbsoluteInexactGradient
from ..algorithms import AbsoluteInexactFW
from common.objectives import MSE
from common.lmo import MinLinearDirectionL2BallLMO

rng = np.random.default_rng(2025)


def test_absolute_inexact_frank_wolfe():
    """It should return results for absolute inexact Frank-Wolfe algorithm."""
    A = np.array([[1, 2], [4, 5]])
    b = np.array([1, 2])
    objective = MSE(A=A, b=b)
    algorithm = AbsoluteInexactFW(
        obj=objective,
        inexact_grad=AbsoluteInexactGradient(
            obj=objective,
            rng=rng,
        ),
        lmo=MinLinearDirectionL2BallLMO(radius=1.0),
        L=np.linalg.eigvals(A).max(),
    )
    algorithm._verbose = True

    result = algorithm.run(x0=np.zeros(A.shape[1]))

    assert result
