import numpy as np

from common.objectives import MSE
from common.lmo import ShiftedBallLMO

from ..algorithm import FrankWolfe
from ..step_size import DiminishingStepSizeStrategy


def test_frank_wolfe():
    under_test = FrankWolfe(
        obj=MSE(A=np.array([[1, 2], [3, 4]]), b=[1, 2]),
        lmo=ShiftedBallLMO(center=1.0),
        step_size=DiminishingStepSizeStrategy(),
        L=0.01,
    )

    under_test.run(x0=np.zeros(2))

    assert len(under_test.history) == 1001
