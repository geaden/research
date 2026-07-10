import numpy as np
from numpy.testing import assert_array_almost_equal

from common.objectives import MSE
from .. import ComparisonOracle


def test_comparison_oracle():
    under_test = ComparisonOracle(
        obj=MSE(A=np.array([[1, 2], [3, 4]]), b=np.array([1, 2])),
        gamma=0.5,
        delta=0.1,
        nu=0.5,
    )

    value = under_test(x=np.array([1, 2]), L=1)

    assert_array_almost_equal(value, np.array([-0.999998, -0.001953]))
