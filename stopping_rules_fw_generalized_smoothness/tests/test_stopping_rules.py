import pytest

import numpy as np

from common.objectives import Objective, MSE
from ..stopping_rules import (
    DualGapStoppingRuleStrategy,
    ConvergenceRateStoppingRuleStrategy,
)

_DIMENSION = 2


@pytest.fixture()
def objective() -> Objective:
    A = np.array(
        [
            [2, 1],
            [1, 2],
        ]
    )
    b = np.ones(_DIMENSION)
    return MSE(A, b)


def test_convergence_rate_stopping_rule(objective: Objective):
    """It should run convergence rate stopping rule."""

    stopping_rule = ConvergenceRateStoppingRuleStrategy(
        x0=np.zeros(_DIMENSION),
        obj=objective,
        strong_convexity_const=1,
        tol=1e-4,
    )

    assert not stopping_rule.check(dual_gap=1)

    assert stopping_rule.check(
        x=np.zeros(_DIMENSION), x_next=np.zeros(_DIMENSION), alpha=0.5, L0=1, L1=1
    )
    assert stopping_rule._T1 == 1

    assert stopping_rule.check(
        x=np.zeros(_DIMENSION), x_next=np.zeros(_DIMENSION), alpha=0.5, L0=3, L1=0.5
    )
    assert stopping_rule._T2 == 1

    assert stopping_rule.check(
        x=np.zeros(_DIMENSION), x_next=np.zeros(_DIMENSION), alpha=1, L0=1, L1=0.5
    )
    assert stopping_rule._T3 == 1


def test_dual_gap_stopping_rule():
    """It should run dual gap stopping rule."""

    stopping_rule = DualGapStoppingRuleStrategy(tol=0.5)

    assert stopping_rule.check(dual_gap=0.2)
    assert not stopping_rule.check(dual_gap=0.6)
