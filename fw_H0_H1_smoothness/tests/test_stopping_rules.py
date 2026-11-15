import numpy as np
import pytest

from ..stopping_rules import GapClippedStoppingRule, PolyakResidualStoppingRule

_GRAD = np.array([3.0, 4.0])  # ||grad|| = 5
_D = np.array([-1.0, 0.0])  # ||d|| = 1, G = -<grad, d> = 3


def test_gap_clipped_stopping_rule_metrics():
    """It should compute G, r_check = f - f_low, and phi = min(G, r_check)."""
    rule = GapClippedStoppingRule(tol=0.5, f_low=0.2)

    m1 = rule.compute_metrics(f_val=1.0, grad=_GRAD, d=_D)
    assert m1["G"] == pytest.approx(3.0)
    assert m1["r_check"] == pytest.approx(0.8)
    assert m1["phi"] == pytest.approx(0.8)
    assert m1["M"] == pytest.approx(5.0)
    assert m1["D"] == pytest.approx(1.0)
    assert not rule.check(f_val=1.0, grad=_GRAD, d=_D)

    m2 = rule.compute_metrics(f_val=0.6, grad=_GRAD, d=_D)
    assert m2["phi"] == pytest.approx(0.4)
    assert rule.check(f_val=0.6, grad=_GRAD, d=_D)


def test_gap_clipped_stopping_rule_when_gap_is_smaller():
    """It should use G when G < f - f_low."""
    rule = GapClippedStoppingRule(tol=0.5, f_low=0.0)
    small_d = np.array([-0.05, 0.0])  # G = 0.15

    metrics = rule.compute_metrics(f_val=1.0, grad=_GRAD, d=small_d)
    assert metrics["G"] == pytest.approx(0.15)
    assert metrics["r_check"] == pytest.approx(1.0)
    assert metrics["phi"] == pytest.approx(0.15)
    assert rule.check(f_val=1.0, grad=_GRAD, d=small_d)


def test_polyak_residual_stopping_rule():
    """It should check f(x) - f_up <= tol and validate f_up >= f_low."""
    rule = PolyakResidualStoppingRule(tol=0.1, f_low=0.2, f_up=0.5)

    m = rule.compute_metrics(f_val=0.55, grad=_GRAD, d=_D)
    assert m["r_check"] == pytest.approx(0.35)
    assert m["r_hat"] == pytest.approx(0.05)
    assert rule.check(f_val=0.55, grad=_GRAD, d=_D)
    assert not rule.check(f_val=0.7, grad=_GRAD, d=_D)

    with pytest.raises(AssertionError):
        PolyakResidualStoppingRule(tol=0.1, f_low=0.6, f_up=0.5)
