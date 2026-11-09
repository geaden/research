import numpy as np
import pytest

from ..stopping_rules import InexactDualGapStoppingRule, StandardStoppingRule

_G_HAT = np.array([3.0, 4.0])  # ||ĝ|| = 5
_D = np.array([-1.0, 0.0])  # ||d|| = 1, Ĝ = -<ĝ, d> = 3


def test_inexact_dual_gap():
    """It should compute G̃ = Ĝ - (α M_k + δ) / (1 - α) D_k."""
    rule = InexactDualGapStoppingRule(tol=1.0, alpha=0.25, delta=0.25)

    gap = rule.compute_inexact_gap(_G_HAT, _D)

    assert gap["hat_G"] == pytest.approx(3.0)
    assert gap["M"] == pytest.approx(5.0)
    assert gap["D"] == pytest.approx(1.0)
    assert gap["noise_bound"] == pytest.approx((0.25 * 5.0 + 0.25) / 0.75)
    assert gap["tilde_G"] == pytest.approx(3.0 - 2.0)


def test_inexact_dual_gap_stopping_rule():
    """It should stop when G̃ <= Δ."""
    rule = InexactDualGapStoppingRule(tol=1.0, alpha=0.25, delta=0.25)

    assert rule.check(_G_HAT, _D)
    assert not InexactDualGapStoppingRule(tol=0.9, alpha=0.25, delta=0.25).check(
        _G_HAT, _D
    )


def test_inexact_dual_gap_reduces_to_exact():
    """It should coincide with the oracle gap for α = δ = 0."""
    rule = InexactDualGapStoppingRule(tol=3.0, alpha=0.0, delta=0.0)

    assert rule.compute_inexact_gap(_G_HAT, _D)["tilde_G"] == pytest.approx(3.0)
    assert rule.check(_G_HAT, _D)


@pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.5])
def test_inexact_dual_gap_invalid_alpha(alpha: float):
    """It should reject relative inexactness outside of [0, 1)."""
    with pytest.raises(AssertionError):
        InexactDualGapStoppingRule(tol=1.0, alpha=alpha, delta=0.0)


def test_standard_stopping_rule():
    """It should stop when the oracle gap Ĝ <= tol."""
    rule = StandardStoppingRule(tol=3.0)

    assert rule.compute_gap(_G_HAT, _D) == pytest.approx(3.0)
    assert rule.check(_G_HAT, _D)
    assert not StandardStoppingRule(tol=2.9).check(_G_HAT, _D)
