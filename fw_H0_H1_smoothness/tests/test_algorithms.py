import numpy as np
import pytest

from common.objectives import Objective
from common.oracles.lmo import LMO
from ..algorithms import (
    GapClippedH0H1FW,
    PolyakH0H1FW,
    ClassicalH0H1FW,
    AdaptiveH0H1FW,
)
from ..objectives import QuarticH0H1Problem
from ..stopping_rules import GapClippedStoppingRule, PolyakResidualStoppingRule

_TOL = 1e-3
_MAX_ITER = 200


@pytest.fixture()
def gap_rule() -> GapClippedStoppingRule:
    return GapClippedStoppingRule(tol=_TOL, f_low=0.0)


@pytest.fixture()
def gap_clipped_fw(
    objective: Objective,
    lmo: LMO,
    smoothness: tuple[float, float],
    gap_rule: GapClippedStoppingRule,
) -> GapClippedH0H1FW:
    H0, H1 = smoothness
    return GapClippedH0H1FW(
        obj=objective,
        lmo=lmo,
        H0=H0,
        H1=H1,
        stopping_rule=gap_rule,
        max_iter=_MAX_ITER,
    )


def _values(objective: Objective, history: list[np.ndarray]) -> np.ndarray:
    return np.array([float(objective(x)) for x in history])


def test_gap_clipped_fw_stops_on_rule(
    gap_clipped_fw: GapClippedH0H1FW,
    objective: Objective,
    x0: np.ndarray,
    f_star: float,
):
    """It should stop once the clipped gap phi_k <= tol."""
    result = gap_clipped_fw.run(x0)

    assert result
    k = gap_clipped_fw.stopped_at
    assert k is not None
    assert len(gap_clipped_fw.history) == k + 1
    assert len(gap_clipped_fw.gap_history) == k + 1
    assert len(gap_clipped_fw.step_history) == k
    assert gap_clipped_fw.gap_history[-1]["phi"] <= _TOL
    assert all(g["phi"] > _TOL for g in gap_clipped_fw.gap_history[:-1])
    np.testing.assert_allclose(result.x_opt, gap_clipped_fw.history[-1])
    assert float(objective(result.x_opt)) - f_star <= _TOL + 1e-9
    assert gap_clipped_fw.metrics["iterations"] == k
    assert gap_clipped_fw.T1 + gap_clipped_fw.T2 + gap_clipped_fw.T3 == k


def test_gap_clipped_fw_monotonic_descent(
    gap_clipped_fw: GapClippedH0H1FW,
    objective: Objective,
    x0: np.ndarray,
):
    """It should monotonically decrease the objective with steps in [0, 1]."""
    gap_clipped_fw.run(x0)

    values = _values(objective, gap_clipped_fw.history)
    assert np.all(np.diff(values) <= 1e-12)
    assert all(0.0 <= h <= 1.0 for h in gap_clipped_fw.step_history)


def test_gap_clipped_fw_continues_after_rule(
    objective: Objective,
    lmo: LMO,
    smoothness: tuple[float, float],
    gap_rule: GapClippedStoppingRule,
    x0: np.ndarray,
    f_star: float,
):
    """It should keep iterating when stop_on_rule is False."""
    H0, H1 = smoothness
    algorithm = GapClippedH0H1FW(
        obj=objective,
        lmo=lmo,
        H0=H0,
        H1=H1,
        stopping_rule=gap_rule,
        max_iter=_MAX_ITER,
        stop_on_rule=False,
    )

    algorithm.run(x0)

    assert len(algorithm.history) == _MAX_ITER + 1
    assert algorithm.stopped_at is not None
    assert algorithm.stopped_at < _MAX_ITER
    assert float(objective(algorithm.history[-1])) - f_star < 1e-6


def test_polyak_fw_stops_and_guarantees_quality(
    objective: Objective,
    lmo: LMO,
    smoothness: tuple[float, float],
    x0: np.ndarray,
    f_star: float,
):
    """It should stop when f(x^k) - f_up <= tol and satisfy f(x^k) - f* <= tol + delta_+."""
    H0, H1 = smoothness
    delta_plus = 2e-3
    rule = PolyakResidualStoppingRule(
        tol=_TOL, f_low=0.0, f_up=f_star + delta_plus
    )
    algorithm = PolyakH0H1FW(
        obj=objective,
        lmo=lmo,
        H0=H0,
        H1=H1,
        stopping_rule=rule,
        max_iter=_MAX_ITER,
    )

    result = algorithm.run(x0)

    assert algorithm.stopped_at is not None
    assert float(objective(result.x_opt)) - f_star <= _TOL + delta_plus + 1e-9
    assert algorithm.T1 + algorithm.T2 + algorithm.T3 == algorithm.stopped_at


def test_polyak_fw_zero_step_when_residual_non_positive(
    objective: Objective,
    lmo: LMO,
    smoothness: tuple[float, float],
    x0: np.ndarray,
):
    """It should take h_k = 0 when f(x^k) - f_up <= 0."""
    H0, H1 = smoothness
    rule = PolyakResidualStoppingRule(tol=-1.0, f_low=0.0, f_up=10.0)
    algorithm = PolyakH0H1FW(
        obj=objective,
        lmo=lmo,
        H0=H0,
        H1=H1,
        stopping_rule=rule,
        max_iter=5,
        stop_on_rule=False,
    )

    algorithm.run(x0)

    assert algorithm.step_history == [0.0] * 5
    for x in algorithm.history:
        np.testing.assert_allclose(x, x0)


def test_classical_fw_with_and_without_rule(
    objective: Objective,
    lmo: LMO,
    gap_rule: GapClippedStoppingRule,
    x0: np.ndarray,
):
    """It should run classical FW and optionally record when the rule fires."""
    alg_with = ClassicalH0H1FW(
        obj=objective, lmo=lmo, max_iter=30, stopping_rule=gap_rule
    )
    alg_with.run(x0)
    assert len(alg_with.history) == 31
    assert alg_with.stopped_at is not None
    assert alg_with.metrics["iterations"] == 30

    alg_without = ClassicalH0H1FW(obj=objective, lmo=lmo, max_iter=10)
    alg_without.run(x0)
    assert alg_without.stopped_at is None
    assert len(alg_without.gap_history) == 10


def test_adaptive_fw_convergence(
    objective: Objective,
    lmo: LMO,
    gap_rule: GapClippedStoppingRule,
    x0: np.ndarray,
):
    """It should adapt H0, H1 via backtracking and decrease f."""
    algorithm = AdaptiveH0H1FW(
        obj=objective,
        lmo=lmo,
        H0=1e-6,
        H1=1e-6,
        stopping_rule=gap_rule,
        max_iter=100,
    )

    result = algorithm.run(x0)

    assert result
    values = _values(objective, algorithm.history)
    assert np.all(np.diff(values) <= 1e-12)
    assert values[-1] < values[0]
    assert algorithm.stopped_at is not None
    assert algorithm.metrics["iterations"] == algorithm.stopped_at


def test_quartic_h0_h1_objective():
    """It should evaluate value, gradient, and optimal point of QuarticH0H1Problem."""
    prob = QuarticH0H1Problem(dim=8, H0=0.5, H1=3.0, shift_scale=1.0, seed=7)
    x_opt = prob.x_optimal
    assert prob(x_opt) == pytest.approx(0.0)
    np.testing.assert_allclose(prob.grad(x_opt), np.zeros(8), atol=1e-12)
    x = np.zeros(8)
    assert prob(x) > 0.0
    assert np.linalg.norm(prob.grad(x)) > 0.0
