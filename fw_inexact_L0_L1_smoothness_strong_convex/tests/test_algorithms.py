import numpy as np
import pytest

from common.objectives import Objective
from common.oracles.lmo import LMO
from ..algorithms import (
    ProposedMixedInexactFW,
    ClassicalMixedInexactFW,
    AdaptiveMixedInexactFW,
)
from ..oracles import MixedInexactOracle, MixedInexactObjectiveWrapper
from ..stopping_rules import InexactDualGapStoppingRule

_ALPHA = 0.05
_DELTA = 1e-3
_TOL = 1e-3
_MAX_ITER = 200


@pytest.fixture()
def inexact_objective(objective: Objective) -> MixedInexactObjectiveWrapper:
    oracle = MixedInexactOracle(
        obj=objective,
        alpha=_ALPHA,
        delta=_DELTA,
        rng=np.random.default_rng(2026),
    )
    return MixedInexactObjectiveWrapper(objective, oracle)


@pytest.fixture()
def stopping_rule() -> InexactDualGapStoppingRule:
    return InexactDualGapStoppingRule(tol=_TOL, alpha=_ALPHA, delta=_DELTA)


@pytest.fixture()
def proposed(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    smoothness: tuple[float, float],
    stopping_rule: InexactDualGapStoppingRule,
) -> ProposedMixedInexactFW:
    L0, L1 = smoothness
    return ProposedMixedInexactFW(
        obj=inexact_objective,
        lmo=lmo,
        L0=L0,
        L1=L1,
        stopping_rule=stopping_rule,
        max_iter=_MAX_ITER,
    )


def _values(objective: Objective, history: list[np.ndarray]) -> np.ndarray:
    return np.array([objective(x) for x in history])


def test_proposed_frank_wolfe_stops_on_rule(
    proposed: ProposedMixedInexactFW,
    objective: Objective,
    x0: np.ndarray,
    f_star: float,
):
    """It should stop once the inexact dual gap is below the target precision."""
    result = proposed.run(x0)

    assert result
    k = proposed.stopped_at
    assert k is not None
    assert len(proposed.history) == k + 1
    assert len(proposed.gap_history) == k + 1
    assert len(proposed.step_history) == k
    assert proposed.gap_history[-1]["tilde_G"] <= _TOL
    assert all(g["tilde_G"] > _TOL for g in proposed.gap_history[:-1])
    np.testing.assert_allclose(result.x_opt, proposed.history[-1])
    # Quality of the solution: f(x^k) - f* <= Δ + 2 ε_k D.
    eps_k = proposed.gap_history[-1]["noise_bound"]
    assert objective(result.x_opt) - f_star <= _TOL + 2 * eps_k * 2.0
    assert proposed.metrics["iterations"] == k
    assert proposed.metrics["stopped_at"] == k


def test_proposed_frank_wolfe_descent(
    proposed: ProposedMixedInexactFW,
    objective: Objective,
    x0: np.ndarray,
):
    """It should monotonically decrease the objective with steps in [0, 1]."""
    proposed.run(x0)

    values = _values(objective, proposed.history)
    assert np.all(np.diff(values) <= 1e-12)
    assert all(0.0 <= h <= 1.0 for h in proposed.step_history)


def test_proposed_frank_wolfe_continues_after_rule(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    smoothness: tuple[float, float],
    stopping_rule: InexactDualGapStoppingRule,
    objective: Objective,
    x0: np.ndarray,
):
    """It should keep running after the rule fires and record the moment it fired."""
    L0, L1 = smoothness
    algorithm = ProposedMixedInexactFW(
        obj=inexact_objective,
        lmo=lmo,
        L0=L0,
        L1=L1,
        stopping_rule=stopping_rule,
        max_iter=_MAX_ITER,
        stop_on_rule=False,
    )

    algorithm.run(x0)

    assert len(algorithm.history) == _MAX_ITER + 1
    assert algorithm.stopped_at is not None
    assert algorithm.stopped_at < _MAX_ITER
    assert algorithm.gap_history[algorithm.stopped_at]["tilde_G"] <= _TOL
    values = _values(objective, algorithm.history)
    assert values[-1] <= values[algorithm.stopped_at]


def test_proposed_frank_wolfe_zero_step_for_non_positive_gap(
    objective: Objective,
    lmo: LMO,
    smoothness: tuple[float, float],
    x0: np.ndarray,
):
    """It should not move when the inexact dual gap is non-positive."""
    L0, L1 = smoothness
    # Large absolute inexactness makes G̃(x^k) negative at every point.
    delta = 10.0
    oracle = MixedInexactOracle(obj=objective, alpha=0.0, delta=0.0)
    algorithm = ProposedMixedInexactFW(
        obj=MixedInexactObjectiveWrapper(objective, oracle),
        lmo=lmo,
        L0=L0,
        L1=L1,
        stopping_rule=InexactDualGapStoppingRule(tol=-np.inf, alpha=0.0, delta=delta),
        max_iter=5,
    )

    algorithm.run(x0)

    assert algorithm.stopped_at is None
    assert algorithm.step_history == [0.0] * 5
    for x in algorithm.history:
        np.testing.assert_allclose(x, x0)
    assert algorithm.metrics["iterations"] == 5


def test_proposed_frank_wolfe_exact_gradient_linear_rate(
    objective: Objective,
    lmo: LMO,
    smoothness: tuple[float, float],
    x0: np.ndarray,
    f_star: float,
):
    """It should converge fast on the l2-ball with the exact gradient."""
    L0, L1 = smoothness
    oracle = MixedInexactOracle(obj=objective, alpha=0.0, delta=0.0)
    algorithm = ProposedMixedInexactFW(
        obj=MixedInexactObjectiveWrapper(objective, oracle),
        lmo=lmo,
        L0=L0,
        L1=L1,
        stopping_rule=InexactDualGapStoppingRule(tol=0.0, alpha=0.0, delta=0.0),
        max_iter=300,
        stop_on_rule=False,
    )

    algorithm.run(x0)

    assert objective(algorithm.history[-1]) - f_star < 1e-6


def test_proposed_frank_wolfe_invalid_constants(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    stopping_rule: InexactDualGapStoppingRule,
):
    """It should reject negative smoothness constants."""
    with pytest.raises(AssertionError):
        ProposedMixedInexactFW(
            obj=inexact_objective, lmo=lmo, L0=-1.0, L1=1.0,
            stopping_rule=stopping_rule,
        )
    with pytest.raises(AssertionError):
        ProposedMixedInexactFW(
            obj=inexact_objective, lmo=lmo, L0=1.0, L1=-1.0,
            stopping_rule=stopping_rule,
        )


def test_proposed_frank_wolfe_smoothness_bound(proposed: ProposedMixedInexactFW):
    """It should compute L̂_k = L0 + L1 (M_k + δ) / (1 - α)."""
    M_k = 2.0
    expected = proposed.L0 + proposed.L1 * (M_k + _DELTA) / (1.0 - _ALPHA)

    assert proposed._smoothness_bound(M_k) == pytest.approx(expected)


def test_classical_frank_wolfe(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    stopping_rule: InexactDualGapStoppingRule,
    x0: np.ndarray,
):
    """It should run classical Frank-Wolfe for max_iter iterations."""
    algorithm = ClassicalMixedInexactFW(
        obj=inexact_objective,
        lmo=lmo,
        alpha=_ALPHA,
        delta=_DELTA,
        max_iter=50,
        stopping_rule=stopping_rule,
    )

    result = algorithm.run(x0)

    assert result
    assert len(algorithm.history) == 51
    assert len(algorithm.gap_history) == 50
    assert algorithm.stopped_at is not None
    assert algorithm.gap_history[algorithm.stopped_at]["tilde_G"] <= _TOL
    # The first step h_0 = 1 moves the iterate to the LMO output.
    assert np.linalg.norm(algorithm.history[1]) == pytest.approx(1.0)
    assert algorithm.metrics["iterations"] == 50
    assert algorithm.metrics["stopped_at"] == algorithm.stopped_at


def test_classical_frank_wolfe_without_stopping_rule(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    x0: np.ndarray,
):
    """It should track the inexact gap without a stopping rule."""
    algorithm = ClassicalMixedInexactFW(
        obj=inexact_objective, lmo=lmo, alpha=_ALPHA, delta=_DELTA, max_iter=10
    )

    algorithm.run(x0)

    assert algorithm.stopped_at is None
    assert len(algorithm.history) == 11
    for gap in algorithm.gap_history:
        noise = (_ALPHA * gap["M"] + _DELTA) / (1.0 - _ALPHA)
        assert gap["tilde_G"] == pytest.approx(gap["hat_G"] - noise * gap["D"])


def test_adaptive_frank_wolfe(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    smoothness: tuple[float, float],
    stopping_rule: InexactDualGapStoppingRule,
    objective: Objective,
    x0: np.ndarray,
):
    """It should run adaptive Frank-Wolfe and stop on the rule."""
    L0, L1 = smoothness
    algorithm = AdaptiveMixedInexactFW(
        obj=inexact_objective,
        lmo=lmo,
        L0=L0,
        L1=L1,
        stopping_rule=stopping_rule,
        max_iter=_MAX_ITER,
    )

    result = algorithm.run(x0)

    assert result
    assert algorithm.stopped_at is not None
    assert len(algorithm.history) == algorithm.stopped_at + 1
    assert algorithm.gap_history[-1]["tilde_G"] <= _TOL
    assert all(0.0 <= h <= 1.0 for h in algorithm.step_history)
    assert objective(result.x_opt) < objective(x0)
    assert algorithm.metrics["iterations"] == algorithm.stopped_at


def test_adaptive_frank_wolfe_continues_after_rule(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    smoothness: tuple[float, float],
    stopping_rule: InexactDualGapStoppingRule,
    x0: np.ndarray,
):
    """It should keep running after the rule fires when stop_on_rule is False."""
    L0, L1 = smoothness
    algorithm = AdaptiveMixedInexactFW(
        obj=inexact_objective,
        lmo=lmo,
        L0=L0,
        L1=L1,
        stopping_rule=stopping_rule,
        max_iter=30,
        stop_on_rule=False,
    )

    algorithm.run(x0)

    assert len(algorithm.history) == 31
    assert len(algorithm.step_history) == 30


def test_adaptive_frank_wolfe_backtracking(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    stopping_rule: InexactDualGapStoppingRule,
    objective: Objective,
    x0: np.ndarray,
):
    """It should increase underestimated constants and still decrease f."""
    algorithm = AdaptiveMixedInexactFW(
        obj=inexact_objective,
        lmo=lmo,
        L0=1e-8,
        L1=1e-8,
        stopping_rule=stopping_rule,
        max_iter=_MAX_ITER,
    )

    algorithm.run(x0)

    values = _values(objective, algorithm.history)
    assert np.all(np.diff(values) <= 1e-12)
    assert values[-1] < values[0]


def test_adaptive_frank_wolfe_invalid_parameters(
    inexact_objective: MixedInexactObjectiveWrapper,
    lmo: LMO,
    stopping_rule: InexactDualGapStoppingRule,
):
    """It should reject invalid parameters."""
    with pytest.raises(AssertionError):
        AdaptiveMixedInexactFW(
            obj=inexact_objective, lmo=lmo, L0=1.0, L1=1.0,
            stopping_rule=stopping_rule, rho=0.5,
        )
