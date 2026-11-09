import numpy as np
import pytest

from common.objectives import Objective
from ..objectives import ShiftedPointProblem
from ..oracles import (
    MixedInexactOracle,
    MixedInexactObjectiveWrapper,
    LowPrecisionCompressedOracle,
    ShiftedPointOracle,
)

_POINTS = 50


def _points(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Random points in the unit l2-ball."""
    x = rng.standard_normal(size=(_POINTS, dim))
    return x * rng.uniform(0, 1, size=(_POINTS, 1)) / np.linalg.norm(
        x, axis=1, keepdims=True
    )


def _assert_composite_error(oracle, objective: Objective, points: np.ndarray):
    """||ĝ(x) - ∇f(x)|| <= α ||∇f(x)|| + δ at all points."""
    alpha, delta = oracle.parameters["alpha"], oracle.parameters["delta"]
    for x in points:
        g = objective.grad(x)
        error = np.linalg.norm(oracle(x) - g)
        assert error <= alpha * np.linalg.norm(g) + delta + 1e-12


@pytest.mark.parametrize(
    "alpha,delta", [(0.0, 0.0), (0.05, 0.0), (0.0, 1e-2), (0.05, 1e-3), (0.4, 0.1)]
)
def test_mixed_inexact_oracle(
    objective: Objective, rng: np.random.Generator, alpha: float, delta: float
):
    """It should satisfy the composite inexactness condition."""
    oracle = MixedInexactOracle(
        obj=objective, alpha=alpha, delta=delta, rng=np.random.default_rng(1)
    )

    assert oracle.parameters == {"alpha": alpha, "delta": delta}
    _assert_composite_error(oracle, objective, _points(rng, 10))


def test_mixed_inexact_oracle_inexact_gradient_bound(
    objective: Objective, rng: np.random.Generator
):
    """It should satisfy ||ĝ - ∇f|| <= (α ||ĝ|| + δ) / (1 - α)."""
    alpha, delta = 0.3, 1e-2
    oracle = MixedInexactOracle(obj=objective, alpha=alpha, delta=delta, rng=rng)

    for x in _points(rng, 10):
        g_hat = oracle(x)
        error = np.linalg.norm(g_hat - objective.grad(x))
        assert error <= (alpha * np.linalg.norm(g_hat) + delta) / (1 - alpha) + 1e-12


def test_mixed_inexact_oracle_exact(objective: Objective, rng: np.random.Generator):
    """It should return the exact gradient for α = δ = 0."""
    oracle = MixedInexactOracle(obj=objective, alpha=0.0, delta=0.0)

    for x in _points(rng, 10):
        np.testing.assert_allclose(oracle(x), objective.grad(x))


@pytest.mark.parametrize("alpha,delta", [(-0.1, 0.0), (1.0, 0.0), (0.1, -1.0)])
def test_mixed_inexact_oracle_invalid_parameters(
    objective: Objective, alpha: float, delta: float
):
    """It should reject invalid inexactness parameters."""
    with pytest.raises(AssertionError):
        MixedInexactOracle(obj=objective, alpha=alpha, delta=delta)


@pytest.mark.parametrize("bits,q", [(10, 1e-5), (6, 1e-4), (3, 3e-3), (4, 0.0)])
def test_low_precision_compressed_oracle(
    objective: Objective, rng: np.random.Generator, bits: int, q: float
):
    """It should produce composite error with α = 2^{-t} and δ = q √n / 2."""
    oracle = LowPrecisionCompressedOracle(
        obj=objective, mantissa_bits=bits, q=q, dim=10
    )

    assert oracle.parameters["alpha"] == pytest.approx(2.0 ** (-bits))
    assert oracle.parameters["delta"] == pytest.approx(q * np.sqrt(10) / 2)
    _assert_composite_error(oracle, objective, _points(rng, 10))


def test_low_precision_compressed_oracle_rounding():
    """It should round to the given number of mantissa bits."""
    oracle = LowPrecisionCompressedOracle(
        obj=ShiftedPointProblem(dim=2), mantissa_bits=2, q=0.0, dim=2
    )

    # 0.8 = 0.8 * 2^0 -> 0.75 * 2^0; 5.5 = 0.6875 * 2^3 -> 0.75 * 2^3.
    np.testing.assert_allclose(oracle._round(np.array([0.8, 5.5])), [0.75, 6.0])


def test_low_precision_compressed_oracle_invalid_parameters(objective: Objective):
    """It should reject invalid parameters."""
    with pytest.raises(AssertionError):
        LowPrecisionCompressedOracle(obj=objective, mantissa_bits=0, q=0.0, dim=10)
    with pytest.raises(AssertionError):
        LowPrecisionCompressedOracle(obj=objective, mantissa_bits=4, q=-1.0, dim=10)


@pytest.mark.parametrize("eps", [1e-4, 1e-3, 1e-2])
def test_shifted_point_oracle(
    objective: Objective,
    smoothness: tuple[float, float],
    rng: np.random.Generator,
    eps: float,
):
    """It should produce composite error α = L1 ε e^{L1 ε}, δ = L0 ε e^{L1 ε}."""
    L0, L1 = smoothness
    oracle = ShiftedPointOracle(
        obj=objective, eps=eps, L0=L0, L1=L1, dim=10, rng=rng
    )

    factor = eps * np.exp(L1 * eps)
    assert np.linalg.norm(oracle.shift) == pytest.approx(eps)
    assert oracle.parameters["alpha"] == pytest.approx(L1 * factor)
    assert oracle.parameters["delta"] == pytest.approx(L0 * factor)
    _assert_composite_error(oracle, objective, _points(rng, 10))


def test_shifted_point_oracle_too_large_shift(objective: Objective):
    """It should reject shifts producing α >= 1/2."""
    with pytest.raises(AssertionError):
        ShiftedPointOracle(obj=objective, eps=1.0, L0=1.0, L1=1.0, dim=10)
    with pytest.raises(AssertionError):
        ShiftedPointOracle(obj=objective, eps=-1.0, L0=1.0, L1=1.0, dim=10)


def test_mixed_inexact_objective_wrapper(
    objective: Objective, rng: np.random.Generator
):
    """It should evaluate the exact objective and return the inexact gradient."""
    oracle = MixedInexactOracle(
        obj=objective, alpha=0.1, delta=1e-2, rng=np.random.default_rng(3)
    )
    wrapper = MixedInexactObjectiveWrapper(objective, oracle)
    x = rng.standard_normal(10)

    assert wrapper(x) == pytest.approx(objective(x))
    np.testing.assert_allclose(wrapper.exact_grad(x), objective.grad(x))
    assert not np.allclose(wrapper.grad(x), objective.grad(x))
    assert wrapper.oracle_params == {"alpha": 0.1, "delta": 1e-2}


def test_shifted_point_problem():
    """It should define the (L0, L1)-smooth shifted point problem."""
    problem = ShiftedPointProblem(dim=5, L0=0.5, L1=2.0, shift_scale=1.2, seed=1)
    x_star = problem.x_optimal

    assert np.linalg.norm(x_star) == pytest.approx(1.2)
    assert problem(x_star) == pytest.approx(0.0)
    np.testing.assert_allclose(problem.grad(x_star), np.zeros(5))

    x = np.zeros(5)
    r = np.linalg.norm(x - x_star)
    assert problem(x) == pytest.approx(0.25 * r**2 + 2.0 / 3.0 * r**3)
    # Gradient agrees with central finite differences.
    h = 1e-6
    numerical = np.array(
        [(problem(x + h * e) - problem(x - h * e)) / (2 * h) for e in np.eye(5)]
    )
    np.testing.assert_allclose(problem.grad(x), numerical, rtol=1e-6)
