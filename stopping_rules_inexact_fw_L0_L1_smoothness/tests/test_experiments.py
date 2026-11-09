import numpy as np
import pytest

from ..experiments import (
    ExperimentConfig,
    _figure_name,
    _project_l2_ball,
    _smoothness_constants,
    setup_l2ball_problem,
    run_proposed,
    run_classical,
)
from ..oracles import MixedInexactOracle


def test_figure_name():
    """It should name figures as denisov{n}.pgf."""
    assert _figure_name(1) == "denisov1.pgf"
    assert _figure_name(2) == "denisov2.pgf"


def test_project_l2_ball():
    """It should project onto the l2-ball."""
    project = _project_l2_ball(2.0)

    np.testing.assert_allclose(project(np.array([0.5, 0.5])), [0.5, 0.5])
    np.testing.assert_allclose(project(np.array([3.0, 4.0])), [1.2, 1.6])


def test_smoothness_constants():
    """It should compute L1 = max ||a_i||, L = ||A||^2 / (4n) and L0 = 0.01 L."""
    A = np.array([[3.0, 4.0], [0.0, 1.0]])

    L0, L1, L = _smoothness_constants(A)

    assert L1 == pytest.approx(5.0)
    assert L == pytest.approx(np.linalg.norm(A, 2) ** 2 / 8)
    assert L0 == pytest.approx(1e-2 * L)


def test_run_methods_on_l2ball():
    """It should run the proposed and classical methods on the l2-ball problem."""
    config = ExperimentConfig(n=20, max_iter=100)
    problem = setup_l2ball_problem(config)
    oracle = MixedInexactOracle(
        obj=problem.obj,
        alpha=config.alpha,
        delta=config.delta,
        rng=np.random.default_rng(config.seed),
    )

    assert problem.D == pytest.approx(2.0)
    assert np.isfinite(problem.f_star)

    proposed = run_proposed(
        problem, oracle, config.target_precision, config.max_iter, "proposed"
    )
    classical = run_classical(
        problem, oracle, config.target_precision, config.max_iter, "classical"
    )

    for curve in (proposed, classical):
        assert len(curve.residuals) == config.max_iter + 1
        assert curve.stopped_at is not None
        assert min(curve.residuals) >= 0.0
    # Linear convergence of the proposed method on the strongly convex set.
    assert proposed.residuals[-1] < classical.residuals[-1]
