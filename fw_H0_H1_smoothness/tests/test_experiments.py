import numpy as np
import pytest

from ..experiments import (
    ExperimentConfig,
    _figure_name,
    _project_l2_ball,
    _smoothness_constants,
    setup_l2ball_problem,
    run_gap_clipped,
    run_polyak,
    run_classical,
)


def test_figure_name():
    """It should name figures as denisov{n}.pgf."""
    assert _figure_name(1) == "denisov1.pgf"
    assert _figure_name(2) == "denisov2.pgf"
    assert _figure_name(3) == "denisov3.pgf"


def test_project_l2_ball():
    """It should project onto the l2-ball."""
    project = _project_l2_ball(2.0)

    np.testing.assert_allclose(project(np.array([0.5, 0.5])), [0.5, 0.5])
    np.testing.assert_allclose(project(np.array([3.0, 4.0])), [1.2, 1.6])


def test_smoothness_constants():
    """It should compute H1 = max ||a_i|| and H0 = L = ||A||^2 / (4n)."""
    A = np.array([[3.0, 4.0], [0.0, 1.0]])

    H0, H1, L = _smoothness_constants(A)

    assert H1 == pytest.approx(5.0)
    assert L == pytest.approx(np.linalg.norm(A, 2) ** 2 / 8)
    assert H0 == pytest.approx(L)


def test_run_methods_on_l2ball():
    """It should run gap-clipped, Polyak, and classical FW on the l2-ball."""
    config = ExperimentConfig(n=20, max_iter=120)
    problem = setup_l2ball_problem(config)

    assert problem.D == pytest.approx(2.0)
    assert np.isfinite(problem.f_star)

    c_gap = run_gap_clipped(
        problem, 0.0, config.target_precision, config.max_iter, "gap"
    )
    c_polyak = run_polyak(
        problem,
        0.0,
        problem.f_star,
        config.target_precision,
        config.max_iter,
        "polyak",
    )
    c_classical = run_classical(
        problem, config.target_precision, config.max_iter, "classical"
    )

    for curve in (c_gap, c_polyak, c_classical):
        assert len(curve.residuals) == config.max_iter + 1
        assert curve.stopped_at is not None
        assert min(curve.residuals) >= 0.0
    assert c_gap.residuals[-1] < c_classical.residuals[-1]
