"""Experiments for Frank-Wolfe under (H0, H1)-smoothness on logistic regression.

The feasible set is the Euclidean l2-ball (a strongly convex set).

Produced figures (PGF, ``images/`` directory), named ``denisov{n}.pgf``:
    * ``denisov1.pgf`` -- comparison of gap-clipped (H0, H1)-FW, Polyak-type
      (H0, H1)-FW, unclipped numerator baseline, and classical FW, plus
      step-size warm-up dynamics.
    * ``denisov2.pgf`` -- influence of the lower bound f_low <= f*
      (underestimation gap delta_- = f* - f_low).
    * ``denisov3.pgf`` -- influence of the upper bound f_up = f* + delta_+
      in the Polyak-type step size.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from common.math_utils import non_singular_matrix
from common.objectives import Objective, LogisticRegression
from common.oracles.lmo import LMO, L2BallLMO
from common.plot_utils import preamble, do_show_plot
from common.utils import log

try:
    from .algorithms import (
        GapClippedH0H1FW,
        PolyakH0H1FW,
        ClassicalH0H1FW,
    )
    from .stopping_rules import (
        GapClippedStoppingRule,
        PolyakResidualStoppingRule,
    )
except ImportError:
    from algorithms import (
        GapClippedH0H1FW,
        PolyakH0H1FW,
        ClassicalH0H1FW,
    )
    from stopping_rules import (
        GapClippedStoppingRule,
        PolyakResidualStoppingRule,
    )


_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
_FIGSIZE = (14, 5.5)
_DPI = 100
_FLOOR = 1e-10


@dataclass
class ExperimentConfig:
    """Configuration of the logistic regression experiments."""

    n: int = 250
    target_precision: float = 5e-3  # \Delta
    radius: float = 1.0
    max_iter: int = 300
    seed: int = 74


@dataclass
class LogRegProblem:
    """Logistic regression problem on the l2-ball."""

    obj: Objective
    lmo: LMO
    project: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    H0: float
    H1: float
    L: float
    D: float
    f_star: float = field(default=np.nan)


@dataclass
class Curve:
    """Trajectory of a single run."""

    label: str
    residuals: list[float]
    steps: list[float]
    stopped_at: Optional[int]
    T1: int = 0
    T2: int = 0
    T3: int = 0


def _logreg_data(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    A = non_singular_matrix(config.n, 0.75, 1.0, -1.0, 1.0, rng).astype(
        np.float64
    )
    y = rng.choice([-1, 1], size=(config.n,))
    return A, y


def _smoothness_constants(A: np.ndarray) -> tuple[float, float, float]:
    """Return (H0, H1, L) for logistic regression on the l2-ball.

    Each summand satisfies ||nabla^2 phi_i(x)|| <= ||a_i||^2 phi_i(x), and at
    the constrained minimum x* on the ball the Hessian is bounded by the global
    Lipschitz constant L = ||A||_2^2 / (4n), so we set H0 = L and
    H1 = max_i ||a_i||.
    """
    n = A.shape[0]
    row_norms = np.linalg.norm(A, axis=1)
    H1 = float(row_norms.max())
    L = float(np.linalg.norm(A, 2) ** 2 / (4 * n))
    H0 = L
    return H0, H1, L


def _project_l2_ball(radius: float) -> Callable[[np.ndarray], np.ndarray]:
    def project(x: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(x)
        return x if nrm <= radius else x * (radius / nrm)

    return project


def _reference_optimum(
    problem: LogRegProblem, iterations: int = 20_000
) -> float:
    """Compute f* by accelerated projected gradient (FISTA)."""
    x = problem.x0.copy()
    z = x.copy()
    t = 1.0
    step = 1.0 / problem.L
    for _ in range(iterations):
        x_next = problem.project(z - step * problem.obj.grad(z))
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_next + (t - 1.0) / t_next * (x_next - x)
        x, t = x_next, t_next
    return float(problem.obj(x))


def setup_l2ball_problem(config: ExperimentConfig) -> LogRegProblem:
    A, y = _logreg_data(config)
    H0, H1, L = _smoothness_constants(A)
    problem = LogRegProblem(
        obj=LogisticRegression(A, y),
        lmo=L2BallLMO(radius=config.radius),
        project=_project_l2_ball(config.radius),
        x0=np.zeros(config.n),
        H0=H0,
        H1=H1,
        L=L,
        D=2.0 * config.radius,
    )
    problem.f_star = _reference_optimum(problem)
    return problem


def _curve_from(label: str, algorithm, problem: LogRegProblem) -> Curve:
    residuals = [
        max(_FLOOR, float(problem.obj(x)) - problem.f_star)
        for x in algorithm.history
    ]
    return Curve(
        label=label,
        residuals=residuals,
        steps=list(getattr(algorithm, "step_history", [])),
        stopped_at=algorithm.stopped_at,
        T1=getattr(algorithm, "T1", 0),
        T2=getattr(algorithm, "T2", 0),
        T3=getattr(algorithm, "T3", 0),
    )


def run_gap_clipped(
    problem: LogRegProblem,
    f_low: float,
    target_precision: float,
    max_iter: int,
    label: str,
) -> Curve:
    """Run Variant 2 (gap-clipped step) with lower bound f_low <= f*."""
    stopping_rule = GapClippedStoppingRule(tol=target_precision, f_low=f_low)
    algorithm = GapClippedH0H1FW(
        obj=problem.obj,
        lmo=problem.lmo,
        H0=problem.H0,
        H1=problem.H1,
        stopping_rule=stopping_rule,
        max_iter=max_iter,
        stop_on_rule=False,
    )
    algorithm.run(problem.x0)
    return _curve_from(label, algorithm, problem)


def run_polyak(
    problem: LogRegProblem,
    f_low: float,
    f_up: float,
    target_precision: float,
    max_iter: int,
    label: str,
) -> Curve:
    """Run Variant 1 (Polyak-type step) with f_low <= f* <= f_up."""
    stopping_rule = PolyakResidualStoppingRule(
        tol=target_precision, f_low=f_low, f_up=f_up
    )
    algorithm = PolyakH0H1FW(
        obj=problem.obj,
        lmo=problem.lmo,
        H0=problem.H0,
        H1=problem.H1,
        stopping_rule=stopping_rule,
        max_iter=max_iter,
        stop_on_rule=False,
    )
    algorithm.run(problem.x0)
    return _curve_from(label, algorithm, problem)


def run_classical(
    problem: LogRegProblem,
    target_precision: float,
    max_iter: int,
    label: str,
) -> Curve:
    """Run classical Frank-Wolfe with step 2 / (k + 2)."""
    stopping_rule = GapClippedStoppingRule(tol=target_precision, f_low=0.0)
    algorithm = ClassicalH0H1FW(
        obj=problem.obj,
        lmo=problem.lmo,
        max_iter=max_iter,
        stopping_rule=stopping_rule,
    )
    algorithm.run(problem.x0)
    return _curve_from(label, algorithm, problem)


def _style(i: int) -> dict:
    return {
        "color": _COLORS[i % len(_COLORS)],
        "linestyle": _LINESTYLES[i % len(_LINESTYLES)],
        "linewidth": 1.8,
    }


def _mark_stop(
    ax: plt.Axes, curve: Curve, values: list[float], i: int
) -> None:
    if curve.stopped_at is None:
        return
    k = curve.stopped_at
    color = _COLORS[i % len(_COLORS)]
    ax.axvline(k, color=color, linestyle=":", linewidth=1.4, alpha=0.9)
    ax.plot(k, values[min(k, len(values) - 1)], "*", color=color, markersize=12)


def _stop_label(curve: Curve) -> str:
    if curve.stopped_at is None:
        return curve.label + r" (правило не сработало)"
    return curve.label + rf" ($N={curve.stopped_at}$)"


def plot_residuals(ax: plt.Axes, curves: list[Curve], title: str) -> None:
    for i, curve in enumerate(curves):
        ax.plot(
            np.arange(len(curve.residuals)),
            curve.residuals,
            label=_stop_label(curve),
            **_style(i),
        )
        _mark_stop(ax, curve, curve.residuals, i)
    ax.set_yscale("log")
    ax.set_xlabel(r"Итерация $k$")
    ax.set_ylabel(r"$f(x^k) - f^*$")
    ax.set_title(title)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)


def _figure_name(n: int) -> str:
    return f"denisov{n}.pgf"


def _save(filename: str, interactive: bool, verbose: bool) -> None:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(module_dir, "images"), exist_ok=True)
    original_dir = os.getcwd()
    try:
        os.chdir(module_dir)
        plt.tight_layout()
        do_show_plot(
            filename=filename,
            show_plot=True,
            interactive=interactive,
            should_draw=not interactive,
        )
        log(f"Saved images/{filename}", verbose=verbose)
    finally:
        os.chdir(original_dir)
    plt.close("all")


def _log_curves(curves: list[Curve], verbose: bool) -> None:
    for curve in curves:
        k = curve.stopped_at
        res = curve.residuals[k] if k is not None else float("nan")
        log(
            f"  {curve.label}: stop at {k} (T1={curve.T1}, T2={curve.T2}, T3={curve.T3}), "
            f"residual at stop {res:.3e}, final residual {curve.residuals[-1]:.3e}",
            verbose=verbose,
        )


def experiment_methods_and_warmup(
    problem: LogRegProblem,
    config: ExperimentConfig,
    filename: str,
    interactive: bool,
    verbose: bool,
) -> list[Curve]:
    """Experiment 1: comparison of step strategies and warm-up of h_k."""
    log("Experiment 1: comparison of methods and warm-up on l2-ball", verbose=verbose)

    curves = [
        run_gap_clipped(
            problem,
            0.0,
            config.target_precision,
            config.max_iter,
            r"Вариант~2 ($\underline{f}=0$, $\Phi_k$)",
        ),
        run_polyak(
            problem,
            problem.f_star,
            problem.f_star,
            config.target_precision,
            config.max_iter,
            r"Вариант~1 ($\underline{f}=\overline{f}=f^*$)",
        ),
        run_polyak(
            problem,
            0.0,
            0.0,
            config.target_precision,
            config.max_iter,
            r"Без отсечения ($\overline{f}=\underline{f}=0$)",
        ),
        run_classical(
            problem,
            config.target_precision,
            config.max_iter,
            r"FW, шаг $\frac{2}{k+2}$",
        ),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_FIGSIZE, dpi=_DPI)
    plot_residuals(ax1, curves, r"(а) Невязка по функции $f(x^k) - f^*$")

    # Plot step-size evolution (warm-up and decay vs oscillation)
    k_max_show = min(150, config.max_iter)
    for idx in (0, 1, 2):
        c = curves[idx]
        ax2.plot(
            np.arange(k_max_show),
            c.steps[:k_max_show],
            label=c.label,
            **_style(idx),
        )
    ax2.set_xlabel(r"Итерация $k$")
    ax2.set_ylabel(r"Величина шага $h_k$")
    ax2.set_title(r"(б) Динамика шага $h_k$ (разогрев и затухание)")
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)

    _save(filename, interactive, verbose)
    _log_curves(curves, verbose)
    return curves


def experiment_lower_bound_influence(
    problem: LogRegProblem,
    config: ExperimentConfig,
    filename: str,
    interactive: bool,
    verbose: bool,
) -> None:
    """Experiment 2: influence of the lower estimate f_low <= f*."""
    log("Experiment 2: influence of lower bound f_low <= f*", verbose=verbose)

    f_low_values = [0.0, 0.25, 0.45, problem.f_star]
    curves = []
    for f_low in f_low_values:
        lbl = (
            r"$\underline{f}=f^*$"
            if np.isclose(f_low, problem.f_star)
            else rf"$\underline{{f}}={f_low:g}$"
        )
        curves.append(
            run_gap_clipped(
                problem,
                f_low,
                config.target_precision,
                config.max_iter,
                lbl,
            )
        )

    f_low_grid = np.linspace(0.0, problem.f_star, 17)
    delta_minus_grid = problem.f_star - f_low_grid
    stop_iters = []
    residual_at_stop = []
    for f_low in f_low_grid:
        c = run_gap_clipped(
            problem, f_low, config.target_precision, config.max_iter, ""
        )
        k = c.stopped_at if c.stopped_at is not None else config.max_iter
        stop_iters.append(k)
        residual_at_stop.append(c.residuals[k])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_FIGSIZE, dpi=_DPI)
    plot_residuals(ax1, curves, r"(а) Невязка при различных $\underline{f} \le f^*$")

    ax2.semilogy(
        delta_minus_grid,
        residual_at_stop,
        "o-",
        color=_COLORS[0],
        label=r"$f(x^N) - f^*$ в момент остановки",
    )
    ax2.axhline(
        config.target_precision,
        linestyle="--",
        color=_COLORS[1],
        label=r"Порог $\Delta$",
    )
    ax2.set_xlabel(r"Занижение $\delta_- = f^* - \underline{f}$")
    ax2.set_ylabel(r"$f(x^N) - f^*$")
    ax2.grid(True, alpha=0.3)

    ax2r = ax2.twinx()
    ax2r.plot(
        delta_minus_grid,
        stop_iters,
        "s:",
        color=_COLORS[2],
        label=r"Число итераций $N$",
    )
    ax2r.set_ylabel(r"$N$")

    lines = ax2.get_lines() + ax2r.get_lines()
    ax2.legend(
        lines,
        [ln.get_label() for ln in lines],
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=9,
    )
    ax2.set_title(r"(б) Число итераций $N$ и точность от $\delta_-$")

    _save(filename, interactive, verbose)
    _log_curves(curves, verbose)


def experiment_upper_bound_polyak(
    problem: LogRegProblem,
    config: ExperimentConfig,
    filename: str,
    interactive: bool,
    verbose: bool,
) -> None:
    """Experiment 3: influence of the upper estimate f_up = f* + delta_+ in Polyak step."""
    log("Experiment 3: influence of upper bound f_up >= f* in Polyak step", verbose=verbose)

    delta_plus_values = [0.0, 5e-4, 2e-3, 5e-3]
    curves = []
    for dp in delta_plus_values:
        curves.append(
            run_polyak(
                problem,
                0.0,
                problem.f_star + dp,
                config.target_precision,
                config.max_iter,
                rf"$\delta_+={dp:g}$",
            )
        )

    dp_grid = np.linspace(0.0, 1e-2, 17)
    residual_at_stop = []
    bound = []
    stop_iters = []
    for dp in dp_grid:
        c = run_polyak(
            problem,
            0.0,
            problem.f_star + dp,
            config.target_precision,
            config.max_iter,
            "",
        )
        k = c.stopped_at if c.stopped_at is not None else config.max_iter
        stop_iters.append(k)
        residual_at_stop.append(c.residuals[k])
        bound.append(config.target_precision + dp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_FIGSIZE, dpi=_DPI)
    plot_residuals(
        ax1,
        curves,
        r"(а) Шаг типа Поляка ($\underline{f}=0$, $\overline{f}=f^*+\delta_+$)",
    )

    ax2.semilogy(
        dp_grid,
        residual_at_stop,
        "o-",
        color=_COLORS[0],
        label=r"$f(x^N) - f^*$ в момент остановки",
    )
    ax2.semilogy(
        dp_grid,
        bound,
        "--",
        color=_COLORS[1],
        label=r"Гарантия $\Delta + \delta_+$",
    )
    ax2.set_xlabel(r"Завышение $\delta_+ = \overline{f} - f^*$")
    ax2.set_ylabel(r"$f(x^N) - f^*$")
    ax2.grid(True, alpha=0.3)

    ax2r = ax2.twinx()
    ax2r.plot(
        dp_grid,
        stop_iters,
        "s:",
        color=_COLORS[2],
        label=r"Число итераций $N$",
    )
    ax2r.set_ylabel(r"$N$")

    lines = ax2.get_lines() + ax2r.get_lines()
    ax2.legend(
        lines,
        [ln.get_label() for ln in lines],
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=9,
    )
    ax2.set_title(r"(б) Качество решения и момент остановки от $\delta_+$")

    _save(filename, interactive, verbose)
    _log_curves(curves, verbose)


def run_experiments(verbose: bool = False, interactive: bool = False):
    """Run all experiments and generate PGF figures."""
    log("=" * 80, verbose=verbose)
    log("Frank-Wolfe under (H0, H1)-smoothness, logistic regression", verbose=verbose)
    log("=" * 80, verbose=verbose)

    if not interactive:
        preamble()
    else:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            log("TkAgg backend not available", verbose=verbose)

    config = ExperimentConfig()
    l2ball = setup_l2ball_problem(config)
    log(
        f"l2-ball: H0={l2ball.H0:.3g}, H1={l2ball.H1:.3g}, L={l2ball.L:.3g}, "
        f"f(x0)={float(l2ball.obj(l2ball.x0)):.6f}, f*={l2ball.f_star:.6f}",
        verbose=verbose,
    )

    experiment_methods_and_warmup(
        l2ball, config, _figure_name(1), interactive, verbose
    )
    experiment_lower_bound_influence(
        l2ball, config, _figure_name(2), interactive, verbose
    )
    experiment_upper_bound_polyak(
        l2ball, config, _figure_name(3), interactive, verbose
    )

    log("Done.", verbose=verbose)
