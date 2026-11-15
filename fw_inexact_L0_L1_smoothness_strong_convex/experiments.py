"""Experiments for Frank-Wolfe with composite (mixed) gradient inexactness and
(L0, L1)-smoothness on the logistic regression problem.

The feasible set is the Euclidean l2-ball (a strongly convex set).

All methods keep running after the stopping rule G̃(x^k) ≤ Δ fires; the
iteration at which the rule fired is marked on the plots by a vertical line
(of the same color as the corresponding curve) and a star marker.

Produced figures (PGF, ``images/`` directory), named ``denisov{n}.pgf`` where
``n`` is the number of the experiment:
    * ``denisov1.pgf`` -- proposed vs classical FW on the l2-ball.
    * ``denisov2.pgf`` -- influence of the absolute inexactness δ.
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
    # Try relative imports first (when run as module)
    from .oracles import MixedInexactOracle, MixedInexactObjectiveWrapper
    from .algorithms import ProposedMixedInexactFW, ClassicalMixedInexactFW
    from .stopping_rules import InexactDualGapStoppingRule
except ImportError:
    # Fall back to absolute imports (when run directly)
    from oracles import MixedInexactOracle, MixedInexactObjectiveWrapper
    from algorithms import ProposedMixedInexactFW, ClassicalMixedInexactFW
    from stopping_rules import InexactDualGapStoppingRule


_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
_FIGSIZE = (14, 5.5)
_FIGSIZE_SINGLE = (8, 5.5)
_DPI = 100
_FLOOR = 1e-10


@dataclass
class ExperimentConfig:
    """Configuration of the logistic regression experiments."""

    n: int = 250  # problem dimension (= number of samples)
    alpha: float = 0.05  # relative inexactness
    delta: float = 1e-3  # absolute inexactness
    target_precision: float = 5e-3  # Δ
    radius: float = 1.0  # radius of the l2-ball
    max_iter: int = 300
    seed: int = 74


@dataclass
class LogRegProblem:
    """Logistic regression problem on a feasible set."""

    obj: Objective
    lmo: LMO
    project: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    L0: float
    L1: float
    L: float
    D: float
    f_star: float = field(default=np.nan)


@dataclass
class Curve:
    """Trajectory of a single run."""

    label: str
    residuals: list[float]
    stopped_at: Optional[int]


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


def _logreg_data(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    A = non_singular_matrix(config.n, 0.75, 1.0, -1.0, 1.0, rng).astype(np.float64)
    y = rng.choice([-1, 1], size=(config.n,))
    return A, y


def _smoothness_constants(A: np.ndarray) -> tuple[float, float, float]:
    """Return (L0, L1, L) for the logistic regression.

    For every summand φ_i(x) = log(1 + exp(-y_i <a_i, x>)) one has
    ||∇²φ_i(x)|| ≤ ||a_i|| ||∇φ_i(x)||, i.e. (0, ||a_i||)-smoothness;
    we use L1 = max_i ||a_i|| and a small L0, as in the experiments of
    the (L0, L1)-Frank-Wolfe paper. L is the global Lipschitz constant.
    """
    n = A.shape[0]
    row_norms = np.linalg.norm(A, axis=1)
    L1 = float(row_norms.max())
    L = float(np.linalg.norm(A, 2) ** 2 / (4 * n))
    L0 = 1e-2 * L
    return L0, L1, L


def _project_l2_ball(radius: float) -> Callable[[np.ndarray], np.ndarray]:
    def project(x: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(x)
        return x if nrm <= radius else x * (radius / nrm)

    return project


def _reference_optimum(problem: LogRegProblem, iterations: int = 20_000) -> float:
    """Compute f* by accelerated projected gradient (FISTA) with exact gradients."""
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
    L0, L1, L = _smoothness_constants(A)
    problem = LogRegProblem(
        obj=LogisticRegression(A, y),
        lmo=L2BallLMO(radius=config.radius),
        project=_project_l2_ball(config.radius),
        x0=np.zeros(config.n),
        L0=L0,
        L1=L1,
        L=L,
        D=2.0 * config.radius,
    )
    problem.f_star = _reference_optimum(problem)
    return problem


# ---------------------------------------------------------------------------
# Running methods
# ---------------------------------------------------------------------------


def _curve_from(
    label: str, algorithm, problem: LogRegProblem
) -> Curve:
    residuals = [max(_FLOOR, problem.obj(x) - problem.f_star) for x in algorithm.history]
    return Curve(label=label, residuals=residuals, stopped_at=algorithm.stopped_at)


def run_proposed(
    problem: LogRegProblem,
    oracle,
    target_precision: float,
    max_iter: int,
    label: str,
) -> Curve:
    """Run the proposed method; it does NOT stop when the rule fires."""
    params = oracle.parameters
    stopping_rule = InexactDualGapStoppingRule(
        tol=target_precision, alpha=params["alpha"], delta=params["delta"]
    )
    algorithm = ProposedMixedInexactFW(
        obj=MixedInexactObjectiveWrapper(problem.obj, oracle),
        lmo=problem.lmo,
        L0=problem.L0,
        L1=problem.L1,
        stopping_rule=stopping_rule,
        max_iter=max_iter,
        stop_on_rule=False,
    )
    algorithm.run(problem.x0)
    return _curve_from(label, algorithm, problem)


def run_classical(
    problem: LogRegProblem,
    oracle,
    target_precision: float,
    max_iter: int,
    label: str,
) -> Curve:
    """Run classical FW with h_k = 2/(k+2); the stopping rule is only marked."""
    params = oracle.parameters
    stopping_rule = InexactDualGapStoppingRule(
        tol=target_precision, alpha=params["alpha"], delta=params["delta"]
    )
    algorithm = ClassicalMixedInexactFW(
        obj=MixedInexactObjectiveWrapper(problem.obj, oracle),
        lmo=problem.lmo,
        alpha=params["alpha"],
        delta=params["delta"],
        max_iter=max_iter,
        stopping_rule=stopping_rule,
    )
    algorithm.run(problem.x0)
    return _curve_from(label, algorithm, problem)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _style(i: int) -> dict:
    return {
        "color": _COLORS[i % len(_COLORS)],
        "linestyle": _LINESTYLES[i % len(_LINESTYLES)],
        "linewidth": 1.8,
    }


def _mark_stop(ax: plt.Axes, curve: Curve, values: list[float], i: int) -> None:
    """Mark the iteration at which the stopping rule fired."""
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
    """Name of the PGF file of the n-th experiment."""
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


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def experiment_proposed_vs_classical(
    problem: LogRegProblem,
    config: ExperimentConfig,
    filename: str,
    interactive: bool,
    verbose: bool,
) -> list[Curve]:
    """Experiment 1: proposed method vs classical FW on the l2-ball."""
    log("Experiment 1: proposed vs classical on the l2-ball", verbose=verbose)

    def oracle(seed_shift: int) -> MixedInexactOracle:
        return MixedInexactOracle(
            obj=problem.obj,
            alpha=config.alpha,
            delta=config.delta,
            rng=np.random.default_rng(config.seed + seed_shift),
        )

    curves = [
        run_proposed(
            problem,
            oracle(0),
            config.target_precision,
            config.max_iter,
            r"Алгоритм~1, шаг $h_k$",
        ),
        run_classical(
            problem,
            oracle(1),
            config.target_precision,
            config.max_iter,
            r"FW, шаг $\frac{2}{k+2}$",
        ),
    ]

    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_SINGLE, dpi=_DPI)
    plot_residuals(ax, curves, r"Невязка по функции")
    _save(filename, interactive, verbose)
    _log_curves(curves, verbose)
    return curves


def experiment_absolute_inexactness(
    problem: LogRegProblem,
    config: ExperimentConfig,
    filename: str,
    interactive: bool,
    verbose: bool,
) -> None:
    """Experiment 2: influence of the absolute inexactness δ with fixed α."""
    log("Experiment 2: influence of absolute inexactness", verbose=verbose)

    deltas_curves = [0.0, 1e-3, 1e-2, 3e-2]
    curves = []
    for delta in deltas_curves:
        oracle = MixedInexactOracle(
            obj=problem.obj,
            alpha=config.alpha,
            delta=delta,
            rng=np.random.default_rng(config.seed),
        )
        curves.append(
            run_proposed(
                problem,
                oracle,
                config.target_precision,
                config.max_iter,
                rf"$\delta={delta:g}$",
            )
        )

    # Residual at the moment the stopping rule fires vs δ and the
    # theoretical guarantee f(x^N) - f* ≤ Δ + 2 ε D.
    deltas_grid = np.linspace(0.0, 4e-2, 17)
    residual_at_stop = []
    bound = []
    stop_iters = []
    for delta in deltas_grid:
        oracle = MixedInexactOracle(
            obj=problem.obj,
            alpha=config.alpha,
            delta=delta,
            rng=np.random.default_rng(config.seed),
        )
        curve = run_proposed(
            problem, oracle, config.target_precision, config.max_iter, ""
        )
        k = curve.stopped_at if curve.stopped_at is not None else config.max_iter
        stop_iters.append(k)
        residual_at_stop.append(curve.residuals[k])
        M = np.linalg.norm(problem.obj.grad(problem.x0)) * (1 + config.alpha) + delta
        eps = (config.alpha * M + delta) / (1.0 - config.alpha)
        bound.append(config.target_precision + 2.0 * eps * problem.D)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_FIGSIZE, dpi=_DPI)
    plot_residuals(ax1, curves, r"(а) Невязка при различных $\delta$")

    ax2.semilogy(
        deltas_grid, residual_at_stop, "o-", color=_COLORS[0],
        label=r"$f(x^N) - f^*$ в момент остановки",
    )
    ax2.semilogy(
        deltas_grid, bound, "--", color=_COLORS[1],
        label=r"Оценка $\Delta + 2\varepsilon D$",
    )
    ax2.set_xlabel(r"$\delta$")
    ax2.set_ylabel(r"$f(x^N) - f^*$")
    ax2.grid(True, alpha=0.3)
    ax2r = ax2.twinx()
    ax2r.plot(
        deltas_grid, stop_iters, "s:", color=_COLORS[2],
        label=r"Число итераций $N$",
    )
    ax2r.set_ylabel(r"$N$")
    lines = ax2.get_lines() + ax2r.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines],
               frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax2.set_title(r"(б) Качество решения и момент остановки")
    _save(filename, interactive, verbose)
    _log_curves(curves, verbose)


def _log_curves(curves: list[Curve], verbose: bool) -> None:
    for curve in curves:
        k = curve.stopped_at
        res = curve.residuals[k] if k is not None else float("nan")
        log(
            f"  {curve.label}: stop at {k}, residual at stop {res:.3e}, "
            f"final residual {curve.residuals[-1]:.3e}",
            verbose=verbose,
        )


def run_experiments(verbose: bool = False, interactive: bool = False):
    """
    Run experiments and generate plots.

    Args:
        verbose: Whether to print detailed progress
        interactive: Whether to show plots interactively
    """
    log("=" * 80, verbose=verbose)
    log("Frank-Wolfe with composite gradient inexactness, logistic regression",
        verbose=verbose)
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
        f"l2-ball: L0={l2ball.L0:.3g}, L1={l2ball.L1:.3g}, L={l2ball.L:.3g}, "
        f"f*={l2ball.f_star:.6f}",
        verbose=verbose,
    )

    experiment_proposed_vs_classical(
        l2ball, config, _figure_name(1), interactive, verbose
    )
    experiment_absolute_inexactness(
        l2ball, config, _figure_name(2), interactive, verbose
    )

    log("Done.", verbose=verbose)
