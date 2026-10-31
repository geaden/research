"""Experiments for Frank-Wolfe with mixed inexactness and (L0, L1)-smoothness."""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple

from common.oracles.lmo import SimplexLMO
from common.plotting import LineStyle
from common.plot_utils import preamble, do_show_plot
from common.utils import log

try:
    # Try relative imports first (when run as module)
    from .objectives import ShiftedPointProblem
    from .oracles import MixedInexactOracle, MixedInexactObjectiveWrapper
    from .algorithms import ProposedMixedInexactFW, ClassicalMixedInexactFW
    from .stopping_rules import InexactDualGapStoppingRule
except ImportError:
    # Fall back to absolute imports (when run directly)
    from objectives import ShiftedPointProblem
    from oracles import MixedInexactOracle, MixedInexactObjectiveWrapper
    from algorithms import ProposedMixedInexactFW, ClassicalMixedInexactFW
    from stopping_rules import InexactDualGapStoppingRule


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""

    dim: int = 50
    L0: float = 1e-4
    L1: float = 2.5
    shift_scale: float = 1.2
    alpha: float = 0.04
    delta: float = 1e-3
    target_precision: float = 2e-3
    max_iter: int = 120
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from running an algorithm."""

    algorithm_name: str
    history: List[np.ndarray]
    gap_history: List[Dict]
    iterations: int
    final_x: np.ndarray
    objective_values: List[float]


def setup_problem(
    config: ExperimentConfig,
) -> Tuple[ShiftedPointProblem, SimplexLMO, np.ndarray]:
    """
    Set up the optimization problem.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (problem, lmo, initial_point)
    """
    problem = ShiftedPointProblem(
        dim=config.dim,
        L0=config.L0,
        L1=config.L1,
        shift_scale=config.shift_scale,
        seed=config.seed,
    )

    lmo = SimplexLMO(radius=1.0)
    x0 = np.ones(config.dim) / float(config.dim)

    return problem, lmo, x0


def run_proposed_method(
    problem: ShiftedPointProblem,
    lmo: SimplexLMO,
    x0: np.ndarray,
    config: ExperimentConfig,
    verbose: bool = False,
) -> ExperimentResult:
    """
    Run the proposed Frank-Wolfe method with adaptive step.

    Args:
        problem: Optimization problem
        lmo: Linear minimization oracle
        x0: Initial point
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        Experiment results
    """
    # Create oracle and wrapped objective
    oracle = MixedInexactOracle(
        obj=problem,
        alpha=config.alpha,
        delta=config.delta,
        rng=np.random.default_rng(config.seed),
    )

    wrapped_obj = MixedInexactObjectiveWrapper(problem, oracle)

    # Create stopping rule
    stopping_rule = InexactDualGapStoppingRule(
        tol=config.target_precision, alpha=config.alpha, delta=config.delta
    )

    # Run algorithm
    algorithm = ProposedMixedInexactFW(
        obj=wrapped_obj,
        lmo=lmo,
        L0=config.L0,
        L1=config.L1,
        stopping_rule=stopping_rule,
        max_iter=config.max_iter,
    )

    result = algorithm.run(x0)
    metrics = algorithm.metrics

    # Compute objective values
    obj_values = [problem(x) for x in algorithm.history]

    if verbose:
        log(
            f"Proposed method stopped at iteration {metrics['iterations']}",
            verbose=verbose,
        )
        log(f"Final objective value: {obj_values[-1]:.6f}", verbose=verbose)

    return ExperimentResult(
        algorithm_name="Предложенный (короткий шаг h_k)",
        history=algorithm.history,
        gap_history=metrics["gap_history"],
        iterations=metrics["iterations"],
        final_x=result.x_opt,
        objective_values=obj_values,
    )


def run_classical_method(
    problem: ShiftedPointProblem,
    lmo: SimplexLMO,
    x0: np.ndarray,
    config: ExperimentConfig,
    verbose: bool = False,
) -> ExperimentResult:
    """
    Run classical Frank-Wolfe with 2/(k+2) step.

    Args:
        problem: Optimization problem
        lmo: Linear minimization oracle
        x0: Initial point
        config: Experiment configuration
        verbose: Whether to print progress

    Returns:
        Experiment results
    """
    # Create oracle and wrapped objective
    oracle = MixedInexactOracle(
        obj=problem,
        alpha=config.alpha,
        delta=config.delta,
        rng=np.random.default_rng(
            config.seed + 1
        ),  # Different seed for fair comparison
    )

    wrapped_obj = MixedInexactObjectiveWrapper(problem, oracle)

    # Run algorithm
    algorithm = ClassicalMixedInexactFW(
        obj=wrapped_obj,
        lmo=lmo,
        alpha=config.alpha,
        delta=config.delta,
        max_iter=config.max_iter,
    )

    result = algorithm.run(x0)
    metrics = algorithm.metrics

    # Compute objective values
    obj_values = [problem(x) for x in algorithm.history]

    if verbose:
        log(
            f"Classical method ran for {metrics['iterations']} iterations",
            verbose=verbose,
        )
        log(f"Final objective value: {obj_values[-1]:.6f}", verbose=verbose)

    return ExperimentResult(
        algorithm_name="Классический FW (шаг 2/(k+2))",
        history=algorithm.history,
        gap_history=metrics["gap_history"],
        iterations=metrics["iterations"],
        final_x=result.x_opt,
        objective_values=obj_values,
    )


def plot_convergence(
    proposed_result: ExperimentResult,
    classical_result: ExperimentResult,
    config: ExperimentConfig,
    ax: plt.Axes,
    verbose: bool = False,
):
    """
    Plot convergence of objective function values.

    Args:
        proposed_result: Results from proposed method
        classical_result: Results from classical method
        config: Experiment configuration
        ax: Matplotlib axes for plotting
        verbose: Whether to print progress
    """
    # Compute f_star as minimum across both methods
    f_star = (
        min(
            min(proposed_result.objective_values),
            min(classical_result.objective_values),
        )
        - 1e-5
    )

    # Prepare data for proposed method
    iters_p = list(range(len(proposed_result.objective_values)))
    residuals_p = [max(1e-7, f - f_star) for f in proposed_result.objective_values]

    # Prepare data for classical method
    iters_c = list(range(len(classical_result.objective_values)))
    residuals_c = [max(1e-7, f - f_star) for f in classical_result.objective_values]

    # Plot convergence curves
    ax.plot(
        iters_p,
        residuals_p,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        label=proposed_result.algorithm_name,
        markersize=4,
    )
    ax.plot(
        iters_c,
        residuals_c,
        "s--",
        color="#d62728",
        linewidth=1.8,
        alpha=0.85,
        label=classical_result.algorithm_name,
        markersize=3,
    )

    # Mark stopping point for proposed method
    stop_k = proposed_result.iterations
    ax.axvline(stop_k, color="#2ca02c", linestyle=":", linewidth=2.0)
    ax.plot(
        stop_k,
        residuals_p[stop_k] if stop_k < len(residuals_p) else residuals_p[-1],
        "*",
        color="#2ca02c",
        markersize=14,
        label=f"Остановка (k={stop_k})",
    )

    # Configure axes
    ax.set_yscale("log")
    ax.set_xlabel("Итерация $k$", fontsize=11)
    ax.set_ylabel(r"$f(x^k) - f^*$", fontsize=11)
    ax.set_title("(а) Сходимость невязки целевой функции", fontsize=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    if verbose:
        log(f"Plotted convergence curves", verbose=verbose)


def plot_dual_gap(
    proposed_result: ExperimentResult,
    classical_result: ExperimentResult,
    config: ExperimentConfig,
    ax: plt.Axes,
    verbose: bool = False,
):
    """
    Plot inexact dual gap trajectory.

    Args:
        proposed_result: Results from proposed method
        classical_result: Results from classical method
        config: Experiment configuration
        ax: Matplotlib axes for plotting
        verbose: Whether to print progress
    """
    # Extract gap values
    iters_p = list(range(len(proposed_result.gap_history)))
    gap_p = [g["tilde_G"] for g in proposed_result.gap_history]

    iters_c = list(range(len(classical_result.gap_history)))
    gap_c = [g["tilde_G"] for g in classical_result.gap_history]

    # Plot gap trajectories
    ax.plot(
        iters_p,
        gap_p,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        label=r"$\tilde{G}$ ($(L_0, L_1)$-обобщенная гладкость)",
        markersize=4,
    )
    ax.plot(
        iters_c,
        gap_c,
        "s--",
        color="#d62728",
        linewidth=1.8,
        alpha=0.85,
        label=r"$\tilde{G}$ ($h_k=\frac{2}{k+2}$)",
        markersize=3,
    )

    # Add horizontal lines for reference
    ax.axhline(
        config.target_precision,
        color="#2ca02c",
        linestyle="-",
        linewidth=2.0,
        label=rf"Точность $\Delta = {config.target_precision}$",
    )
    ax.axhline(
        0.0,
        color="gray",
        linestyle="-.",
        linewidth=1.2,
        label=r"Порог шума ($\tilde{G} \leq 0$)",
    )

    # Configure axes
    ax.set_xlabel("Итерация $k$", fontsize=11)
    ax.set_ylabel(r"$\tilde{G}(x^k)$", fontsize=11)
    ax.set_title("(б) Динамика неточного зазора двойственности", fontsize=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    if verbose:
        log(f"Plotted dual gap trajectories", verbose=verbose)


def run_experiments(verbose: bool = False, interactive: bool = False):
    """
    Run experiments and generate plots.

    Args:
        verbose: Whether to print detailed progress
        interactive: Whether to show plots interactively
    """
    log("=" * 80, verbose=verbose)
    log("ЭКСПЕРИМЕНТЫ: Метод Франк-Вульфа со смешанной неточностью", verbose=verbose)
    log("и (L0, L1)-обобщенной гладкостью", verbose=verbose)
    log("=" * 80, verbose=verbose)

    # Configure matplotlib
    if not interactive:
        preamble()
    else:
        try:
            matplotlib.use("TkAgg")
        except:
            log("TkAgg backend not available", verbose=verbose)

    # Set up experiment configuration
    config = ExperimentConfig()

    log(f"\nПараметры эксперимента:", verbose=verbose)
    log(f"  Размерность: d = {config.dim}", verbose=verbose)
    log(f"  Константы гладкости: L0 = {config.L0}, L1 = {config.L1}", verbose=verbose)
    log(f"  Параметры шума: α = {config.alpha}, δ = {config.delta}", verbose=verbose)
    log(f"  Целевая точность: Δ = {config.target_precision}", verbose=verbose)
    log(f"  Максимум итераций: {config.max_iter}\n", verbose=verbose)

    # Set up problem
    problem, lmo, x0 = setup_problem(config)

    # Run proposed method
    log("Запуск предложенного метода...", verbose=verbose)
    proposed_result = run_proposed_method(problem, lmo, x0, config, verbose)

    # Run classical method
    log("Запуск классического метода...", verbose=verbose)
    classical_result = run_classical_method(problem, lmo, x0, config, verbose)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100)

    # Plot convergence
    plot_convergence(proposed_result, classical_result, config, ax1, verbose)

    # Plot dual gap
    plot_dual_gap(proposed_result, classical_result, config, ax2, verbose)

    # Set overall title
    plt.suptitle(
        r"Франк-Вульф со смешанной неточностью и $(L_0, L_1)$-гладкостью",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Create images directory in module location
    module_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(module_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save plots using the common function
    # Note: do_show_plot expects filename relative to 'images/' directory
    # We need to save in our module's images directory
    original_dir = os.getcwd()
    try:
        # Change to module directory so 'images/' path works correctly
        os.chdir(module_dir)

        # Save as PGF for LaTeX inclusion
        do_show_plot(
            filename="denisov1.pgf",
            show_plot=True,
            interactive=interactive,
            should_draw=False,
        )

        if verbose:
            log(
                f"График сохранен в {images_dir}/mixed_inexact_fw_experiments.pgf",
                verbose=verbose,
            )
    finally:
        # Restore original directory
        os.chdir(original_dir)

    if not interactive:
        plt.close()

    # Print summary statistics
    log("\n" + "=" * 80, verbose=verbose)
    log("ИТОГИ ЭКСПЕРИМЕНТА", verbose=verbose)
    log("=" * 80, verbose=verbose)
    log(f"Предложенный метод:", verbose=verbose)
    log(f"  - Остановка на итерации: {proposed_result.iterations}", verbose=verbose)
    log(
        f"  - Финальная невязка: {proposed_result.objective_values[-1]:.6e}",
        verbose=verbose,
    )
    log(f"Классический метод:", verbose=verbose)
    log(f"  - Выполнено итераций: {classical_result.iterations}", verbose=verbose)
    log(
        f"  - Финальная невязка: {classical_result.objective_values[-1]:.6e}",
        verbose=verbose,
    )
    log("=" * 80, verbose=verbose)
