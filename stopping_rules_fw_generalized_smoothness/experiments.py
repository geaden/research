# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from common.algorithmx import BaseAlgorithm
from common.experiment_utils import title
from common.oracles.lmo import LMO, SimplexLMO, LinfBallLMO
from common.math_utils import non_singular_matrix, ensure_non_zero
from common.objectives import Objective, MSE, LogisticRegression
from common.regularization import L1Regular, L2Regular
from common.plotting import LineStyle
from common.plot_utils import preamble, do_show_plot
from common.utils import log

from algorithms import FrankWolfe, FrankWolfeL0L1, AdaptiveFrankWolfeL0L1
from stopping_rules import (
    StoppingRuleStrategy,
    DualGapStoppingRuleStrategy,
    ConvergenceRateStoppingRuleStrategy,
)

rng = np.random.default_rng(74)


_TOLERANCE = 1e-20
_ITERATIONS_COUNT = 2_000
_DPI = 60


class PlotTitle:

    def __init__(self) -> None:
        self._current = "а"

    def next(self) -> str:
        current = self._current
        self._current = chr(ord(self._current) + 1)
        return current


_plot_title = PlotTitle()


@dataclass
class ExperimentData:
    label: str
    obj: Objective
    algorithm: Callable[[np.float64], BaseAlgorithm]
    x0: np.float64


def make_experiments(
    obj: Objective,
    lmo: LMO,
    L: float,
    L0: float,
    L1: float,
    x0: np.ndarray,
    stopping_rule: Callable[
        [float], StoppingRuleStrategy
    ] = DualGapStoppingRuleStrategy,
) -> list[ExperimentData]:
    return [
        ExperimentData(
            label=r"FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfe(
                obj=obj,
                lmo=lmo,
                L=L,
                stopping_rule=DualGapStoppingRuleStrategy(tol=delta),
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        ),
        ExperimentData(
            label=r"($L_0$, $L_1$)-FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfeL0L1(
                obj=obj,
                lmo=lmo,
                L0=L0,
                L1=L1,
                stopping_rule=stopping_rule(delta),
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        ),
        ExperimentData(
            label=r"Adapt ($L_0$, $L_1$)-FW",
            obj=obj,
            algorithm=lambda delta: AdaptiveFrankWolfeL0L1(
                obj=obj,
                lmo=lmo,
                L0=L0,
                L1=L1,
                stopping_rule=DualGapStoppingRuleStrategy(tol=delta),
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        ),
    ]


def setup_experiments(
    verbose: bool, _: bool, stopping_rule: Callable[[float], StoppingRuleStrategy]
) -> list[ExperimentData]:
    n: int = 250  # problem dimension

    A = non_singular_matrix(n, 0.75, 1.0, -1.0, 1.0, rng).astype(np.float64)

    x0 = np.zeros(n)
    y = rng.choice([-1, 1], size=(n,))
    obj = LogisticRegression(A, y)
    lmo = SimplexLMO()
    L = np.linalg.norm(A, axis=1).max() ** 2
    L0 = ensure_non_zero(0)
    L1 = np.linalg.norm(A, axis=1).max()
    log(f"{L=}, {L0=}, {L1=}", verbose=verbose)
    return make_experiments(obj, lmo, L, L0, L1, x0, stopping_rule)


def _run_convergence_rate_stopping_rule(
    experiments: list[ExperimentData],
    verbose: bool,
    ax: plt.Axes,
) -> None:
    log(title("Convergence rate stopping rule"), verbose=verbose)
    style = LineStyle()

    for data in experiments:
        algorithm = data.algorithm(_TOLERANCE)
        algorithm.run(data.x0)
        ax.plot(
            np.arange(len(algorithm.history)),
            list(map(data.obj, algorithm.history)),
            label=data.label,
            **style.next(),
        )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$f(x)$")
    ax.legend()
    ax.set_title(f"({_plot_title.next()})")
    ax.grid()

    style.reset()


def _run_delta_iterations(
    experiments: list[ExperimentData],
    verbose: bool,
    ax: plt.Axes,
) -> None:
    log(title("Delta iterations"), verbose=verbose)

    deltas = np.linspace(ensure_non_zero(0), 5e-3, 100)

    style = LineStyle()

    for data in experiments:

        def iterations_count(delta: float) -> int:
            algorithm = data.algorithm(delta)
            algorithm.run(data.x0)
            return len(algorithm.history) - 1

        iterations: list[int] = list(map(iterations_count, deltas))
        ax.plot(
            deltas,
            iterations,
            label=data.label,
            **style.next(),
        )

    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$k$")
    ax.legend()
    ax.set_title(f"({_plot_title.next()})")
    ax.grid()
    style.reset()


def _run_l1_regularization_logreg(verbose: bool, ax: plt.Axes) -> None:
    log(title("L1 regularization LogReg Delta - iterations"), verbose=verbose)

    n: int = 250  # problem dimension

    A = non_singular_matrix(n, 0.75, 1.0, -1.0, 1.0, rng).astype(np.float64)

    x0 = np.zeros(n)

    y = rng.choice([-1, 1], size=(n,))
    lam = 1e-1 * np.linalg.norm(np.dot(A, y)) / (2 * n)
    log(f"{lam=}", verbose=verbose)
    obj = L1Regular(LogisticRegression(A, y), lam=lam)
    lmo = SimplexLMO()
    L = np.linalg.norm(A, axis=1).max() ** 2
    L0 = ensure_non_zero(0)
    L1 = np.linalg.norm(A, axis=1).max()
    log(f"{L=}, {L0=}, {L1=}", verbose=verbose)

    experiments = make_experiments(obj, lmo, L, L0, L1, x0)

    deltas = np.linspace(ensure_non_zero(0), 5e-3, 100)

    style = LineStyle()

    for data in experiments:

        def iterations_count(delta: float) -> int:
            algorithm = data.algorithm(delta)
            algorithm.run(data.x0)
            return len(algorithm.history) - 1

        iterations: list[int] = list(map(iterations_count, deltas))
        ax.plot(
            deltas,
            iterations,
            label=data.label,
            **style.next(),
        )

    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$k$")
    ax.legend()
    plot_title = obj.__doc__
    log(plot_title, verbose=verbose)
    ax.set_title(f"({_plot_title.next()})")
    ax.grid()

    style.reset()


def _run_l1_regularization_mse_linf_ball(verbose: bool, ax: plt.Axes) -> None:
    log(title("L1 regularization MSE Delta - iterations"), verbose=verbose)

    n: int = 750  # problem dimension

    A = 0.05 * non_singular_matrix(n, 0.2, 1.0, -1.0, 1.0, rng).astype(np.float64)
    b = 0.25 * rng.random(n).astype(np.float64)

    x0 = -np.ones(n, dtype=np.float64)
    lam = ensure_non_zero(1e-6)
    log(f"{lam=}", verbose=verbose)
    obj = L1Regular(MSE(A, b), lam=lam)

    L = np.linalg.norm(A, axis=1).max() ** 2
    L0 = ensure_non_zero(0)
    L1 = np.linalg.norm(A, axis=1).max()
    log(f"{L=}, {L0=}, {L1=}", verbose=verbose)

    lmo = LinfBallLMO(radius=1.0, center=1.0)

    experiments = make_experiments(obj, lmo, L, L0, L1, x0)

    deltas = np.linspace(ensure_non_zero(0), 5e-3, 100)
    style = LineStyle()

    for data in experiments:

        def iterations_count(delta: float) -> int:
            algorithm = data.algorithm(delta)
            algorithm.run(data.x0)
            return len(algorithm.history) - 1

        iterations: list[int] = list(map(iterations_count, deltas))
        ax.plot(
            deltas,
            iterations,
            label=data.label,
            **style.next(),
        )

    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$k$")
    ax.legend()

    plot_title = obj.__doc__ + rf", {n=}, {lmo}"
    log(plot_title, verbose=verbose)
    ax.set_title(f"({_plot_title.next()})")
    ax.grid()
    style.reset()


def _run_delta_iterations_convergence_rate_stopping_rule_strategy(
    verbose: bool, ax: plt.Axes
) -> None:
    log(
        title(
            "L2 regularization MSE with convergence stopping rule Delta - iterations"
        ),
        verbose=verbose,
    )

    n: int = 750  # problem dimension

    A = 0.05 * non_singular_matrix(n, 0.2, 1.0, -1.0, 1.0, rng).astype(np.float64)
    b = 0.25 * rng.random(n).astype(np.float64)

    x0 = -np.ones(n, dtype=np.float64)
    lam = ensure_non_zero(1e-6)
    log(f"{lam=}", verbose=verbose)
    obj = L2Regular(MSE(A, b), lam=lam)

    L = np.linalg.norm(A, axis=1).max() ** 2
    L0 = ensure_non_zero(0)
    L1 = np.linalg.norm(A, axis=1).max()
    log(f"{L=}, {L0=}, {L1=}", verbose=verbose)

    lmo = LinfBallLMO(radius=1.0, center=1.0)

    experiments = make_experiments(
        obj,
        lmo,
        L,
        L0,
        L1,
        x0,
        lambda delta: ConvergenceRateStoppingRuleStrategy(
            x0=x0, obj=obj, strong_convexity_const=0.5, tol=delta
        ),
    )

    deltas = np.linspace(ensure_non_zero(0), 1, 100)
    style = LineStyle()

    for data in experiments:

        def iterations_count(delta: float) -> int:
            algorithm = data.algorithm(delta)
            algorithm.run(data.x0)
            return len(algorithm.history) - 1

        iterations: list[int] = list(map(iterations_count, deltas))
        ax.plot(
            deltas,
            iterations,
            label=data.label,
            **style.next(),
        )

    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$k$")
    ax.legend()

    plot_title = obj.__doc__ + rf", {n=}, {lmo}"
    log(plot_title, verbose=verbose)
    ax.set_title(f"({_plot_title.next()})")
    ax.grid()
    style.reset()


def run_experiments(verbose: bool, interactive: bool):
    log("Running experiments...", verbose=verbose)

    experiments = setup_experiments(
        verbose, interactive, lambda delta: DualGapStoppingRuleStrategy(tol=delta)
    )

    if not interactive:
        preamble()
    else:
        try:
            matplotlib.use("TkAgg")
        except:
            log("TkAgg failed", verbose=verbose)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=_DPI)

    _run_convergence_rate_stopping_rule(experiments, verbose, axs[0, 0])
    _run_delta_iterations(experiments, verbose, axs[0, 1])
    _run_l1_regularization_logreg(verbose, axs[0, 2])
    _run_l1_regularization_mse_linf_ball(verbose, axs[1, 0])
    _run_delta_iterations_convergence_rate_stopping_rule_strategy(verbose, axs[1, 1])
    fig.delaxes(axs[1, 2])

    plt.tight_layout()
    do_show_plot(
        filename="denisov_combined.pgf", show_plot=True, interactive=interactive
    )
    plt.close(fig)

    log("Done.", verbose=verbose)
