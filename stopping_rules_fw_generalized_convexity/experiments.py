from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from common.algorithms import BaseAlgorithm
from common.experiment_utils import title
from common.latex_utils import latex_table
from common.lmo import LMO, MinLinearDirectionL2BallLMO, SimplexLMO, LinfBallLMO
from common.math_utils import non_singular_matrix, significant_figures, ensure_non_zero
from common.objectives import Objective, MSE, LogisticRegression
from common.regularization import L1Regular
from common.plot_utils import preamble, do_show_plot
from common.utils import log

from algorithms import FrankWolfe, FrankWolfeL0L1

rng = np.random.default_rng(74)


_TOLERANCE = 1e-6
_ITERATIONS_COUNT = 2_000


@dataclass
class ExperimentData:
    label: str
    obj: Objective
    algorithm: Callable[[np.float64], BaseAlgorithm]
    x0: np.float64


def setup_experiments(verbose: bool, _: bool) -> list[ExperimentData]:
    n: int = 250  # problem dimension

    A = non_singular_matrix(n, 0.75, 1.0, -1.0, 1.0, rng).astype(np.float64)

    x0 = np.zeros(n)

    experiments: list[ExperimentData] = []

    y = rng.choice([-1, 1], size=(n,))
    obj = LogisticRegression(A, y)
    lmo = SimplexLMO()
    L = np.linalg.norm(A, axis=1).max() ** 2
    log(f"{L=}", verbose=verbose)
    experiments.append(
        ExperimentData(
            label="Classic FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfe(
                obj=obj,
                lmo=lmo,
                L=L,
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        ),
    )
    L0 = ensure_non_zero(0)
    L1 = np.linalg.norm(A, axis=1).max()
    log(f"{L0=}, {L1=}", verbose=verbose)
    experiments.append(
        ExperimentData(
            label=r"$(L_0, L_1)$-FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfeL0L1(
                obj=obj,
                lmo=lmo,
                L0=L0,
                L1=L1,
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        )
    )

    return experiments


def _run_convegence_rate_stopping_rule(
    experiments: list[ExperimentData],
    verbose: bool,
    interactive: bool,
) -> None:
    log(title("Convergence rate stopping rule"), verbose=verbose)
    plt.figure(figsize=(12, 6), dpi=100)
    for data in experiments:
        algorithm = data.algorithm(_TOLERANCE)
        algorithm.run(data.x0)
        plt.plot(
            np.arange(len(algorithm.history)),
            list(map(data.obj, algorithm.history)),
            label=data.label,
        )
        plt.title(rf"{data.obj.__doc__}")

    plt.xlabel(r"$k$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.grid()

    do_show_plot(filename="denisov1.pgf", show_plot=True, interactive=interactive)


def _run_delta_iterations(
    experiments: list[ExperimentData],
    verbose: bool,
    interactive: bool,
) -> None:
    log(title("Delta iterations"), verbose=verbose)

    deltas = np.linspace(ensure_non_zero(0), 0.0005, 100)

    plt.figure(figsize=(12, 6), dpi=100)

    for data in experiments:

        def iterations_count(delta: float) -> int:
            algorithm = data.algorithm(delta)
            algorithm.run(data.x0)
            return len(algorithm.history) - 1

        iterations: list[int] = list(map(iterations_count, deltas))
        plt.plot(deltas, iterations, label=data.label)

    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$k$")
    plt.legend()
    plt.grid()

    do_show_plot(filename="denisov2.pgf", show_plot=True, interactive=interactive)


def _run_l1_regularization_mse_linf_ball(
    experiments: list[ExperimentData],
    verbose: bool,
    interactive: bool,
) -> None:
    log(title("L1 regularization MSE Delta - iterations"), verbose=verbose)

    n: int = 510  # problem dimension

    scale_factor = 0.03
    A = scale_factor * non_singular_matrix(n, 0.2, 1.0, -1.0, 1.0, rng).astype(
        np.float64
    )
    b = 0.25 * np.ones(n, dtype=np.float64)

    center = 1.0
    x0 = -np.ones(n, dtype=np.float64)
    lam = ensure_non_zero(1e-6)
    obj = L1Regular(MSE(A, b), lam=lam)

    experiments = []

    L = np.linalg.norm(A, axis=1).max() ** 2

    L0 = ensure_non_zero(0)
    L1 = np.linalg.norm(A, axis=1).max()

    log(f"{L=}, {L0=}, {L1=}", verbose=verbose)

    lmo = LinfBallLMO(radius=1.0, center=center)

    experiments.append(
        ExperimentData(
            label="Classic FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfe(
                obj=obj,
                lmo=lmo,
                L=L,
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        ),
    )
    experiments.append(
        ExperimentData(
            label=r"$(L_0, L_1)$-FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfeL0L1(
                obj=obj,
                lmo=lmo,
                L0=L0,
                L1=L1,
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        ),
    )

    deltas = np.linspace(ensure_non_zero(0), 2e-4, 100)

    plt.figure(figsize=(12, 6), dpi=100)

    for data in experiments:

        def iterations_count(delta: float) -> int:
            algorithm = data.algorithm(delta)
            algorithm.run(data.x0)
            return len(algorithm.history) - 1

        iterations: list[int] = list(map(iterations_count, deltas))
        plt.plot(deltas, iterations, label=data.label)

    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$k$")
    plt.legend()

    plot_title = obj.__doc__ + rf", {n=}, {lmo}"
    log(plot_title, verbose=verbose)
    plt.title(plot_title)
    plt.grid()

    do_show_plot(filename="denisov3.pgf", show_plot=True, interactive=interactive)


def _run_l1_regularization_logreg(
    experiments: list[ExperimentData],
    verbose: bool,
    interactive: bool,
) -> None:
    log(title("L1 regularization LogReg Delta - iterations"), verbose=verbose)

    n: int = 250  # problem dimension

    A = non_singular_matrix(n, 0.75, 1.0, -1.0, 1.0, rng).astype(np.float64)

    x0 = np.zeros(n)

    experiments: list[ExperimentData] = []

    y = rng.choice([-1, 1], size=(n,))
    lam = 1e-1 * np.linalg.norm(np.dot(A, y)) / (2 * n)
    log(f"{lam=}", verbose=verbose)
    obj = L1Regular(LogisticRegression(A, y), lam=lam)
    lmo = SimplexLMO()
    L = np.linalg.norm(A, axis=1).max() ** 2
    log(f"{L=}", verbose=verbose)

    # TODO(geaden): Remove duplication.
    experiments: list[ExperimentData] = []

    lmo = SimplexLMO()
    L = np.linalg.norm(A, axis=1).max() ** 2
    log(f"{L=}", verbose=verbose)
    experiments.append(
        ExperimentData(
            label="Classic FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfe(
                obj=obj,
                lmo=lmo,
                L=L,
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        ),
    )
    L0 = 1e-12
    L1 = np.linalg.norm(A, axis=1).max()
    log(f"{L0=}, {L1=}", verbose=verbose)
    experiments.append(
        ExperimentData(
            label=r"$(L_0, L_1)$-FW",
            obj=obj,
            algorithm=lambda delta: FrankWolfeL0L1(
                obj=obj,
                lmo=lmo,
                L0=L0,
                L1=L1,
                iterations_count=_ITERATIONS_COUNT,
                tol=delta,
            ),
            x0=x0,
        )
    )

    deltas = np.linspace(ensure_non_zero(0), 0.0005, 100)

    plt.figure(figsize=(12, 6), dpi=100)

    for data in experiments:

        def iterations_count(delta: float) -> int:
            algorithm = data.algorithm(delta)
            algorithm.run(data.x0)
            return len(algorithm.history) - 1

        iterations: list[int] = list(map(iterations_count, deltas))
        plt.plot(deltas, iterations, label=data.label)

    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$k$")
    plt.legend()
    plot_title = obj.__doc__
    log(plot_title, verbose=verbose)
    plt.title(plot_title)
    plt.grid()

    do_show_plot(filename="denisov4.pgf", show_plot=True, interactive=interactive)


def run_experiments(verbose: bool, interactive: bool):
    log("Running experiments...", verbose=verbose)

    experiments = setup_experiments(verbose, interactive)

    if not interactive:
        preamble()
    else:
        try:
            matplotlib.use("TkAgg")
        except:
            log("TkAgg failed", verbose=verbose)

    _run_convegence_rate_stopping_rule(experiments, verbose, interactive)
    _run_delta_iterations(experiments, verbose, interactive)
    _run_l1_regularization_logreg(experiments, verbose, interactive)
    _run_l1_regularization_mse_linf_ball(experiments, verbose, interactive)

    log("Done.", verbose=verbose)
