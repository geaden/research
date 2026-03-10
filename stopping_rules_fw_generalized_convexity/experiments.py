from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from common.algorithms import BaseAlgorithm
from common.experiment_utils import title
from common.latex_utils import latex_table
from common.lmo import LMO, SimplexLMO
from common.math_utils import non_singular_matrix, significant_figures
from common.objectives import Objective, MSE, LogisticRegression
from common.plot_utils import preamble, do_show_plot
from common.utils import log

from algorithms import FrankWolfe, FrankWolfeL0L1

rng = np.random.default_rng(74)


_TOLERANCE = 1e-6
_ITERATIONS_COUNT = 5_000


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

    deltas = np.linspace(1e-12, 0.0005, 100)

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


def run_experiments(verbose: bool, interactive: bool):
    log("Running experiments...", verbose=verbose)

    experiments = setup_experiments(verbose, interactive)

    if not interactive:
        preamble()

    _run_convegence_rate_stopping_rule(experiments, verbose, interactive)
    _run_delta_iterations(experiments, verbose, interactive)

    log("Done.", verbose=verbose)
