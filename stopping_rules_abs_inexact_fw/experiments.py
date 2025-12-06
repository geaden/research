"""Module contains experiment main entry points."""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from common.objectives import Objective, MSE, LogisticRegression
from common.utils import non_singular_matrix, log, significant_figures
from estimate import Estimate, InductionConvexEstimate, GapSumEstimate
from algorithms import AbsoluteInexactFW, AbsoluteInexactGradient
from common.lmo import SimplexLMO
from common.plot_utils import preamble, do_show_plot, SOLID_LINE
from common.experiment_utils import title
from common.latex_utils import latex_table


@dataclass
class ExperimentData:
    """Representation of an experiment."""

    title: str
    filename_prefix: str
    obj: Objective
    gradient: AbsoluteInexactGradient
    x0: np.ndarray
    L: np.float64
    D: np.float64
    N: int
    delta: float


def setup_experiments(verbose: bool, _: bool) -> list[ExperimentData]:
    rng = np.random.default_rng(2025)
    n = 200
    N = 350
    delta = 0.1
    A = non_singular_matrix(n, 0.75, 1.0, -1.0, 1.0, rng).astype(np.float64)
    b = rng.random(n).astype(np.float64)
    L0 = np.linalg.eigvals(A).max().astype(np.float64)
    log(f"{L0=}", verbose)
    x0 = rng.random(n).astype(np.float64)

    experiments = []

    # 1. MSE
    obj = MSE(A, b)
    experiments.append(
        ExperimentData(
            title=r"MSE",
            filename_prefix="mse",
            obj=obj,
            gradient=AbsoluteInexactGradient(obj, rng, delta),
            x0=x0,
            D=1.0,
            L=L0,
            N=N,
            delta=delta,
        )
    )

    # 2. Logistic regression
    y = rng.choice([-1, 1], size=(n,))
    obj = LogisticRegression(A, y)
    experiments.append(
        ExperimentData(
            title="Функция ошибки\nлогистической регрессии",
            filename_prefix="logreg",
            obj=obj,
            gradient=AbsoluteInexactGradient(obj, rng, delta),
            x0=x0,
            D=1.0,
            L=L0,
            N=N,
            delta=delta,
        )
    )

    return experiments


def _run_experiment__theoretical_convergence_iteration_numbers(
    experiments: list[ExperimentData],
    verbose: bool,
    interactive: bool,
):
    log(title("Theoretical convergence iterations number"), verbose=verbose)

    data = experiments[-1]

    if not interactive:
        preamble()

    deltas = [0.01, 0.5, 0.8]

    for delta in deltas:
        estimates: list[Estimate] = [
            InductionConvexEstimate(
                L=data.L,
                D=data.D,
                delta=delta,
            ),
        ]

        # GapSum
        alg = AbsoluteInexactFW(
            obj=data.obj,
            inexact_grad=data.gradient,
            lmo=SimplexLMO(),
            L=data.L,
            N=data.N,
            delta=delta,
        )
        result = alg.run(data.x0)
        estimates.append(
            GapSumEstimate(
                obj=data.obj,
                L=data.L,
                D=data.D,
                delta=delta,
                x0=result.x0,
                x_opt=result.x_opt,
            )
        )

        # Plot
        count = 1

        plt.figure(plt.figure(figsize=(12, 6), dpi=100))

        for estimate in estimates:
            show_ylabel = count == 1

            xvals = range(1, data.N + 1)
            yvals = [estimate.run(x) for x in xvals]
            plt.subplot(1, len(estimates), count)
            # Theoretical
            plt.plot(xvals, yvals, color="r", linestyle="--")

            plt.xlabel(r"Количество итераций, $N$")
            if show_ylabel:
                plt.ylabel(r"$f(x_N)-f^*$")

            plt.title(rf"{estimate.__doc__} $\Delta={data.delta}$", pad=20)
            plt.legend(deltas)

            do_show_plot(
                filename="convergence_iterations.pgf",
                show_plot=count == len(estimates) * len(deltas),
                interactive=interactive,
            )

            count += 1


def _run_experiment__convergence_based_on_delta(
    experiments: list[ExperimentData], verbose: bool, interactive: bool
):
    log(title("Convergence based on delta"), verbose=verbose)

    data = experiments[-1]

    deltas = [
        0,
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    deltas = [significant_figures(delta, 3) for delta in deltas]

    count = 1

    values = []

    for delta in deltas:
        clear_figure = count == 1
        show_ylabel = count == 1

        if clear_figure:
            plt.figure(plt.figure(figsize=(12, 6), dpi=100))

        data.gradient._delta = delta
        alg = AbsoluteInexactFW(
            obj=data.obj,
            inexact_grad=data.gradient,
            lmo=SimplexLMO(),
            L=data.L,
            N=data.N,
            delta=delta,
        )
        alg.run(data.x0)

        xvals = range(len(alg.history))
        yvals = [data.obj(x) for x in alg.history]

        plt.xlabel(r"Количество итераций, $k$")
        if show_ylabel:
            plt.ylabel(r"$f(x^k)$")

        plt.plot(xvals, yvals)

        plt.legend(deltas)

        do_show_plot(
            filename="iterations_based_on_delta.pgf",
            show_plot=count == len(deltas),
            interactive=interactive,
        )

        log(f"{delta=}, N={len(alg.history)}", verbose=verbose)
        values.append((delta, len(alg.history) - 1))

        count += 1

    # Print table
    print(
        latex_table(
            caption=r"Полученное количество итераций в зависимости от значений $\Delta$",
            values=values,
            headers=[r"$\Delta$", "$N$"],
        )
    )


def run_experiments(verbose: bool, interactive: bool):
    print("Running experiments...")
    experiments = setup_experiments(verbose, interactive)
    _run_experiment__theoretical_convergence_iteration_numbers(
        experiments, verbose, interactive
    )
    _run_experiment__convergence_based_on_delta(experiments, verbose, interactive)
    print("Done.")
