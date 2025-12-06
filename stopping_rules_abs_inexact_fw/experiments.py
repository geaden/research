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


def setup_experiments(verbose: bool, interactive: bool) -> list[ExperimentData]:
    rng = np.random.default_rng(2025)
    n = 200
    N = 250
    delta = 1e-4
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

    estimates: list[Estimate] = [
        InductionConvexEstimate(
            L=data.L,
            D=data.D,
            delta=data.delta,
        ),
    ]

    # GapSum
    alg = AbsoluteInexactFW(
        obj=data.obj,
        inexact_grad=data.gradient,
        lmo=SimplexLMO(),
        L=data.L,
        N=data.N,
        delta=data.delta,
    )
    result = alg.run(data.x0)
    estimates.append(
        GapSumEstimate(
            obj=data.obj,
            L=data.L,
            D=data.D,
            delta=data.delta,
            x0=result.x0,
            x_opt=result.x_opt,
        )
    )

    # Plot
    count = 1

    if not interactive:
        preamble()

    for estimate in estimates:
        clear_figure = count == 1
        show_ylabel = count == 1

        if clear_figure:
            plt.figure(plt.figure(figsize=(12, 6), dpi=100))

        xvals = range(1, data.N + 1)
        yvals = [estimate.run(N) for N in xvals]
        plt.subplot(1, len(estimates), count)
        # Theoretical
        plt.plot(xvals, yvals, color="r", linestyle="--")
        # TODO(geaden): Bring back experimental values
        # Experimental
        # plt.plot(
        #     range(len(alg.history)),
        #     [data.obj(x) - data.obj(result.x_opt) for x in alg.history],
        #     color="b",
        #     linestyle="-",
        # )

        plt.xlabel(r"Количество итераций, $N$")
        if show_ylabel:
            plt.ylabel(r"$f(x_N)-f^*$")

        plt.title(estimate.__doc__, pad=20)

        do_show_plot(
            filename="theoretical_convergence.pgf",
            show_plot=count == len(estimates),
            interactive=interactive,
        )

        count += 1


def _run_experiment__convergence_based_on_delta(
    experiments: list[ExperimentData], verbose: bool, interactive: bool
):
    log(title("Convergence based on delta"), verbose=verbose)

    lines = [("-", "b"), ("--", "orange"), ("-.", "g"), (":", "r")]

    data = experiments[-1]

    deltas = [0.0001, 0.001, 0.01, 0.1]
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
            N=250,
            delta=delta,
        )
        alg.run(data.x0)

        xvals = range(len(alg.history))
        yvals = [data.obj(x) for x in alg.history]

        plt.xlabel(r"Количество итераций, $k$")
        if show_ylabel:
            plt.ylabel(r"f(x^k)")

        linestyle, color = lines.pop()
        plt.plot(xvals, yvals, color=color, linestyle=linestyle)

        plt.legend(deltas)

        do_show_plot(
            filename="iterations_based_on_delta.pgf",
            show_plot=count == len(deltas),
            interactive=interactive,
        )

        log(f"{delta=}, N={len(alg.history)}", verbose=verbose)
        values.append((delta, len(alg.history)))

        count += 1

    # Print table
    print(r"\begin{table}[!h]")
    print(r" \caption{Количество итераций для различных $\Delta$}")
    print(r"  \begin{center}")
    print(r"    \begin{tabular}{||c c||}")
    print(r"        \hline")
    print(r"        $\Delta$ & N \\ [0.5ex]")
    print(r"        \hline\hline")
    for delta, N in values:
        print(rf"        {delta} & {N} \\")
        print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"  \end{center}")
    print(r"\end{table}")


def run_experiments(verbose: bool, interactive: bool):
    print("Running experiments...")
    experiments = setup_experiments(verbose, interactive)
    _run_experiment__theoretical_convergence_iteration_numbers(
        experiments, verbose, interactive
    )
    _run_experiment__convergence_based_on_delta(experiments, verbose, interactive)
    print("Done.")
