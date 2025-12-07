"""Module contains experiment main entry points."""

from dataclasses import dataclass
import itertools
import numpy as np
import matplotlib.pyplot as plt

from algorithms import AbsoluteInexactFW, AbsoluteInexactGradient
from common.experiment_utils import title
from common.utils import log
from common.latex_utils import latex_table
from common.lmo import SimplexLMO, MinLinearDirectionL2BallLMO, ShiftedBallLMO
from common.math_utils import non_singular_matrix, significant_figures
from common.objectives import Objective, MSE, LogisticRegression
from common.plot_utils import preamble, do_show_plot
from estimate import Estimate, InductionConvexEstimate, GapSumEstimate

rng = np.random.default_rng(73)

# Absolute noise
_DELTA = 0.1

_LMOS = [
    SimplexLMO(),
    MinLinearDirectionL2BallLMO(radius=1.0),
    MinLinearDirectionL2BallLMO(radius=10.0),
    ShiftedBallLMO(center=1.0, radius=5.0),
]


@dataclass
class ExperimentData:
    """Representation of an experiment."""

    title: str
    filename_prefix: str
    obj: Objective
    x0: np.ndarray
    L: np.float64
    D: np.float64
    N: int
    delta: float


def _create_deltas(num: int, start: float, end: float) -> list[float]:
    deltas = np.linspace(start, end, endpoint=False, num=num)
    deltas = [significant_figures(delta, 3) for delta in deltas]
    return deltas


def setup_experiments(verbose: bool, _: bool) -> list[ExperimentData]:
    n: int = 200  # problem dimension
    N: int = 350  # maximum number of iterations if stopping rule is unreached

    A = non_singular_matrix(n, 0.75, 1.0, -1.0, 1.0, rng).astype(np.float64)
    b = rng.random(n).astype(np.float64)
    L0 = np.linalg.eigvals(A).max().astype(np.float64)
    log(f"{L0=}", verbose)

    x0 = rng.random(n).astype(np.float64)

    experiments: list[ExperimentData] = []

    # 1. MSE
    obj = MSE(A, b)
    experiments.append(
        ExperimentData(
            title="MSE",
            filename_prefix="mse",
            obj=obj,
            x0=x0,
            D=1.0,
            L=L0,
            N=500,
            delta=_DELTA,
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
            x0=x0,
            D=1.0,
            L=L0,
            N=N,
            delta=_DELTA,
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

    # By induction
    estimates: list[Estimate] = [
        InductionConvexEstimate(
            L=data.L,
            D=data.D,
            delta=data.delta,
        ),
    ]

    # By sum of gaps
    alg = AbsoluteInexactFW(
        obj=data.obj,
        inexact_grad=AbsoluteInexactGradient(data.obj, rng, data.delta),
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
    if not interactive:
        preamble()

    count = 1

    for estimate in estimates:
        show_ylabel = count == 1
        clear_figure = count == 1

        if clear_figure:
            plt.figure(figsize=(12, 6), dpi=100)

        xvals = range(1, data.N + 1)
        yvals = [estimate.run(x) for x in xvals]
        plt.subplot(1, len(estimates), count)
        # Theoretical
        plt.plot(xvals, yvals, color="r", linestyle="--")

        plt.xlabel(r"Количество итераций, $N$")
        if show_ylabel:
            plt.ylabel(r"$f(x_N)-f^*$")
        plt.title(estimate.__doc__, pad=20)
        plt.suptitle(rf"$\Delta={data.delta}$")

        do_show_plot(
            filename=f"convergence_iterations.pgf",
            show_plot=count == len(estimates),
            interactive=interactive,
        )

        count += 1


def _run_experiment__convergence_based_on_delta(
    experiments: list[ExperimentData], verbose: bool, interactive: bool
):
    log(title("Convergence based on delta"), verbose=verbose)

    deltas = _create_deltas(num=12, start=0.1, end=0.45)
    graph_deltas = _create_deltas(num=3, start=0.1, end=1.0)

    for data in experiments:
        log(data.obj.__doc__, verbose=verbose)

        # Table
        table_values = []
        for delta in deltas:
            alg = AbsoluteInexactFW(
                obj=data.obj,
                inexact_grad=AbsoluteInexactGradient(data.obj, rng, delta),
                lmo=SimplexLMO(),
                L=data.L,
                N=data.N,
                delta=delta,
            )
            alg.run(data.x0)

            N = len(alg.history) - 1
            log(f"{delta=}, {N=}", verbose=verbose)

            table_values.append((delta, N))

        # Graph
        count = 1
        for delta, marker in zip(graph_deltas, itertools.cycle("^s*o")):
            clear_figure = count == 1
            show_ylabel = count == 1

            if clear_figure:
                plt.figure(figsize=(12, 6), dpi=100)

            alg = AbsoluteInexactFW(
                obj=data.obj,
                inexact_grad=AbsoluteInexactGradient(data.obj, rng, delta),
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

            plt.plot(xvals, yvals, marker=marker, markevery=10)

            plt.legend(graph_deltas)
            plt.suptitle(data.obj.__doc__)

            do_show_plot(
                filename=f"iterations_based_on_delta_{data.obj.__class__.__name__.lower()}.pgf",
                show_plot=count == len(graph_deltas),
                interactive=interactive,
            )

            count += 1

        table_values.sort(key=lambda x: x[1])

        # Print table of obtained values.
        print(
            latex_table(
                caption=r"Полученное количество итераций в зависимости от значений $\Delta$",
                values=table_values,
                headers=[r"$\Delta$", "$N$"],
            )
        )


def _run_experiments__different_lmo(
    experiments: list[ExperimentData], verbose: bool, interactive: bool
) -> None:
    for data in experiments:
        log(data.obj.__doc__, verbose=verbose)

        # Graph
        count = 1
        for lmo, marker in zip(_LMOS, itertools.cycle("^s*o")):
            clear_figure = count == 1
            show_ylabel = count == 1

            if clear_figure:
                plt.figure(figsize=(12, 6), dpi=100)

            alg = AbsoluteInexactFW(
                obj=data.obj,
                inexact_grad=AbsoluteInexactGradient(data.obj, rng, data.delta),
                lmo=lmo,
                L=data.L,
                N=data.N,
                delta=data.delta,
            )
            alg.run(data.x0)

            xvals = range(len(alg.history))
            yvals = [data.obj(x) for x in alg.history]

            plt.xlabel(r"Количество итераций, $k$")
            if show_ylabel:
                plt.ylabel(r"$f(x^k)$")

            plt.plot(xvals, yvals, marker=marker, markevery=10)

            plt.legend([str(lmo) for lmo in _LMOS])
            plt.suptitle(data.obj.__doc__)

            do_show_plot(
                filename=f"lmos_iterations_{data.obj.__class__.__name__.lower()}.pgf",
                show_plot=count == len(_LMOS),
                interactive=interactive,
            )

            count += 1


def _run_experiments__norm_gradient(
    experiments: list[ExperimentData], verbose: bool, interactive: bool
) -> None:
    for data in experiments:
        log(data.obj.__doc__, verbose=verbose)

        # Graph
        count = 1
        for lmo, marker in zip(_LMOS, itertools.cycle("^s*o")):
            clear_figure = count == 1
            show_ylabel = count == 1

            if clear_figure:
                plt.figure(figsize=(12, 6), dpi=100)

            alg = AbsoluteInexactFW(
                obj=data.obj,
                inexact_grad=AbsoluteInexactGradient(data.obj, rng, data.delta),
                lmo=lmo,
                L=data.L,
                N=data.N,
                delta=data.delta,
            )
            alg.run(data.x0)

            xvals = range(len(alg.history))
            yvals = [np.linalg.norm(data.obj.grad(x)) for x in alg.history]

            plt.xlabel(r"Количество итераций, $k$")
            if show_ylabel:
                plt.ylabel(r"$\left\Vert f(x^k) \right\Vert$")

            plt.plot(xvals, yvals, marker=marker, markevery=10)

            plt.legend([str(lmo) for lmo in _LMOS])
            plt.suptitle(data.obj.__doc__)

            do_show_plot(
                filename=f"norm_gradient_{data.obj.__class__.__name__.lower()}.pgf",
                show_plot=count == len(_LMOS),
                interactive=interactive,
            )

            count += 1


def run_experiments(verbose: bool, interactive: bool):
    print("Running experiments...")
    experiments = setup_experiments(verbose, interactive)
    _run_experiment__theoretical_convergence_iteration_numbers(
        experiments, verbose, interactive
    )
    _run_experiment__convergence_based_on_delta(experiments, verbose, interactive)
    _run_experiments__different_lmo(experiments, verbose, interactive)
    _run_experiments__norm_gradient(experiments, verbose, interactive)
    print("Done.")
