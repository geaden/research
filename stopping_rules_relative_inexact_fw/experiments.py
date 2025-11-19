"""Module contains numerical experiments."""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Type
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from step import ShortStepSize, DecayingStepSize
from utils import log, non_singular_matrix, significant_figures, ensure_non_zero
from objectives import Objective, MSE, LogisticRegression
from lmo import random_euclidean_ball, MinLinearDirectionL2Ball
from algorithms import BaseInexactFW, InexactFW, AdaptiveInexactFW


@dataclass
class ExperimentData:
    """Representation of an experiment."""

    title: str
    filename_prefix: str
    radius_grid: list[float]
    obj: Objective
    alpha: float
    R: int
    L: np.float64
    N: int
    delta: float


def setup(verbose: Optional[bool] = False) -> list[ExperimentData]:
    """
    Setup experiment environment.
    """
    np.random.seed(2025)
    R = 10
    N = 150
    delta = 1e-4
    A = non_singular_matrix(R, 0.75, 1.0, -1.0, 1.0)
    b = np.random.randn(R)
    L0 = np.linalg.eigvals(A).max().astype(np.longdouble)
    log(f"{L0=}", verbose)

    experiments = []

    # 1. MSE
    experiments.append(
        ExperimentData(
            title=r"MSE (на шаре)",
            filename_prefix="mse_on_ball",
            radius_grid=np.linspace(0.2, 0.7, num=N),
            obj=MSE(A, b),
            alpha=0.02,
            R=R,
            L=L0,
            N=N,
            delta=delta,
        )
    )

    # 2. Logistic regression
    y = np.random.choice([-1, 1], size=(R,))
    experiments.append(
        ExperimentData(
            title="Функция ошибки\nлогистической регрессии (на шаре)",
            filename_prefix="logreg_on_ball",
            radius_grid=np.linspace(1.0, 5.0, num=N),
            obj=LogisticRegression(A, y),
            alpha=0.03,
            R=R,
            L=L0,
            N=N,
            delta=delta,
        )
    )

    return experiments


def run_experiments(verbose: bool, interactive: bool):
    experiments_data = setup(verbose=verbose)

    for t in [InexactFW, AdaptiveInexactFW]:
        _run_experiment_iterations_per_ball_size(
            t, experiments_data, interactive, verbose
        )

    _run_adaptive_convergence_based_on_alpha(experiments_data, interactive, verbose)
    _run_comparison_convergence_non_adaptive_and_adaptive(experiments_data, interactive)


def _run_experiment_iterations_per_ball_size(
    algorithm_cls: Type[BaseInexactFW],
    experiments_data: list[ExperimentData],
    interactive: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> None:
    """
    Comparison of non-adaptive relative inexact FW practical and theoretical
    number of iterations.
    """
    count = 1
    for data in experiments_data:
        X, Y, theory_Y = [], [], []
        for ball_radius in data.radius_grid:

            x0 = random_euclidean_ball(data.R, ball_radius)

            algorithm = algorithm_cls(
                obj=data.obj,
                lmo=MinLinearDirectionL2Ball(ball_radius),
                step_size=ShortStepSize(data.alpha),
                N=data.N,
                L=data.L,
                alpha=data.alpha,
                delta=data.delta,
            )

            result = algorithm.run(x0)
            log(f"x^*={result.x_opt}", verbose=verbose)
            history = algorithm.history

            X.append(ball_radius)
            Y.append(len(history))

            N_t = algorithm.theoretical_iterations(result)
            theory_Y.append(N_t)

        _do_plot_iterations_per_ball_size(
            data.title,
            data.filename_prefix + f"_iterations_{algorithm_cls.__name__.lower()}.pgf",
            ball_size=X,
            iterations=Y,
            theoretical=theory_Y,
            interactive=interactive,
            count=count,
        )

        count += 1


def _run_adaptive_convergence_based_on_alpha(
    experiments_data: list[ExperimentData],
    interactive: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    """
    Convergence based on alpha.
    """
    expected_iterations_stack = [35, 60]
    count = 1
    for data in experiments_data:
        hist = []
        radius = np.max(data.radius_grid)
        x0 = random_euclidean_ball(data.R, radius=radius)

        result = AdaptiveInexactFW(
            data.obj,
            lmo=MinLinearDirectionL2Ball(radius=radius),
            step_size=ShortStepSize(data.alpha),
            N=data.N,
            L=data.L,
            alpha=data.alpha,
            delta=data.delta,
        ).run(x0)

        # Select optimal alpha
        eps = 8 * np.sqrt(data.delta)
        x = (eps - np.sqrt(data.delta)) / (2 * result.M * result.D)
        alpha = ensure_non_zero((-(1 + x) + np.sqrt((1 + x) ** 2 + 4 * x)) / 2)
        log(f"{count}: optimal alpha={alpha}", verbose=verbose)

        alphas = np.linspace(ensure_non_zero(0), 2 * alpha + eps, num=7)[1:]
        alphas = list(set([significant_figures(alpha) for alpha in alphas]))
        alphas.sort()

        for alpha in alphas:
            algorithm = AdaptiveInexactFW(
                data.obj,
                lmo=MinLinearDirectionL2Ball(radius=radius),
                step_size=ShortStepSize(data.alpha),
                N=data.N,
                L=data.L,
                alpha=data.alpha,
                delta=data.delta,
            )
            algorithm.run(x0)

            hist.append([data.obj(x) for x in algorithm.history])

        _do_plot_convergence(
            data.title.replace("на шаре", r"на шаре $r = {}$".format(radius)),
            data.filename_prefix + "_convergence.pgf",
            hist,
            alpha=alphas,
            interactive=interactive,
            limit=expected_iterations_stack.pop(),
            count=count,
        )

        count += 1


def _run_comparison_convergence_non_adaptive_and_adaptive(
    experiments_data: list[ExperimentData], interactive: Optional[bool] = False
) -> None:
    """
    Comparison of non-adaptive and adaptive convergence.
    """
    expected_iterations_stack = [60, 60]
    data = dataclasses.replace(experiments_data[1])
    data.filename_prefix = "nonadaptive_adaptive_comparison_convergence.pgf"

    algorithms = [
        (
            r"Неадаптивный вариант с шагом $\frac{k}{k+2}$ на шаре",
            InexactFW(
                data.obj,
                lmo=MinLinearDirectionL2Ball(radius=8.0),
                step_size=DecayingStepSize(),
                N=data.N,
                L=data.L,
                alpha=data.alpha,
                delta=data.delta,
            ),
        ),
        (
            r"Адаптивный вариант на шаре",
            AdaptiveInexactFW(
                data.obj,
                lmo=MinLinearDirectionL2Ball(radius=2.0),
                step_size=ShortStepSize(data.alpha),
                N=data.N,
                L=data.L,
                alpha=data.alpha,
                delta=data.delta,
            ),
        ),
    ]

    count = 1
    for title, algorithm in algorithms:
        radius = algorithm._lmo._r
        x0 = random_euclidean_ball(data.R, radius=radius)

        algorithm.run(x0)

        _do_plot_convergence(
            title.replace("на шаре", r"на шаре $r = {}$".format(radius)),
            data.filename_prefix,
            [[data.obj(x) for x in algorithm.history]],
            alpha=[],
            interactive=interactive,
            limit=expected_iterations_stack.pop(),
            count=count,
        )

        count += 1


def _preamble():
    matplotlib.use("pgf")
    params = {
        "text.latex.preamble": (
            r"\usepackage[T1, T2A]{fontenc}"
            r"\usepackage[utf8]{inputenc}"
            r"\usepackage[main=russian,english]{babel}"
        ),
        "text.usetex": True,
        "font.size": 11,
        "font.family": "lmodern",
    }
    plt.rcParams.update(params)


def _do_plot_iterations_per_ball_size(
    label: str,
    filename: str,
    ball_size: list[int],
    iterations: list[int],
    theoretical: list[int],
    interactive: bool = False,
    count: int = 1,
) -> None:
    clear_figure = count == 1
    show_ylabel = count == 1
    show_plot = count == 2

    if not interactive:
        _preamble()

    if clear_figure:
        plt.figure(plt.figure(figsize=(12, 6)))

    lines = [("--", "r"), ("solid", "b")]
    plt.subplot(1, 2, count)

    for iters in [iterations, theoretical]:
        style, color = lines.pop()
        plt.plot(ball_size, iters, color=color, linestyle=style)

    plt.title(label)
    plt.yscale("log")

    plt.xlabel(r"Радиус шара, $r$")
    if show_ylabel:
        plt.ylabel(r"$\log$ числа итераций")

    plt.legend(["Фактическое число итераций", "Теоретическая оценка"])

    if show_plot:
        plt.show()

        if not interactive:
            plt.savefig("images/" + filename)


def _do_plot_convergence(
    label: str,
    filename: str,
    fx: list[np.ndarray],
    alpha: list[float],
    interactive: bool = False,
    limit: int = 100,
    count: int = 1,
) -> None:
    clear_figure = count == 1
    show_ylabel = count == 1

    if not interactive:
        _preamble()

    if clear_figure:
        plt.figure(plt.figure(figsize=(12, 6)))

    plt.subplot(1, 2, count)
    for f in fx:
        plt.plot(
            range(0, len(f[:limit])),
            f[:limit],
        )
        plt.title(label)
        plt.xlabel(r"Номер итерации, $k$")
        if show_ylabel:
            plt.ylabel(r"$f(x^k)$")
        plt.xticks(range(0, limit, 5 if limit < 50 else 10))

    if alpha:
        plt.legend([r"$\alpha={}$".format(a) for a in alpha])

    if count == 2:
        plt.show()
        if not interactive:
            plt.savefig("images/" + filename)
