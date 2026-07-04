from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from common.objectives import Lasso, Objective
from common.algorithmx.fw import BaseFrankWolfe, FrankWolfe, DiminishingStepSizeStrategy
from common.experiment_utils import title
from common.objectives import LogisticRegression
from common.oracles.lmo import L2BallLMO
from common.plot_utils import do_show_plot, preamble
from common.plotting import LineStyle
from common.math_utils import ensure_non_zero, non_singular_matrix
from common.utils import log

from algorithms import (
    AdaptiveFrankWolfeRobustRelativeInexactness,
    AdaptiveFrankWolfeRobustComparison,
)


@dataclass
class ExperimentsData:
    obj: Objective
    algorithm: BaseFrankWolfe
    title: str
    x0: np.ndarray


_MAX_ITERATIONS = 30
_TOLERANCE = 1e-2
_DELTA = 1e-3


def _setup_experiments(verbose: bool) -> list[ExperimentsData]:
    """
    Setup experiment environment.
    """
    _SEED = 71

    np.random.seed(_SEED)

    # Dimension of problem
    dim = 20

    A = non_singular_matrix(dim, 0.75, 1.0, -1.0, 1.0, rng=np.random.default_rng(_SEED))
    y = np.random.choice([-1, 1], size=(dim,))

    obj = LogisticRegression(A, y)
    lmo = L2BallLMO(radius=1.0)

    L = np.linalg.norm(A, ord=2) ** 2 / (4 * dim)
    log(f"{L=}", verbose=verbose)

    x0 = np.zeros(dim)

    experiments: list[BaseFrankWolfe] = [
        ExperimentsData(
            obj=obj,
            algorithm=lambda alpha: AdaptiveFrankWolfeRobustRelativeInexactness(
                obj=obj,
                lmo=lmo,
                L=L,
                alpha=alpha,
                max_iter=_MAX_ITERATIONS,
                tol=_TOLERANCE,
            ),
            title="FW-Relative Inexactness)",
            x0=x0,
        ),
        ExperimentsData(
            obj=obj,
            algorithm=lambda alpha: AdaptiveFrankWolfeRobustComparison(
                obj=obj,
                lmo=lmo,
                L=L,
                alpha=alpha,
                max_iter=_MAX_ITERATIONS,
                tol=_TOLERANCE,
            ),
            title="FW-Comparison Oracle",
            x0=x0,
        ),
    ]

    return experiments


def _run_experiment_log_regression(verbose: bool, interactive: bool) -> None:
    experiments = _setup_experiments(verbose=verbose)

    log(title("Log iterations per alpha"), verbose=verbose)

    if not interactive:
        preamble()

    style = LineStyle()
    plt.figure(figsize=(12, 6), dpi=100)

    alphas = np.linspace(ensure_non_zero(0), 0.999, 100)

    for data in experiments:
        algorithms = [data.algorithm(alpha) for alpha in alphas]

        def exec_algorithm(algorithm: BaseFrankWolfe) -> int:
            algorithm.run(data.x0)
            return len(algorithm.history)

        plt.semilogy(
            alphas,
            list(map(exec_algorithm, algorithms)),
            label=data.title,
            **style.next(),
        )

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\log(K)$")
    plt.legend()
    plt.grid()

    style.reset()
    do_show_plot(filename="denisov1.pgf", show_plot=True, interactive=interactive)

    log("Done.", verbose=verbose)


def _run_experiment_lasso(verbose: bool, interactive: bool) -> None:
    np.random.seed(2026)

    # Dimension of the problem
    dim = 20

    A = non_singular_matrix(dim, 0.75, 1.0, -1.0, 1.0, rng=np.random.default_rng(2026))
    b = np.random.rand(dim)

    obj = Lasso(A, b, lam=0.5)
    lmo = L2BallLMO(radius=1.0)

    L = np.linalg.norm(A, ord=2) ** 2
    log(f"{L=}", verbose=verbose)

    x0 = np.zeros(dim)

    experiments: list[BaseFrankWolfe] = [
        ExperimentsData(
            obj=obj,
            algorithm=lambda alpha: AdaptiveFrankWolfeRobustRelativeInexactness(
                obj=obj,
                lmo=lmo,
                L=L,
                alpha=alpha,
                max_iter=_MAX_ITERATIONS,
                tol=_TOLERANCE,
            ),
            title="FW-Relative Inexactness)",
            x0=x0,
        ),
        ExperimentsData(
            obj=obj,
            algorithm=lambda alpha: AdaptiveFrankWolfeRobustComparison(
                obj=obj,
                lmo=lmo,
                L=L,
                alpha=alpha,
                delta=_DELTA,
                max_iter=_MAX_ITERATIONS,
                tol=_TOLERANCE,
            ),
            title="Robust FW (Comparison)",
            x0=x0,
        ),
    ]
    log(title("Lasso. Iterations per alpha."), verbose=verbose)

    if not interactive:
        preamble()

    style = LineStyle()
    plt.figure(figsize=(12, 6), dpi=100)

    alphas = np.linspace(ensure_non_zero(0), 0.5, 100)

    for data in experiments:
        algorithms = [data.algorithm(alpha) for alpha in alphas]

        def exec_algorithm(algorithm: BaseFrankWolfe) -> int:
            algorithm.run(data.x0)
            return len(algorithm.history)

        plt.semilogy(
            alphas,
            list(map(exec_algorithm, algorithms)),
            label=data.title,
            **style.next(),
        )

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\log(K)$")
    plt.legend()
    plt.grid()
    plt.title(obj.__doc__)

    style.reset()
    do_show_plot(filename="denisov2.pgf", show_plot=True, interactive=interactive)

    log("Done.", verbose=verbose)


def run_experiments(verbose: bool, interactive: bool) -> None:
    _run_experiment_log_regression(verbose, interactive)
    _run_experiment_lasso(verbose, interactive)
