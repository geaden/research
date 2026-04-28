from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from common.objectives import Objective
from common.algorithmx import Result
from common.algorithmx.fw import BaseFrankWolfe, FrankWolfe, DiminishingStepSizeStrategy
from common.experiment_utils import title
from common.objectives import LogisticRegression
from common.lmo import MinLinearDirectionL2BallLMO
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


def _setup_experiments(verbose: bool) -> list[ExperimentsData]:
    """
    Setup experiment environment.
    """
    np.random.seed(2026)

    # Dimension of problem
    dim = 100

    A = non_singular_matrix(dim, 0.75, 1.0, -1.0, 1.0, rng=np.random.default_rng(2026))
    y = np.random.choice([-1, 1], size=(dim,))

    obj = LogisticRegression(A, y)
    lmo = MinLinearDirectionL2BallLMO(radius=1.0)

    L = 1 / (4 * dim) * np.linalg.eigvals(A).max()
    log(f"{L=}", verbose=verbose)

    x0 = np.zeros(dim)

    experiments: list[BaseFrankWolfe] = [
        ExperimentsData(
            obj=obj,
            algorithm=lambda alpha: FrankWolfe(
                obj=obj,
                lmo=lmo,
                step_size=DiminishingStepSizeStrategy(),
                L=L,
                max_iter=_MAX_ITERATIONS,
            ),
            title="Classical FW",
            x0=x0,
        ),
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
            title="Robust FW (Relative Inexactness)",
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
            title="Robust FW (Comparison)",
            x0=x0,
        ),
    ]

    return experiments


def run_experiments(verbose: bool, interactive: bool) -> None:
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
