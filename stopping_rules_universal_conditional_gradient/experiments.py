"""Module contains experiment implementations."""

import numpy as np
import matplotlib.pyplot as plt

from common.oracles.lmo import L2BallLMO
from common.plot_utils import preamble
from common.plotting.line_style import LineStyle
from stopping_rules_robust_fw.algorithms import AdaptiveFrankWolfeRobustComparison
from algorithms import NormalizedGradientMethodHoelder
from objectives import GasNetworkObjective
from common.utils import log
from common.plot_utils import do_show_plot

_SEED = 103


def _run_gas_network_experiment(verbose: bool, interactive: bool):
    """
    Implements Example 1 from .
    """
    # Problem parameters from Example 1
    # f(x) = sum(a_i * x_i^3)
    # The dual of this problem has a Hoelder continuous gradient with nu = 1/2.
    # However, the request is to use the comparison oracle with a nu value.
    # The paper mentions the dual function belongs to C^1,1/2.
    # Let's use nu=0.5 as per the paper's analysis of the dual.
    nu = 0.5

    if not interactive:
        preamble()

    plt.figure(figsize=(12, 6), dpi=100)

    # Setup a sample problem
    m = 10  # number of pipelines
    n = 5  # number of nodes
    rng = np.random.default_rng(_SEED)
    a = rng.random(m)
    A = rng.random((n, m))
    d = rng.random(n)

    obj = GasNetworkObjective(a)
    # The LMO would solve argmin over a set {x | Ax=d}.
    # For simplicity, we'll use a ball constraint LMO.
    lmo = L2BallLMO(radius=m)  # Assuming a feasible set diameter

    # Algorithm parameters
    L_nu = 1.0  # Placeholder, this would be L_nu
    alpha = 1e-4
    delta = 1e-3
    x0 = rng.random(m) * 1e-2

    # Note: AdaptiveFrankWolfeRobustComparison needs to be updated to pass `nu`
    # to its internal ComparisonOracle for a proper run.
    algo_fw = AdaptiveFrankWolfeRobustComparison(obj, lmo, L_nu, alpha, delta, nu=nu)

    algo_ngm = NormalizedGradientMethodHoelder(
        obj=obj,
        L_nu=L_nu,
        nu=nu,
        epsilon=alpha,
        delta=delta,
        max_iter=100,
    )

    style = LineStyle()

    log(
        "Running Gas Network Experiment from Universal gradient methods for convex optimization problems",
        verbose=verbose,
    )
    log("Running Normalized Gradient Method (Algorithm 3)...", verbose=verbose)
    result_ngm = algo_ngm.run(x0)
    log(f"Optimal value (NGM): {obj(result_ngm.x_opt)}", verbose=verbose)
    print(algo_ngm.history)
    plt.plot(
        np.arange(len(algo_ngm.history)),
        list(map(obj, algo_ngm.history)),
        label="NGM",
        **style.next(),
    )

    # log("\nRunning Adaptive Frank-Wolfe with Comparison Oracle...", verbose=verbose)
    # result_fw = algo_fw.run(x0)
    # log(f"Optimal value (FW): {obj(result_fw.x_opt)}", verbose=verbose)
    # plt.plot(
    #     np.arange(len(algo_fw.history)),
    #     list(map(obj, algo_fw.history)),
    #     label="Adaptive FW",
    #     **style.next(),
    # )

    style.reset()

    plt.xlabel(r"$k$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.grid()

    style.reset()
    do_show_plot(filename="denisov1.pgf", show_plot=True, interactive=interactive)

    log("Done.", verbose=verbose)


def run_experiments(verbose: bool, interactive: bool):
    _run_gas_network_experiment(verbose, interactive)
