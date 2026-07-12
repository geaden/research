"""Module contains experiment implementations."""

import numpy as np
import matplotlib.pyplot as plt

from common.oracles.lmo import L2BallLMO
from common.plotting.line_style import LineStyle
from stopping_rules_robust_fw.algorithms import AdaptiveFrankWolfeRobustComparison
from algorithms import NormalizedGradientMethodHoelder
from common.utils import log
from common.objectives import LogisticRegression
from common.plot_utils import do_show_plot, preamble

from objectives import GasNetworkObjective, TwoLayerNetwork

_SEED = 103


def _run_gas_network_experiment(verbose: bool, interactive: bool):
    nu = 0.5

    if not interactive:
        preamble()

    plt.figure(figsize=(12, 6), dpi=100)

    # Setup a sample problem
    m = 10  # number of pipelines
    n = 5  # number of nodes
    rng = np.random.default_rng(_SEED)
    # Generate alpha from a stable range to avoid near-zero values,
    # which would cause the theoretical L_nu to become excessively large
    # and the algorithm's step size to become near-zero.
    alpha_vec = rng.uniform(0.1, 1.0, size=m)
    A = rng.random((n, m))
    d = rng.random(n)

    p = 3
    obj = GasNetworkObjective(alpha=alpha_vec, A=A, d=d, p=p)

    # Algorithm parameters
    L_nu = (
        (2 / (3 * np.sqrt(3)))
        * (1 / p) ** 1.5
        * (1 / (p - 1))
        * np.max(1 / np.sqrt(alpha_vec))
        * (np.linalg.norm(A, ord=2) ** 1.5)
    )
    # The theoretical bound for L_nu can be very conservative in practice,
    # leading to a step size that is too small. We scale it down to
    # achieve a more practical step size for the experiment.
    L_nu /= 1e2
    assert L_nu > 0
    alpha = 1e-4
    delta = 1e-3
    x0 = rng.random(n) * 1e-2  # Initial point for the dual variable y in R^n

    algo_ngm = NormalizedGradientMethodHoelder(
        obj=obj,
        L_nu=L_nu,
        nu=nu,
        epsilon=alpha,
        delta=delta,
        max_iter=1000,
    )

    style = LineStyle()

    log("Running Gas Network Experiment...", verbose=verbose)
    log("Running Normalized Gradient Method (Algorithm 3)...", verbose=verbose)
    result_ngm = algo_ngm.run(x0)
    log(f"Optimal value (NGM): {obj(result_ngm.x_opt)}", verbose=verbose)

    f_star = obj(result_ngm.x_opt)

    def calculate_diff(x: np.ndarray) -> np.float64:
        return obj(x) - f_star

    plt.plot(
        np.arange(len(algo_ngm.history)),
        list(map(calculate_diff, algo_ngm.history)),
        **style.next(),
    )

    style.reset()

    plt.xlabel(r"$k$")
    plt.ylabel(r"$f^k-f^\ast$")
    plt.legend()
    plt.grid()

    style.reset()
    do_show_plot(filename="ngm.pgf", show_plot=True, interactive=interactive)


def _run_logistic_regression_experiment(verbose: bool, interactive: bool):
    """
    Runs a logistic regression experiment using Normalized Gradient Method.
    """
    nu = 1.0

    if not interactive:
        preamble()

    plt.figure(figsize=(12, 6), dpi=100)

    # Setup a sample problem
    n_samples = 100
    n_features = 20
    rng = np.random.default_rng(_SEED)
    A = rng.random((n_samples, n_features))
    y = rng.choice([-1, 1], size=n_samples)

    obj = LogisticRegression(A=A, y=y)

    # Algorithm parameters
    # For logistic regression, L can be estimated as 0.25 * ||A||_2^2 / n
    L_nu = 0.25 * np.linalg.norm(A, ord=2) ** 2 / n_samples
    epsilon = 1e-4
    delta = 1e-3
    x0 = np.zeros(n_features)

    algo_ngm = NormalizedGradientMethodHoelder(
        obj=obj,
        L_nu=L_nu,
        nu=nu,
        epsilon=epsilon,
        delta=delta,
        max_iter=5000,
    )

    style = LineStyle()

    log("Running Logistic Regression Experiment...", verbose=verbose)
    result_ngm = algo_ngm.run(x0)
    log(f"Optimal value (NGM): {obj(result_ngm.x_opt)}", verbose=verbose)

    f_star = obj(result_ngm.x_opt)

    def calculate_diff(x: np.ndarray) -> np.float64:
        return obj(x) - f_star

    plt.plot(
        np.arange(len(algo_ngm.history)),
        list(map(calculate_diff, algo_ngm.history)),
        **style.next(),
    )

    plt.xlabel(r"$k$")
    plt.ylabel(r"$f^k-f^\ast$")
    plt.grid()
    do_show_plot(filename="ngm_logreg.pgf", show_plot=True, interactive=interactive)


def _run_two_layer_network_experiment(verbose: bool, interactive: bool):
    """
    Runs a two-layer network experiment using Normalized Gradient Method.
    This uses a random feature model, where the first layer is fixed and random,
    and only the second layer is trained.
    """
    nu = 1.0  # The objective is MSE on random features, which has a Lipschitz gradient.

    if not interactive:
        preamble()

    plt.figure(figsize=(12, 6), dpi=100)

    # Setup a sample problem
    n_samples = 100
    n_features = 20
    n_hidden = 50
    rng = np.random.default_rng(_SEED)
    X = rng.random((n_samples, n_features))

    # Generate a true function and then the targets
    true_w1 = rng.standard_normal((n_hidden, n_features))
    true_b1 = rng.standard_normal((n_hidden,))
    true_w2 = rng.standard_normal((n_hidden + 1,))  # with bias
    H_true = np.maximum(0, X @ true_w1.T + true_b1)
    H_true = np.c_[H_true, np.ones(n_samples)]
    y = H_true @ true_w2 + rng.normal(0, 0.1, size=n_samples)  # add some noise

    obj = TwoLayerNetwork(X=X, y=y, n_hidden=n_hidden, seed=_SEED)

    # Algorithm parameters
    L_nu = np.linalg.norm(obj.A, ord=2) ** 2
    epsilon = 1e-4
    delta = 1e-3
    x0 = np.zeros(n_hidden + 1)

    algo_ngm = NormalizedGradientMethodHoelder(
        obj=obj,
        L_nu=L_nu,
        nu=nu,
        epsilon=epsilon,
        delta=delta,
        max_iter=5000,
    )

    style = LineStyle()

    log("Running Two-Layer Network Experiment...", verbose=verbose)
    result_ngm = algo_ngm.run(x0)
    log(f"Optimal value (NGM): {obj(result_ngm.x_opt)}", verbose=verbose)

    f_star = obj(result_ngm.x_opt)

    def calculate_diff(x: np.ndarray) -> np.float64:
        return obj(x) - f_star

    plt.plot(
        np.arange(len(algo_ngm.history)),
        list(map(calculate_diff, algo_ngm.history)),
        **style.next(),
    )

    plt.xlabel(r"$k$")
    plt.ylabel(r"$f^k-f^\ast$")
    plt.grid()
    do_show_plot(filename="ngm_2layer.pgf", show_plot=True, interactive=interactive)


def run_experiments(verbose: bool, interactive: bool):
    _run_gas_network_experiment(verbose, interactive)
    _run_logistic_regression_experiment(verbose, interactive)
    _run_two_layer_network_experiment(verbose, interactive)

    log("Done.", verbose=verbose)
