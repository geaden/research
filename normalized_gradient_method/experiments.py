"""Module contains experiment implementations."""

from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from common.math_utils import ensure_non_zero
from common.latex_utils import latex_table
from common.math_utils import ensure_non_zero, significant_figures
from common.oracles.lmo import L2BallLMO
from common.plotting.line_style import LineStyle
from stopping_rules_robust_fw.algorithms import AdaptiveFrankWolfeRobustComparison
from algorithms import NormalizedGradientMethodHoelder
from common.utils import log
from common.objectives import LogisticRegression
from common.plot_utils import do_show_plot, preamble

from objectives import GasNetworkObjective, TwoLayerNetwork

_SEED = 103

_EPSILON = 1e-4
_DELTA = 1e-3


def _plot_iterations_vs_epsilon(
    ax: plt.Axes,
    history: list[np.ndarray],
    obj: Any,
    f_star: float,
    delta: float,
):
    """Helper to plot iterations vs epsilon comparison."""

    n = len(history)

    practical_diff = np.array([obj(x) - f_star for x in history])

    positive_diffs = practical_diff[practical_diff > 1e-12]
    if len(positive_diffs) < 2:
        return  # Not enough data to plot

    min_eps = np.min(positive_diffs)
    max_eps = np.max(positive_diffs)

    epsilons = np.logspace(np.log10(min_eps), np.log10(max_eps), num=20)

    practical_iterations = []
    valid_epsilons = []

    for eps in epsilons:
        found_k = np.where(practical_diff <= eps)[0]
        if len(found_k) > 0:
            practical_iterations.append(found_k[0])
            valid_epsilons.append(eps)

    if len(valid_epsilons) < 2:
        return  # Not enough points to plot

    valid_epsilons = np.array(valid_epsilons)
    practical_iterations = np.array(practical_iterations)

    K_upper_bound = np.max(practical_iterations * valid_epsilons)

    theoretical_base_term = n * np.log(n / delta)
    C_estimated = K_upper_bound / ensure_non_zero(theoretical_base_term)

    theoretical_iterations = C_estimated * (1 / valid_epsilons) * theoretical_base_term

    if len(valid_epsilons) > 0:
        num_samples = min(10, len(valid_epsilons))
        indices = np.linspace(0, len(valid_epsilons) - 1, num_samples, dtype=int)

        table_values = []
        for i in indices:
            eps_val = valid_epsilons[i]
            N_practical = practical_iterations[i]
            N_theoretical = theoretical_iterations[i]
            exponent = int(np.floor(np.log10(eps_val)))
            mantissa = eps_val / (10**exponent)
            eps_formatted = rf"${significant_figures(mantissa, 3)} \cdot 10^{{{exponent}}}$"
            table_values.append(
                (
                    eps_formatted,
                    N_practical,
                    int(np.ceil(N_theoretical)),
                )
            )

        print(
            latex_table(
                caption=r"Количество итераций $N$ для заданной точности $\varepsilon$",
                values=table_values,
                headers=[
                    r"$\varepsilon$",
                    r"$N$ (практическое)",
                    r"$N$ (теоретическое)",
                ],
            )
        )

    ax.plot(valid_epsilons, practical_iterations, label="Practical")

    ax.plot(
        valid_epsilons,
        theoretical_iterations,
        label=r"Theoretical $O(\frac{1}{\varepsilon} n \log \frac{n}{\delta})$",
        linestyle="--",
    )

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$N$")
    ax.set_xscale("log")
    ax.set_yscale("log")


def _run_gas_network_experiment(verbose: bool, interactive: bool):
    nu = 0.5

    if not interactive:
        preamble()

    _, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    m = 10  # number of pipelines
    n = 5  # number of nodes
    rng = np.random.default_rng(_SEED)
    alpha_vec = rng.uniform(0.1, 1.0, size=m)
    A = rng.random((n, m))
    d = rng.random(n)

    p = 3
    obj = GasNetworkObjective(alpha=alpha_vec, A=A, d=d, p=p)

    L_nu = (
        (2 / (3 * np.sqrt(3)))
        * (1 / p) ** 1.5
        * (1 / (p - 1))
        * np.max(1 / np.sqrt(alpha_vec))
        * (np.linalg.norm(A, ord=2) ** 1.5)
    )
    L_nu /= 1e4
    log(f"{L_nu=}", verbose=verbose)
    assert L_nu > 0
    x0 = rng.random(n) * 1e5

    algo_ngm = NormalizedGradientMethodHoelder(
        obj=obj,
        L_nu=L_nu,
        nu=nu,
        epsilon=_EPSILON,
        delta=_DELTA,
        max_iter=50,
    )

    style = LineStyle()

    log("Running Gas Network Experiment...", verbose=verbose)
    log("Running Normalized Gradient Method...", verbose=verbose)
    result_ngm = algo_ngm.run(x0)
    log(f"Optimal value (NGM): {obj(result_ngm.x_opt)}", verbose=verbose)

    f_star = obj(result_ngm.x_opt)

    def calculate_diff(x: np.ndarray) -> np.float64:
        return obj(x) - f_star

    axs[0].plot(
        np.arange(len(algo_ngm.history)),
        list(map(calculate_diff, algo_ngm.history)),
        **style.next(),
    )

    axs[0].set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$f^k-f^\ast$")
    axs[0].grid()

    _plot_iterations_vs_epsilon(axs[1], algo_ngm.history, obj, f_star, _DELTA)

    plt.tight_layout()
    do_show_plot(filename="ngm.pgf", show_plot=True, interactive=interactive)


def _run_logistic_regression_experiment(verbose: bool, interactive: bool):
    """
    Runs a logistic regression experiment using Normalized Gradient Method.
    """
    nu = 1.0

    if not interactive:
        preamble()

    _, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    n_samples = 100
    n_features = 20
    rng = np.random.default_rng(_SEED)
    A = rng.random((n_samples, n_features))
    y = rng.choice([-1, 1], size=n_samples)

    obj = LogisticRegression(A=A, y=y)

    L_nu = 0.25 * np.linalg.norm(A, ord=2) ** 2 / n_samples
    log(f"{L_nu=}", verbose=verbose)
    assert L_nu > 0
    x0 = np.zeros(n_features)

    algo_ngm = NormalizedGradientMethodHoelder(
        obj=obj,
        L_nu=L_nu,
        nu=nu,
        epsilon=_EPSILON,
        delta=_DELTA,
        max_iter=1000,
    )

    style = LineStyle()

    log("Running Logistic Regression Experiment...", verbose=verbose)
    result_ngm = algo_ngm.run(x0)
    log(f"Optimal value (NGM): {obj(result_ngm.x_opt)}", verbose=verbose)

    f_star = obj(result_ngm.x_opt)

    def calculate_diff(x: np.ndarray) -> np.float64:
        return obj(x) - f_star

    axs[0].plot(
        np.arange(len(algo_ngm.history)),
        list(map(calculate_diff, algo_ngm.history)),
        **style.next(),
    )

    axs[0].set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$f^k-f^\ast$")
    axs[0].grid()

    _plot_iterations_vs_epsilon(axs[1], algo_ngm.history, obj, f_star, _DELTA)
    plt.tight_layout()
    do_show_plot(filename="ngm_logreg.pgf", show_plot=True, interactive=interactive)


def _run_two_layer_network_experiment(verbose: bool, interactive: bool):
    """
    Runs a two-layer network experiment using Normalized Gradient Method.
    This uses a random feature model, where the first layer is fixed and random,
    and only the second layer is trained.
    """
    nu = 1.0

    if not interactive:
        preamble()

    _, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    n_samples = 400
    n_features = 20
    n_hidden = 20
    rng = np.random.default_rng(_SEED)
    X = rng.random((n_samples, n_features))

    true_w1 = rng.standard_normal((n_hidden, n_features))
    true_b1 = rng.standard_normal((n_hidden,))
    true_w2 = rng.standard_normal((n_hidden + 1,))
    H_true = np.maximum(0, X @ true_w1.T + true_b1)
    H_true = np.c_[H_true, np.ones(n_samples)]
    y = H_true @ true_w2 + rng.normal(0, 0.1, size=n_samples)

    obj = TwoLayerNetwork(X=X, y=y, n_hidden=n_hidden, seed=_SEED)

    L_nu = np.linalg.norm(obj.A, ord=2) ** 2 / n_samples
    log(f"{L_nu=}", verbose=verbose)
    assert L_nu > 0
    x0 = np.zeros(n_hidden + 1)

    algo_ngm = NormalizedGradientMethodHoelder(
        obj=obj,
        L_nu=L_nu,
        nu=nu,
        epsilon=_EPSILON,
        delta=_DELTA,
        max_iter=10000,
    )

    style = LineStyle()

    log("Running Two-Layer Network Experiment...", verbose=verbose)
    result_ngm = algo_ngm.run(x0)
    log(f"Optimal value (NGM): {obj(result_ngm.x_opt)}", verbose=verbose)

    f_star = obj(result_ngm.x_opt)

    def calculate_diff(x: np.ndarray) -> np.float64:
        return obj(x) - f_star

    axs[0].plot(
        np.arange(len(algo_ngm.history)),
        list(map(calculate_diff, algo_ngm.history)),
        **style.next(),
    )

    axs[0].set_xlabel(r"$k$")
    axs[0].set_ylabel(r"$f^k-f^\ast$")
    axs[0].grid()

    _plot_iterations_vs_epsilon(axs[1], algo_ngm.history, obj, f_star, _DELTA)
    plt.tight_layout()
    do_show_plot(filename="ngm_2layer.pgf", show_plot=True, interactive=interactive)


def run_experiments(verbose: bool, interactive: bool):
    _run_gas_network_experiment(verbose, interactive)
    _run_logistic_regression_experiment(verbose, interactive)
    _run_two_layer_network_experiment(verbose, interactive)

    log("Done.", verbose=verbose)
