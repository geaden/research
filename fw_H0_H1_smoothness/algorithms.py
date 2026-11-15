"""Frank-Wolfe algorithms for convex optimization under (H0, H1)-smoothness."""

from typing import Optional, Union
import numpy as np

from common.algorithmx import Result
from common.algorithmx.fw import BaseFrankWolfe
from common.objectives import Objective
from common.oracles.lmo import LMO
from common.math_utils import ensure_non_zero

try:
    from .stopping_rules import GapClippedStoppingRule, PolyakResidualStoppingRule
except ImportError:
    from stopping_rules import GapClippedStoppingRule, PolyakResidualStoppingRule


StoppingRuleType = Union[GapClippedStoppingRule, PolyakResidualStoppingRule]


class GapClippedH0H1FW(BaseFrankWolfe):
    r"""Frank-Wolfe method for (H0, H1)-smooth functions with gap-clipped step.

    Given a lower bound f_low <= f* (e.g., f_low = 0 for non-negative losses),
    computes:
        \check{r}_k = f(x^k) - f_low >= f(x^k) - f*,
        \Phi_k = min(G(x^k), \check{r}_k),
        \widehat{H}_k = H0 + H1 * \check{r}_k,
        h_k = min(1, \Phi_k / (e * \widehat{H}_k * ||d^k||^2)).

    Since \Phi_k <= \check{r}_k and h_k <= 1, the step satisfies
    H1 * ||x^{k+1} - x^k||^2 <= 1/e < 1 automatically on every iteration.
    """

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        H0: float,
        H1: float,
        stopping_rule: GapClippedStoppingRule,
        max_iter: int = 300,
        tol: float = 1e-14,
        stop_on_rule: bool = True,
    ):
        super().__init__(obj=obj, lmo=lmo)
        assert H0 >= 0, "H0 must be non-negative"
        assert H1 >= 0, "H1 must be non-negative"

        self.H0 = H0
        self.H1 = H1
        self.stopping_rule = stopping_rule
        self.max_iter = max_iter
        self.tol = tol
        self.stop_on_rule = stop_on_rule

        self.gap_history: list[dict[str, float]] = []
        self.step_history: list[float] = []
        self.stopped_at: Optional[int] = None
        self.T1: int = 0
        self.T2: int = 0
        self.T3: int = 0

    def _smoothness_bound(self, r_check: float) -> float:
        """Compute upper curvature bound H0 + H1 * (f(x^k) - f_low)."""
        return self.H0 + self.H1 * max(0.0, r_check)

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        self.track(x)
        self.stopped_at = None
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0

        for k in range(self.max_iter):
            f_val = float(self._obj(x))
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x
            D_k = float(np.linalg.norm(d_k))

            info = self.stopping_rule.compute_metrics(f_val, grad, d_k)
            self.gap_history.append(info)

            if self.stopped_at is None and self.stopping_rule.check(f_val, grad, d_k):
                self.stopped_at = k
                if self.stop_on_rule:
                    break

            r_check = info["r_check"]
            phi_k = info["phi"]
            H_hat_k = self._smoothness_bound(r_check)

            if D_k > 1e-12 and phi_k > 0:
                h_raw = phi_k / (np.e * ensure_non_zero(H_hat_k) * (D_k ** 2))
                h_k = min(1.0, h_raw)
            else:
                h_k = 0.0

            if self.stopped_at is None or k < self.stopped_at:
                if h_k < 1.0:
                    if self.H1 * r_check >= self.H0:
                        self.T1 += 1
                    else:
                        self.T2 += 1
                else:
                    self.T3 += 1

            self.step_history.append(h_k)
            x = x + h_k * d_k
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @property
    def metrics(self) -> dict:
        return {
            "gap_history": self.gap_history,
            "step_history": self.step_history,
            "stopped_at": self.stopped_at,
            "T1": self.T1,
            "T2": self.T2,
            "T3": self.T3,
            "iterations": (
                self.stopped_at
                if self.stopped_at is not None
                else len(self.history) - 1
            ),
        }


class PolyakH0H1FW(BaseFrankWolfe):
    r"""Polyak-type Frank-Wolfe method for (H0, H1)-smooth functions.

    Uses a lower bound f_low <= f* in the denominator and an upper bound
    f_up >= f* in the numerator:
        \hat{r}_k = f(x^k) - f_up <= f(x^k) - f* <= G(x^k),
        \check{r}_k = f(x^k) - f_low >= f(x^k) - f*,
        \widehat{H}_k = H0 + H1 * \check{r}_k,
        h_k = min(1, \hat{r}_k / (e * \widehat{H}_k * ||d^k||^2)).

    When f_low = f_up = f*, this is the exact Polyak-type Frank-Wolfe step.
    """

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        H0: float,
        H1: float,
        stopping_rule: PolyakResidualStoppingRule,
        max_iter: int = 300,
        tol: float = 1e-14,
        stop_on_rule: bool = True,
    ):
        super().__init__(obj=obj, lmo=lmo)
        assert H0 >= 0, "H0 must be non-negative"
        assert H1 >= 0, "H1 must be non-negative"

        self.H0 = H0
        self.H1 = H1
        self.stopping_rule = stopping_rule
        self.max_iter = max_iter
        self.tol = tol
        self.stop_on_rule = stop_on_rule

        self.gap_history: list[dict[str, float]] = []
        self.step_history: list[float] = []
        self.stopped_at: Optional[int] = None
        self.T1: int = 0
        self.T2: int = 0
        self.T3: int = 0

    def _smoothness_bound(self, r_check: float) -> float:
        return self.H0 + self.H1 * max(0.0, r_check)

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        self.track(x)
        self.stopped_at = None
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0

        for k in range(self.max_iter):
            f_val = float(self._obj(x))
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x
            D_k = float(np.linalg.norm(d_k))

            info = self.stopping_rule.compute_metrics(f_val, grad, d_k)
            self.gap_history.append(info)

            if self.stopped_at is None and self.stopping_rule.check(f_val, grad, d_k):
                self.stopped_at = k
                if self.stop_on_rule:
                    break

            r_check = info["r_check"]
            r_hat = info["r_hat"]
            H_hat_k = self._smoothness_bound(r_check)

            if D_k > 1e-12 and r_hat > 0:
                h_raw = r_hat / (np.e * ensure_non_zero(H_hat_k) * (D_k ** 2))
                h_k = min(1.0, h_raw)
            else:
                h_k = 0.0

            if self.stopped_at is None or k < self.stopped_at:
                if h_k < 1.0:
                    if self.H1 * r_check >= self.H0:
                        self.T1 += 1
                    else:
                        self.T2 += 1
                else:
                    self.T3 += 1

            self.step_history.append(h_k)
            x = x + h_k * d_k
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @property
    def metrics(self) -> dict:
        return {
            "gap_history": self.gap_history,
            "step_history": self.step_history,
            "stopped_at": self.stopped_at,
            "T1": self.T1,
            "T2": self.T2,
            "T3": self.T3,
            "iterations": (
                self.stopped_at
                if self.stopped_at is not None
                else len(self.history) - 1
            ),
        }


class ClassicalH0H1FW(BaseFrankWolfe):
    """Classical Frank-Wolfe with step h_k = 2 / (k + 2) for baseline comparison."""

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        max_iter: int = 300,
        tol: float = 1e-14,
        stopping_rule: Optional[StoppingRuleType] = None,
    ):
        super().__init__(obj=obj, lmo=lmo)
        self.max_iter = max_iter
        self.tol = tol
        self.stopping_rule = stopping_rule

        self.gap_history: list[dict[str, float]] = []
        self.step_history: list[float] = []
        self.stopped_at: Optional[int] = None

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        self.track(x)
        self.stopped_at = None

        for k in range(self.max_iter):
            f_val = float(self._obj(x))
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x

            if self.stopping_rule is not None:
                info = self.stopping_rule.compute_metrics(f_val, grad, d_k)
                self.gap_history.append(info)
                if self.stopped_at is None and self.stopping_rule.check(
                    f_val, grad, d_k
                ):
                    self.stopped_at = k
            else:
                G_k = float(-np.dot(grad, d_k))
                self.gap_history.append(
                    {
                        "G": G_k,
                        "D": float(np.linalg.norm(d_k)),
                        "M": float(np.linalg.norm(grad)),
                    }
                )

            h_k = 2.0 / (k + 2.0)
            self.step_history.append(h_k)
            x = x + h_k * d_k
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @property
    def metrics(self) -> dict:
        return {
            "gap_history": self.gap_history,
            "step_history": self.step_history,
            "stopped_at": self.stopped_at,
            "iterations": len(self.history) - 1,
        }


class AdaptiveH0H1FW(BaseFrankWolfe):
    """Backtracking adaptive (H0, H1)-Frank-Wolfe algorithm."""

    _H_0_max = 1e10
    _H_1_max = 1e10

    def __init__(
        self,
        obj: Objective,
        lmo: LMO,
        H0: float,
        H1: float,
        stopping_rule: GapClippedStoppingRule,
        rho: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-14,
        stop_on_rule: bool = True,
    ):
        super().__init__(obj=obj, lmo=lmo)
        assert rho >= 1.0, "rho must be >= 1"
        assert H0 >= 0, "H0 must be non-negative"
        assert H1 >= 0, "H1 must be non-negative"
        self.H0 = H0
        self.H1 = H1
        self.rho = rho
        self.stopping_rule = stopping_rule
        self.max_iter = max_iter
        self.tol = tol
        self.stop_on_rule = stop_on_rule

        self.gap_history: list[dict[str, float]] = []
        self.step_history: list[float] = []
        self.stopped_at: Optional[int] = None

    @staticmethod
    def _conv_factor(r_check: float, H0: float, H1: float) -> float:
        return ensure_non_zero(H0 + H1 * max(0.0, r_check))

    def run(self, x0: np.ndarray) -> Result:
        x = x0.copy()
        self.track(x)
        self.stopped_at = None
        H0 = ensure_non_zero(self.H0)
        H1 = ensure_non_zero(self.H1)
        flip = False

        for _ in range(self.max_iter):
            f_val = float(self._obj(x))
            grad = self._obj.grad(x)
            s_k = self._lmo(grad)
            d_k = s_k - x
            D_k = float(np.linalg.norm(d_k))

            info = self.stopping_rule.compute_metrics(f_val, grad, d_k)
            self.gap_history.append(info)
            phi_k = info["phi"]
            r_check = info["r_check"]

            if self.stopped_at is None and self.stopping_rule.check(f_val, grad, d_k):
                self.stopped_at = len(self.gap_history) - 1
                if self.stop_on_rule:
                    break

            if D_k <= 1e-12 or phi_k <= 0:
                self.step_history.append(0.0)
                self.track(x)
                continue

            conv_factor = self._conv_factor(r_check, H0, H1)
            H0 /= self.rho + H0 / conv_factor
            H1 /= self.rho + H1 * r_check / conv_factor

            for _ in range(60):
                conv_factor = self._conv_factor(r_check, H0, H1)
                h_k = min(1.0, phi_k / (np.e * conv_factor * (D_k ** 2)))
                x_next = x + h_k * d_k
                if (
                    float(self._obj(x_next))
                    <= f_val
                    - h_k * phi_k
                    + 0.5 * np.e * conv_factor * (h_k ** 2) * (D_k ** 2)
                ):
                    break
                if not flip:
                    H0 = min(H0 * (self.rho - H0 / conv_factor), self._H_0_max)
                else:
                    H1 = min(
                        H1 * (self.rho - H1 * r_check / conv_factor),
                        self._H_1_max,
                    )
                flip = not flip

            self.step_history.append(h_k)
            x = x_next
            self.track(x)

        return Result(x0=x0.copy(), x_opt=x.copy())

    @property
    def metrics(self) -> dict:
        return {
            "gap_history": self.gap_history,
            "step_history": self.step_history,
            "stopped_at": self.stopped_at,
            "iterations": (
                self.stopped_at
                if self.stopped_at is not None
                else len(self.history) - 1
            ),
        }
