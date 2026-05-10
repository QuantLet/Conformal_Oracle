"""Predictive distribution types used throughout the package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from scipy import stats


@dataclass
class SampleDistribution:
    """Predictive distribution represented as Monte Carlo samples."""

    samples: np.ndarray

    def quantile(self, alpha: float) -> float:
        return float(np.quantile(self.samples, alpha))

    def expected_shortfall(self, alpha: float) -> float:
        q = self.quantile(alpha)
        tail = self.samples[self.samples <= q]
        if len(tail) == 0:
            return q
        return float(np.mean(tail))

    def cdf(self, x: float) -> float:
        return float(np.mean(self.samples <= x))

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class QuantileGridDistribution:
    """Predictive distribution as a finite quantile grid."""

    levels: np.ndarray
    quantiles: np.ndarray

    def quantile(
        self, alpha: float, completion: str = "student_t"
    ) -> float:
        idx_exact = np.where(np.isclose(self.levels, alpha))[0]
        if len(idx_exact) > 0:
            return float(self.quantiles[idx_exact[0]])

        if self.levels.min() <= alpha <= self.levels.max():
            return float(np.interp(alpha, self.levels, self.quantiles))

        return self._tail_completion_quantile(alpha, completion)

    def expected_shortfall(
        self, alpha: float, completion: str = "student_t"
    ) -> float:
        if completion == "linear":
            return self._es_linear(alpha)
        params = self._fit_parametric(completion)
        if completion == "student_t":
            df, loc, scale = params
            rv = stats.t(df=df, loc=loc, scale=scale)
        else:
            loc, scale = params
            rv = stats.norm(loc=loc, scale=scale)
        q = rv.ppf(alpha)
        tail_samples = rv.rvs(size=10000, random_state=42)
        tail_samples = tail_samples[tail_samples <= q]
        if len(tail_samples) == 0:
            return float(q)
        return float(np.mean(tail_samples))

    def cdf(self, x: float, completion: str = "student_t") -> float:
        """CDF evaluated at x. Within the grid range, interpolates
        between grid points. Outside the grid, uses the same parametric
        tail completion as quantile() (Student-t by default). Users who
        want strict in-grid-only behaviour can check
        ``min(levels) <= cdf_value <= max(levels)`` themselves."""
        below = self.quantiles[self.quantiles <= x]
        if len(below) == 0 and x < self.quantiles.min():
            params = self._fit_parametric(completion)
            if completion == "student_t":
                df, loc, scale = params
                return float(stats.t.cdf(x, df=df, loc=loc, scale=scale))
            else:
                loc, scale = params
                return float(stats.norm.cdf(x, loc=loc, scale=scale))
        if x >= self.quantiles.max():
            return 1.0
        return float(np.interp(x, self.quantiles, self.levels))

    def _tail_completion_quantile(self, alpha: float, method: str) -> float:
        if method == "linear":
            if alpha < self.levels.min():
                slope = (self.quantiles[1] - self.quantiles[0]) / (
                    self.levels[1] - self.levels[0]
                )
                return float(
                    self.quantiles[0] + slope * (alpha - self.levels[0])
                )
            else:
                slope = (self.quantiles[-1] - self.quantiles[-2]) / (
                    self.levels[-1] - self.levels[-2]
                )
                return float(
                    self.quantiles[-1] + slope * (alpha - self.levels[-1])
                )

        params = self._fit_parametric(method)
        if method == "student_t":
            df, loc, scale = params
            return float(stats.t.ppf(alpha, df=df, loc=loc, scale=scale))
        else:
            loc, scale = params
            return float(stats.norm.ppf(alpha, loc=loc, scale=scale))

    def _fit_parametric(self, method: str) -> tuple[float, ...]:
        if method == "student_t":
            df, loc, scale = stats.t.fit(
                self.quantiles,
                floc=float(np.median(self.quantiles)),
            )
            return (df, loc, scale)
        else:
            loc = float(np.mean(self.quantiles))
            scale = float(np.std(self.quantiles))
            return (loc, scale)

    def _es_linear(self, alpha: float) -> float:
        q = self.quantile(alpha, completion="linear")
        below = self.quantiles[self.levels <= alpha]
        if len(below) == 0:
            return float(q)
        return float(np.mean(np.append(below, q)))


@dataclass
class ParametricDistribution:
    """Predictive distribution from a closed-form parametric family."""

    location: float
    scale: float
    family: Literal["normal", "student_t", "skewed_t"]
    df: float | None = None
    skew: float | None = None

    def quantile(self, alpha: float) -> float:
        rv = self._distribution()
        return float(rv.ppf(alpha))

    def expected_shortfall(self, alpha: float) -> float:
        rv = self._distribution()
        q = rv.ppf(alpha)
        samples = rv.rvs(size=50000, random_state=42)
        tail = samples[samples <= q]
        if len(tail) == 0:
            return float(q)
        return float(np.mean(tail))

    def cdf(self, x: float) -> float:
        rv = self._distribution()
        return float(rv.cdf(x))

    def _distribution(self) -> stats.rv_continuous:
        if self.family == "normal":
            return stats.norm(loc=self.location, scale=self.scale)
        elif self.family == "student_t":
            if self.df is None:
                raise ValueError("df required for student_t family")
            return stats.t(df=self.df, loc=self.location, scale=self.scale)
        elif self.family == "skewed_t":
            if self.df is None or self.skew is None:
                raise ValueError("df and skew required for skewed_t family")
            # Hansen's skewed-t: use scipy's nct as approximation
            return stats.nct(
                df=self.df,
                nc=self.skew,
                loc=self.location,
                scale=self.scale,
            )
        raise ValueError(f"Unknown family: {self.family}")


PredictiveDistribution = Union[
    SampleDistribution,
    QuantileGridDistribution,
    ParametricDistribution,
]
