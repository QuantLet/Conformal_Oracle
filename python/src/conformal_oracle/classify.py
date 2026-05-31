"""First-class regime classification entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from conformal_oracle._protocols import Forecaster


@dataclass
class RegimeVerdict:
    """Result of regime classification."""

    regime: Literal["signal-preserving", "replacement"]
    R: float
    R_bootstrap_ci: tuple[float, float]
    persistence_days: int | None
    basel_zone: Literal["green", "yellow", "red"]


def classify_regime(
    returns: pd.Series,
    *,
    forecast: pd.Series | None = None,
    forecaster: Forecaster | None = None,
    alpha: float = 0.01,
    mode: Literal["static", "rolling"] = "rolling",
    **kwargs: object,
) -> RegimeVerdict:
    """Classify a forecaster or quantile path as signal-preserving
    or replacement.

    Supply exactly one of ``forecast`` or ``forecaster``.

    Returns a :class:`RegimeVerdict` summarising the regime.
    """
    from conformal_oracle.audit import audit

    result = audit(
        returns,
        forecaster=forecaster,
        forecast=forecast,
        alpha=alpha,
        mode=mode,
        **kwargs,
    )

    if mode == "static":
        R = result.replacement_ratio
        R_ci = result.q_v_stat_ci
        persistence = None
    else:
        R = float(result.replacement_ratio.mean())
        # Bootstrap CI not directly available for rolling; use qV std
        qv_mean = float(result.q_v_roll.mean())
        qv_std = float(result.q_v_roll.std())
        R_ci = (qv_mean - 1.96 * qv_std, qv_mean + 1.96 * qv_std)
        # Count max consecutive days with R > 1
        above = (result.replacement_ratio > 1.0).astype(int).values
        max_run = 0
        current = 0
        for v in above:
            if v == 1:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        persistence = max_run

    return RegimeVerdict(
        regime=result.regime,
        R=R,
        R_bootstrap_ci=R_ci,
        persistence_days=persistence,
        basel_zone=result.basel_zone_corrected,
    )
