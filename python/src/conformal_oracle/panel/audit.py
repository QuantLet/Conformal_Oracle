"""Panel-level audit orchestration."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from conformal_oracle._protocols import Forecaster
from conformal_oracle.audit.single_rolling import audit_rolling
from conformal_oracle.audit.single_static import audit_static
from conformal_oracle.panel.result import PanelResult


def audit_panel(
    returns: pd.DataFrame,
    forecasters: dict[str, Forecaster],
    alpha: float = 0.01,
    mode: Literal["static", "rolling"] = "rolling",
    calibration_split: float = 0.70,
    window: int = 250,
    seed: int = 2026,
    recalibration: object | None = None,
    **mode_kwargs: object,
) -> PanelResult:
    """Run a panel-level audit across all (forecaster, asset) pairs.

    Each (forecaster, asset) combination gets a deterministic seed
    derived from the base seed and the pair identity, so results
    are reproducible regardless of execution order.
    """
    asset_names = list(returns.columns)
    forecaster_names = list(forecasters.keys())

    results: dict[str, dict[str, object]] = {}

    for fc_name, fc in forecasters.items():
        results[fc_name] = {}
        for asset in asset_names:
            series = returns[asset].dropna()
            pair_seed = (
                seed + hash((fc_name, asset)) % (2**31)
            ) % (2**31)

            if mode == "static":
                r = audit_static(
                    series,
                    fc,
                    alpha=alpha,
                    calibration_split=calibration_split,
                    seed=pair_seed,
                    recalibration=recalibration,
                    **mode_kwargs,
                )
            elif mode == "rolling":
                r = audit_rolling(
                    series,
                    fc,
                    alpha=alpha,
                    window=window,
                    seed=pair_seed,
                    recalibration=recalibration,
                    **mode_kwargs,
                )
            else:
                raise ValueError(
                    f"Unknown mode: {mode!r}. "
                    "Use 'static' or 'rolling'."
                )
            results[fc_name][asset] = r

    return PanelResult(
        results=results,
        forecaster_names=forecaster_names,
        asset_names=asset_names,
        alpha=alpha,
        mode=mode,
        returns=returns,
    )
