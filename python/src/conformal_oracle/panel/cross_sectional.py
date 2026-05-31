"""Cross-sectional correlations of qV with asset characteristics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_asset_characteristics(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Asset-level summary statistics from full-sample returns.

    Computes: annualised_vol, tail_frequency (beyond 3 std),
    autocorrelation (lag-1), excess_kurtosis.
    """
    rows = {}
    for col in returns.columns:
        s = returns[col].dropna()
        vol = float(s.std() * np.sqrt(252))
        threshold = s.mean() - 3 * s.std()
        tail_freq = float((s < threshold).mean())
        ac1 = float(s.autocorr(lag=1)) if len(s) > 1 else 0.0
        kurt = float(s.kurtosis())
        rows[col] = {
            "annualised_vol": vol,
            "tail_frequency": tail_freq,
            "autocorrelation": ac1,
            "excess_kurtosis": kurt,
        }
    return pd.DataFrame(rows).T


def compute_cross_sectional_correlations(
    qv_values: pd.DataFrame,
    asset_characteristics: pd.DataFrame,
) -> pd.DataFrame:
    """Per-forecaster Pearson correlations between qV and characteristics.

    qv_values: rows=forecasters, cols=assets.
    asset_characteristics: rows=assets, cols=characteristics.

    Returns: rows=forecasters, cols=characteristics, values=Pearson rho.
    """
    assets = list(
        set(qv_values.columns) & set(asset_characteristics.index)
    )
    assets.sort()

    result = {}
    for fc in qv_values.index:
        qv = qv_values.loc[fc, assets].values.astype(float)
        row = {}
        for char in asset_characteristics.columns:
            cv = asset_characteristics.loc[assets, char].values.astype(
                float
            )
            mask = np.isfinite(qv) & np.isfinite(cv)
            if mask.sum() < 3:
                row[char] = np.nan
            else:
                row[char] = float(
                    np.corrcoef(qv[mask], cv[mask])[0, 1]
                )
        result[fc] = row
    return pd.DataFrame(result).T
