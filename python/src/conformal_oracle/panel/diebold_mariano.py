"""Panel Diebold-Mariano test with Driscoll-Kraay HAC."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _newey_west_auto_bandwidth(x: np.ndarray) -> int:
    """Andrews (1991) AR(1) automatic bandwidth selection."""
    n = len(x)
    if n < 3:
        return 0
    std = np.std(x)
    if std < 1e-15:
        return 0
    rho_hat = np.corrcoef(x[:-1], x[1:])[0, 1]
    if not np.isfinite(rho_hat):
        return 0
    rho_hat = np.clip(rho_hat, -0.99, 0.99)
    alpha_hat = (4 * rho_hat**2) / ((1 - rho_hat)**2 * (1 + rho_hat)**2)
    bw = int(np.ceil(1.1447 * (alpha_hat * n) ** (1 / 3)))
    return min(bw, n - 1)


def _bartlett_hac_variance(x: np.ndarray, bandwidth: int) -> float:
    """HAC variance estimate with Bartlett kernel."""
    n = len(x)
    x_dm = x - x.mean()
    gamma_0 = float(np.sum(x_dm**2)) / n
    hac = gamma_0
    for k in range(1, bandwidth + 1):
        w = 1.0 - k / (bandwidth + 1)
        gamma_k = float(np.sum(x_dm[k:] * x_dm[:-k])) / n
        hac += 2.0 * w * gamma_k
    return hac / n


def panel_dm_test(
    qs_sequences: dict[str, dict[str, np.ndarray]],
    forecaster_a: str,
    forecaster_b: str,
    bandwidth: int | None = None,
) -> tuple[float, float]:
    """Panel DM test with Driscoll-Kraay HAC.

    1. Per-asset, per-step score differences
    2. Cross-sectional sum S_t at each date
    3. HAC variance with Bartlett kernel
    4. HLN small-sample correction
    """
    assets = sorted(
        set(qs_sequences[forecaster_a].keys())
        & set(qs_sequences[forecaster_b].keys())
    )

    min_len = min(
        len(qs_sequences[forecaster_a][a])
        for a in assets
    )
    min_len = min(
        min_len,
        min(len(qs_sequences[forecaster_b][a]) for a in assets),
    )

    # Cross-sectional sum of differences at each t
    S_t = np.zeros(min_len)
    for asset in assets:
        seq_a = qs_sequences[forecaster_a][asset][-min_len:]
        seq_b = qs_sequences[forecaster_b][asset][-min_len:]
        S_t += seq_a - seq_b

    d_bar = S_t.mean()
    T = len(S_t)

    if bandwidth is None:
        bandwidth = _newey_west_auto_bandwidth(S_t)

    hac_var = _bartlett_hac_variance(S_t, bandwidth)

    if hac_var <= 0:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(hac_var)

    # HLN small-sample correction
    dm_stat *= np.sqrt((T - 1) / T)

    p_value = float(2.0 * stats.norm.sf(abs(dm_stat)))

    return float(dm_stat), p_value
