"""Wild-cluster bootstrap for panel Kupiec and DM tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class WildClusterBootstrapResult:
    """Wild-cluster bootstrap results for Kupiec and DM tests."""

    kupiec_table: pd.DataFrame
    dm_table: pd.DataFrame
    n_bootstrap: int
    n_clusters: int
    seed: int

    def kupiec_latex(self) -> str:
        from conformal_oracle.panel.latex import (
            wildcluster_kupiec_to_latex,
        )
        return wildcluster_kupiec_to_latex(self.kupiec_table)

    def dm_latex(self) -> str:
        from conformal_oracle.panel.latex import (
            wildcluster_dm_to_latex,
        )
        return wildcluster_dm_to_latex(self.dm_table)


def _kupiec_lr(violations: np.ndarray, alpha: float) -> float:
    """Kupiec LR statistic on a single violation array."""
    N = len(violations)
    X = int(np.sum(violations))
    if X == 0 or X == N:
        return 0.0
    pi_hat = X / N
    lr = -2.0 * (
        (N - X) * np.log((1 - alpha) / (1 - pi_hat))
        + X * np.log(alpha / pi_hat)
    )
    return max(lr, 0.0)


def wild_cluster_bootstrap_kupiec(
    violations: dict[str, np.ndarray],
    alpha: float,
    B: int = 999,
    seed: int = 2026,
) -> dict[str, float]:
    """Rademacher wild-cluster bootstrap for panel-pooled Kupiec.

    Clusters = assets. Each bootstrap draw multiplies each
    cluster's centred violations by w_j in {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    assets = sorted(violations.keys())
    J = len(assets)

    all_viol = np.concatenate([violations[a] for a in assets])
    lr_orig = _kupiec_lr(all_viol, alpha)

    pi_hat = float(np.mean(all_viol))
    centred = {
        a: violations[a] - pi_hat for a in assets
    }

    boot_lr = np.empty(B)
    for b in range(B):
        weights = rng.choice([-1, 1], size=J)
        boot_viol = []
        for j, a in enumerate(assets):
            resampled = pi_hat + weights[j] * centred[a]
            resampled = np.clip(resampled, 0.0, 1.0)
            boot_viol.append(resampled)
        boot_all = np.concatenate(boot_viol)
        boot_lr[b] = _kupiec_lr(
            (boot_all > 0.5).astype(int), alpha,
        )

    p_boot = float(np.mean(boot_lr >= lr_orig))

    return {
        "lr_original": lr_orig,
        "p_asymptotic": float(stats.chi2.sf(lr_orig, df=1)),
        "p_bootstrap": p_boot,
        "boot_mean_lr": float(np.mean(boot_lr)),
        "boot_q95_lr": float(np.quantile(boot_lr, 0.95)),
    }


def wild_cluster_bootstrap_dm(
    qs_differences: dict[tuple[str, str], dict[str, np.ndarray]],
    B: int = 999,
    seed: int = 2026,
) -> dict[tuple[str, str], float]:
    """Wild-cluster bootstrap p-values for pairwise DM tests.

    Each key in qs_differences is (forecaster_a, forecaster_b),
    values are dicts of asset -> per-step QS difference arrays.
    """
    rng = np.random.default_rng(seed)
    results: dict[tuple[str, str], float] = {}

    for pair, asset_diffs in qs_differences.items():
        assets = sorted(asset_diffs.keys())
        J = len(assets)

        min_len = min(len(asset_diffs[a]) for a in assets)

        # Original DM statistic (cross-sectional sum)
        S_t = np.zeros(min_len)
        for a in assets:
            S_t += asset_diffs[a][-min_len:]

        d_bar = S_t.mean()
        T = len(S_t)
        var_S = float(np.var(S_t, ddof=1))
        if var_S <= 0:
            results[pair] = 1.0
            continue

        t_orig = abs(d_bar / np.sqrt(var_S / T))

        # Per-asset mean differences (for resampling)
        asset_means = {
            a: asset_diffs[a][-min_len:] for a in assets
        }
        boot_t = np.empty(B)
        for b in range(B):
            weights = rng.choice([-1, 1], size=J)
            S_boot = np.zeros(min_len)
            for j, a in enumerate(assets):
                centred = asset_means[a] - asset_means[a].mean()
                S_boot += weights[j] * centred
            d_boot = S_boot.mean()
            var_boot = float(np.var(S_boot, ddof=1))
            if var_boot <= 0:
                boot_t[b] = 0.0
            else:
                boot_t[b] = abs(d_boot / np.sqrt(var_boot / T))

        results[pair] = float(np.mean(boot_t >= t_orig))

    return results
