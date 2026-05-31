"""
run_drift_diagnostic.py
=======================
Distributional drift diagnostic via empirical total variation distance
on rolling conformal scores (Lag-Llama, S&P 500).

Produces:
  - fig_drift_diagnostic.pdf / .png  (Figure 7 in paper)

Paper reference: §3.2.3 — Rolling Coverage Guarantee (Proposition 3.2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ks_2samp
from pathlib import Path

# ── Configuration ──
ALPHA = 0.01
W = 250
F_CAL = 0.70
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "cfp_ijf_data"
OUT_DIR = SCRIPT_DIR


def empirical_tv(x, y, n_bins=None):
    """Empirical total variation distance via histogram binning."""
    if n_bins is None:
        n_bins = int(np.sqrt(len(x) + len(y)))
    all_vals = np.concatenate([x, y])
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    p1, _ = np.histogram(x, bins=bins, density=True)
    p2, _ = np.histogram(y, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    return 0.5 * np.sum(np.abs(p1 - p2)) * bin_width


def main():
    # ── Load data ──
    returns = pd.read_csv(DATA_DIR / "returns" / "SP500.csv",
                          index_col=0, parse_dates=True)
    forecasts = pd.read_parquet(DATA_DIR / "lagllama" / "SP500.parquet")

    common_idx = returns.index.intersection(forecasts.index)
    r = returns.loc[common_idx, "log_return"]
    var_01 = forecasts.loc[common_idx, "VaR_0.01"]

    print(f"Aligned series: {len(r)} observations")
    print(f"Date range: {r.index[0].date()} to {r.index[-1].date()}")

    # ── Conformity scores ──
    scores = var_01.values - r.values
    dates = r.index

    print(f"\nScore statistics:")
    print(f"  Mean:   {np.mean(scores):.6f}")
    print(f"  Std:    {np.std(scores):.6f}")
    print(f"  Min:    {np.min(scores):.6f}")
    print(f"  Max:    {np.max(scores):.6f}")

    # ── Compute δ̂_w(t) via empirical TV and KS test ──
    n = len(scores)
    n_bins = int(np.sqrt(W))
    delta_hat_tv = np.full(n, np.nan)
    ks_pvals = np.full(n, np.nan)

    for t in range(W, n):
        window = scores[t - W : t]
        half = W // 2
        first_half = window[:half]
        second_half = window[half:]
        delta_hat_tv[t] = empirical_tv(first_half, second_half, n_bins=n_bins)
        _, pval = ks_2samp(first_half, second_half)
        ks_pvals[t] = pval

    drift_dates = dates[W:]
    drift_vals = delta_hat_tv[W:]
    drift_pvals = ks_pvals[W:]

    print(f"\nDrift diagnostic computed for {len(drift_vals)} windows "
          f"({n_bins} bins)")
    print(f"  Mean δ̂_w(t):   {np.nanmean(drift_vals):.4f}")
    print(f"  Max  δ̂_w(t):   {np.nanmax(drift_vals):.4f}")
    print(f"  Windows with KS p < 0.05: "
          f"{np.sum(drift_pvals < 0.05)} / {len(drift_pvals)} "
          f"({100*np.mean(drift_pvals < 0.05):.1f}%)")

    # ── Key events ──
    events = {
        "GFC":      ("2008-09-15", "2009-03-09"),
        "COVID-19": ("2020-02-20", "2020-04-30"),
        "SVB":      ("2023-03-08", "2023-03-20"),
    }

    covid_mask = (drift_dates >= "2020-02-01") & (drift_dates <= "2020-06-30")
    if covid_mask.any():
        peak_idx = np.argmax(drift_vals[covid_mask])
        peak_date = drift_dates[covid_mask][peak_idx]
        peak_val = drift_vals[covid_mask][peak_idx]
        print(f"\nPeak drift around COVID: δ̂ = {peak_val:.4f} "
              f"on {peak_date.date()}")

    # ── Figure: Drift diagnostic ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [1.2, 1]},
                                    sharex=True)
    fig.patch.set_alpha(0.0)

    # Panel A: empirical TV distance
    ax1.plot(drift_dates, drift_vals, color="#1A3A6E", lw=0.8, alpha=0.9)
    ax1.axhline(0.10, color="#CD0000", ls="--", lw=1.0, alpha=0.7,
                label="Warning threshold (0.10)")

    colors_ev = {"GFC": "#E07B39", "COVID-19": "#CD0000", "SVB": "#8E44AD"}
    for name, (start, end) in events.items():
        ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color=colors_ev[name], label=name)

    ax1.set_ylabel(r"$\hat{\delta}_w(t)$ (empirical TV)", fontsize=18)
    ax1.set_title(r"Distributional Drift Diagnostic: $\hat{\delta}_w(t)$ "
                  f"(Lag-Llama, S&P 500, w={W})",
                  fontweight="bold", fontsize=22)
    ax1.set_ylim(0, min(0.65, np.nanmax(drift_vals) * 1.15))
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: KS p-value
    ax2.scatter(drift_dates, drift_pvals, s=1.5, c="#1A3A6E", alpha=0.4)
    ax2.axhline(0.05, color="#CD0000", ls="--", lw=1.0, alpha=0.7,
                label="p = 0.05")
    for name, (start, end) in events.items():
        ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color=colors_ev[name])

    ax2.set_ylabel("KS p-value", fontsize=18)
    ax2.set_xlabel("Date", fontsize=18)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(axis="y", alpha=0.3)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=14, frameon=False)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    FIG_DIR = SCRIPT_DIR.parent.parent / "figures"
    SLIDE_DIR = SCRIPT_DIR.parent.parent / "ICFS 2026"
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_drift_diagnostic.{ext}",
                    bbox_inches="tight", dpi=600, transparent=True)
        fig.savefig(FIG_DIR / f"fig_drift_diagnostic.{ext}",
                    bbox_inches="tight", dpi=600, transparent=True)
        fig.savefig(SLIDE_DIR / f"fig_drift_diagnostic.{ext}",
                    bbox_inches="tight", dpi=600, transparent=True)
    plt.close(fig)
    print("\nSaved: fig_drift_diagnostic.pdf/.png")

    # ── Summary statistics by regime ──
    regimes = {
        "Full sample":   (drift_dates[0], drift_dates[-1]),
        "Pre-GFC":       ("2003-01-01", "2008-09-14"),
        "GFC":           ("2008-09-15", "2009-03-09"),
        "Post-GFC calm": ("2013-01-01", "2019-12-31"),
        "COVID-19":      ("2020-02-20", "2020-04-30"),
        "Post-COVID":    ("2021-01-01", "2023-01-01"),
        "SVB crisis":    ("2023-03-08", "2023-03-31"),
    }

    print(f'\n{"Regime":<20s} {"Mean δ̂":>10s} {"Max δ̂":>10s} '
          f'{"KS p<.05":>10s} {"N":>6s}')
    print("-" * 60)
    for name, (s, e) in regimes.items():
        mask = ((drift_dates >= pd.Timestamp(s)) &
                (drift_dates <= pd.Timestamp(e)))
        if mask.any():
            vals = drift_vals[mask]
            pvals = drift_pvals[mask]
            print(f"{name:<20s} {np.mean(vals):10.4f} {np.max(vals):10.4f} "
                  f"{100*np.mean(pvals < 0.05):9.1f}% {len(vals):6d}")

    # ── Implication for coverage bound ──
    Delta_w = np.sqrt(np.log(W) / W)
    print(f"\nTheoretical parameters (w = {W}):")
    print(f"  Δ_w (β-mixing remainder): {Delta_w:.4f}")
    print(f"  α = {ALPHA}")

    calm_delta = np.nanmean(
        drift_vals[(drift_dates >= "2013-01-01") &
                   (drift_dates <= "2019-12-31")])
    covid_delta = np.nanmax(
        drift_vals[(drift_dates >= "2020-02-01") &
                   (drift_dates <= "2020-06-30")])

    print(f"\nCoverage lower bound (calm, δ̂ ≈ {calm_delta:.3f}):")
    print(f"  >= 1 - {ALPHA} - {calm_delta:.3f} - {Delta_w:.3f} = "
          f"{1 - ALPHA - calm_delta - Delta_w:.3f} "
          f"({100*(1 - ALPHA - calm_delta - Delta_w):.1f}%)")

    print(f"\nCoverage lower bound (COVID peak, δ̂ ≈ {covid_delta:.3f}):")
    print(f"  >= 1 - {ALPHA} - {covid_delta:.3f} - {Delta_w:.3f} = "
          f"{1 - ALPHA - covid_delta - Delta_w:.3f} "
          f"({100*(1 - ALPHA - covid_delta - Delta_w):.1f}%)")

    print("\nEmpirical rolling coverage from Table 5: 99.1% >> theoretical "
          "bound,")
    print("confirming the bound is conservative but informative about "
          "regime shifts.")


if __name__ == "__main__":
    main()
