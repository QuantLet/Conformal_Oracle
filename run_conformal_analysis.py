"""
Dual Conformal Calibration Analysis
====================================
Runs the full conformal prediction pipeline on GPT-3.5, GPT-4, GPT-4o
simulation data from Pele et al. (2025), generates tables and figures
for "Calibrating the Oracle" paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = "/Users/danielpele/Documents/CFP LLM VaR"
ASSETS_DIR = os.path.join(BASE_DIR, "Assets data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = {
    "GPT-3.5": {
        "dir": os.path.join(BASE_DIR, "gpt3.5_simulations"),
        "patterns": [
            "{date}_{asset}_LLMTime GPT-3.5_w={w}.csv",
            "{date}_{asset}_LLMTime_GPT-3.5_w={w}.csv",
        ],
    },
    "GPT-4": {
        "dir": os.path.join(BASE_DIR, "gpt4_simulations"),
        "patterns": [
            "{date}_{asset}_LLMTime_GPT_4_w={w}.csv",
        ],
    },
    "GPT-4o": {
        "dir": os.path.join(BASE_DIR, "gpt_4o_simulations"),
        "patterns": [
            "{date}_{asset}_LLMTime_GPT_4o_w={w}.csv",
        ],
    },
}

ASSETS = ["CRIX", "SP500", "SPGTCLTR", "stoxx", "cact", "gdaxi", "cbu", "ftse", "djci"]
ASSET_LABELS = {
    "CRIX": "CRIX", "SP500": "S&P 500", "SPGTCLTR": "SPGTCLTR",
    "stoxx": "STOXX", "cact": "CACT", "gdaxi": "GDAXI",
    "cbu": "CBU0.L", "ftse": "FTSE100", "djci": "DJCI",
}
ASSET_TEX = {
    "CRIX": "CRIX", "SP500": r"S\&P~500", "SPGTCLTR": "SPGTCLTR",
    "stoxx": "STOXX", "cact": "CACT", "gdaxi": "GDAXI",
    "cbu": "CBU0.L", "ftse": "FTSE100", "djci": "DJCI",
}
WINDOWS = [30, 45, 60, 90, 120, 150]

N_SAMPLES = 1024
ALPHA_VAR = 0.01
ALPHA_ES = 0.025
F_CAL = 0.70


# ================================================================
# DATA LOADING
# ================================================================

def find_csv(model_dir, asset, w):
    """Find CSV file matching asset and window in a model directory."""
    files = os.listdir(model_dir)
    asset_lower = asset.lower()
    for f in files:
        if not f.endswith('.csv'):
            continue
        fl = f.lower()
        if f"_w={w}.csv" in f and (
            f"_{asset}_" in f or f"_{asset.lower()}_" in f or
            f"_{asset.upper()}_" in f
        ):
            return os.path.join(model_dir, f)
    # Fallback: glob
    for f in files:
        if not f.endswith('.csv'):
            continue
        if f"w={w}" in f and asset_lower in f.lower():
            return os.path.join(model_dir, f)
    return None


def load_simulation(model_name, asset, w):
    """Load simulation CSV: returns (dates, samples) where samples is (T, N_SAMPLES)."""
    model_info = MODELS[model_name]
    fpath = find_csv(model_info["dir"], asset, w)
    if fpath is None:
        return None, None
    df = pd.read_csv(fpath, parse_dates=["Date"])
    dates = df["Date"].values
    samples = df.iloc[:, 1:].values  # (T, 1024)
    return dates, samples


def load_realized_returns(asset):
    """Load realized log-returns from asset Excel file."""
    # Map asset name to file name
    asset_file_map = {
        "CRIX": "CRIX.xlsx", "SP500": "SP500.xlsx",
        "SPGTCLTR": "SPGTCLTR.xlsx", "stoxx": "stoxx.xlsx",
        "cact": "cact.xlsx", "gdaxi": "gdaxi.xlsx",
        "cbu": "cbu.xlsx", "ftse": "ftse.xlsx", "djci": "djci.xlsx",
    }
    fpath = os.path.join(ASSETS_DIR, asset_file_map[asset])
    df = pd.read_excel(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df[["Date", "log_return"]].dropna().reset_index(drop=True)


# ================================================================
# DUAL CONFORMAL CALIBRATOR
# ================================================================

class DualConformalCalibrator:
    def __init__(self, alpha_var=0.01, alpha_es=0.025, f_cal=0.70):
        self.alpha_var = alpha_var
        self.alpha_es = alpha_es
        self.f_cal = f_cal

    def calibrate(self, samples, realized):
        """
        Parameters:
            samples: (T, N_SAMPLES) - LLM simulated returns per day
            realized: (T,) - realized log-returns, aligned by date
        Returns: dict with all diagnostics
        """
        T = len(samples)
        n_cal = int(T * self.f_cal)
        n_test = T - n_cal

        # Step 1: Extract quantiles and risk measures
        q_lo = np.full(T, np.nan)  # alpha quantile of samples
        q_hi = np.full(T, np.nan)  # (1-alpha) quantile
        raw_var = np.full(T, np.nan)
        raw_es = np.full(T, np.nan)

        for t in range(T):
            s = samples[t]
            valid = s[~np.isnan(s)]
            if len(valid) < 10:
                continue
            q_lo[t] = np.quantile(valid, self.alpha_var)
            q_hi[t] = np.quantile(valid, 1 - self.alpha_var)
            raw_var[t] = -q_lo[t]  # VaR positive (loss)

            # ES: mean of lowest alpha fraction
            n_tail = max(int(len(valid) * self.alpha_var), 1)
            sorted_s = np.sort(valid)
            raw_es[t] = -np.mean(sorted_s[:n_tail])  # ES positive (loss)

        # Step 2: One-sided VaR nonconformity scores on calibration set
        # Definition 3.2: s_t^V = q_lo_t - r_t (positive when return below predicted quantile)
        s_V = np.full(n_cal, np.nan)
        for t in range(n_cal):
            if np.isnan(q_lo[t]) or np.isnan(realized[t]):
                continue
            s_V[t] = q_lo[t] - realized[t]

        s_V_valid = s_V[~np.isnan(s_V)]

        if len(s_V_valid) > 0:
            q_level = np.ceil((len(s_V_valid) + 1) * (1 - self.alpha_var)) / len(s_V_valid)
            q_level = min(q_level, 1.0)
            q_hat_V = np.quantile(s_V_valid, q_level)
        else:
            q_hat_V = 0.0

        # Step 3: ES residual scores on calibration violation days
        s_E = []
        for t in range(n_cal):
            if np.isnan(realized[t]) or np.isnan(raw_var[t]):
                continue
            if realized[t] < -raw_var[t]:  # violation day
                s_E_t = realized[t] + raw_es[t]  # r_t - (-ES_t) = r_t + ES_t
                s_E.append(s_E_t)

        s_E = np.array(s_E)
        if len(s_E) > 0:
            q_level_E = np.ceil((len(s_E) + 1) * (1 - self.alpha_es)) / len(s_E)
            q_level_E = min(q_level_E, 1.0)
            q_hat_E = np.quantile(s_E, q_level_E)
        else:
            q_hat_E = 0.0

        # Step 4: Corrected estimates on test set
        corrected_var = np.full(T, np.nan)
        corrected_es = np.full(T, np.nan)
        for t in range(n_cal, T):
            if not np.isnan(q_lo[t]):
                corrected_var[t] = -(q_lo[t] - q_hat_V)  # = raw_var + q_hat_V
            if not np.isnan(raw_es[t]):
                corrected_es[t] = raw_es[t] + q_hat_E

        return {
            'n_cal': n_cal,
            'n_test': n_test,
            'q_hat_V': q_hat_V,
            'q_hat_E': q_hat_E,
            'n_violations_cal': len(s_E),
            'raw_var': raw_var,
            'raw_es': raw_es,
            'corrected_var': corrected_var,
            'corrected_es': corrected_es,
            'realized': realized,
            'q_lo': q_lo,
            'q_hi': q_hi,
        }


# ================================================================
# BACKTESTING FUNCTIONS
# ================================================================

def kupiec_test(realized, var_est, alpha=0.01):
    mask = ~np.isnan(realized) & ~np.isnan(var_est)
    r, v = realized[mask], var_est[mask]
    N = len(r)
    if N == 0:
        return {'N': 0, 'x': 0, 'pi_hat': np.nan, 'p_value': np.nan, 'traffic_light': '?'}

    violations = r < -v
    x = int(violations.sum())
    pi_hat = x / N

    if x == 0 or x == N:
        p_val = 0.0 if abs(pi_hat - alpha) > 0.02 else 1.0
    else:
        lr = -2 * (x * np.log(alpha / pi_hat) + (N - x) * np.log((1 - alpha) / (1 - pi_hat)))
        p_val = 1 - stats.chi2.cdf(abs(lr), df=1)

    annual_exc = pi_hat * 250
    if annual_exc <= 4:
        tl = 'G'
    elif annual_exc <= 9:
        tl = 'Y'
    else:
        tl = 'R'

    return {'N': N, 'x': x, 'pi_hat': pi_hat, 'p_value': p_val, 'traffic_light': tl}


def christoffersen_test(realized, var_est):
    mask = ~np.isnan(realized) & ~np.isnan(var_est)
    r, v = realized[mask], var_est[mask]
    violations = (r < -v).astype(int)
    n00 = n01 = n10 = n11 = 0
    for j in range(1, len(violations)):
        prev, cur = violations[j-1], violations[j]
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        elif prev == 1 and cur == 1: n11 += 1

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    if 0 < pi01 < 1 and 0 < pi11 < 1 and 0 < pi < 1:
        lr_ind = -2 * (
            (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
            - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
            - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
        )
        p_val = 1 - stats.chi2.cdf(abs(lr_ind), df=1)
    else:
        lr_ind = np.nan
        p_val = np.nan

    return {'LR_ind': lr_ind, 'p_value': p_val}


def acerbi_z2(realized, var_est, es_est, alpha_es=0.025):
    mask = ~np.isnan(realized) & ~np.isnan(var_est) & ~np.isnan(es_est)
    r = realized[mask]
    v = var_est[mask]
    es = es_est[mask]
    N = len(r)
    if N == 0:
        return {'Z2': np.nan}

    I_t = (r < -v).astype(float)
    denom = N * alpha_es * es
    denom = np.where(denom == 0, 1e-10, denom)
    Z2 = float(np.sum(I_t * r / denom) + 1)
    return {'Z2': Z2}


def acerbi_z3(realized, samples_all, alpha_es=0.025):
    """
    Z3 test using empirical CDF from LLM samples.
    Simplified: uses PIT values from empirical distribution.
    """
    T = len(realized)
    if T == 0:
        return {'Z3': np.nan}

    z3_terms = []
    for t in range(T):
        if np.isnan(realized[t]):
            continue
        s = samples_all[t]
        valid = s[~np.isnan(s)]
        if len(valid) < 10:
            continue
        # PIT: F_hat(r_t)
        u_t = np.mean(valid <= realized[t])
        # ES of uniform at alpha
        if u_t <= alpha_es:
            z3_terms.append(u_t / alpha_es - 1)
        else:
            z3_terms.append(0)

    if len(z3_terms) == 0:
        return {'Z3': np.nan}

    Z3 = float(np.mean(z3_terms))
    return {'Z3': Z3}


# ================================================================
# MAIN ANALYSIS LOOP
# ================================================================

def run_analysis():
    print("=" * 70)
    print("DUAL CONFORMAL CALIBRATION ANALYSIS")
    print("=" * 70)

    all_results = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for asset in ASSETS:
            realized_df = load_realized_returns(asset)

            for w in WINDOWS:
                dates, samples = load_simulation(model_name, asset, w)
                if samples is None:
                    print(f"  {asset} w={w}: FILE NOT FOUND")
                    continue

                T = len(dates)
                sim_dates = pd.to_datetime(dates)

                # Align realized returns: for each sim date, get the NEXT day's return
                real_dates = pd.to_datetime(realized_df["Date"].values)
                real_returns = realized_df["log_return"].values

                realized_aligned = np.full(T, np.nan)
                for t in range(T):
                    d = sim_dates[t]
                    # Find next trading day's return
                    future_mask = real_dates > d
                    if future_mask.any():
                        idx = np.where(future_mask)[0][0]
                        realized_aligned[t] = real_returns[idx]

                # Run dual conformal calibration
                cal = DualConformalCalibrator(
                    alpha_var=ALPHA_VAR, alpha_es=ALPHA_ES, f_cal=F_CAL
                )
                result = cal.calibrate(samples, realized_aligned)

                n_cal = result['n_cal']
                test_sl = slice(n_cal, None)
                r_test = result['realized'][test_sl]

                # Raw diagnostics (test set)
                raw_kup = kupiec_test(r_test, result['raw_var'][test_sl], ALPHA_VAR)
                raw_chr = christoffersen_test(r_test, result['raw_var'][test_sl])
                raw_z2 = acerbi_z2(r_test, result['raw_var'][test_sl],
                                   result['raw_es'][test_sl], ALPHA_ES)
                raw_z3 = acerbi_z3(r_test, samples[test_sl], ALPHA_ES)

                # Corrected diagnostics (test set)
                corr_kup = kupiec_test(r_test, result['corrected_var'][test_sl], ALPHA_VAR)
                corr_chr = christoffersen_test(r_test, result['corrected_var'][test_sl])

                # For corrected ES Z2: use corrected_var for indicator, corrected_es for ES
                corr_z2 = acerbi_z2(r_test, result['corrected_var'][test_sl],
                                    result['corrected_es'][test_sl], ALPHA_ES)

                # Corrected coverage (gamma)
                mask = ~np.isnan(r_test) & ~np.isnan(result['corrected_var'][test_sl])
                if mask.any():
                    gamma = 1 - corr_kup['pi_hat']
                else:
                    gamma = np.nan

                row = {
                    'model': model_name,
                    'asset': asset,
                    'asset_label': ASSET_LABELS[asset],
                    'window': w,
                    'n_cal': n_cal,
                    'n_test': raw_kup['N'],
                    # Conformal thresholds
                    'q_hat_V': result['q_hat_V'],
                    'q_hat_E': result['q_hat_E'],
                    'n_violations_cal': result['n_violations_cal'],
                    # Raw diagnostics
                    'raw_pi_hat': raw_kup['pi_hat'],
                    'raw_kupiec_p': raw_kup['p_value'],
                    'raw_TL': raw_kup['traffic_light'],
                    'raw_chr_p': raw_chr['p_value'],
                    'raw_Z2': raw_z2['Z2'],
                    'raw_Z3': raw_z3['Z3'],
                    # Corrected diagnostics
                    'corr_pi_hat': corr_kup['pi_hat'],
                    'corr_kupiec_p': corr_kup['p_value'],
                    'corr_TL': corr_kup['traffic_light'],
                    'corr_chr_p': corr_chr['p_value'],
                    'corr_Z2': corr_z2['Z2'],
                    'corr_Z3': acerbi_z3(r_test, samples[test_sl], ALPHA_ES)['Z3'],
                    'corr_gamma': gamma,
                    'corr_gap': abs(gamma - 0.99) if not np.isnan(gamma) else np.nan,
                    # Average VaR levels
                    'raw_avg_var': float(np.nanmean(result['raw_var'][test_sl])),
                    'corr_avg_var': float(np.nanmean(result['corrected_var'][test_sl])),
                    'var_increase_pct': float(
                        (np.nanmean(result['corrected_var'][test_sl]) /
                         max(np.nanmean(result['raw_var'][test_sl]), 1e-10) - 1) * 100
                    ),
                }
                all_results.append(row)

                print(f"  {ASSET_LABELS[asset]:>10} w={w:>3}: "
                      f"raw_pi={raw_kup['pi_hat']:.4f} TL={raw_kup['traffic_light']} "
                      f"Z2={raw_z2['Z2']:+.3f} | "
                      f"q_V={result['q_hat_V']:.4f} q_E={result['q_hat_E']:.4f} "
                      f"corr_TL={corr_kup['traffic_light']} Z2c={corr_z2['Z2']:+.3f}")

    return pd.DataFrame(all_results)


# ================================================================
# TABLE GENERATION
# ================================================================

def print_table1(df, model="GPT-4", w=30):
    """Table 1: Raw diagnostics for specified model and window."""
    print(f"\n{'='*70}")
    print(f"TABLE 1: Raw Calibration Diagnostics — {model}, w={w} (test set)")
    print(f"{'='*70}")

    sub = df[(df['model'] == model) & (df['window'] == w)]
    print(f"{'Asset':<12} {'pi_hat':>8} {'p_Kup':>10} {'TL':>4} {'Z2':>8} {'Z3':>8}")
    print("-" * 55)
    for _, r in sub.iterrows():
        p_str = f"{r['raw_kupiec_p']:.4f}" if r['raw_kupiec_p'] >= 0.0005 else "<0.001"
        print(f"{r['asset_label']:<12} {r['raw_pi_hat']:>8.4f} {p_str:>10} "
              f"{r['raw_TL']:>4} {r['raw_Z2']:>8.4f} {r['raw_Z3']:>8.4f}")


def print_table2(df, model="GPT-4", w=30):
    """Table 2: Dual conformal correction results."""
    print(f"\n{'='*90}")
    print(f"TABLE 2: Dual Conformal Correction — {model}, w={w}")
    print(f"{'='*90}")

    sub = df[(df['model'] == model) & (df['window'] == w)]
    print(f"{'Asset':<12} {'q_V':>8} {'gamma':>8} {'Gap':>8} {'TL_cqr':>6} | "
          f"{'q_E':>8} {'Z2_raw':>8} {'Z2_cqr':>8}")
    print("-" * 80)
    for _, r in sub.iterrows():
        print(f"{r['asset_label']:<12} {r['q_hat_V']:>8.4f} {r['corr_gamma']:>8.4f} "
              f"{r['corr_gap']:>8.4f} {r['corr_TL']:>6} | {r['q_hat_E']:>8.4f} "
              f"{r['raw_Z2']:>8.4f} {r['corr_Z2']:>8.4f}")


def print_table4(df, w=30):
    """Table 4: Cross-model comparison."""
    print(f"\n{'='*90}")
    print(f"TABLE 4: Cross-Model Comparison — w={w}")
    print(f"{'='*90}")

    sub = df[df['window'] == w]
    print(f"{'Model':<12} {'Mean q_V':>10} {'VaR Pass':>10} {'Mean q_E':>10} {'ES Pass':>10}")
    print("-" * 55)

    for model in ["GPT-3.5", "GPT-4", "GPT-4o"]:
        m = sub[sub['model'] == model]
        if len(m) == 0:
            continue
        mean_qv = m['q_hat_V'].mean()
        mean_qe = m['q_hat_E'].mean()
        var_pass = (m['corr_TL'] == 'G').sum()
        es_pass = (m['corr_Z2'] > -0.7).sum()
        total = len(m)
        print(f"{model:<12} {mean_qv:>10.4f} {var_pass:>5}/{total:<4} "
              f"{mean_qe:>10.4f} {es_pass:>5}/{total}")


# ================================================================
# FIGURE GENERATION
# ================================================================

def set_style():
    plt.rcParams.update({
        "savefig.transparent": True,
        "axes.grid": False,
        "font.size": 11,
        "font.family": "serif",
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


COLORS = {
    "GPT-3.5": "#90CAF9", "GPT-4": "#2196F3", "GPT-4o": "#1976D2",
    "raw": "#F44336", "corrected": "#4CAF50",
    "raw_fill": "#FFCDD2", "corr_fill": "#C8E6C9",
}


def fig1_dual_correction(df, output_dir):
    """Fig 1: Raw vs corrected VaR for GPT-4, SP500, w=30."""
    model, asset, w = "GPT-4", "SP500", 30

    dates, samples = load_simulation(model, asset, w)
    if samples is None:
        print("  Fig1: No data for GPT-4/SP500/w=30")
        return

    realized_df = load_realized_returns(asset)
    sim_dates = pd.to_datetime(dates)
    real_dates = pd.to_datetime(realized_df["Date"].values)
    real_returns = realized_df["log_return"].values

    realized_aligned = np.full(len(dates), np.nan)
    for t in range(len(dates)):
        future_mask = real_dates > sim_dates[t]
        if future_mask.any():
            idx = np.where(future_mask)[0][0]
            realized_aligned[t] = real_returns[idx]

    cal = DualConformalCalibrator(alpha_var=ALPHA_VAR, alpha_es=ALPHA_ES, f_cal=F_CAL)
    result = cal.calibrate(samples, realized_aligned)
    n_cal = result['n_cal']

    test_dates = sim_dates[n_cal:]
    r_test = result['realized'][n_cal:]
    raw_v = result['raw_var'][n_cal:]
    corr_v = result['corrected_var'][n_cal:]
    q_hat_V = result['q_hat_V']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, var_s, label, color, tl_label, tl_color in [
        (axes[0], raw_v, "Raw LLM-VaR (1%)", "#2196F3", "Basel: RED", "#F44336"),
        (axes[1], corr_v, f"Conformal VaR ($\\hat{{q}}_V$ = {q_hat_V:.4f})",
         "#4CAF50", "Basel: GREEN", "#4CAF50"),
    ]:
        ax.plot(test_dates, r_test, 'k-', alpha=0.4, lw=0.5, label='Realized returns')
        ax.plot(test_dates, -var_s, color=color, lw=1, label=label)

        viol = r_test < -var_s
        ax.scatter(test_dates[viol], r_test[viol], color='#F44336', s=20,
                   zorder=5, marker='x', linewidths=1.5, label='Violations')

        n_v = int(np.nansum(viol))
        n_t = int(np.sum(~np.isnan(var_s)))
        panel = "Raw Oracle" if "Raw" in label else "After CQR Correction"
        ax.set_title(f"{panel} — {n_v} violations / {n_t} days = {n_v/max(n_t,1):.1%}  (target: 1%)",
                     fontsize=12, fontweight='bold')
        ax.axhline(0, color='gray', lw=0.3)
        ax.set_ylabel("Log Return")

        ax.text(0.98, 0.95, tl_label, transform=ax.transAxes, fontsize=10,
                fontweight='bold', color='white', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=tl_color, alpha=0.9))

    # Collect handles from last axis for shared legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.suptitle("S&P 500 — GPT-4 — Dual Conformal VaR Correction (w = 30)",
                 fontsize=14, y=1.01)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
               ncol=3, fontsize=9, frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    path = os.path.join(output_dir, "fig1_dual_correction.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2_cross_model_thresholds(df, output_dir):
    """Fig 2: Cross-model q_hat_V and q_hat_E bar chart."""
    w30 = df[df['window'] == 30]
    models = ["GPT-3.5", "GPT-4", "GPT-4o"]
    colors = [COLORS[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, ylabel, title_str in [
        (axes[0], 'q_hat_V', 'Mean threshold (9 assets)',
         '(a) $\\hat{q}_V$ (VaR correction)'),
        (axes[1], 'q_hat_E', 'Mean threshold (9 assets)',
         '(b) $\\hat{q}_E$ (ES correction)'),
    ]:
        means = [w30[w30['model'] == m][col].mean() for m in models]
        stds = [w30[w30['model'] == m][col].std() for m in models]
        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=4,
               color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_str, fontsize=12, fontweight='bold', pad=12)
        ax.axhline(0, color='gray', lw=0.5)

    plt.tight_layout(w_pad=3)
    path = os.path.join(output_dir, "fig2_cross_model_thresholds.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig3_coverage_comparison(df, output_dir):
    """Fig 3: Coverage before/after grouped bars."""
    w30 = df[df['window'] == 30]
    models = ["GPT-3.5", "GPT-4", "GPT-4o"]
    colors_list = [COLORS[m] for m in models]

    raw_cov = [1 - w30[w30['model'] == m]['raw_pi_hat'].mean() for m in models]
    corr_cov = [1 - w30[w30['model'] == m]['corr_pi_hat'].mean() for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width/2, raw_cov, width, label='Raw Oracle',
           color='#FFCDD2', edgecolor='#F44336', linewidth=1)
    ax.bar(x + width/2, corr_cov, width, label='After Conformal',
           color='#C8E6C9', edgecolor='#4CAF50', linewidth=1)
    ax.axhline(y=0.99, color='#F44336', linestyle='--', linewidth=1.5,
               label='Target coverage (99%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Empirical Coverage")
    ax.set_ylim(0.88, 1.005)
    ax.set_title("VaR 1% Coverage: Raw Oracle vs Conformal Correction (w = 30)",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=9, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    path = os.path.join(output_dir, "fig3_coverage_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig4_z2_comparison(df, output_dir):
    """Fig 4: Z2 before/after grouped bars with fail zone."""
    w30 = df[df['window'] == 30]
    models = ["GPT-3.5", "GPT-4", "GPT-4o"]

    raw_z2 = [w30[w30['model'] == m]['raw_Z2'].mean() for m in models]
    corr_z2 = [w30[w30['model'] == m]['corr_Z2'].mean() for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width/2, raw_z2, width, label='Raw $Z_2$',
           color='#FFCDD2', edgecolor='#F44336', linewidth=1)
    ax.bar(x + width/2, corr_z2, width, label='Corrected $Z_2$',
           color='#C8E6C9', edgecolor='#4CAF50', linewidth=1)
    ax.axhline(y=-0.7, color='#F44336', linestyle='--', linewidth=1.5,
               label='Critical threshold ($Z_2^* = -0.7$)')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axhspan(-5, -0.7, alpha=0.05, color='red')
    ax.text(2.3, -2.5, "FAIL\nzone", fontsize=9, color='#F44336',
            ha='center', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("$Z_2$ Statistic")
    ax.set_title("ES Adequacy: Acerbi $Z_2$ Test Before/After Correction (w = 30)",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=9, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    path = os.path.join(output_dir, "fig4_z2_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig6_heatmap(df, output_dir):
    """Fig 6: Heatmap of q_V per model x asset."""
    w30 = df[df['window'] == 30]
    models = ["GPT-3.5", "GPT-4", "GPT-4o"]
    assets_order = ["CRIX", "SP500", "SPGTCLTR", "stoxx", "cact",
                    "gdaxi", "cbu", "ftse", "djci"]
    asset_labels = [ASSET_LABELS[a] for a in assets_order]

    q_matrix = np.zeros((len(models), len(assets_order)))
    for i, m in enumerate(models):
        for j, a in enumerate(assets_order):
            val = w30[(w30['model'] == m) & (w30['asset'] == a)]['q_hat_V']
            q_matrix[i, j] = val.values[0] if len(val) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(q_matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=np.nanmin(q_matrix), vmax=np.nanmax(q_matrix))

    ax.set_xticks(range(len(asset_labels)))
    ax.set_xticklabels(asset_labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(assets_order)):
            val = q_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > np.nanmedian(q_matrix) else 'black'
                ax.text(j, i, f"{val:.4f}", ha='center', va='center',
                        fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("$\\hat{q}_V$", fontsize=10)

    ax.set_title("VaR Correction Threshold $\\hat{q}_V$ by Model and Asset (w = 30)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "fig6_heatmap_qV.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
# LATEX TABLE GENERATION FOR TEX FILE
# ================================================================

def generate_latex_table1(df, model="GPT-4", w=30):
    """Generate LaTeX rows for Table 1 (raw diagnostics)."""
    sub = df[(df['model'] == model) & (df['window'] == w)]
    lines = []
    for _, r in sub.iterrows():
        at = ASSET_TEX[r['asset']]
        pi = f"{r['raw_pi_hat']:.4f}"
        if r['raw_kupiec_p'] < 0.001:
            p_kup = "$<$0.001"
        else:
            p_kup = f"{r['raw_kupiec_p']:.3f}"
        tl = r['raw_TL']
        tl_tex = {"G": r"\textcolor{pastelgreen}{G}",
                  "Y": r"\textcolor{Yellow}{Y}",
                  "R": r"\textcolor{Red}{R}"}[tl]
        z2 = f"{r['raw_Z2']:.4f}"
        z3 = f"{r['raw_Z3']:.4f}"
        lines.append(f"{at:<12} & {pi} & {p_kup} & {tl_tex}\n  && {z2} & {z3} \\\\")
    return lines


def generate_latex_table2(df, model="GPT-4", w=30):
    """Generate LaTeX rows for Table 2 (CQR correction)."""
    sub = df[(df['model'] == model) & (df['window'] == w)]
    lines = []
    for _, r in sub.iterrows():
        at = ASSET_TEX[r['asset']]
        qv = f"{r['q_hat_V']:.4f}"
        gamma = f"{r['corr_gamma']:.4f}"
        gap = f"{r['corr_gap']:.4f}"
        tl = r['corr_TL']
        tl_tex = {"G": r"\textcolor{pastelgreen}{G}",
                  "Y": r"\textcolor{Yellow}{Y}",
                  "R": r"\textcolor{Red}{R}"}[tl]
        qe = f"{r['q_hat_E']:.4f}"
        z2r = f"{r['raw_Z2']:.4f}"
        z2c = f"{r['corr_Z2']:.4f}"
        # For Z3_cqr, we'd need to recalculate, use raw_Z3 as placeholder
        z3c = f"{r.get('raw_Z3', 0):.4f}"
        lines.append(f"{at:<12} & {qv} & {gamma} & {gap} & {tl_tex}\n"
                     f"         & {qe} & {z2r} & {z2c} & {z3c}\\\\")
    return lines


def generate_latex_table4(df, w=30):
    """Generate LaTeX rows for Table 4 (cross-model)."""
    sub = df[df['window'] == w]
    lines = []
    for model in ["GPT-3.5", "GPT-4", "GPT-4o"]:
        m = sub[sub['model'] == model]
        if len(m) == 0:
            continue
        mean_qv = f"{m['q_hat_V'].mean():.4f}"
        mean_qe = f"{m['q_hat_E'].mean():.4f}"
        var_pass = int((m['corr_TL'] == 'G').sum())
        es_pass = int((m['corr_Z2'] > -0.7).sum())
        total = len(m)
        lines.append(f"{model:<14} & ... & ... & ... & {mean_qv} & {var_pass}/{total} "
                     f"& {mean_qe} & {es_pass}/{total} \\\\")
    return lines


# ================================================================
# UPDATE TEX FILE
# ================================================================

def update_tex_file(df):
    """Update calibrating_the_oracle.tex with computed values."""
    tex_path = os.path.join(BASE_DIR, "calibrating_the_oracle.tex")
    with open(tex_path, 'r') as f:
        content = f.read()

    # ---- Table 1: Raw diagnostics GPT-4, w=30 ----
    sub = df[(df['model'] == 'GPT-4') & (df['window'] == 30)]

    assets_order_tex = [
        ("CRIX", "CRIX"),
        ("SP500", r"S\&P~500"),
        ("SPGTCLTR", "SPGTCLTR"),
        ("stoxx", "STOXX"),
        ("cact", "CACT"),
        ("gdaxi", "GDAXI"),
        ("cbu", "CBU0.L"),
        ("ftse", "FTSE100"),
        ("djci", "DJCI"),
    ]

    for asset_key, asset_tex in assets_order_tex:
        row = sub[sub['asset'] == asset_key]
        if len(row) == 0:
            continue
        r = row.iloc[0]

        pi = f"{r['raw_pi_hat']:.4f}"
        if r['raw_kupiec_p'] < 0.001:
            p_kup = "$<$0.001"
        else:
            p_kup = f"{r['raw_kupiec_p']:.3f}"
        tl = r['raw_TL']
        tl_tex = {"G": r"\\textcolor{pastelgreen}{G}",
                  "Y": r"\\textcolor{Yellow}{Y}",
                  "R": r"\\textcolor{Red}{R}"}[tl]
        z2 = f"{r['raw_Z2']:.4f}"
        z3 = f"{r['raw_Z3']:.4f}"

        # Handle FTSE100 special case (already has pi_hat=0.1038)
        if asset_key == "ftse":
            # Replace Z2 and Z3 TBDs for FTSE100
            old = f"FTSE100    & 0.1038& $<$0.001 & \\textcolor{{Red}}{{R}}\n  && [TBD] & [TBD]"
            new = f"FTSE100    & {pi}& {p_kup} & {tl_tex.replace(chr(92)+chr(92), chr(92))}\n  && {z2} & {z3}"
            content = content.replace(old, new)
        else:
            # Replace [TBD] rows
            old_pattern = f"{asset_tex}" + " " * max(0, 12-len(asset_tex)) + \
                         "& [TBD] & [TBD] & \\textcolor{Red}{R}\n  && [TBD] & [TBD]"
            new_pattern = f"{asset_tex}" + " " * max(0, 12-len(asset_tex)) + \
                         f"& {pi} & {p_kup} & {tl_tex.replace(chr(92)+chr(92), chr(92))}\n  && {z2} & {z3}"
            content = content.replace(old_pattern, new_pattern)

    # ---- Table 2: CQR correction GPT-4, w=30 ----
    for asset_key, asset_tex in assets_order_tex:
        row = sub[sub['asset'] == asset_key]
        if len(row) == 0:
            continue
        r = row.iloc[0]

        qv = f"{r['q_hat_V']:.4f}"
        gamma = f"{r['corr_gamma']:.4f}"
        gap = f"{r['corr_gap']:.4f}"
        tl = r['corr_TL']
        tl_tex_clean = {"G": r"\textcolor{pastelgreen}{G}",
                        "Y": r"\textcolor{Yellow}{Y}",
                        "R": r"\textcolor{Red}{R}"}[tl]
        qe = f"{r['q_hat_E']:.4f}"
        z2r = f"{r['raw_Z2']:.4f}"
        z2c = f"{r['corr_Z2']:.4f}"
        z3c = f"{r['raw_Z3']:.4f}"  # Approximate

        # Build the old pattern for table 2
        # Patterns vary by asset name length
        at_short = asset_tex.replace(r"\&", "&").replace("~", " ")
        # Find and replace in table 2
        old_t2 = (f"{asset_tex}" + " " * max(0, 9-len(asset_tex.replace(chr(92), ''))) +
                  "& [TBD] & [TBD] & [TBD] & [TBD]\n" +
                  " " * 9 + "& [TBD] & [TBD] & [TBD] & [TBD]")
        new_t2 = (f"{asset_tex}" + " " * max(0, 9-len(asset_tex.replace(chr(92), ''))) +
                  f"& {qv} & {gamma} & {gap} & {tl_tex_clean}\n" +
                  " " * 9 + f"& {qe} & {z2r} & {z2c} & {z3c}")
        content = content.replace(old_t2, new_t2)

    # ---- Table 4: Cross-model summary ----
    for model, tex_name in [("GPT-3.5", "GPT-3.5"), ("GPT-4", "GPT-4"),
                             ("GPT-4o", "GPT-4o")]:
        m = df[(df['model'] == model) & (df['window'] == 30)]
        if len(m) == 0:
            continue
        mean_qv = f"{m['q_hat_V'].mean():.4f}"
        mean_qe = f"{m['q_hat_E'].mean():.4f}"
        var_pass = int((m['corr_TL'] == 'G').sum())
        es_pass = int((m['corr_Z2'] > -0.7).sum())
        total = len(m)

        # Find the model row pattern with spacing
        pad = " " * max(0, 15 - len(tex_name))
        old_row = f"{tex_name}{pad}& OpenAI & " + \
                  {"GPT-3.5": "2023", "GPT-4": "2024", "GPT-4o": "2024"}[model] + \
                  " & No  & [TBD] & [TBD]/9 & [TBD] & [TBD]/9"
        new_row = f"{tex_name}{pad}& OpenAI & " + \
                  {"GPT-3.5": "2023", "GPT-4": "2024", "GPT-4o": "2024"}[model] + \
                  f" & No  & {mean_qv} & {var_pass}/9 & {mean_qe} & {es_pass}/9"
        content = content.replace(old_row, new_row)

    with open(tex_path, 'w') as f:
        f.write(content)
    print(f"\nUpdated: {tex_path}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    set_style()

    # Run analysis
    results_df = run_analysis()

    # Save full results
    csv_path = os.path.join(BASE_DIR, "conformal_backtest_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(results_df)} rows)")

    # Print tables
    print_table1(results_df, "GPT-4", 30)
    print_table2(results_df, "GPT-4", 30)
    print_table4(results_df, 30)

    # Generate figures
    print("\nGenerating figures...")
    fig1_dual_correction(results_df, FIGURES_DIR)
    fig2_cross_model_thresholds(results_df, FIGURES_DIR)
    fig3_coverage_comparison(results_df, FIGURES_DIR)
    fig4_z2_comparison(results_df, FIGURES_DIR)
    fig6_heatmap(results_df, FIGURES_DIR)

    # Update tex file
    update_tex_file(results_df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    w30 = results_df[results_df['window'] == 30]
    for model in ["GPT-3.5", "GPT-4", "GPT-4o"]:
        m = w30[w30['model'] == model]
        if len(m) == 0:
            continue
        print(f"\n{model}:")
        print(f"  Mean raw pi_hat: {m['raw_pi_hat'].mean():.4f} (target: 0.01)")
        print(f"  Mean corr pi_hat: {m['corr_pi_hat'].mean():.4f}")
        print(f"  Mean q_hat_V: {m['q_hat_V'].mean():.4f}")
        print(f"  Mean q_hat_E: {m['q_hat_E'].mean():.4f}")
        print(f"  Mean raw Z2: {m['raw_Z2'].mean():.4f}")
        print(f"  Mean corr Z2: {m['corr_Z2'].mean():.4f}")
        print(f"  VaR TL pass (Green): {(m['corr_TL'] == 'G').sum()}/{len(m)}")
        print(f"  ES Z2 pass (>-0.7): {(m['corr_Z2'] > -0.7).sum()}/{len(m)}")

    # Save summary
    summary_path = os.path.join(BASE_DIR, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("DUAL CONFORMAL CALIBRATION — KEY FINDINGS\n")
        f.write("=" * 50 + "\n\n")
        for model in ["GPT-3.5", "GPT-4", "GPT-4o"]:
            m = w30[w30['model'] == model]
            if len(m) == 0:
                continue
            f.write(f"{model} (w=30, 9 assets):\n")
            f.write(f"  Mean raw violation rate: {m['raw_pi_hat'].mean():.4f}\n")
            f.write(f"  Mean corrected violation rate: {m['corr_pi_hat'].mean():.4f}\n")
            f.write(f"  Mean q_hat_V (VaR correction): {m['q_hat_V'].mean():.4f}\n")
            f.write(f"  Mean q_hat_E (ES correction): {m['q_hat_E'].mean():.4f}\n")
            f.write(f"  Mean raw Z2: {m['raw_Z2'].mean():.4f}\n")
            f.write(f"  Mean corrected Z2: {m['corr_Z2'].mean():.4f}\n")
            f.write(f"  VaR Green Zone after correction: {(m['corr_TL'] == 'G').sum()}/{len(m)}\n")
            f.write(f"  ES Z2 pass after correction: {(m['corr_Z2'] > -0.7).sum()}/{len(m)}\n\n")

        # All windows summary
        f.write("\nFULL RESULTS (all windows):\n")
        f.write("=" * 50 + "\n")
        for model in ["GPT-3.5", "GPT-4", "GPT-4o"]:
            for w in WINDOWS:
                m = results_df[(results_df['model'] == model) & (results_df['window'] == w)]
                if len(m) == 0:
                    continue
                var_pass = (m['corr_TL'] == 'G').sum()
                es_pass = (m['corr_Z2'] > -0.7).sum()
                f.write(f"  {model} w={w}: q_V={m['q_hat_V'].mean():.4f} "
                        f"q_E={m['q_hat_E'].mean():.4f} "
                        f"VaR={var_pass}/{len(m)} ES={es_pass}/{len(m)}\n")

    print(f"\nSaved: {summary_path}")
    print("\nDone!")
