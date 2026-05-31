"""
Conformal Correction for GARCH/EWMA Benchmarks
================================================
Applies the IDENTICAL conformal procedure used for LLMs
(run_conformal_analysis_v2.py) to GARCH and EWMA forecasts.

Methods:
  - GARCH-N(250): Normal innovations, 250-day window
  - GAS-N(250): GAS with Normal innovations, 250-day window
  - GAS-t(250): GAS with Student-t innovations, 250-day window
  - GARCH-LPA: Local Parametric Approach (adaptive window)
  - EWMA-N(120): EWMA Normal, 120-day window
  - EWMA-DCS(120): EWMA Dynamic Conditional Score, 120-day window
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(BASE_DIR, "Assets data")

ALPHA_VAR = 0.01
ALPHA_ES  = 0.025
F_CAL     = 0.70

ASSET_ORDER = ["CRIX", "SP500", "SPGTCLTR", "stoxx", "fchi", "gdaxi", "cbu", "ftse", "djci"]
ASSET_LABELS = {
    "CRIX": "CRIX", "SP500": "S&P 500", "SPGTCLTR": "SPGTCLTR",
    "stoxx": "STOXX", "fchi": "FCHI", "gdaxi": "GDAXI",
    "cbu": "CBU0.L", "ftse": "FTSE100", "djci": "DJCI",
}
ASSET_TEX = {
    "CRIX": "CRIX", "SP500": r"S\&P~500", "SPGTCLTR": "SPGTCLTR",
    "stoxx": "STOXX", "fchi": "FCHI", "gdaxi": "GDAXI",
    "cbu": "CBU0.L", "ftse": "FTSE100", "djci": "DJCI",
}

# LPA folder mapping
LPA_FOLDERS = {
    "CRIX": "CRIX_20240923_203024",
    "SP500": "SP500_20240923_234055",
    "SPGTCLTR": "SPGTCLTR_20240924_000606",
    "stoxx": "stoxx_20240924_003112",
    "fchi": "fchi_20240923_194011",
    "gdaxi": "gdaxi_20240923_231441",
    "cbu": "cbu_20240923_200649",
    "ftse": "ftse_20240923_224833",
    "djci": "djci_20240923_222332",
}

ASSET_FILES = {
    "CRIX": "CRIX.xlsx", "SP500": "SP500.xlsx", "SPGTCLTR": "SPGTCLTR.xlsx",
    "stoxx": "stoxx.xlsx", "fchi": "fchi.xlsx", "gdaxi": "gdaxi.xlsx",
    "cbu": "cbu.xlsx", "ftse": "ftse.xlsx", "djci": "djci.xlsx",
}


# ================================================================
# DATA LOADING
# ================================================================

def load_realized_returns(asset):
    """Load realized log-returns from asset Excel file."""
    fpath = os.path.join(ASSETS_DIR, ASSET_FILES[asset])
    df = pd.read_excel(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df[["Date", "log_return"]].dropna().reset_index(drop=True)


def load_garch_data(asset, model_type, window=250):
    """Load GARCH/GAS data from pickle.
    Returns DataFrame with index=dates, columns: VaR_0.0100, ES_0.0100
    """
    pkl_path = os.path.join(BASE_DIR, f"GARCH/GARCH_Simulations_final_{window}_window.pkl")
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    if asset not in d or model_type not in d[asset]:
        return None
    df = d[asset][model_type]
    # Check for unreasonable values (GARCH-t bug)
    if df['VaR_0.0100'].dropna().abs().mean() > 1.0:
        return None  # Skip broken models
    return df[['VaR_0.0100', 'ES_0.0100']].copy()


def load_lpa_data(asset):
    """Load LPA results from CSV."""
    folder = LPA_FOLDERS.get(asset)
    if folder is None:
        return None
    csv_path = os.path.join(BASE_DIR, "GARCH", "LPA", folder, "results.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Unnamed: 0": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    # Rename to match GARCH format
    df = df.rename(columns={"VaR_0.01": "VaR_0.0100", "ES_0.01": "ES_0.0100"})
    return df[['VaR_0.0100', 'ES_0.0100']].copy()


def load_ewma_data(asset, model_type, window=120):
    """Load EWMA data from pickle."""
    pkl_path = os.path.join(BASE_DIR, f"EWMA/EWMA_Simulations_{window}_window.pkl")
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    if asset not in d or model_type not in d[asset]:
        return None
    df = d[asset][model_type]
    return df[['VaR_0.0100', 'ES_0.0100']].copy()


# ================================================================
# CONFORMAL CALIBRATION (identical to LLM pipeline)
# ================================================================

def apply_conformal(dates, q_lo, raw_var, raw_es, realized, alpha_var=0.01, alpha_es=0.025, f_cal=0.70):
    """
    Apply conformal correction to pre-computed VaR/ES forecasts.

    Parameters:
        dates: array of dates
        q_lo: array of lower quantile forecasts (negative values)
        raw_var: array of VaR (positive = loss)
        raw_es: array of ES (positive = loss)
        realized: array of realized returns (negative = loss)
    """
    T = len(q_lo)
    n_cal = int(T * f_cal)

    # Step 1: One-sided VaR nonconformity scores (Definition 3.2)
    # s_t^V = q_lo_t - r_t
    s_V = np.full(n_cal, np.nan)
    for t in range(n_cal):
        if np.isnan(q_lo[t]) or np.isnan(realized[t]):
            continue
        s_V[t] = q_lo[t] - realized[t]

    s_V_valid = s_V[~np.isnan(s_V)]

    if len(s_V_valid) > 0:
        q_level = np.ceil((len(s_V_valid) + 1) * (1 - alpha_var)) / len(s_V_valid)
        q_level = min(q_level, 1.0)
        q_hat_V = np.quantile(s_V_valid, q_level)
    else:
        q_hat_V = 0.0

    # Step 2: ES residual scores on calibration violation days
    s_E = []
    for t in range(n_cal):
        if np.isnan(realized[t]) or np.isnan(raw_var[t]):
            continue
        if realized[t] < -raw_var[t]:  # violation day
            s_E_t = realized[t] + raw_es[t]
            s_E.append(s_E_t)

    s_E = np.array(s_E)
    if len(s_E) > 0:
        q_level_E = np.ceil((len(s_E) + 1) * (1 - alpha_es)) / len(s_E)
        q_level_E = min(q_level_E, 1.0)
        q_hat_E = np.quantile(s_E, q_level_E)
    else:
        q_hat_E = 0.0

    # Step 3: Corrected estimates on test set
    corrected_var = np.full(T, np.nan)
    corrected_es = np.full(T, np.nan)
    for t in range(n_cal, T):
        if not np.isnan(q_lo[t]):
            corrected_var[t] = -(q_lo[t] - q_hat_V)  # = raw_var + q_hat_V
        if not np.isnan(raw_es[t]):
            corrected_es[t] = raw_es[t] + q_hat_E

    return {
        'n_cal': n_cal,
        'n_test': T - n_cal,
        'q_hat_V': q_hat_V,
        'q_hat_E': q_hat_E,
        'n_violations_cal': len(s_E),
        'raw_var': raw_var,
        'raw_es': raw_es,
        'corrected_var': corrected_var,
        'corrected_es': corrected_es,
        'realized': realized,
        'q_lo': q_lo,
    }


# ================================================================
# BACKTESTING (identical to LLM pipeline)
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


# ================================================================
# MAIN ANALYSIS
# ================================================================

def align_returns(forecast_df, realized_df):
    """Align realized returns with forecast dates.
    For each forecast date, get the NEXT day's realized return.
    """
    forecast_dates = pd.to_datetime(forecast_df.index)
    real_dates = pd.to_datetime(realized_df["Date"].values)
    real_returns = realized_df["log_return"].values

    realized_aligned = np.full(len(forecast_dates), np.nan)
    for t, d in enumerate(forecast_dates):
        future_mask = real_dates > d
        if future_mask.any():
            idx = np.where(future_mask)[0][0]
            realized_aligned[t] = real_returns[idx]

    return realized_aligned


def run_method(method_name, load_func, asset):
    """Run conformal analysis for one method and one asset."""
    forecast_df = load_func(asset)
    if forecast_df is None:
        return None

    # Drop rows with NaN in VaR
    forecast_df = forecast_df.dropna(subset=['VaR_0.0100'])
    if len(forecast_df) < 20:
        return None

    realized_df = load_realized_returns(asset)
    realized = align_returns(forecast_df, realized_df)

    dates = forecast_df.index
    q_lo = forecast_df['VaR_0.0100'].values  # negative values
    raw_var = -q_lo  # positive (loss)
    raw_es = forecast_df['ES_0.0100'].values  # already positive

    result = apply_conformal(dates, q_lo, raw_var, raw_es, realized,
                              alpha_var=ALPHA_VAR, alpha_es=ALPHA_ES, f_cal=F_CAL)

    n_cal = result['n_cal']
    test_sl = slice(n_cal, None)
    r_test = result['realized'][test_sl]

    # Raw diagnostics (test set)
    raw_kup = kupiec_test(r_test, result['raw_var'][test_sl], ALPHA_VAR)
    raw_z2 = acerbi_z2(r_test, result['raw_var'][test_sl], result['raw_es'][test_sl], ALPHA_ES)

    # Corrected diagnostics (test set)
    corr_kup = kupiec_test(r_test, result['corrected_var'][test_sl], ALPHA_VAR)
    corr_z2 = acerbi_z2(r_test, result['corrected_var'][test_sl], result['corrected_es'][test_sl], ALPHA_ES)

    return {
        'method': method_name,
        'asset': asset,
        'asset_label': ASSET_LABELS[asset],
        'n_total': len(forecast_df),
        'n_cal': n_cal,
        'n_test': raw_kup['N'],
        'q_hat_V': result['q_hat_V'],
        'q_hat_E': result['q_hat_E'],
        'n_violations_cal': result['n_violations_cal'],
        'raw_pi_hat': raw_kup['pi_hat'],
        'raw_TL': raw_kup['traffic_light'],
        'raw_Z2': raw_z2['Z2'],
        'corr_pi_hat': corr_kup['pi_hat'],
        'corr_TL': corr_kup['traffic_light'],
        'corr_Z2': corr_z2['Z2'],
        'raw_avg_var': float(np.nanmean(result['raw_var'][test_sl])),
        'corr_avg_var': float(np.nanmean(result['corrected_var'][test_sl])),
    }


def main():
    print("=" * 70)
    print("CONFORMAL CORRECTION FOR GARCH/EWMA BENCHMARKS")
    print("=" * 70)

    # Define methods
    methods = {
        'GARCH-N(250)': lambda asset: load_garch_data(asset, 'Norm', 250),
        'GAS-N(250)':   lambda asset: load_garch_data(asset, 'GAS', 250),
        'GAS-t(250)':   lambda asset: load_garch_data(asset, 'GAS_T', 250),
        'GARCH-LPA':    lambda asset: load_lpa_data(asset),
        'EWMA-N(120)':  lambda asset: load_ewma_data(asset, 'Norm', 120),
        'EWMA-DCS(120)': lambda asset: load_ewma_data(asset, 'DCS', 120),
    }

    all_results = []

    for method_name, load_func in methods.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        for asset in ASSET_ORDER:
            result = run_method(method_name, load_func, asset)
            if result is None:
                print(f"  {ASSET_LABELS[asset]:>10}: SKIPPED (no data or broken)")
                continue

            all_results.append(result)
            print(f"  {result['asset_label']:>10}: "
                  f"raw_pi={result['raw_pi_hat']:.4f} TL={result['raw_TL']} "
                  f"Z2={result['raw_Z2']:+.3f} | "
                  f"q_V={result['q_hat_V']:+.5f} q_E={result['q_hat_E']:+.5f} "
                  f"corr_TL={result['corr_TL']} Z2c={result['corr_Z2']:+.3f}")

    df = pd.DataFrame(all_results)

    # ================================================================
    # SUMMARY TABLES
    # ================================================================

    print(f"\n{'='*100}")
    print("TABLE 1: CROSS-METHOD SUMMARY (mean across 9 assets)")
    print(f"{'='*100}")
    print(f"{'Method':<18} {'Raw pi':>8} {'q_V':>10} {'Corr pi':>8} {'Green':>7} {'Z2 Pass':>8} {'Avg VaR':>10}")
    print("-" * 80)

    summary_rows = []
    for method_name in methods:
        m = df[df['method'] == method_name]
        if len(m) == 0:
            continue
        mean_raw_pi = m['raw_pi_hat'].mean()
        mean_qv = m['q_hat_V'].mean()
        mean_corr_pi = m['corr_pi_hat'].mean()
        green = int((m['corr_TL'] == 'G').sum())
        z2_pass = int((m['corr_Z2'] > -0.7).sum())
        total = len(m)
        avg_var = m['corr_avg_var'].mean()

        print(f"{method_name:<18} {mean_raw_pi:>8.4f} {mean_qv:>+10.5f} {mean_corr_pi:>8.4f} "
              f"{green:>3}/{total:<3} {z2_pass:>4}/{total:<3} {avg_var:>10.5f}")

        summary_rows.append({
            'method': method_name,
            'mean_raw_pi': mean_raw_pi,
            'mean_q_hat_V': mean_qv,
            'mean_q_hat_E': m['q_hat_E'].mean(),
            'mean_corr_pi': mean_corr_pi,
            'green_count': green,
            'z2_pass_count': z2_pass,
            'total': total,
            'mean_corr_avg_var': avg_var,
        })

    # Add LLM results from memory for comparison
    llm_results = [
        {'method': 'GPT-3.5+CP', 'mean_raw_pi': 0.004, 'mean_q_hat_V': 0.002,
         'mean_q_hat_E': 0.0, 'mean_corr_pi': 0.006, 'green_count': 8,
         'z2_pass_count': 9, 'total': 9, 'mean_corr_avg_var': 0.0},
        {'method': 'GPT-4+CP', 'mean_raw_pi': 0.086, 'mean_q_hat_V': 0.024,
         'mean_q_hat_E': 0.021, 'mean_corr_pi': 0.002, 'green_count': 9,
         'z2_pass_count': 9, 'total': 9, 'mean_corr_avg_var': 0.0},
        {'method': 'GPT-4o+CP', 'mean_raw_pi': 0.057, 'mean_q_hat_V': 0.020,
         'mean_q_hat_E': 0.009, 'mean_corr_pi': 0.002, 'green_count': 9,
         'z2_pass_count': 9, 'total': 9, 'mean_corr_avg_var': 0.0},
    ]

    print()
    print("LLM results (from prior analysis, w=30):")
    for lr in llm_results:
        print(f"{lr['method']:<18} {lr['mean_raw_pi']:>8.4f} {lr['mean_q_hat_V']:>+10.5f} {lr['mean_corr_pi']:>8.4f} "
              f"{lr['green_count']:>3}/{lr['total']:<3} {lr['z2_pass_count']:>4}/{lr['total']:<3}")

    # ================================================================
    # PER-ASSET DETAIL TABLE
    # ================================================================

    print(f"\n{'='*100}")
    print("TABLE 2: PER-ASSET DETAIL FOR EACH METHOD")
    print(f"{'='*100}")

    for method_name in methods:
        m = df[df['method'] == method_name]
        if len(m) == 0:
            continue
        print(f"\n--- {method_name} ---")
        print(f"{'Asset':>10} {'Raw pi':>8} {'q_V':>10} {'Corr pi':>8} {'Raw TL':>7} {'Corr TL':>8} {'Raw Z2':>8} {'Corr Z2':>8}")
        for _, row in m.iterrows():
            print(f"{row['asset_label']:>10} {row['raw_pi_hat']:>8.4f} {row['q_hat_V']:>+10.5f} "
                  f"{row['corr_pi_hat']:>8.4f} {row['raw_TL']:>7} {row['corr_TL']:>8} "
                  f"{row['raw_Z2']:>+8.3f} {row['corr_Z2']:>+8.3f}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    # Detailed CSV
    detail_path = os.path.join(BASE_DIR, "garch_conformal_detail.csv")
    df.to_csv(detail_path, index=False, float_format="%.6f")
    print(f"\nSaved: {detail_path}")

    # Summary CSV
    all_summary = summary_rows + llm_results
    summary_df = pd.DataFrame(all_summary)
    summary_path = os.path.join(BASE_DIR, "conformal_all_methods_summary.csv")
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Saved: {summary_path}")

    # ================================================================
    # LATEX TABLE
    # ================================================================

    generate_latex_table(df, summary_rows, llm_results)

    # ================================================================
    # FIGURE
    # ================================================================

    generate_figure(df, summary_rows, llm_results)

    return df, summary_rows


def generate_latex_table(df, summary_rows, llm_results):
    """Generate LaTeX table: tab_conformal_all_methods.tex"""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Conformal correction across model classes. $\overline{\hat{\pi}}_{\mathrm{raw}}$ and $\overline{\hat{\pi}}_{\mathrm{corr}}$ are mean violation rates (raw and corrected) across nine assets. $\overline{\qV}$ is the mean conformal correction threshold. Green denotes Basel Green Zone ($\leq 4$ annual exceptions).}")
    lines.append(r"\label{tab:conformal_all_methods}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r c c}")
    lines.append(r"\toprule")
    lines.append(r"Method & $\overline{\hat{\pi}}_{\mathrm{raw}}$ & $\overline{\qV}$ & $\overline{\hat{\pi}}_{\mathrm{corr}}$ & Green & $Z_2$ Pass \\")
    lines.append(r"\midrule")

    # Parametric methods
    for row in summary_rows:
        name = row['method']
        green_str = f"{row['green_count']}/{row['total']}"
        z2_str = f"{row['z2_pass_count']}/{row['total']}"
        lines.append(f"  {name} & {row['mean_raw_pi']:.4f} & {row['mean_q_hat_V']:+.4f} & {row['mean_corr_pi']:.4f} & {green_str} & {z2_str} \\\\")

    lines.append(r"\midrule")

    # LLM methods
    for row in llm_results:
        name = row['method']
        green_str = f"{row['green_count']}/{row['total']}"
        z2_str = f"{row['z2_pass_count']}/{row['total']}"
        lines.append(f"  {name} & {row['mean_raw_pi']:.4f} & {row['mean_q_hat_V']:+.4f} & {row['mean_corr_pi']:.4f} & {green_str} & {z2_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(BASE_DIR, "tab_conformal_all_methods.tex")
    with open(tex_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved: {tex_path}")


def generate_figure(df, summary_rows, llm_results):
    """Generate the money figure: mean |q_V| per method."""

    # Collect data
    method_names = []
    mean_qvs = []
    colors = []

    # Parametric (blue) and semi-parametric (green)
    parametric_methods = ['GARCH-N(250)', 'GAS-N(250)', 'GAS-t(250)', 'GARCH-LPA']
    semi_methods = ['EWMA-N(120)', 'EWMA-DCS(120)']
    llm_methods = ['GPT-3.5+CP', 'GPT-4+CP', 'GPT-4o+CP']

    for row in summary_rows:
        method_names.append(row['method'])
        mean_qvs.append(row['mean_q_hat_V'])
        if row['method'] in parametric_methods:
            colors.append('#2166AC')  # blue
        else:
            colors.append('#4DAF4A')  # green

    for row in llm_results:
        method_names.append(row['method'])
        mean_qvs.append(row['mean_q_hat_V'])
        colors.append('#E31A1C')  # red

    # Short names for x-axis
    short_names = {
        'GARCH-N(250)': 'GARCH-N',
        'GAS-N(250)': 'GAS-N',
        'GAS-t(250)': 'GAS-t',
        'GARCH-LPA': 'GARCH-LPA',
        'EWMA-N(120)': 'EWMA-N',
        'EWMA-DCS(120)': 'EWMA-DCS',
        'GPT-3.5+CP': 'GPT-3.5',
        'GPT-4+CP': 'GPT-4',
        'GPT-4o+CP': 'GPT-4o',
    }
    x_labels = [short_names.get(m, m) for m in method_names]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(method_names))
    bars = ax.bar(x, mean_qvs, color=colors, width=0.6, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, mean_qvs):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., y + 0.0005,
                f'{val:+.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=14)
    ax.set_ylabel(r'Mean $\hat{q}_V$ (conformal correction)', fontsize=18)
    ax.set_title(r'Conformal Correction Magnitude Across Model Classes', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2166AC', edgecolor='black', label='Parametric (GARCH/GAS)'),
        Patch(facecolor='#4DAF4A', edgecolor='black', label='Semi-parametric (EWMA)'),
        Patch(facecolor='#E31A1C', edgecolor='black', label='LLM-based'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    fig.tight_layout()

    # Save
    png_path = os.path.join(BASE_DIR, "fig_conformal_agnostic.png")
    pdf_path = os.path.join(BASE_DIR, "fig_conformal_agnostic.pdf")
    fig.savefig(png_path, dpi=600, transparent=True, bbox_inches='tight')
    fig.savefig(pdf_path, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    df, summary = main()
