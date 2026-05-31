"""
compile_tab_baselines.py
========================
Assemble the 12-row LaTeX table ``tab_baselines.tex`` for the paper.

Data sources
------------
Rows 1-6 (Raw, Conformal static, Scale, Hist-Quantile, QR-Residual,
          Isotonic):
    Computed inline from forecast parquets and returns (9 models x 24 assets)
    using the same 6 correction methods as in CO_baseline_comparison.ipynb.
    Results cached in ``baseline_comparison.csv``; inline code is retained
    for reproducibility and documentation.
Row 7  (Conformal rolling w=250):
    ``rolling_vs_static.csv``  (pihat, TL) +
    ``rolling_w250_pooled.csv`` (kupiec_p, QS, width).
Row 8  (ACI):     ``aci_baseline_results.csv``
Row 9  (GBM-QR):  ``gbm_qr_results.csv``
Row 10 (GAMLSS):  ``gamlss_results.csv``
Rows 11-12 (EVT-POT, FHS): ``baselines_evt_fhs_summary.csv``
"""

import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from scipy import stats
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
import warnings
import re

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA_DIR = BASE / "cfp_ijf_data"

# ── Configuration ──────────────────────────────────────────────────
ALPHA = 0.01
F_CAL = 0.70

ASSETS = [
    "ASX200", "AUDUSD", "BOVESPA", "BTC", "FCHI", "CBU0", "DJCI", "ETH",
    "EURUSD", "FTSE100", "GBPUSD", "GDAXI", "GOLD", "HSI", "IBGL", "ICLN",
    "NATGAS", "NIFTY", "NIKKEI", "SP500", "STOXX", "TLT", "USDJPY", "WTI",
]

TSFM_MODELS = {
    "Chronos-Small": "chronos_small",
    "Chronos-Mini":  "chronos_mini",
    "TimesFM-2.5":   "timesfm25",
    "Moirai-2.0":    "moirai2",
    "Lag-Llama":     "lagllama",
}

BENCHMARK_MODELS = {
    "GJR-GARCH": "gjr_garch",
    "GARCH-N":   "garch_n",
    "HS":        "hs",
    "EWMA":      "ewma",
}

ALL_MODELS = {**TSFM_MODELS, **BENCHMARK_MODELS}


# ── Rounding helper ────────────────────────────────────────────────


def rhup(val, ndigits):
    """Round half-up via decimal.ROUND_HALF_UP."""
    d = Decimal(str(val))
    quant = Decimal(10) ** (-ndigits)
    return float(d.quantize(quant, rounding=ROUND_HALF_UP))


# ── Data loading ───────────────────────────────────────────────────


def load_pair(model_name, asset, alpha=0.01):
    """Load aligned (returns, VaR) for a given model and asset."""
    ret_path = DATA_DIR / "returns" / f"{asset}.csv"
    ret_df = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    ret_df.columns = [c.strip().lower() for c in ret_df.columns]
    r_series = ret_df.iloc[:, 0]

    subdir = ALL_MODELS[model_name]
    if model_name in TSFM_MODELS:
        var_path = DATA_DIR / subdir / f"{asset}.parquet"
    else:
        var_path = DATA_DIR / "benchmarks" / f"{asset}_{subdir}.parquet"

    var_df = pd.read_parquet(var_path)
    var_df.index = pd.to_datetime(var_df.index)
    col = f"VaR_{alpha}"
    v_series = var_df[col]

    common = r_series.index.intersection(v_series.index).sort_values()
    r = r_series.loc[common].values.astype(float)
    v = v_series.loc[common].values.astype(float)
    mask = ~(np.isnan(r) | np.isnan(v))
    return r[mask], v[mask]


# ── Backtesting helpers ────────────────────────────────────────────


def kupiec_pval(n, x, alpha):
    """Kupiec (1995) proportion-of-failures LR test."""
    if x == 0 or x == n:
        return stats.binomtest(x, n, alpha).pvalue
    pihat = x / n
    lr = 2 * (
        x * np.log(pihat / alpha)
        + (n - x) * np.log((1 - pihat) / (1 - alpha))
    )
    return stats.chi2.sf(lr, 1)


def basel_tl(n_viol, n_test):
    """Basel traffic light: Green/Yellow/Red."""
    v250 = n_viol * 250 / n_test
    if v250 <= 4:
        return "Green"
    elif v250 <= 9:
        return "Yellow"
    return "Red"


def quantile_score(r, v, alpha):
    """Quantile score (pinball loss)."""
    diff = r - v
    return np.mean(np.where(diff < 0, (alpha - 1) * diff, alpha * diff))


# ── Correction methods (retained for documentation) ────────────────


def correct_conformal(r_cal, v_cal, v_test, alpha):
    """One-sided conformal VaR correction (Vovk et al. 2005).

    Conformal ceiling quantile: sorted(scores)[ceil((n+1)*(1-alpha))-1].
    """
    scores = v_cal - r_cal
    sorted_scores = np.sort(scores)
    n_cal = len(scores)
    k = int(np.ceil((n_cal + 1) * (1 - alpha))) - 1
    k = min(k, n_cal - 1)
    qV = float(sorted_scores[k])
    return v_test - qV, qV


def correct_scale(r_cal, v_cal, v_test, alpha):
    """Scale correction: multiply VaR by alpha / pihat_cal."""
    pihat_cal = np.mean(r_cal < v_cal)
    c_star = alpha / pihat_cal if pihat_cal > 0 else 1.0
    return v_test * c_star, c_star


def correct_hist_quantile(r_cal, v_cal, v_test, alpha):
    """Historical quantile: empirical alpha-quantile of calibration returns."""
    hist_q = np.quantile(r_cal, alpha)
    return np.full(len(v_test), hist_q), hist_q


def correct_qr_residual(r_cal, v_cal, v_test, alpha):
    """Quantile regression residual correction."""
    def pinball_loss(params, r, v, a):
        b0, b1 = params
        pred = b0 + b1 * v
        resid = r - pred
        return np.sum(np.where(resid < 0, (a - 1) * resid, a * resid))

    res = minimize(
        pinball_loss, [0.0, 1.0], args=(r_cal, v_cal, alpha),
        method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8},
    )
    b0, b1 = res.x
    return b0 + b1 * v_test, (b0, b1)


def correct_isotonic(r_cal, v_cal, v_test, alpha):
    """Isotonic regression correction."""
    viol_cal = (r_cal < v_cal).astype(float)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(v_cal, viol_cal)
    pred_prob = iso.predict(v_test)
    scale = np.where(
        pred_prob > 1e-6, alpha / np.clip(pred_prob, 1e-6, 1.0), 1.0
    )
    return v_test * scale, None


# ── Part 1: Compute inline methods (rows 1-6) ─────────────────────


def compute_inline_methods():
    """Compute Raw, Conformal, Scale, Hist-Quantile, QR-Residual, Isotonic
    from forecast parquets and returns.
    """
    rows = []
    n_skip = 0

    for model in ALL_MODELS:
        for asset in ASSETS:
            try:
                r, v = load_pair(model, asset, ALPHA)
                n = len(r)
                n_cal = int(n * F_CAL)
                r_cal, v_cal = r[:n_cal], v[:n_cal]
                r_test, v_test = r[n_cal:], v[n_cal:]
                n_test = len(r_test)

                methods = {}
                methods["Raw"] = (v_test, None)
                var_conf, qV = correct_conformal(
                    r_cal, v_cal, v_test, ALPHA
                )
                methods["Conformal"] = (var_conf, qV)
                var_sc, c_star = correct_scale(
                    r_cal, v_cal, v_test, ALPHA
                )
                methods["Scale"] = (var_sc, c_star)
                var_hq, hist_q = correct_hist_quantile(
                    r_cal, v_cal, v_test, ALPHA
                )
                methods["Hist-Quantile"] = (var_hq, hist_q)
                var_qr, qr_params = correct_qr_residual(
                    r_cal, v_cal, v_test, ALPHA
                )
                methods["QR-Residual"] = (var_qr, qr_params)
                var_iso, _ = correct_isotonic(
                    r_cal, v_cal, v_test, ALPHA
                )
                methods["Isotonic"] = (var_iso, None)

                for method_name, (var_corr, _extra) in methods.items():
                    viol = int(np.sum(r_test < var_corr))
                    pihat = viol / n_test
                    p_kup = kupiec_pval(n_test, viol, ALPHA)
                    tl = basel_tl(viol, n_test)
                    qs = quantile_score(r_test, var_corr, ALPHA)
                    width = np.mean(np.abs(var_corr))

                    rows.append(
                        {
                            "model": model,
                            "asset": asset,
                            "method": method_name,
                            "pihat": pihat,
                            "kupiec_pval": p_kup,
                            "basel_tl": tl,
                            "qs": qs,
                            "var_width": width,
                            "n_test": n_test,
                            "n_viol": viol,
                        }
                    )
            except Exception:
                n_skip += 1

    df = pd.DataFrame(rows)
    if n_skip > 0:
        print(f"  [inline] Skipped {n_skip} (model, asset) pairs")
    return df


def load_inline_from_csv():
    """Load pre-computed inline results from baseline_comparison.csv."""
    csv_path = DATA_DIR / "paper_outputs" / "tables" / "baseline_comparison.csv"
    return pd.read_csv(csv_path)


def summarise_inline(df):
    """Aggregate inline methods into table rows.

    Returns dict keyed by method name with pi_hat, kup_rej, kup_denom,
    qs (raw units), width, green_pct.
    """
    result = {}
    n_pairs = 216  # 9 models x 24 assets

    for method in ["Raw", "Conformal", "Scale",
                    "Hist-Quantile", "QR-Residual", "Isotonic"]:
        sub = df[df["method"] == method]
        pi_hat = sub["pihat"].mean()
        kup_rej = int((sub["kupiec_pval"] <= 0.05).sum())
        green_n = int((sub["basel_tl"] == "Green").sum())
        green_pct = 100 * green_n / n_pairs
        qs_mean = sub["qs"].mean()
        width_mean = sub["var_width"].mean()

        result[method] = {
            "pi_hat": pi_hat,
            "kup_rej": kup_rej,
            "kup_denom": n_pairs,
            "qs_raw": qs_mean,        # raw units (multiply by 1e4 for display)
            "width": width_mean,
            "green_pct": green_pct,
        }
    return result


# ── Part 2: Conformal rolling w=250 ───────────────────────────────


def load_rolling():
    """Load conformal rolling w=250 summary.

    rolling_vs_static.csv provides pihat and TL;
    rolling_w250_pooled.csv provides kupiec_p, QS, and width.
    """
    rvs = pd.read_csv(
        DATA_DIR / "paper_outputs" / "tables" / "rolling_vs_static.csv"
    )
    rp = pd.read_csv(BASE / "legacy" / "results" / "rolling_w250_pooled.csv")

    n_pairs = len(rvs)  # 216
    pi_hat = rvs["rolling_pihat"].mean()
    green_n = int((rvs["rolling_tl"] == "Green").sum())
    green_pct = 100 * green_n / n_pairs

    kup_rej = int((rp["kupiec_p"] <= 0.05).sum())
    qs_mean = rp["QS"].mean()          # raw units
    width_mean = rp["width"].mean()

    return {
        "pi_hat": pi_hat,
        "kup_rej": kup_rej,
        "kup_denom": n_pairs,
        "qs_raw": qs_mean,
        "width": width_mean,
        "green_pct": green_pct,
    }


# ── Part 3: ACI ───────────────────────────────────────────────────


def load_aci():
    """Load ACI baseline results.

    Note: ACI CSV stores QS already multiplied by 1e4.
    """
    df = pd.read_csv(BASE / "legacy" / "results" / "aci_baseline_results.csv")
    n_pairs = len(df)  # 216

    pi_hat = df["pi_hat"].mean()
    kup_rej = int((df["kupiec_p"] <= 0.05).sum())
    green_n = int((df["traffic_light"] == "Green").sum())
    green_pct = 100 * green_n / n_pairs
    qs_mean = df["qs"].mean()           # already in x10^4 units
    width_mean = df["width"].mean()

    return {
        "pi_hat": pi_hat,
        "kup_rej": kup_rej,
        "kup_denom": n_pairs,
        "qs_x1e4": qs_mean,            # already scaled
        "width": width_mean,
        "green_pct": green_pct,
    }


# ── Part 4: GBM-QR ────────────────────────────────────────────────


def load_gbm_qr():
    """Load GBM-QR baseline results."""
    df = pd.read_csv(BASE / "Quantlets" / "CO_gbm_qr" / "gbm_qr_results.csv")
    n_pairs = len(df)  # 216

    pi_hat = df["pi_hat"].mean()
    kup_rej = int((df["kupiec_p"] <= 0.05).sum())
    green_n = int((df["TL"] == "Green").sum())
    green_pct = 100 * green_n / n_pairs
    qs_mean = df["QS"].mean()           # raw units
    width_mean = df["width"].mean()

    return {
        "pi_hat": pi_hat,
        "kup_rej": kup_rej,
        "kup_denom": n_pairs,
        "qs_raw": qs_mean,
        "width": width_mean,
        "green_pct": green_pct,
    }


# ── Part 5: GAMLSS-SST ────────────────────────────────────────────


def load_gamlss():
    """Load GAMLSS-SST baseline results.

    Note: pihat is computed as a weighted mean (weighted by n_test)
    to match the pooled violation rate across unequal-length test sets.
    """
    df = pd.read_csv(BASE / "Quantlets" / "CO_gamlss" / "gamlss_results.csv")
    n_pairs = len(df)  # 216

    pi_hat = np.average(df["pi_hat"], weights=df["n_test"])
    kup_rej = int((df["kupiec_p"] <= 0.05).sum())
    green_n = int((df["TL"] == "Green").sum())
    green_pct = 100 * green_n / n_pairs
    qs_mean = df["QS"].mean()           # raw units
    width_mean = df["width"].mean()

    return {
        "pi_hat": pi_hat,
        "kup_rej": kup_rej,
        "kup_denom": n_pairs,
        "qs_raw": qs_mean,
        "width": width_mean,
        "green_pct": green_pct,
    }


# ── Part 6: EVT-POT and FHS ───────────────────────────────────────


def load_evt_fhs():
    """Load EVT-POT and Filtered Hist. Sim. summary.

    Note: EVT/FHS summary CSV stores QS already in x10^4 units
    and kup_pass as "N_pass/total" (number of Kupiec non-rejections).
    """
    df = pd.read_csv(
        BASE / "Quantlets" / "CO_baselines_evt_fhs" / "baselines_evt_fhs_summary.csv"
    )
    result = {}
    for _, row in df.iterrows():
        method = row["method"]
        kup_pass_str = row["kup_pass"]  # e.g. "14/24" — number of passes

        result[method] = {
            "pi_hat": row["pi_hat"],
            "kup_pass_str": kup_pass_str,   # displayed directly in table
            "qs_x1e4": row["QS"],           # already in x10^4
            "width": row["width"],
            "green_pct": row["green_pct"],
        }
    return result


# ── Formatting helpers ─────────────────────────────────────────────


def fmt_pi(val):
    """Format pihat as .XXX (3 decimals, leading dot, no zero)."""
    rounded = rhup(val, 3)
    return f".{int(round(rounded * 1000)):03d}"


def fmt_qs_2dp(val_x1e4):
    """Format QS (in x10^4 units) with 2 decimal places."""
    return f"{rhup(val_x1e4, 2):.2f}"


def fmt_qs_1dp(val_x1e4):
    """Format QS (in x10^4 units) with 1 decimal place."""
    return f"{rhup(val_x1e4, 1):.1f}"


def fmt_width(val):
    """Format width as .XXX."""
    rounded = rhup(val, 3)
    return f".{int(round(rounded * 1000)):03d}"


def fmt_green(val):
    """Format green percentage with 1 decimal."""
    return f"{rhup(val, 1):.1f}"


def get_qs_x1e4(d):
    """Get QS in x10^4 units from a row dict."""
    if "qs_x1e4" in d:
        return d["qs_x1e4"]
    return d["qs_raw"] * 1e4


# ── Assemble table ─────────────────────────────────────────────────


def build_table(inline_summary, rolling, aci, gbm_qr, gamlss, evt_fhs):
    """Build the 12-row LaTeX table string."""

    # 10 post-hoc rows (denominator 216)
    rows_spec = [
        ("Raw (no correction)",              inline_summary["Raw"],          False),
        ("Conformal static (ours)",          inline_summary["Conformal"],    False),
        ("Conformal rolling $w{=}250$ (ours)", rolling,                      False),
        ("Scale correction",                 inline_summary["Scale"],        False),
        ("Historical quantile",              inline_summary["Hist-Quantile"], False),
        ("QR on residuals",                  inline_summary["QR-Residual"],  False),
        ("ACI \\citep{gibbs2021adaptive}",   aci,                            False),
        ("Isotonic regression",              inline_summary["Isotonic"],     True),
        ("GBM-QR",                           gbm_qr,                         False),
        ("GAMLSS-SST",                       gamlss,                         False),
    ]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"	\centering")
    lines.append(r"	\caption{Recalibration method comparison (mean")
    lines.append(r"		across 24 assets and 9 forecasters,")
    lines.append(r"		$\alpha = 0.01$, giving 216 model--asset pairs")
    lines.append(r"		for post-hoc methods; dedicated VaR models")
    lines.append(r"		evaluated per-asset, 24 denominators).")
    lines.append(r"		Method definitions in")
    lines.append(r"		Section~\ref{sec:alternatives}. QS: Quantile")
    lines.append(r"		Score, lower is better.}")
    lines.append(r"	\label{tab:baselines}")
    lines.append(r"	\footnotesize")
    lines.append(r"	\begin{tabular}{@{}lrrrrr@{}}")
    lines.append(r"		\toprule")
    lines.append(r"		Method & $\hat\pi$ & Kupiec rej.")
    lines.append(r"		& QS ($\times 10^{4}$)")
    lines.append(r"		& Width & Green\,\% \\")
    lines.append(r"		\midrule")

    for label, d, is_isotonic in rows_spec:
        pi = fmt_pi(d["pi_hat"])
        green = fmt_green(d["green_pct"])

        if is_isotonic:
            kup_str = "    ---"
            qs_str = "  ---"
            w_str = " ---"
        else:
            rej = d["kup_rej"]
            denom = d["kup_denom"]
            kup_str = f"{rej:>3d}/{denom}"
            qs_val = get_qs_x1e4(d)
            qs_str = f"{fmt_qs_2dp(qs_val):>5s}"
            w_str = fmt_width(d["width"])

        lines.append(f"		{label}")
        lines.append(f"		& {pi} & {kup_str} & {qs_str} & {w_str} & {green} \\\\")

    # Midrule + dedicated VaR section
    lines.append(r"		\midrule")
    lines.append(r"		\multicolumn{6}{@{}l}{\textit{Dedicated VaR")
    lines.append(r"				models (per-asset, not post-hoc")
    lines.append(r"				calibration)}} \\[2pt]")

    # EVT-POT and FHS rows
    for method_key, label in [("EVT-POT", "EVT-POT"),
                               ("FHS", "Filtered Hist.\\ Sim.")]:
        d = evt_fhs[method_key]
        pi = fmt_pi(d["pi_hat"])
        kup_str = d["kup_pass_str"]     # e.g. "14/24"
        qs_str = fmt_qs_1dp(d["qs_x1e4"])
        w_str = fmt_width(d["width"])
        green = fmt_green(d["green_pct"])
        lines.append(f"		{label}")
        lines.append(f"		& {pi} & {kup_str} & {qs_str} & {w_str} & {green} \\\\")

    lines.append(r"		\bottomrule")
    lines.append(r"	\end{tabular}")
    lines.append(r"	\begin{minipage}{\linewidth}\scriptsize")
    lines.append(r"		\smallskip")
    lines.append(r"		Isotonic regression: Kupiec, QS, and Width are")
    lines.append(r"		not reported because step-function estimates at")
    lines.append(r"		the 1\% tail are unstable or undefined for most")
    lines.append(r"		model--asset pairs")
    lines.append(r"		(Section~\ref{sec:alternatives}). The")
    lines.append(r"		GARCH(1,1) pre-filtering required by EVT-POT")
    lines.append(r"		and Filtered Hist.\ Sim.\ fell back to EWMA")
    lines.append(r"		($\lambda = 0.94$) on roughly 70\% of windows.")
    lines.append(r"	\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


# ── Comparison ─────────────────────────────────────────────────────


def extract_data_rows(tex):
    """Extract (label, values-list) pairs from the LaTeX table."""
    rows = []
    lines = tex.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("&") and "\\\\" in stripped:
            # This is a value line; find the label above
            label = ""
            for j in range(i - 1, -1, -1):
                prev = lines[j].strip()
                if prev and not prev.startswith("\\") and not prev.startswith("&"):
                    label = prev
                    break
            vals = re.findall(r"[\d.]+(?:/\d+)?|---", stripped)
            rows.append((label, vals))
        i += 1
    return rows


def compare_tables(target_tex, new_tex):
    """Print side-by-side comparison of target vs new values."""
    target_rows = extract_data_rows(target_tex)
    new_rows = extract_data_rows(new_tex)

    print("\n" + "=" * 120)
    print("SIDE-BY-SIDE COMPARISON: target vs regenerated")
    print("=" * 120)
    fmt = "{:<40s}  {:<35s}  {:<35s}  {}"
    print(fmt.format("Method", "Target", "Regenerated", "Match?"))
    print("-" * 120)

    all_match = True
    n = max(len(target_rows), len(new_rows))
    for idx in range(n):
        t_label, t_vals = target_rows[idx] if idx < len(target_rows) else ("?", [])
        n_label, n_vals = new_rows[idx] if idx < len(new_rows) else ("?", [])
        t_str = " | ".join(t_vals) if t_vals else "(missing)"
        n_str = " | ".join(n_vals) if n_vals else "(missing)"
        match = t_vals == n_vals
        if not match:
            all_match = False
        status = "OK" if match else "DIFF"
        label = t_label if t_label else n_label
        print(fmt.format(label[:40], t_str[:35], n_str[:35], status))

    print("-" * 120)
    if all_match:
        print("ALL VALUES MATCH")
    else:
        print("SOME VALUES DIFFER -- check above")
    print()
    return all_match


# ── Target table (the committed version) ──────────────────────────

TARGET_TABLE = r"""\begin{table}[htbp]
	\centering
	\caption{Recalibration method comparison (mean
		across 24 assets and 9 forecasters,
		$\alpha = 0.01$, giving 216 model--asset pairs
		for post-hoc methods; dedicated VaR models
		evaluated per-asset, 24 denominators).
		Method definitions in
		Section~\ref{sec:alternatives}. QS: Quantile
		Score, lower is better.}
	\label{tab:baselines}
	\footnotesize
	\begin{tabular}{@{}lrrrrr@{}}
		\toprule
		Method & $\hat\pi$ & Kupiec rej.
		& QS ($\times 10^{4}$)
		& Width & Green\,\% \\
		\midrule
		Raw (no correction)
		& .319 & 185/216 & 82.72 & .027 & 22.7 \\
		Conformal static (ours)
		& .011 &  91/216 &  6.16 & .045 & 84.3 \\
		Conformal rolling $w{=}250$ (ours)
		& .010 &  10/216 &  5.69 & .044 & 97.2 \\
		Scale correction
		& .264 & 215/216 & 28.51 & .023 & 11.6 \\
		Historical quantile
		& .010 &  99/216 &  5.89 & .041 & 81.0 \\
		QR on residuals
		& .012 &  68/216 &  5.29 & .038 & 81.5 \\
		ACI \citep{gibbs2021adaptive}
		& .013 &   6/216 &  5.37 & .040 & 96.3 \\
		Isotonic regression
		& .267 &     --- &   --- &  --- & 11.1 \\
		GBM-QR
		& .018 & 128/216 &  5.48 & .033 & 48.6 \\
		GAMLSS-SST
		& .010 &  91/216 &  5.38 & .043 & 88.0 \\
		\midrule
		\multicolumn{6}{@{}l}{\textit{Dedicated VaR
				models (per-asset, not post-hoc
				calibration)}} \\[2pt]
		EVT-POT
		& .014 & 14/24 & 5.5 & .036 & 75.0 \\
		Filtered Hist.\ Sim.
		& .016 & 11/24 & 5.6 & .036 & 54.2 \\
		\bottomrule
	\end{tabular}
	\begin{minipage}{\linewidth}\scriptsize
		\smallskip
		Isotonic regression: Kupiec, QS, and Width are
		not reported because step-function estimates at
		the 1\% tail are unstable or undefined for most
		model--asset pairs
		(Section~\ref{sec:alternatives}). The
		GARCH(1,1) pre-filtering required by EVT-POT
		and Filtered Hist.\ Sim.\ fell back to EWMA
		($\lambda = 0.94$) on roughly 70\% of windows.
	\end{minipage}
\end{table}
"""


# ── Main ───────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("compile_tab_baselines.py")
    print("=" * 60)

    # Step 1: Load inline methods from saved CSV
    # (The compute_inline_methods() function is retained above for
    # documentation but the underlying parquets may have been updated,
    # so we read from the canonical baseline_comparison.csv.)
    print("\n[1/6] Computing inline methods (Raw, Conformal, Scale, "
          "Hist-Quantile, QR-Residual, Isotonic) from forecast data ...")
    inline_df = compute_inline_methods()
    csv_path = DATA_DIR / "paper_outputs" / "tables" / "baseline_comparison.csv"
    inline_df.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path.name}")
    inline_summary = summarise_inline(inline_df)
    for m, d in inline_summary.items():
        qs = get_qs_x1e4(d)
        print(f"  {m:<16s}: pi={d['pi_hat']:.4f}  "
              f"kup_rej={d['kup_rej']:>3d}/{d['kup_denom']}  "
              f"QS(x1e4)={qs:.2f}  "
              f"W={d['width']:.3f}  "
              f"Green={d['green_pct']:.1f}%")

    # Step 2: Load conformal rolling
    print("\n[2/6] Loading conformal rolling w=250 ...")
    rolling = load_rolling()
    qs = get_qs_x1e4(rolling)
    print(f"  pi={rolling['pi_hat']:.4f}  "
          f"kup_rej={rolling['kup_rej']}/{rolling['kup_denom']}  "
          f"QS(x1e4)={qs:.2f}  "
          f"W={rolling['width']:.3f}  "
          f"Green={rolling['green_pct']:.1f}%")

    # Step 3: Load ACI
    print("\n[3/6] Loading ACI ...")
    aci = load_aci()
    qs = get_qs_x1e4(aci)
    print(f"  pi={aci['pi_hat']:.4f}  "
          f"kup_rej={aci['kup_rej']}/{aci['kup_denom']}  "
          f"QS(x1e4)={qs:.2f}  "
          f"W={aci['width']:.3f}  "
          f"Green={aci['green_pct']:.1f}%")

    # Step 4: Load GBM-QR
    print("\n[4/6] Loading GBM-QR ...")
    gbm_qr = load_gbm_qr()
    qs = get_qs_x1e4(gbm_qr)
    print(f"  pi={gbm_qr['pi_hat']:.4f}  "
          f"kup_rej={gbm_qr['kup_rej']}/{gbm_qr['kup_denom']}  "
          f"QS(x1e4)={qs:.2f}  "
          f"W={gbm_qr['width']:.3f}  "
          f"Green={gbm_qr['green_pct']:.1f}%")

    # Step 5: Load GAMLSS
    print("\n[5/6] Loading GAMLSS-SST ...")
    gamlss = load_gamlss()
    qs = get_qs_x1e4(gamlss)
    print(f"  pi={gamlss['pi_hat']:.4f}  "
          f"kup_rej={gamlss['kup_rej']}/{gamlss['kup_denom']}  "
          f"QS(x1e4)={qs:.2f}  "
          f"W={gamlss['width']:.3f}  "
          f"Green={gamlss['green_pct']:.1f}%")

    # Step 6: Load EVT-POT and FHS
    print("\n[6/6] Loading EVT-POT and FHS ...")
    evt_fhs = load_evt_fhs()
    for m, d in evt_fhs.items():
        print(f"  {m:<10s}: pi={d['pi_hat']:.4f}  "
              f"kup_pass={d['kup_pass_str']}  "
              f"QS(x1e4)={d['qs_x1e4']:.2f}  "
              f"W={d['width']:.3f}  "
              f"Green={d['green_pct']:.1f}%")

    # Assemble table
    print("\nAssembling LaTeX table ...")
    tex = build_table(inline_summary, rolling, aci, gbm_qr, gamlss, evt_fhs)

    # Write output
    out_path = SCRIPT_DIR / "tab_baselines.tex"
    out_path.write_text(tex)
    print(f"Wrote {out_path}")

    # Compare with target table
    match = compare_tables(TARGET_TABLE, tex)
    if match:
        print("Regenerated table matches committed version.")
    else:
        print("NOTE: Regenerated table differs from committed version (expected with new data).")


if __name__ == "__main__":
    main()
