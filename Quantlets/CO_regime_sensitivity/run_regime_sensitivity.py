"""
CO_regime_sensitivity -- run_regime_sensitivity.py
===================================================
Regime sensitivity analysis for the conformal oracle paper.

Classifies each model-asset pair as "replacement" vs "signal-preserving"
based on the rolling ratio  R_t = |q_{V,t}| / |VaR_{raw,t}|.

A pair is labelled *replacement* if R_t exceeds a threshold R for at
least `persistence` consecutive trading days anywhere in the test period.

The script produces a LaTeX table (tab_regime_sensitivity.tex) with:
  - Panel A: 3x3 grid (R thresholds x persistence thresholds) showing
    N_rep / 240 replacement-regime pairs.
  - Panel B: per-model classification for each of the 9 cells.

Input:  cfp_ijf_data/{model_dir}/{asset}.parquet  (VaR forecasts)
        cfp_ijf_data/returns/{asset}.csv           (log-returns)
        cfp_ijf_data/paper_outputs/tables/moirai11_results.csv  (for Moirai 1.1)
Output: tab_regime_sensitivity.tex
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA_DIR = BASE / 'cfp_ijf_data'
OUT_DIR = SCRIPT_DIR

# ── Assets (24) ───────────────────────────────────────────────────
ASSETS = [
    'ASX200', 'AUDUSD', 'BOVESPA', 'BTC', 'FCHI', 'CBU0', 'DJCI', 'ETH',
    'EURUSD', 'FTSE100', 'GBPUSD', 'GDAXI', 'GOLD', 'HSI', 'IBGL', 'ICLN',
    'NATGAS', 'NIFTY', 'NIKKEI', 'SP500', 'STOXX', 'TLT', 'USDJPY', 'WTI',
]

# ── Models ─────────────────────────────────────────────────────────
MODEL_DIRS = {
    'Chronos-Small': ('chronos_small', '{asset}.parquet'),
    'Chronos-Mini':  ('chronos_mini',  '{asset}.parquet'),
    'TimesFM-2.5':   ('timesfm25',     '{asset}.parquet'),
    'Moirai-2.0':    ('moirai2',       '{asset}.parquet'),
    'Moirai-1.1':    ('moirai',        '{asset}.parquet'),
    'Lag-Llama':     ('lagllama',      '{asset}.parquet'),
    'GJR-GARCH':     ('benchmarks',    '{asset}_gjr_garch.parquet'),
    'GARCH-N':       ('benchmarks',    '{asset}_garch_n.parquet'),
    'Hist-Sim':      ('benchmarks',    '{asset}_hs.parquet'),
    'EWMA':          ('benchmarks',    '{asset}_ewma.parquet'),
}

# Canonical ordering for display
MODEL_ORDER = [
    'Moirai-1.1', 'Lag-Llama', 'GJR-GARCH', 'GARCH-N',
    'Hist-Sim', 'EWMA',
    'Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5', 'Moirai-2.0',
]
MODEL_DISPLAY = {
    'Moirai-1.1': 'Moirai 1.1', 'Lag-Llama': 'Lag-Llama',
    'GJR-GARCH': 'GJR-GARCH', 'GARCH-N': 'GARCH-N',
    'Hist-Sim': r'Hist.\ Sim.', 'EWMA': 'EWMA',
    'Chronos-Small': 'Chronos-Small', 'Chronos-Mini': 'Chronos-Mini',
    'TimesFM-2.5': 'TimesFM 2.5', 'Moirai-2.0': 'Moirai 2.0',
}

ALPHA = 0.01
W_ROLL = 250        # rolling calibration window for conformity scores
F_CAL = 0.70        # fraction used for initial calibration (skip this warm-up)

R_THRESHOLDS = [1.2, 1.5, 2.0]
PERSIST_THRESHOLDS = [10, 20, 50]


# ── Data loading ──────────────────────────────────────────────────

def load_pair(model, asset):
    """Load returns and lower-quantile forecasts, align by date.

    Returns
    -------
    dates  : DatetimeIndex
    r      : np.ndarray  (daily log-returns)
    q_lo   : np.ndarray  (raw lower quantile forecast, negative numbers)
    """
    subdir, pattern = MODEL_DIRS[model]
    fname = pattern.format(asset=asset)

    ret = pd.read_csv(DATA_DIR / 'returns' / f'{asset}.csv',
                       index_col=0, parse_dates=True)
    fcast = pd.read_parquet(DATA_DIR / subdir / fname)

    col = f'VaR_{ALPHA}'
    if col not in fcast.columns:
        raise KeyError(f'{col} not in {subdir}/{fname}')

    common = ret.index.intersection(fcast.index).sort_values()
    if len(common) < W_ROLL + 50:
        raise ValueError(f'Only {len(common)} overlapping dates for '
                         f'{model}/{asset}')

    return common, ret.loc[common, 'log_return'].values, \
           fcast.loc[common, col].values


# ── Rolling R_t computation ───────────────────────────────────────

def compute_rolling_R(returns, q_lo, w=W_ROLL, alpha=ALPHA):
    """Compute rolling R_t = |q_{V,t}| / |VaR_{raw,t}| for each day t.

    At each day t (for t >= w), the conformity scores are computed over
    the window [t-w, t-1]:
        s_tau = q_lo_tau - r_tau    for tau in [t-w, t-1]
    Then:
        q_{V,t} = quantile(s, ceil((w+1)*(1-alpha))/w)
        VaR_{raw,t} = -q_lo_t
        R_t = |q_{V,t}| / |VaR_{raw,t}|

    Returns
    -------
    R_t : np.ndarray of shape (n,) with NaN for t < w
    """
    n = len(returns)
    R_t = np.full(n, np.nan)

    # Conformity scores for entire series
    scores = q_lo - returns   # s_t = q_lo_t - r_t

    # Quantile level for conformal ceiling
    k_level = np.ceil((w + 1) * (1 - alpha)) / w
    k_level = min(k_level, 1.0)

    for t in range(w, n):
        window_scores = scores[t - w:t]
        valid = window_scores[~np.isnan(window_scores)]
        if len(valid) < 10:
            continue

        q_V_t = np.quantile(valid, k_level)
        var_raw_t = abs(q_lo[t])

        if var_raw_t < 1e-12:
            # Degenerate forecast; treat as extreme replacement
            R_t[t] = np.inf
        else:
            R_t[t] = abs(q_V_t) / var_raw_t

    return R_t


def has_consecutive_run(series, threshold, persistence):
    """Check if `series` exceeds `threshold` for at least `persistence`
    consecutive non-NaN entries.

    Parameters
    ----------
    series : np.ndarray
    threshold : float
    persistence : int  (number of consecutive days)

    Returns
    -------
    bool
    """
    run = 0
    for v in series:
        if np.isnan(v):
            run = 0
            continue
        if v > threshold:
            run += 1
            if run >= persistence:
                return True
        else:
            run = 0
    return False


# ── Main computation ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("REGIME SENSITIVITY ANALYSIS")
    print(f"Models: {len(MODEL_ORDER)} | Assets: {len(ASSETS)} | "
          f"Pairs: {len(MODEL_ORDER) * len(ASSETS)}")
    print(f"R thresholds: {R_THRESHOLDS}")
    print(f"Persistence thresholds: {PERSIST_THRESHOLDS}")
    print(f"Rolling window: {W_ROLL} days | Alpha: {ALPHA}")
    print("=" * 70)

    # Store rolling R_t classification for each (model, asset) pair
    # For each pair, store: dict of (R_thresh, persist) -> bool (replacement)
    results = {}   # (model, asset) -> {(R_thresh, persist): bool}
    n_total = len(MODEL_ORDER) * len(ASSETS)
    errors = []
    done = 0

    for model in MODEL_ORDER:
        for asset in ASSETS:
            done += 1
            try:
                dates, r, q_lo = load_pair(model, asset)

                # Skip initial calibration warm-up: start rolling from
                # after the first F_CAL fraction (matching the paper's
                # calibration/test split)
                n = len(r)
                n_cal = int(n * F_CAL)
                start_idx = max(W_ROLL, n_cal)

                R_t = compute_rolling_R(r, q_lo, w=W_ROLL, alpha=ALPHA)
                # Only evaluate on the test period
                R_test = R_t[start_idx:]

                pair_results = {}
                for R_thresh in R_THRESHOLDS:
                    for persist in PERSIST_THRESHOLDS:
                        is_replacement = has_consecutive_run(
                            R_test, R_thresh, persist)
                        pair_results[(R_thresh, persist)] = is_replacement

                results[(model, asset)] = pair_results

                if done % 24 == 0:
                    print(f"  [{done:3d}/{n_total}] {model:16s} -- "
                          f"median R_t = {np.nanmedian(R_test):.3f}")

            except Exception as e:
                errors.append(f"{model}/{asset}: {e}")
                # Fill with NaN/False
                pair_results = {}
                for R_thresh in R_THRESHOLDS:
                    for persist in PERSIST_THRESHOLDS:
                        pair_results[(R_thresh, persist)] = False
                results[(model, asset)] = pair_results

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")

    # ── Aggregate counts ──────────────────────────────────────────

    # Panel A: total replacement count per cell
    grid = {}
    for R_thresh in R_THRESHOLDS:
        for persist in PERSIST_THRESHOLDS:
            count = sum(1 for (m, a), pr in results.items()
                        if pr.get((R_thresh, persist), False))
            grid[(R_thresh, persist)] = count

    print("\n" + "=" * 50)
    print("PANEL A: Replacement-regime pairs (out of 240)")
    print("=" * 50)
    header = f"{'R \\\\ Persist':>14s}"
    for p in PERSIST_THRESHOLDS:
        header += f"  {p:>6d}d"
    print(header)
    print("-" * 40)
    for R_thresh in R_THRESHOLDS:
        row = f"  R > {R_thresh:4.1f}    "
        for persist in PERSIST_THRESHOLDS:
            row += f"  {grid[(R_thresh, persist)]:>5d}"
        print(row)

    # Panel B: per-model classification
    print("\n" + "=" * 60)
    print("PANEL B: Per-model regime classification")
    print("=" * 60)
    model_class = {}
    for model in MODEL_ORDER:
        for R_thresh in R_THRESHOLDS:
            for persist in PERSIST_THRESHOLDS:
                n_rep = sum(1 for a in ASSETS
                            if results.get((model, a), {}).get(
                                (R_thresh, persist), False))
                model_class[(model, R_thresh, persist)] = n_rep

    for R_thresh in R_THRESHOLDS:
        for persist in PERSIST_THRESHOLDS:
            print(f"\n  R > {R_thresh}, persist = {persist}d:")
            for model in MODEL_ORDER:
                n_rep = model_class[(model, R_thresh, persist)]
                label = "R" if n_rep > 12 else "SP"  # majority rule
                print(f"    {model:16s}: {n_rep:2d}/24 assets -> {label}")

    # ── Generate LaTeX table ──────────────────────────────────────

    generate_latex(grid, model_class)

    print(f"\nDone. Saved tab_regime_sensitivity.tex")


def generate_latex(grid, model_class):
    """Generate tab_regime_sensitivity.tex."""

    lines = []

    # ── Panel A ───────────────────────────────────────────────────
    lines.append(r'\begin{tabular}{@{}l ccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'& \multicolumn{3}{c}{Persistence threshold (consecutive days)} \\')
    lines.append(r'\cmidrule(lr){2-4}')
    lines.append(r'$R$ threshold & 10\,d & 20\,d & 50\,d \\')
    lines.append(r'\midrule')

    for R_thresh in R_THRESHOLDS:
        cells = []
        for persist in PERSIST_THRESHOLDS:
            n_rep = grid[(R_thresh, persist)]
            cells.append(f'{n_rep}/240')
        lines.append(f'$R > {R_thresh:.1f}$ & '
                      + ' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append('')
    lines.append(r'\bigskip')
    lines.append('')

    # ── Panel B: per-model classification ─────────────────────────
    # Columns: Model | then 9 cells (3 R-thresh x 3 persist)
    # Cell value = number of assets (out of 24) classified as replacement.
    # Bold = SP model unexpectedly classified as replacement (>12/24).

    lines.append(r'\begin{tabular}{@{}l' + ' c' * 9 + r'@{}}')
    lines.append(r'\toprule')

    # Multi-level column header
    lines.append(r'& \multicolumn{3}{c}{$R > 1.2$}'
                 r' & \multicolumn{3}{c}{$R > 1.5$}'
                 r' & \multicolumn{3}{c}{$R > 2.0$} \\')
    lines.append(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}')
    lines.append(r'Model & 10\,d & 20\,d & 50\,d'
                 r' & 10\,d & 20\,d & 50\,d'
                 r' & 10\,d & 20\,d & 50\,d \\')
    lines.append(r'\midrule')

    # Signal-preserving panel header
    lines.append(r'\multicolumn{10}{@{}l}{\textit{'
                 r'Panel~A\@: Signal-preserving models}} \\[2pt]')

    sp_models = ['Moirai-1.1', 'Lag-Llama', 'GJR-GARCH', 'GARCH-N',
                 'Hist-Sim', 'EWMA']
    rep_models = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5', 'Moirai-2.0']

    for model in sp_models:
        label = MODEL_DISPLAY.get(model, model)
        cells = []
        for R_thresh in R_THRESHOLDS:
            for persist in PERSIST_THRESHOLDS:
                n_rep = model_class[(model, R_thresh, persist)]
                # Bold if unexpectedly classified as replacement (>12/24)
                if n_rep > 12:
                    cells.append(r'\textbf{' + f'{n_rep}' + r'}')
                else:
                    cells.append(f'{n_rep}')
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\midrule')
    lines.append(r'\multicolumn{10}{@{}l}{\textit{'
                 r'Panel~B\@: Replacement models}} \\[2pt]')

    for model in rep_models:
        label = MODEL_DISPLAY.get(model, model)
        cells = []
        for R_thresh in R_THRESHOLDS:
            for persist in PERSIST_THRESHOLDS:
                n_rep = model_class[(model, R_thresh, persist)]
                cells.append(f'{n_rep}')
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\multicolumn{10}{@{}p{0.95\linewidth}}{'
                 r'\footnotesize Cells show the number of assets '
                 r'(out of 24) for which the model--asset pair is '
                 r'classified as replacement regime. '
                 r'Bold marks signal-preserving models that switch '
                 r'to majority-replacement ($>12/24$).} \\')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines) + '\n'
    tex_path = OUT_DIR / 'tab_regime_sensitivity.tex'
    tex_path.write_text(tex)
    print(f"\nSaved: {tex_path}")
    print("\nGenerated LaTeX:")
    print(tex)


if __name__ == '__main__':
    main()
