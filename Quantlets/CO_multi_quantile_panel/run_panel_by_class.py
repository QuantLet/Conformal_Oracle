"""
CO_multi_quantile_panel — Panel-pooled backtest by asset class.
Produces tab_panel_by_class.tex  (Table 7 in the paper).

For each of five asset classes (Equity, Bond, Commodity, Crypto, FX),
reports N_assets, N_panel (sum of test observations across all
model-asset pairs), mean corrected coverage rate, and Green-zone rate
under Basel traffic-light classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
OUT_DIR  = Path(__file__).resolve().parent

ALPHA = 0.01

CLASS_MAP = {
    'SP500': 'Equity', 'STOXX': 'Equity', 'GDAXI': 'Equity',
    'FCHI': 'Equity', 'FTSE100': 'Equity', 'ICLN': 'Equity',
    'NIKKEI': 'Equity', 'HSI': 'Equity', 'BOVESPA': 'Equity',
    'NIFTY': 'Equity', 'ASX200': 'Equity',
    'CBU0': 'Bond', 'TLT': 'Bond', 'IBGL': 'Bond',
    'DJCI': 'Commodity', 'GOLD': 'Commodity', 'WTI': 'Commodity',
    'NATGAS': 'Commodity',
    'BTC': 'Crypto', 'ETH': 'Crypto',
    'EURUSD': 'FX', 'GBPUSD': 'FX', 'USDJPY': 'FX', 'AUDUSD': 'FX',
}
CLASS_ORDER = ['Equity', 'Bond', 'Commodity', 'Crypto', 'FX']

# ── Load ───────────────────────────────────────────────────────────
df = pd.read_csv(RES_DIR / 'all_results.csv')
d01 = df[df['alpha'] == ALPHA].copy()
d01['asset_class'] = d01['symbol'].map(CLASS_MAP)
assert d01['asset_class'].notna().all(), 'Unmapped symbols in all_results.csv'

print(f'Loaded {len(d01)} rows at alpha={ALPHA} '
      f'({d01["model"].nunique()} models, '
      f'{d01["symbol"].nunique()} assets)')

# ── Compute per-class statistics ──────────────────────────────────
# Aggregation: equal-weighted mean of pihat_cp across all
# (model, asset) pairs in each class. This matches the
# manuscript's convention and differs from the true pooled
# violation rate (which would weight each pair by its test
# observation count). The difference is small but non-zero in
# classes with heterogeneous warmup constraints (e.g., Equity:
# 0.010 equal-weighted vs 0.011 pooled).
rows = []
for cls in CLASS_ORDER:
    sub = d01[d01['asset_class'] == cls]
    n_assets = sub['symbol'].nunique()
    n_panel  = int(sub['n_test'].sum())
    pi_mean  = sub['pihat_cp'].mean()
    green    = int((sub['TL_cp'] == 'Green').sum())
    total    = len(sub)

    rows.append({
        'asset_class': cls, 'n_assets': n_assets,
        'n_panel': n_panel, 'pi_mean': pi_mean,
        'green': green, 'total': total,
    })
    print(f'  {cls:12s}  N={n_assets}  N_panel={n_panel:>9,}  '
          f'pi={pi_mean:.4f}  Green={green}/{total} '
          f'({100*green/total:.0f}%)')

result = pd.DataFrame(rows)

# ── Save CSV ──────────────────────────────────────────────────────
result.to_csv(OUT_DIR / 'tab_panel_by_class.csv', index=False)

# ── Format helpers ────────────────────────────────────────────────
def fmt_n(n):
    return f'{int(n):,}'.replace(',', '{,}')

def fmt_pi(x):
    d = Decimal(str(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    return '.' + format(d, '.3f')[2:]

# ── Build LaTeX ───────────────────────────────────────────────────
lines = [
    r'\begin{tabular}{@{}lrrrr@{}}',
    r'\toprule',
    r'Asset class & $N$ assets & $N_{\text{panel}}$',
    r'& Corr.\ $\hat\pi$ & Green rate \\',
    r'\midrule',
]

for _, r in result.iterrows():
    pct = int(round(100 * r['green'] / r['total']))
    line = (f'{r["asset_class"]:10s} & {r["n_assets"]:2d}'
            f' & {fmt_n(r["n_panel"])} & {fmt_pi(r["pi_mean"])}\n'
            f'& {r["green"]}/{r["total"]} ({pct}\\%) \\\\')
    lines.append(line)

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_panel_by_class.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
