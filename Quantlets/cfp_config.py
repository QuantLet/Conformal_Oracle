"""Single source of truth for the model set, asset set and split conventions.

Every script in this package used to carry its own copy of the model dictionary.
That is why adding Moirai-1.1 had to be discovered three separate times -- in the
multi-alpha results table, in the block-bootstrap CIs, and in the rolling
intermediate behind Table 3 -- and why a fourth would have been found the same
way. Import from here instead.

    from cfp_config import MODELS, SYMBOLS, ALPHA, F_CAL, n_cal_for, rng_for

Conventions recovered from the published tables and fixed here so they cannot
drift again:

  n_cal = floor(F_CAL * n)      the house split; one legacy file used ceil on
                                half its assets, which moved qV by up to 52%
  qhat  = S_(k), k = ceil((n+1)(1-alpha))
                                the conformal order statistic Theorem 3.3
                                analyses -- NOT np.quantile, which interpolates
                                below it and carries no finite-sample guarantee
  Kupiec pass: p >= 0.05
  R     = mean over assets of |qV| / |VaR_raw|, per-pair ratios then averaged.
          Keep the sign separately: GJR-GARCH is over-conservative (qV < 0 on
          23 of 24 assets) and the absolute column cannot express that.
"""

from __future__ import annotations

import hashlib
from math import ceil

import numpy as np

ALPHA = 0.01
ALPHAS = [0.01, 0.025, 0.05, 0.10]
F_CAL = 0.70
W_ROLL = 250

SYMBOLS = [
    "SP500", "STOXX", "GDAXI", "FCHI", "FTSE100", "ICLN",
    "NIKKEI", "HSI", "BOVESPA", "NIFTY", "ASX200", "CBU0",
    "TLT", "IBGL", "DJCI", "GOLD", "WTI", "NATGAS",
    "BTC", "ETH", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
]

# The CAC 40 series was stored as CACT before the data rebuild and is FCHI now.
# Six pipeline scripts kept the old ticker and, because a missing returns file is
# skipped rather than raised, silently produced 23-asset tables that looked
# complete. Published figures predate the rename and are unaffected.
LEGACY_SYMBOL_ALIASES = {"CACT": "FCHI"}

ASSET_CLASS = {
    **{s: "Equity" for s in ("SP500", "STOXX", "GDAXI", "FCHI", "FTSE100",
                             "ICLN", "NIKKEI", "HSI", "BOVESPA", "NIFTY",
                             "ASX200")},
    **{s: "Bond" for s in ("CBU0", "TLT", "IBGL")},
    **{s: "Commodity" for s in ("DJCI", "GOLD", "WTI", "NATGAS")},
    **{s: "Crypto" for s in ("BTC", "ETH")},
    **{s: "FX" for s in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")},
}

# (subdirectory under cfp_ijf_data, benchmark suffix or None)
MODELS = {
    "Chronos-Small": ("chronos_small", None),
    "Chronos-Mini":  ("chronos_mini",  None),
    "TimesFM-2.5":   ("timesfm25",     None),
    "Moirai-2.0":    ("moirai2",       None),
    "Moirai-1.1":    ("moirai",        None),
    "Lag-Llama":     ("lagllama",      None),
    "GJR-GARCH":     ("benchmarks",    "gjr_garch"),
    "GARCH-N":       ("benchmarks",    "garch_n"),
    "Hist-Sim":      ("benchmarks",    "hs"),
    "EWMA":          ("benchmarks",    "ewma"),
}

# The nine models of the original analysis, before Moirai-1.1 was added as the
# within-family control. Scripts that must reproduce a nine-forecaster figure
# should say so explicitly by importing this rather than by omission.
MODELS_9 = {k: v for k, v in MODELS.items() if k != "Moirai-1.1"}

# Quantile-grid interfaces, whose raw 1% violation rates are ~99%.
GRID_INTERFACE = {"TimesFM-2.5", "Moirai-2.0"}


def n_cal_for(n: int, f_cal: float = F_CAL) -> int:
    """House calibration size: floor, never ceil."""
    return int(n * f_cal)


def qhat_ceil(scores, alpha: float = ALPHA) -> float:
    """Conformal order statistic. Do not substitute an interpolated quantile."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = min(int(ceil((n + 1) * (1 - alpha))) - 1, n - 1)
    return float(s[k])


def rng_for(*keys, seed: int = 42) -> np.random.Generator:
    """A generator seeded from the cell identity, not from call order.

    A single shared stream makes every result depend on how many models happen
    to precede it in a dictionary: adding one model changed 30 of 36 published
    bootstrap intervals. Deriving the seed from (asset, model, ...) makes each
    cell independent and reproducible whatever else is in the loop.
    """
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return np.random.default_rng(seed + int.from_bytes(h[:8], "big"))
