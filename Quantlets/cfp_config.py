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


# --------------------------------------------------------------------------- #
# The conformal shift, in one place
#
# The convention was documented in the module docstring above and implemented
# nowhere, so every producer wrote it again. Four of them wrote it differently:
# the plain empirical quantile in gap_ablation.py and run_robustness_mc.py, and
# np.quantile at level k/n -- which interpolates -- in fz_scores.py and
# run_simulation_study.py. A fifth was a matter of time.
#
# scripts/audit_qv_convention.py fails the build when a site computes a quantile
# of nonconformity scores without going through this function and without an
# entry in analysis/provenance/QV_CONVENTION_SITES.tsv saying why.
# --------------------------------------------------------------------------- #

def conformal_quantile(scores, alpha=None):
    """The finite-sample split-conformal shift: S_(k), k = ceil((n+1)(1-alpha)).

    NOT np.quantile(scores, 1-alpha), which interpolates below the order
    statistic, and NOT np.quantile(scores, k/n), which interpolates above it.
    Both gaps are O(1/n) asymptotically and neither is O(1/n) in a short window.

    When k >= n the shift IS the sample maximum, an extreme-value statistic with
    no stable variance. At alpha = 0.01 that happens for every calibration sample
    of 125 observations or fewer, which is inside the range of rolling windows a
    referee may ask for. `conformal_index` reports it.
    """
    a = ALPHA if alpha is None else alpha
    x = np.sort(np.asarray(scores, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("conformal_quantile: empty score sample")
    k = ceil((n + 1) * (1.0 - a))
    return float(x[min(k, n) - 1])


def conformal_index(n, alpha=None):
    """(k, is_sample_maximum) for a calibration sample of size n."""
    a = ALPHA if alpha is None else alpha
    k = ceil((n + 1) * (1.0 - a))
    return k, k >= n

# --------------------------------------------------------------------------- #
# Stale-price screen
#
# A trailing window in which most returns are exactly zero contains no
# identifiable price process, and every model family breaks on it in its own
# way. CBU0 in 2013 is 97.65% exactly-zero returns: Chronos correctly forecasts
# "no movement" (median VaR_0.01 = -0.0001) and takes a 41% violation rate,
# the GJR-GARCH-t optimiser fails to converge on 35% of windows, and the
# skewed-t variant returns |VaR| up to 2.2e6. One data-quality defect, four
# pathologies, three unrelated model families -- and all of them vanish from
# 2015, once the 250-day estimation window clears the stale period.
#
# WHAT THIS IS: the exclusion of two identified NON-TRADING PERIODS, not a
# threshold on an engineered feature. Two series in the panel have pre-2015 eras
# in which the asset barely traded:
#
#   CBU0  2013  78.8% of returns exactly zero      2014  45.5%
#   IBGL  2013  22.5%                              2014  20.2%
#
# The rule below is how those periods are identified reproducibly. It is not a
# tuned cutoff, and the panel demonstrates rather than asserts this. Across all
# 525 asset-years in the panel the distribution of the zero-return fraction is
#
#   [0,1%) 450   [1,2%) 45   [2,5%) 20   [5,10%) 6   [10,20%) 0   >=20% 4
#
# -- the 10-20% band is EMPTY. On the trailing-250-day measure the screen
# actually uses, the ordering is CBU0 0.980, IBGL 0.404, then DJCI 0.112, ICLN
# 0.084, and everything else below 0.05. Any threshold in (0.12, 0.40) selects
# exactly CBU0 and IBGL and removes exactly 719 of 134,211 observations (0.54%).
# A threshold at 0.10 would sit below the void and pull in DJCI.
#
# DISCLOSURE: this criterion was written after observing that CBU0 failed the
# promotion gate's coverage check, not before. It is not pre-registered and must
# not be described as such. What can be said is that it is defined purely on an
# input property, that it lands in an empty region of the panel's own
# distribution, and that results are reported with and without it.
#
# THE BOUNDARY DOES NOT PARTITION THE SYMPTOMS CLEANLY, and this was checked
# rather than assumed:
#
#                  GJR-t degenerate   divergences   Chronos-A pihat(1%)
#   CBU0 excluded      214/3515            52          0.0559  out of band
#   IBGL excluded      109/4348             1          0.0135  fine
#   DJCI retained        9/2572             1          0.0169  fine
#
# Only CBU0 is severe across all three model families. IBGL's exclusion is not
# corroborated by its Chronos coverage, and DJCI -- retained, at 0.112, just
# below the void -- shows the same pathologies as IBGL at lower intensity. The
# rule is kept as stated rather than narrowed to CBU0: a criterion defined on
# the data and applied blind to symptoms is defensible in a way that "exclude
# the asset that showed problems" is not. The gradient is disclosed instead of
# being smoothed into a cliff.
ZERO_RETURN_MAX = 0.20     # max fraction of exactly-zero returns in the window
ZERO_RETURN_WINDOW = 250   # same window the estimators use


def stale_mask(returns, max_zero: float = ZERO_RETURN_MAX,
               window: int = ZERO_RETURN_WINDOW):
    """True where the trailing window is too stale to define a price process.

    `returns` is a pandas Series indexed by date. Rows before a full window are
    not screened, since the estimators do not produce forecasts for them.
    """
    z = (returns == 0).rolling(window, min_periods=window).mean()
    return (z > max_zero).fillna(False)


# --------------------------------------------------------------------------- #
# Plausibility bound on a one-step conditional forecast
#
# A maximum-likelihood GARCH fit can fail to converge and return a forecast that
# is not merely wrong but absurd, WITHOUT raising and, on some optimisers,
# without setting a convergence flag. Observed on CBU0: conditional means of
# +/-12000 and sigma up to 4.2e8, on an asset whose daily returns are ~0.3%. The
# resulting VaR reached 7.6e8, and the series still passed every median- and
# fraction-based check in the promotion gate, because 56 bad days out of 5800
# move no median. Mean width was 9493 against a median of 0.008.
#
# The stale-price screen removes today's trigger. It does not remove the failure
# mode: any sufficiently degenerate window can produce this, and the next one
# need not be stale. So the bound is applied at the point of fitting, and the
# gate's `extremes` check is the independent second layer. Neither is a
# substitute for the other -- this one prevents the value being written, that
# one prevents it being promoted if some other path writes it anyway.
FORECAST_SANITY_K = 10.0


def forecast_is_plausible(mu: float, sigma: float, window_sd: float,
                          k: float = FORECAST_SANITY_K) -> bool:
    """Is a one-step (mu, sigma) forecast credible given the window it was fit on?

    No converged one-step-ahead fit places the conditional mean or scale more
    than k realised window standard deviations away. Rejecting on this catches
    non-convergence whether or not the optimiser reports it.
    """
    import math
    if not (math.isfinite(mu) and math.isfinite(sigma)) or sigma <= 0:
        return False
    if not math.isfinite(window_sd) or window_sd <= 0:
        return False
    return sigma <= k * window_sd and abs(mu) <= k * window_sd


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
    "GJR-GARCH-t":   ("benchmarks",    "gjr_t"),
    "GARCH-N":       ("benchmarks",    "garch_n"),
    "Hist-Sim":      ("benchmarks",    "hs"),
    "EWMA":          ("benchmarks",    "ewma"),
    # Chronos read analytically instead of sampled. Both are kept on purpose:
    # the shipped series is the checkpoint default top_k = 50, which truncates
    # the predictive support to 50 of 4094 bins, and the comparison between the
    # two IS the configuration finding rather than a correction folded into a
    # rerun. See analysis/chronos_sampling/.
    "Chronos-Small-A": ("chronos_small_analytic", None),
    "Chronos-Mini-A":  ("chronos_mini_analytic",  None),
}

# Added 2026-08-17. Scripts that must reproduce the ten-forecaster panel of the
# IJF submission should import this rather than achieve it by omission.
MODELS_10 = {k: v for k, v in MODELS.items()
             if k not in ("GJR-GARCH-t", "Chronos-Small-A", "Chronos-Mini-A")}

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
