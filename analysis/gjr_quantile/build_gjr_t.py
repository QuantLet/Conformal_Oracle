#!/usr/bin/env python3
"""GJR-GARCH with a Student-t innovation, as an additional benchmark.

Motivation. Correcting the GJR quantile map to the Gaussian innovation the
manuscript describes leaves the benchmark under-covering: pihat = 0.0194 against
a nominal 0.01, Kupiec passing on 0 of 24 assets. That is the textbook behaviour
of a Gaussian GARCH on daily returns, not a defect -- but it makes a fat-tailed
parametric benchmark the natural comparison, and its absence would be the
obvious referee question.

This is also the model the shipped series was *trying* to be. That series used

    mu + sigma * stats.t.ppf(alpha, 5)

with the degrees of freedom hard-coded at 5 and the RAW t quantile, so the
innovation had variance nu/(nu-2) = 5/3 instead of 1 and every VaR was too wide
by a further factor of 1.29. Here nu is estimated per window and the quantile
comes from arch's standardised StudentsT, whose innovation has unit variance by
construction, so sigma means what the GARCH recursion says it means.

THE BOUNDARY PROBLEM, and why it is handled explicitly.

`arch` bounds the shape parameter below at 2.05, just above the value where the
variance ceases to exist. When the optimiser pins nu at that bound the
standardised quantile diverges. This is not hypothetical: the skewed-t variant
built alongside the Gaussian one produced |VaR_0.01| up to 2.2e6 on CBU0, across
39 dates, with the shape parameter at 2.050 on 29 of them -- a mean width of 61
against a median of 0.008. Averaged into a table that is invisible; it simply
inflates the benchmark's width and flatters everything compared against it.

Rule, fixed in advance rather than after seeing which assets it touches:

    nu <= NU_FLOOR          the fit is degenerate. Carry forward the most recent
                            non-degenerate nu for that asset. If none exists yet
                            (only possible at the very start of a series), fall
                            back to the Gaussian quantile.

Every degenerate window is counted and reported per asset. A benchmark whose
degenerate fraction is material is a benchmark to discuss, not to average.

Outputs (analysis/gjr_quantile/):
    candidate_t/<asset>.parquet     mean, std, nu, degenerate flag, VaR_*
    GJR_T.md, gjr_t_summary.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
ALPHAS = [0.01, 0.025, 0.05, 0.10]
WINDOW = 250
NU_FLOOR = 2.10          # arch's own bound is 2.05; anything at or under this
                         # is the optimiser on the constraint, not an estimate
SANITY_K = 10.0          # a converged one-step forecast cannot have |mu| or
                         # sigma more than this multiple of the window's own
                         # realised volatility; see try_fit()

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import SYMBOLS, forecast_is_plausible  # noqa: E402

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import kupiec_p, traffic_light  # noqa: E402


def load_returns(asset: str) -> pd.Series:
    df = pd.read_csv(DATA / "returns" / f"{asset}.csv", parse_dates=["date"]) \
        .set_index("date").sort_index()
    return df[df["log_return"].abs() <= 0.50]["log_return"]


def main() -> int:
    from arch import arch_model
    from arch.univariate import StudentsT

    dist = StudentsT()
    (OUT / "candidate_t").mkdir(parents=True, exist_ok=True)
    zn = stats.norm.ppf(ALPHAS)
    t0 = time.time()
    summary = []

    score_only = "--score-only" in sys.argv
    for ai, asset in enumerate(SYMBOLS):
        fp_out = OUT / "candidate_t" / f"{asset}.parquet"
        if score_only and fp_out.exists():
            d = pd.read_parquet(fp_out)
            summary.append({"asset": asset, "n_dates": len(d),
                            "n_degenerate": int(d["degenerate"].sum()),
                            "n_failed_fit": int(d["nu"].isna().sum()),
                            "frac_degenerate": float(d["degenerate"].mean()),
                            "nu_median": float(np.nanmedian(d["nu_used"]))})
            continue
        ret = load_returns(asset)
        n = len(ret)
        recs, last_nu, n_deg, n_fail = [], np.nan, 0, 0
        for t in range(WINDOW, n):
            r_win = ret.iloc[t - WINDOW:t] * 100
            win_sd = float(r_win.std()) / 100
            deg = False

            def try_fit(distname):
                """Fitted (mu, sigma, nu), or None if the optimiser did not
                produce a usable forecast.

                Estimating nu jointly with the GJR parameters on a 250-day
                window can fail outright: on CBU0 the unguarded version returned
                conditional means of +/-12000 and sigma up to 4.2e8 on an asset
                whose daily returns are ~0.3%. The innovation distribution was
                never the problem -- the optimisation was. A forecast is
                rejected when the mean or the scale is implausible relative to
                the window's own realised volatility, which no converged fit can
                be, and which catches non-convergence whether or not the
                optimiser admits to it.
                """
                try:
                    am = arch_model(r_win, vol="GARCH", p=1, o=1, q=1,
                                    dist=distname)
                    res = am.fit(disp="off", show_warning=False)
                    if getattr(res, "convergence_flag", 0) != 0:
                        return None
                    fcst = res.forecast(horizon=1, reindex=False)
                    m_ = float(fcst.mean.iloc[-1, 0]) / 100
                    s_ = float(np.sqrt(fcst.variance.iloc[-1, 0])) / 100
                    nu_ = float(res.params.values[-1]) if distname == "t" else np.nan
                    if not (np.isfinite(m_) and np.isfinite(s_)) or s_ <= 0:
                        return None
                    if not forecast_is_plausible(m_, s_, win_sd, SANITY_K):
                        return None
                    return m_, s_, nu_
                except Exception:
                    return None

            got = try_fit("t")
            if got is None:
                n_fail += 1
                got = try_fit("normal")          # the fit that does converge
                if got is None:
                    got = (0.0, win_sd, np.nan)  # last resort: window volatility
            mu, sd, nu = got

            if not np.isfinite(nu) or nu <= NU_FLOOR:
                deg = True
                n_deg += 1
                nu_use = last_nu          # carry forward the last real estimate
            else:
                nu_use = nu
                last_nu = nu

            if np.isfinite(nu_use):
                z = np.asarray(dist.ppf(ALPHAS, np.array([nu_use])), dtype=float)
            else:
                z = zn                    # no estimate yet: Gaussian fallback
            recs.append({"date": ret.index[t], "mean": mu, "std": sd,
                         "nu": nu, "nu_used": nu_use, "degenerate": deg,
                         **{f"VaR_{a:g}": mu + sd * zz for a, zz in zip(ALPHAS, z)}})

        df = pd.DataFrame(recs).set_index("date")
        df.to_parquet(OUT / "candidate_t" / f"{asset}.parquet")
        summary.append({"asset": asset, "n_dates": len(df), "n_degenerate": n_deg,
                        "n_failed_fit": n_fail,
                        "frac_degenerate": n_deg / max(len(df), 1),
                        "nu_median": float(np.nanmedian(df["nu_used"]))})
        print(f"  [{ai + 1:2d}/{len(SYMBOLS)}] {asset:8s} {len(df):5d} dates "
              f"| degenerate {n_deg:4d} | nu~{summary[-1]['nu_median']:.2f} "
              f"| {(time.time() - t0) / 60:.1f} min", file=sys.stderr, flush=True)

    # ------------------------------------------------------------- scoring
    rows = []
    for asset in SYMBOLS:
        fp = OUT / "candidate_t" / f"{asset}.parquet"
        if not fp.exists():
            continue
        fc = pd.read_parquet(fp)
        ret = pd.read_csv(DATA / "returns" / f"{asset}.csv", index_col=0,
                          parse_dates=True)
        ret.columns = ["r"]
        i = ret.index.intersection(fc.index)
        r = ret.loc[i, "r"].values
        sd_real = ret.loc[i, "r"].rolling(250).std().values
        q = fc.loc[i, "VaR_0.01"].values
        m = np.isfinite(q)
        v, nn = int(np.sum(r[m] < q[m])), int(m.sum())
        k = m & np.isfinite(sd_real) & (sd_real > 0)
        A = np.vstack([fc.loc[i, f"VaR_{a:g}"].values for a in ALPHAS]).T
        g = np.all(np.isfinite(A), axis=1)
        rows.append({
            "asset": asset, "n": nn, "pihat": v / nn,
            "p_kupiec": kupiec_p(v, nn, 0.01), "TL": traffic_light(v, nn),
            "width001": float(np.mean(np.abs(q[m]))),
            "max_abs_VaR": float(np.max(np.abs(q[m]))),
            "implied_z": float(np.median((q[m] - fc.loc[i, "mean"].values[m])
                                         / fc.loc[i, "std"].values[m])),
            "scale": float(np.median(q[k] / sd_real[k])),
            "frac_monotone": float(np.mean(np.all(np.diff(A[g], axis=1) > 0, axis=1))),
            **{f"pihat_{a:g}": float(np.mean(r < fc.loc[i, f"VaR_{a:g}"].values))
               for a in ALPHAS}})
    sc = pd.DataFrame(rows)
    su = pd.DataFrame(summary).merge(sc, on="asset", how="outer")
    su.to_csv(OUT / "gjr_t_summary.csv", index=False)

    L = ["# GJR-GARCH with a Student-$t$ innovation", "",
         "Added because correcting GJR to the Gaussian innovation the manuscript "
         "describes leaves it under-covering (π̂ = 0.0194 at a nominal 0.01, "
         "Kupiec 0/24). This is also the model the shipped series was attempting: "
         "it used `stats.t.ppf(alpha, 5)` with the degrees of freedom hard-coded "
         "and the *raw* rather than standardised quantile, so its innovation had "
         "variance 5/3 and every VaR was 1.29× too wide on top of that.", "",
         f"ν is estimated per window. Windows where the optimiser pins ν at its "
         f"lower bound (ν ≤ {NU_FLOOR}, where the variance ceases to exist and the "
         "standardised quantile diverges) are counted as degenerate and the last "
         "non-degenerate ν is carried forward. The rule was fixed before seeing "
         "which assets it touches.", "",
         "| asset | n | ν median | degenerate | failed fits | π̂(0.01) | "
         "Kupiec p | TL | mean width | max abs VaR |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in su.iterrows():
        fd = f"{int(r['n_degenerate'])} ({100 * r['frac_degenerate']:.1f}%)"
        if r["frac_degenerate"] > 0.01:
            fd = f"**{fd}**"
        L.append(f"| {r['asset']} | {int(r['n_dates'])} | {r['nu_median']:.2f} | {fd} | "
                 f"{int(r['n_failed_fit'])} | {r['pihat']:.4f} | "
                 f"{r['p_kupiec']:.4f} | {r['TL']} | {r['width001']:.5f} | "
                 f"{r['max_abs_VaR']:.4f} |")
    tot = su["n_degenerate"].sum()
    L += ["", f"**Panel:** mean π̂(0.01) = {sc['pihat'].mean():.4f}, "
          f"Kupiec pass {int((sc['p_kupiec'] > 0.05).sum())}/{len(sc)}, "
          f"Green {int((sc['TL'] == 'Green').sum())}/{len(sc)}, "
          f"mean width {sc['width001'].mean():.5f}, "
          f"median implied z(1%) {sc['implied_z'].median():+.3f}, "
          f"{int(tot)} degenerate windows in total.", "",
          "Not promoted by this script. `max abs VaR` is reported per asset "
          "because a single diverging window is invisible in a mean and inflates "
          "any width the benchmark is compared on.", ""]
    (OUT / "GJR_T.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-6:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
