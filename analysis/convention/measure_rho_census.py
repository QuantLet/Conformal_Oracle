"""Where the separation gap is unconstrained: a census of rho-hat.

Pre-registered in PREREG_RHO_CENSUS.md. rho-hat comes from the calibration
scores of the pipeline's own load_pair, so the census describes the object the
corollary is applied to and not a re-derivation of it.
"""
import sys, importlib.util
from math import ceil, log
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path("/Users/danpele/dev/cfp-llm-var")
OUT = BASE / "analysis" / "convention" / "rho_census.csv"
TRUNCATED = {"Chronos-Small", "Chronos-Mini"}   # run_k0a.py:22, same definition
sys.path.insert(0, str(BASE / "Quantlets"))
spec = importlib.util.spec_from_file_location(
    "rfe", BASE / "Quantlets" / "CO_full_evaluation" / "run_full_evaluation.py")
rfe = importlib.util.module_from_spec(spec); spec.loader.exec_module(rfe)


def rho_and_gap(scores, n_cal):
    rho = pd.Series(np.asarray(scores)).autocorr(lag=1)
    if rho and 0.0 < rho < 0.999:
        return rho, max(5, int(ceil((1.0 / abs(log(rho))) * log(n_cal)))), False
    return rho, max(5, int(ceil(log(n_cal)))), True


def control():
    """The census must tell a persistent score sequence from an i.i.d. one."""
    rng = np.random.default_rng(0)
    e = rng.standard_normal(2000); ar = np.empty(2000); ar[0] = e[0]
    for i in range(1, 2000):
        ar[i] = 0.6 * ar[i - 1] + e[i]
    r_ar, g_ar, fb_ar = rho_and_gap(ar, 2000)
    r_iid, g_iid, fb_iid = rho_and_gap(rng.standard_normal(2000) * 0 + e * 0 + rng.standard_normal(2000), 2000)
    ok = (r_ar > 0.5) and (not fb_ar) and (g_ar > 5)
    print(f"  ctrl   AR(1) rho=0.6 -> rho_hat {r_ar:.3f}, gap {g_ar}, fallback {fb_ar}")
    print(f"  ctrl   i.i.d.        -> rho_hat {r_iid:+.4f}, gap {g_iid}, fallback {fb_iid}")
    if not ok:
        sys.exit("control failed: the census cannot see persistence")
    return True


def main():
    control()
    rows = []
    for model in rfe.MODELS:
        for sym in sorted(rfe.SYMBOLS):
            try:
                got = rfe.load_pair(model, sym, 0.01)
            except Exception:
                got = None
            if got is None:
                continue
            r, v = np.asarray(got[0]), np.asarray(got[1])
            n_cal = int(len(r) * rfe.F_CAL)
            rho, g, fb = rho_and_gap(v[:n_cal] - r[:n_cal], n_cal)
            rows.append({"model": model, "asset": sym, "n_cal": n_cal,
                         "rho": rho, "gap": g, "fallback": fb,
                         "truncated": model in TRUNCATED})
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    from scipy.stats import fisher_exact
    fb = d[d.fallback]
    p = fb.truncated.mean()
    p0 = d.truncated.mean()
    tab = [[int((fb.truncated).sum()), int((~fb.truncated).sum())],
           [int((d[~d.fallback].truncated).sum()), int((~d[~d.fallback].truncated).sum())]]
    _, pv = fisher_exact(tab)
    print(f"\n  {len(d)} cells at alpha=0.01, {len(fb)} with rho_hat <= 0 "
          f"(or undefined); of those {tab[0][0]} on truncated series")
    print(f"  share on truncated {p:.3f} against panel share {p0:.3f}, "
          f"ratio {p/p0:.2f}; Fisher p = {pv:.3g}")
    claim = (p / p0 >= 2) and (pv < 0.05)
    print(f"  pre-registered rule (ratio >= 2 AND p < 0.05): "
          f"{'concentration claimed' if claim else 'NO concentration --- do not write the sentence'}")
    tr = d[d.truncated]
    print(f"  of the {len(tr)} truncated cells, {int(tr.fallback.sum())} fall back; "
          f"of the {len(d)-len(tr)} others, {int(d[~d.truncated].fallback.sum())}")
    print(f"  median rho_hat: truncated {tr.rho.median():+.4f}, "
          f"other {d[~d.truncated].rho.median():+.4f}")
    print(f"  wrote {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
