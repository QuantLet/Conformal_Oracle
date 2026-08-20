#!/usr/bin/env python3
"""Coverage under the default configuration versus the model's own distribution.

Dispersion showed the mechanism; this gives the number that decides what the
paper is. If Chronos lands near nominal once the predictive distribution is read
rather than censored, then no forecaster in the panel fails and the paper becomes
an account of a configuration trap rather than of model failure.

Three estimators over the same dates and contexts:
  analytic      categorical CDF over all 4093 value bins, no sampling
  top_k=4094    sampling, full support
  top_k=50      sampling, the checkpoint default -- reproduces the shipped series

Usage: run_coverage.py [ASSET] [N_DATES]
Output: analysis/chronos_sampling/coverage_<ASSET>.csv and COVERAGE.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
CONTEXT = 512
ALPHAS = [0.01, 0.025, 0.05, 0.10]

sys.path.insert(0, str(OUT))
from analytic_quantiles import analytic_setup, analytic_quantiles, step1_logits  # noqa: E402


def main() -> int:
    from chronos import ChronosPipeline
    asset = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    n_dates = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    df = pd.read_csv(DATA / "returns" / f"{asset}.csv", parse_dates=["date"]) \
        .set_index("date").sort_index()
    ser = df[df["log_return"].abs() <= 0.50]["log_return"]
    vals, dates = ser.values, ser.index

    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",
                                           dtype=torch.float32, device_map=dev)
    centers, n_special = analytic_setup(pipe)

    # most recent n_dates, which is where the paper's test window sits
    idx = np.arange(max(CONTEXT, len(vals) - n_dates), len(vals))
    print(f"{asset}: {len(idx)} dates on {dev}", file=sys.stderr)

    rows = []
    t0 = time.time()
    for k, t in enumerate(idx):
        ctx = torch.tensor(vals[t - CONTEXT:t], dtype=torch.float32).unsqueeze(0)
        lg, scale = step1_logits(pipe, ctx)
        qa, mu, sd = analytic_quantiles(lg[0], scale, centers, n_special)
        row = {"date": dates[t], "realised": float(vals[t]),
               "an_std": sd, **{f"an_{a:g}": qa[a] for a in ALPHAS}}
        for tk in (4094, 50):
            torch.manual_seed(42)
            with torch.no_grad():
                s = pipe.predict(ctx, prediction_length=1, num_samples=1000,
                                 top_k=tk, top_p=1.0, temperature=1.0)
            s = s[0, :, 0].cpu().numpy().astype(float)
            row[f"k{tk}_std"] = float(s.std())
            row[f"k{tk}_distinct"] = int(len(np.unique(s)))
            for a in ALPHAS:
                row[f"k{tk}_{a:g}"] = float(np.percentile(s, a * 100))
        rows.append(row)
        if k % 200 == 0:
            print(f"  {k}/{len(idx)}  {(time.time()-t0)/max(k,1):.2f}s/date",
                  file=sys.stderr, flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"coverage_{asset}.csv", index=False)
    sd_real = float(ser.rolling(250).std().median())

    L = [f"# Chronos coverage: default configuration vs the model's own distribution",
         "", f"Asset {asset}, {len(r)} dates, backend {dev} "
         "(published run: CUDA/A30 — indicative).",
         f" Realised sigma {sd_real:.5f}.", "",
         "| estimator | dispersion | distinct values | "
         + " | ".join(f"π̂({a:g})" for a in ALPHAS) + " |",
         "|---|---|---|" + "---|" * len(ALPHAS)]
    for lab, pre, dcol in (("analytic (all 4093 bins)", "an", None),
                           ("sampled, top_k=4094", "k4094", "k4094_distinct"),
                           ("sampled, top_k=50 (default)", "k50", "k50_distinct")):
        pis = [f"**{float((r['realised'] < r[f'{pre}_{a:g}']).mean()):.4f}**"
               for a in ALPHAS]
        disp = r[f"{pre}_std"].mean() / sd_real
        dis = f"{r[dcol].mean():.0f}" if dcol else "4093 (exact)"
        L.append(f"| {lab} | {disp:.3f} | {dis} | " + " | ".join(pis) + " |")
    L += ["", "Nominal is the α in each column header.", ""]
    (OUT / f"COVERAGE_{asset}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
