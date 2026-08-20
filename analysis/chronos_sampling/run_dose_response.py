#!/usr/bin/env python3
"""Does `top_k` explain the Chronos predictive-dispersion anomaly?

Controlled A/B on identical contexts and seeds, varying only the sampling
configuration. Design and branch readings fixed in advance: PREREGISTRATION.md.

BACKEND CAVEAT: the published run used an A30 (CUDA); this runs on Apple MPS.
Sampling under a different backend is not the same experiment. Results here are
indicative of direction and magnitude and must be confirmed on the A30 before
entering the paper.

Outputs (analysis/chronos_sampling/):
    dose_response.csv     one row per (model, cell, date)
    samples/*.npy         raw sample paths, retained this time
    RESULTS.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
ASSETS = ["SP500", "GOLD", "BTC", "EURUSD"]   # P3: was SP500 only
CONTEXT = 512
N_SAMPLES = 1000
N_DATES = 200
ALPHAS = [0.01, 0.025, 0.05, 0.10]
SEED = 42

MODELS = [("chronos_small", "amazon/chronos-t5-small", "Chronos-Small"),
          ("chronos_mini", "amazon/chronos-t5-mini", "Chronos-Mini")]

# (label, top_k, top_p, temperature)
CELLS = [("top_k=50 (default)", 50, 1.0, 1.0),
         ("top_k=200", 200, 1.0, 1.0),
         ("top_k=1000", 1000, 1.0, 1.0),
         ("top_k=4094 (full vocab)", 4094, 1.0, 1.0),
         ("top_p=0.9 @ k=50", 50, 0.9, 1.0),
         ("top_p=0.99 @ k=50", 50, 0.99, 1.0),
         ("temp=0.5 @ k=50", 50, 1.0, 0.5),
         ("temp=2.0 @ k=50", 50, 1.0, 2.0)]


def load_returns(asset):
    df = pd.read_csv(DATA / "returns" / f"{asset}.csv", parse_dates=["date"]) \
        .set_index("date").sort_index()
    return df[df["log_return"].abs() <= 0.50]["log_return"]


def main() -> int:
    (OUT / "samples").mkdir(parents=True, exist_ok=True)
    from chronos import ChronosPipeline

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"backend: {dev} (published run used CUDA/A30)", file=sys.stderr)

    rets = {a: load_returns(a) for a in ASSETS}
    realised_sd = {a: float(pd.Series(r.values).rolling(250).std().median())
                   for a, r in rets.items()}

    rows = []
    for slug, model_id, label in MODELS:
        print(f"loading {model_id}", file=sys.stderr)
        pipe = ChronosPipeline.from_pretrained(model_id, dtype=torch.float32,
                                               device_map=dev)
        for cell, tk, tp, temp in CELLS:
          for ASSET in ASSETS:
            ret = rets[ASSET]; vals, dates = ret.values, ret.index
            idx = np.linspace(CONTEXT, len(vals) - 1, N_DATES).astype(int)
            paths = []
            for t in idx:
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                ctx = torch.tensor(vals[t - CONTEXT:t],
                                   dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    s = pipe.predict(ctx, prediction_length=1,
                                     num_samples=N_SAMPLES,
                                     top_k=tk, top_p=tp, temperature=temp)
                s = s[0, :, 0].cpu().numpy().astype(float)
                paths.append(s)
                q = {a: float(np.percentile(s, a * 100)) for a in ALPHAS}
                rows.append({
                    "model": label, "asset": ASSET, "cell": cell, "top_k": tk, "top_p": tp,
                    "temperature": temp, "date": str(dates[t].date()),
                    "realised": float(vals[t]),
                    "pred_mean": float(s.mean()), "pred_std": float(s.std()),
                    "n_distinct": int(len(np.unique(s))),
                    **{f"VaR_{a:g}": q[a] for a in ALPHAS},
                })
            np.save(OUT / "samples" / f"{slug}__{ASSET}__{cell.replace(' ', '_').replace('=', '')}.npy",
                    np.array(paths))
            m = np.mean([r["pred_std"] / realised_sd[r["asset"]] for r in rows
                         if r["model"] == label and r["cell"] == cell])
            print(f"  {label:14s} {cell:26s} pred_std/sd={m:.3f}",
                  file=sys.stderr)
        del pipe

    df = pd.DataFrame(rows)
    df["realised_sd"] = df["asset"].map(realised_sd)
    df.to_csv(OUT / "dose_response.csv", index=False)

    df["disp_i"] = df["pred_std"] / df["realised_sd"]
    g = df.groupby(["model", "cell", "top_k", "top_p", "temperature"]).agg(
        pred_std=("disp_i", "mean"), n_distinct=("n_distinct", "mean"),
        **{f"z_{a:g}": (f"VaR_{a:g}", "mean") for a in ALPHAS}).reset_index()
    g["disp"] = g["pred_std"]
    for a in ALPHAS:
        g[f"z_{a:g}"] = (g[f"z_{a:g}"] - df.groupby(
            ["model", "cell"])["pred_mean"].mean().values) / g["pred_std"].values

    L = ["# Chronos sampling dose-response", "",
         f"Asset {ASSET}, {N_DATES} dates, {N_SAMPLES} samples, seed {SEED}, "
         f"backend **{dev}** (published run: CUDA/A30 — indicative only).", "",
         f"Realised sigma (250d median): {realised_sd:.5f}. "
         "Target dispersion is ~1.0; the shipped series sit at 0.117 / 0.109.", "",
         "| Model | cell | top_k | top_p | temp | pred_std/sigma | distinct values |",
         "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {r['model']} | {r['cell']} | {int(r['top_k'])} | "
                 f"{r['top_p']} | {r['temperature']} | **{r['disp']:.3f}** | "
                 f"{r['n_distinct']:.0f} |")
    L.append("")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
