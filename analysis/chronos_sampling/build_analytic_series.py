#!/usr/bin/env python3
"""Chronos VaR from the model's own predictive distribution, for all 24 assets.

The shipped Chronos series were sampled under the checkpoint default
`top_k = 50`, which truncates the categorical predictive distribution to 50 of
its 4094 value bins before any quantile is taken. Coverage came out at 0.39 and
0.42 against a nominal 0.01, and that was read as over-dispersion -- a property
of the model. It is a property of the sampler configuration.

At prediction_length = 1 a Chronos-T5 checkpoint emits a single categorical
distribution over its vocabulary, so no sampling is needed at all:

    logits -> softmax -> probabilities over token ids
    token id -> bin centre (offset by n_special_tokens) -> * tokenizer scale
    sort the support, cumulative-sum the probabilities -> categorical CDF
    VaR(alpha) = smallest value whose CDF >= alpha

This is exact given the model's output, and has no `num_samples`, no `top_k`,
no `top_p`, no seed. It was validated against sampling at full vocabulary in
`analytic_quantiles.py`: the two agree on the standard deviation to within 0.3%.

BOTH series are kept. The paper reports the shipped configuration as the
artefact it is and the analytic quantiles as what the model actually says, so
the comparison is the finding rather than a correction hidden in a rerun.

Batched: contexts are stacked and pushed through the encoder-decoder in blocks,
which is what makes 24 assets x ~6000 dates x 2 model sizes tractable.

Usage:
    python build_analytic_series.py [--model small|mini|both] [--batch 64]
                                    [--assets SP500 GOLD] [--limit N]
Outputs:
    cfp_ijf_data/chronos_small_analytic/<asset>.parquet
    cfp_ijf_data/chronos_mini_analytic/<asset>.parquet
    analysis/chronos_sampling/ANALYTIC_SERIES.md
"""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import SYMBOLS  # noqa: E402

SPECS = {"small": ("amazon/chronos-t5-small", "chronos_small_analytic"),
         "mini": ("amazon/chronos-t5-mini", "chronos_mini_analytic")}


def load_returns(asset: str) -> pd.Series:
    df = pd.read_csv(DATA / "returns" / f"{asset}.csv", parse_dates=["date"]) \
        .set_index("date").sort_index()
    return df[df["log_return"].abs() <= 0.50]["log_return"]


def batch_logits(pipe, ctx: torch.Tensor):
    """First-step logits and tokenizer scale for a batch of contexts."""
    ids, mask, scale = pipe.tokenizer.context_input_transform(ctx)
    dev = pipe.model.device
    ids, mask = ids.to(dev), mask.to(dev)
    dec = torch.zeros((ids.shape[0], 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        out = pipe.model.model(input_ids=ids, attention_mask=mask,
                               decoder_input_ids=dec)
    return (out.logits[:, 0, :].float().cpu().numpy(),
            np.asarray(scale).reshape(-1).astype(float))


def analytic_batch(logits, scales, centers, n_special):
    """Categorical-CDF quantiles, mean and sd for each row of a logit batch."""
    ids = np.arange(logits.shape[1])
    keep = (ids >= n_special) & (ids - n_special < len(centers))
    base = centers[ids[keep] - n_special]
    order = np.argsort(base)
    base_sorted = base[order]

    lg = logits[:, keep]
    p = np.exp(lg - lg.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    p = p[:, order]
    cdf = np.cumsum(p, axis=1)

    vals = base_sorted[None, :] * scales[:, None]        # scale is per row
    mean = np.sum(vals * p, axis=1)
    var = np.sum(p * (vals - mean[:, None]) ** 2, axis=1)
    q = {}
    for a in ALPHAS:
        idx = np.array([np.searchsorted(c, a, side="left") for c in cdf])
        idx = np.clip(idx, 0, vals.shape[1] - 1)
        q[a] = vals[np.arange(vals.shape[0]), idx]
    return q, mean, np.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["small", "mini", "both"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap dates per asset (smoke test only)")
    a = ap.parse_args()

    from chronos import ChronosPipeline
    dev = ("mps" if torch.backends.mps.is_available()
           else ("cuda" if torch.cuda.is_available() else "cpu"))
    assets = a.assets or SYMBOLS
    which = list(SPECS) if a.model == "both" else [a.model]
    print(f"backend {dev} | {len(assets)} assets | models {which} "
          f"| batch {a.batch}", file=sys.stderr)

    summary = []
    for key in which:
        model_id, subdir = SPECS[key]
        dest = DATA / subdir
        dest.mkdir(parents=True, exist_ok=True)
        pipe = ChronosPipeline.from_pretrained(model_id, dtype=torch.float32,
                                               device_map=dev)
        centers = pipe.tokenizer.centers.detach().cpu().numpy()
        n_special = pipe.model.config.n_special_tokens
        t0 = time.time()

        for ai, asset in enumerate(assets):
            ser = load_returns(asset)
            vals, dates = ser.values, ser.index
            idx = np.arange(CONTEXT, len(vals))
            if a.limit:
                idx = idx[-a.limit:]
            rows = []
            for s in range(0, len(idx), a.batch):
                blk = idx[s:s + a.batch]
                ctx = torch.tensor(
                    np.stack([vals[t - CONTEXT:t] for t in blk]),
                    dtype=torch.float32)
                lg, sc = batch_logits(pipe, ctx)
                q, mu, sd = analytic_batch(lg, sc, centers, n_special)
                for j, t in enumerate(blk):
                    rows.append({"date": dates[t], "mean": float(mu[j]),
                                 "std": float(sd[j]),
                                 **{f"VaR_{al:g}": float(q[al][j]) for al in ALPHAS}})
            df = pd.DataFrame(rows).set_index("date")
            df.to_parquet(dest / f"{asset}.parquet")
            el = (time.time() - t0) / 60
            print(f"  [{key}] [{ai + 1:2d}/{len(assets)}] {asset:8s} "
                  f"{len(df):5d} dates | {el:.1f} min", file=sys.stderr, flush=True)
            summary.append({"model": key, "asset": asset, "n": len(df)})
        del pipe

    pd.DataFrame(summary).to_csv(OUT / "analytic_series_summary.csv", index=False)
    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
