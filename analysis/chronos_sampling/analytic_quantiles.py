#!/usr/bin/env python3
"""Read Chronos's predictive distribution directly instead of sampling it.

At prediction_length = 1 a Chronos-T5 checkpoint emits a single categorical
distribution over its 4096-token vocabulary. Sampling that distribution with
`top_k = 50` discards 98.8% of the support before any quantile is computed, and
even at full vocabulary a 1% quantile from 1000 draws is the 10th order
statistic and carries real Monte Carlo error.

The alternative needs no sampling at all:

    logits -> softmax -> probabilities over token ids
    token id -> bin centre (offset by n_special_tokens)
    bin centre * scale  (scale = mean|context|, the tokenizer's own)
    sort support, cumulative-sum the probabilities -> categorical CDF
    quantile(alpha) = smallest value whose CDF >= alpha

This is exact given the model's output, has no `num_samples`, no `top_k`, no
`top_p`, no seed, and costs one forward pass per date.

The script validates the analytic quantiles against the sampling route at full
vocabulary before using them: the two must agree to within Monte Carlo error, or
the reconstruction of the support is wrong.

Output: analysis/chronos_sampling/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
CONTEXT = 512
ALPHAS = [0.01, 0.025, 0.05, 0.10]


def analytic_setup(pipe):
    """Bin centres in scaled space and the special-token offset."""
    tok = pipe.tokenizer
    cfg = pipe.model.config
    return tok.centers.detach().cpu().numpy(), cfg.n_special_tokens


def step1_logits(pipe, ctx: torch.Tensor):
    """Logits of the first predicted token, plus the tokenizer scale."""
    ids, mask, scale = pipe.tokenizer.context_input_transform(ctx)
    dev = pipe.model.device
    ids, mask = ids.to(dev), mask.to(dev)
    inner = pipe.model.model                      # the HF seq2seq model
    dec = torch.zeros((ids.shape[0], 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        out = inner(input_ids=ids, attention_mask=mask, decoder_input_ids=dec)
    return out.logits[:, 0, :].float().cpu().numpy(), float(scale.reshape(-1)[0])


def analytic_quantiles(logits, scale, centers, n_special, alphas=ALPHAS):
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    # token id t maps to centers[t - n_special - 1], which is what the tokenizer's
    # own MeanScaleUniformBins.output_transform does. This file carried the same
    # off-by-one as build_analytic_series.py -- R14 -- and is the script that
    # produced the validation figures Section 4.4 quotes, so those figures were
    # measured on the defective support they were meant to check.
    ids = np.arange(len(p))
    keep = (ids >= n_special + 1) & (ids - n_special - 1 < len(centers))
    vals = centers[ids[keep] - n_special - 1] * scale
    pk = p[keep]
    pk = pk / pk.sum()
    o = np.argsort(vals)
    v, c = vals[o], np.cumsum(pk[o])
    return {a: float(v[int(np.searchsorted(c, a, side="left"))]) for a in alphas}, \
        float(np.sum(v * pk[o])), float(np.sqrt(np.sum(pk[o] * (v - np.sum(v * pk[o])) ** 2)))


def main() -> int:
    from chronos import ChronosPipeline
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    asset = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    n_dates = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    df = pd.read_csv(DATA / "returns" / f"{asset}.csv", parse_dates=["date"]) \
        .set_index("date").sort_index()
    vals = df[df["log_return"].abs() <= 0.50]["log_return"].values

    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small",
                                           dtype=torch.float32, device_map=dev)
    centers, n_special = analytic_setup(pipe)
    print(f"backend {dev} | centers {len(centers)} | n_special {n_special} "
          f"| span [{centers.min():.2f}, {centers.max():.2f}] scaled units",
          file=sys.stderr)

    idx = np.linspace(CONTEXT, len(vals) - 1, n_dates).astype(int)
    rows = []
    for t in idx:
        ctx = torch.tensor(vals[t - CONTEXT:t], dtype=torch.float32).unsqueeze(0)
        lg, scale = step1_logits(pipe, ctx)
        qa, mu, sd = analytic_quantiles(lg[0], scale, centers, n_special)
        torch.manual_seed(42)
        with torch.no_grad():
            s = pipe.predict(ctx, prediction_length=1, num_samples=4000,
                             top_k=4094, top_p=1.0, temperature=1.0)
        s = s[0, :, 0].cpu().numpy().astype(float)
        row = {"t": int(t), "scale": scale, "an_mean": mu, "an_std": sd,
               "sm_mean": float(s.mean()), "sm_std": float(s.std())}
        for a in ALPHAS:
            row[f"an_{a:g}"] = qa[a]
            row[f"sm_{a:g}"] = float(np.percentile(s, a * 100))
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"analytic_validation_{asset}.csv", index=False)

    print("\nValidation: analytic vs sampling at full vocabulary (4000 draws)")
    print(f"{'quantity':10s} {'analytic':>12s} {'sampled':>12s} {'rel.diff':>10s}")
    print(f"{'mean':10s} {r.an_mean.mean():12.6f} {r.sm_mean.mean():12.6f} "
          f"{abs(r.an_mean.mean()-r.sm_mean.mean())/abs(r.sm_mean.mean()):10.3%}")
    print(f"{'std':10s} {r.an_std.mean():12.6f} {r.sm_std.mean():12.6f} "
          f"{abs(r.an_std.mean()-r.sm_std.mean())/r.sm_std.mean():10.3%}")
    for a in ALPHAS:
        A, S = r[f"an_{a:g}"].mean(), r[f"sm_{a:g}"].mean()
        print(f"q({a:<6g}) {A:12.6f} {S:12.6f} {abs(A-S)/abs(S):10.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
