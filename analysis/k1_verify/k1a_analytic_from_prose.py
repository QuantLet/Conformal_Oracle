"""K1a. The analytic Chronos estimator, rebuilt from the manuscript's prose.

Written against Section 4.4 of main_R2.tex (the five numbered steps) and against
the `chronos` package's own tokenizer source. No file in scripts/, pipeline/ or
Quantlets/ was opened while writing this.

The five steps, quoted:
  1. apply the softmax to the logits, giving probabilities over token identifiers;
  2. map each identifier to its bin centre, offset by the special-token count;
  3. multiply by the tokenizer's own scale, the mean absolute value of the context;
  4. sort the support and take the cumulative sum of the probabilities;
  5. read VaR_alpha as the smallest value whose cumulative probability reaches alpha.

Step 1 is ambiguous about the two special tokens, which also carry softmax mass.
Both readings are computed and both are reported.
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

ASSET = "SP500"
CKPT = "amazon/chronos-t5-small"
CONTEXT = 512
BLOCK_START, BLOCK_N = 2939, 200          # pre-registered: 200 contiguous mid-series dates
LEVELS = [0.01, 0.025, 0.05, 0.1]

returns = pd.read_csv(f"cfp_ijf_data/returns/{ASSET}.csv", parse_dates=["date"]).set_index("date")["log_return"]
stored = pd.read_parquet(f"cfp_ijf_data/chronos_small_analytic/{ASSET}.parquet")
print(f"returns {len(returns)} rows, stored {len(stored)} rows, offset {len(returns)-len(stored)}")

dates = stored.index[BLOCK_START:BLOCK_START + BLOCK_N]
print(f"block: {dates[0].date()} .. {dates[-1].date()}  ({len(dates)} dates)")

pipe = ChronosPipeline.from_pretrained(CKPT, device_map="cpu", torch_dtype=torch.float32)
cfg = pipe.tokenizer.config
model = pipe.model.model
n_special = cfg.n_special_tokens
centers = pipe.tokenizer.centers.numpy().astype(np.float64)     # 4093 bin centres, scaled units
n_bins = len(centers)
print(f"vocab {cfg.n_tokens}, special {n_special}, bins {n_bins}, "
      f"centre range [{centers[0]:.4f}, {centers[-1]:.4f}]")

# token id -> bin centre, per the tokenizer's own output_transform: centres[id - n_special - 1]
value_ids = np.arange(n_special + 1, n_special + 1 + n_bins)

pos = {d: returns.index.get_loc(d) for d in dates}
rows = []
BATCH = 20
with torch.no_grad():
    for b0 in range(0, len(dates), BATCH):
        chunk = dates[b0:b0 + BATCH]
        ctxs = torch.stack([torch.tensor(returns.iloc[pos[d] - CONTEXT:pos[d]].to_numpy(), dtype=torch.float32)
                            for d in chunk])
        ids, mask, scale = pipe.tokenizer.context_input_transform(ctxs)
        enc = model.get_encoder()(input_ids=ids, attention_mask=mask)
        dec_in = torch.full((len(chunk), 1), model.config.decoder_start_token_id, dtype=torch.long)
        out = model(encoder_outputs=enc, attention_mask=mask, decoder_input_ids=dec_in)
        logits = out.logits[:, 0, :].double()                       # (B, n_tokens)
        p_all = torch.softmax(logits, dim=-1).numpy()               # step 1
        for j, d in enumerate(chunk):
            p_v = p_all[j, value_ids]                               # step 2: value tokens only
            support = centers * float(scale[j])                     # step 3
            order = np.argsort(support, kind="stable")              # step 4
            s_sorted, p_sorted = support[order], p_v[order]
            rec = {"date": d, "scale": float(scale[j]), "p_special": float(p_all[j, :n_special + 1].sum())}
            for name, p in (("keep", p_sorted), ("renorm", p_sorted / p_sorted.sum())):
                cum = np.cumsum(p)
                for a in LEVELS:                                    # step 5
                    k = int(np.searchsorted(cum, a, side="left"))
                    rec[f"VaR_{a}_{name}"] = s_sorted[min(k, len(s_sorted) - 1)]
            rows.append(rec)
        print(f"  {b0 + len(chunk)}/{len(dates)}", flush=True)

mine = pd.DataFrame(rows).set_index("date")
ref = stored.loc[dates]

def compare(a, b, tag):
    rel = np.abs(a.to_numpy() - b.to_numpy()) / np.abs(b.to_numpy())
    med, mx = float(np.median(rel)), float(np.max(rel))
    ok = med < 1e-3 and mx < 1e-2
    print(f"{tag:44s} median|rel| {med:.3e}   max|rel| {mx:.3e}   {'AGREE' if ok else 'DISAGREE'}")
    return {"tag": tag, "median_rel": med, "max_rel": mx, "agree": bool(ok)}

print(f"\nspecial-token softmax mass: median {mine.p_special.median():.3e}  max {mine.p_special.max():.3e}")
print("\n--- reimplementation vs stored analytic series ---")
res = [compare(mine[f"VaR_{a}_{n}"], ref[f"VaR_{a}"], f"VaR_{a} ({n})") for a in LEVELS for n in ("keep", "renorm")]

print("\n--- NEGATIVE CONTROLS (each must DISAGREE) ---")
bin_w = float(np.diff(centers).mean())
neg = [compare(mine["VaR_0.01_renorm"] + bin_w * mine["scale"], ref["VaR_0.01"], "stored shifted by one bin width")]
sampled = pd.read_parquet(f"cfp_ijf_data/chronos_small/{ASSET}.parquet").loc[dates]
neg.append(compare(mine["VaR_0.01_renorm"], sampled["VaR_0.01"], "vs default-sampled series"))
neg.append(compare(mine["VaR_0.01_renorm"], ref["VaR_0.1"], "vs stored VaR_0.10 (wrong level)"))

controls_ok = all(not n["agree"] for n in neg)
print(f"\nnegative controls all failed as required: {controls_ok}")
if not controls_ok:
    print("!! a negative control AGREED — the comparison rule cannot detect a wrong answer")

mine.to_csv("analysis/k1_verify/k1a_reimplementation.csv")
json.dump({"asset": ASSET, "checkpoint": CKPT, "block": [str(dates[0].date()), str(dates[-1].date())],
           "n_dates": len(dates), "comparisons": res, "negative_controls": neg,
           "negative_controls_all_failed": controls_ok},
          open("analysis/k1_verify/k1a_result.json", "w"), indent=2)
