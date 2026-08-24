"""K1a, part 2. My reimplementation and the stored analytic series differ by exactly
one bin width. Both cannot be right. The model itself adjudicates: draw at full
vocabulary through the library's own sampling path and compare empirical quantiles
against the two candidate analytic supports.

The library's decode is tokenizer.output_transform: id -> centres[id - n_special - 1].
Candidate A (mine)   pairs p[id] with centres[id - n_special - 1].
Candidate B (stored) pairs p[id] with centres[id - n_special].
"""
import numpy as np, pandas as pd, torch, json
from chronos import ChronosPipeline

CONTEXT, LEVELS = 512, [0.01, 0.025, 0.05, 0.1]
out = {}

for ckpt, store, assets in [("amazon/chronos-t5-small", "chronos_small_analytic", ["SP500", "GOLD", "EURUSD"]),
                            ("amazon/chronos-t5-mini",  "chronos_mini_analytic",  ["SP500"])]:
    pipe = ChronosPipeline.from_pretrained(ckpt, device_map="cpu", torch_dtype=torch.float32)
    cfg, model = pipe.tokenizer.config, pipe.model.model
    ns = cfg.n_special_tokens
    centers = pipe.tokenizer.centers.numpy().astype(np.float64)
    binw = float(np.diff(centers).mean())
    for asset in assets:
        r = pd.read_csv(f"cfp_ijf_data/returns/{asset}.csv", parse_dates=["date"]).set_index("date")["log_return"]
        stored = pd.read_parquet(f"cfp_ijf_data/{store}/{asset}.parquet")
        dates = stored.index[3000:3005]
        recs = []
        with torch.no_grad():
            for d in dates:
                i = r.index.get_loc(d)
                ctx = torch.tensor(r.iloc[i-CONTEXT:i].to_numpy(), dtype=torch.float32).unsqueeze(0)
                ids, mask, scale = pipe.tokenizer.context_input_transform(ctx)
                enc = model.get_encoder()(input_ids=ids, attention_mask=mask)
                dec = torch.full((1,1), model.config.decoder_start_token_id, dtype=torch.long)
                p = torch.softmax(model(encoder_outputs=enc, attention_mask=mask,
                                        decoder_input_ids=dec).logits[0,0].double(), -1).numpy()
                sc = float(scale[0])
                pv = p[ns+1: ns+1+len(centers)]
                pv = pv / pv.sum()
                cumA = np.cumsum(pv)                                   # support centres*sc
                rec = {"asset": asset, "ckpt": ckpt, "date": d, "scale": sc}
                for a in LEVELS:
                    k = int(np.searchsorted(cumA, a, side="left"))
                    rec[f"A_{a}"] = centers[min(k, len(centers)-1)] * sc
                    rec[f"B_{a}"] = rec[f"A_{a}"] + binw * sc
                    rec[f"S_{a}"] = float(stored.loc[d, f"VaR_{a}"])
                # the model's own sampler, full vocabulary, 4000 draws
                torch.manual_seed(0)
                fc = pipe.predict(ctx.squeeze(0), prediction_length=1, num_samples=2000,
                                  top_k=4094, top_p=1.0, temperature=1.0, limit_prediction_length=False)
                s = fc[0,:,0].double().numpy()
                for a in LEVELS:
                    rec[f"MC_{a}"] = float(np.quantile(s, a, method="inverted_cdf"))
                rec["n_distinct_full"] = int(len(np.unique(s)))
                recs.append(rec)
        df = pd.DataFrame(recs)
        for a in LEVELS:
            dA = np.abs(df[f"A_{a}"] - df[f"MC_{a}"]).mean()
            dB = np.abs(df[f"B_{a}"] - df[f"MC_{a}"]).mean()
            dS = np.abs(df[f"S_{a}"] - df[f"MC_{a}"]).mean()
            hitA = int((np.abs(df[f"A_{a}"] - df[f"MC_{a}"]) < 1e-12).sum())
            hitB = int((np.abs(df[f"B_{a}"] - df[f"MC_{a}"]) < 1e-12).sum())
            print(f"{asset:7s} {ckpt.split('-')[-1]:6s} a={a:<6} "
                  f"|A-MC| {dA:.3e} (exact {hitA}/5)  |B-MC| {dB:.3e} (exact {hitB}/5)  |stored-MC| {dS:.3e}")
            out[f"{asset}|{ckpt}|{a}"] = {"mean_abs_A_MC": dA, "mean_abs_B_MC": dB, "mean_abs_stored_MC": dS,
                                          "exact_hits_A": hitA, "exact_hits_B": hitB}
        offs = (df["S_0.01"] - df["A_0.01"]) / (binw * df["scale"])
        print(f"        stored minus A, in bin widths: mean {offs.mean():.6f}  sd {offs.std():.2e}")
        out[f"{asset}|{ckpt}|offset_bins"] = {"mean": float(offs.mean()), "sd": float(offs.std())}
        df.to_csv(f"analysis/k1_verify/k1a_adjudicate_{asset}_{ckpt.split('/')[-1]}.csv", index=False)
json.dump(out, open("analysis/k1_verify/k1a_adjudicate.json","w"), indent=2, default=float)
