"""K1a, part 3. Monte Carlo at 2000 draws cannot resolve a one-bin gap: its own
error is about five bins. The mapping is settled deterministically instead.

top_k = 1 forces every draw to the arg-max token. The value the library decodes it
to is then a fact about the library's own decoder, with no sampling error at all.
Candidate A pairs p[id] with centres[id - n_special - 1]; candidate B with
centres[id - n_special].
"""
import numpy as np, pandas as pd, torch, json
from chronos import ChronosPipeline

CONTEXT = 512
pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32)
cfg, model = pipe.tokenizer.config, pipe.model.model
ns, centers = cfg.n_special_tokens, pipe.tokenizer.centers.numpy().astype(np.float64)
binw = float(np.diff(centers).mean())
res = []
for asset in ["SP500", "GOLD", "EURUSD"]:
    r = pd.read_csv(f"cfp_ijf_data/returns/{asset}.csv", parse_dates=["date"]).set_index("date")["log_return"]
    stored = pd.read_parquet(f"cfp_ijf_data/chronos_small_analytic/{asset}.parquet")
    for d in stored.index[3000:3004]:
        i = r.index.get_loc(d)
        ctx = torch.tensor(r.iloc[i-CONTEXT:i].to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            ids, mask, scale = pipe.tokenizer.context_input_transform(ctx.unsqueeze(0))
            enc = model.get_encoder()(input_ids=ids, attention_mask=mask)
            dec = torch.full((1,1), model.config.decoder_start_token_id, dtype=torch.long)
            logits = model(encoder_outputs=enc, attention_mask=mask, decoder_input_ids=dec).logits[0,0].double()
            torch.manual_seed(0)
            greedy = pipe.predict(ctx, prediction_length=1, num_samples=4, top_k=1, top_p=1.0,
                                  temperature=1.0, limit_prediction_length=False)[0,:,0].double().numpy()
        sc = float(scale[0]); j = int(torch.argmax(logits))
        A = centers[j - ns - 1] * sc
        B = centers[min(j - ns, len(centers)-1)] * sc
        v = float(np.unique(greedy)[0]); n_uni = len(np.unique(greedy))
        res.append({"asset": asset, "date": str(d.date()), "argmax_id": j, "n_unique_greedy": n_uni,
                    "decoded": v, "A": A, "B": B,
                    "gap_A_bins": (v - A)/(binw*sc), "gap_B_bins": (v - B)/(binw*sc)})
        print(f"{asset:7s} {d.date()}  argmax id {j:5d}  library decodes to {v: .8f}   "
              f"A {A: .8f} ({(v-A)/(binw*sc):+.3f} bins)   B {B: .8f} ({(v-B)/(binw*sc):+.3f} bins)   "
              f"-> {'A' if abs(v-A) < abs(v-B) else 'B'}")
votes = sum(1 for x in res if abs(x["decoded"]-x["A"]) < abs(x["decoded"]-x["B"]))
print(f"\nthe library's own decoder agrees with candidate A on {votes}/{len(res)} cells")
print("candidate A is what this reimplementation used; the stored analytic series is candidate B.")
json.dump(res, open("analysis/k1_verify/k1a_decide.json","w"), indent=2)
