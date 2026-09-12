"""Read-only arithmetic for the R8 referee review; no estimation or inference.

Run from any directory with Python 3. Outputs only to this review directory.
QS uses the existing posthoc.csv pair supports and is multiplied by 10,000.
These supports must not be confused with the cross-model common-date table.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/review_20260909/calendar/before_adoption/results/posthoc.csv"
payload = SOURCE.read_bytes()
assert hashlib.sha256(payload).hexdigest()=="1e2627dbf59d3172c6bfbf64a454ea99dcfcbc8222819890500ae093b617728a", "The historical referee arithmetic requires its original input"
rows = list(csv.DictReader(payload.decode().splitlines()))
pairs = {}
for row in rows:
    key = (row["model"], row["asset"])
    methods = pairs.setdefault(key, {})
    assert row["method"] not in methods, (key, row["method"])
    methods[row["method"]] = float(row["QS"]) * 10000
assert len(pairs) == 216, len(pairs)
assert all({"Raw", "Conformal", "Scale", "rolling"} <= set(v)
           for v in pairs.values())
models = sorted({key[0] for key in pairs})
methods = sorted(set.intersection(*(set(v) for v in pairs.values())))


def means(selected):
    return {method: mean(pair[method] for pair in selected)
            for method in methods}


overall = means(list(pairs.values()))
by_model = {model: means([v for k, v in pairs.items() if k[0] == model])
            for model in models}
without_lag = [v for k, v in pairs.items() if k[0] != "Lag-Llama"]
total_gain = sum(v["Raw"] - v["Conformal"] for v in pairs.values())
lag_gain = sum(v["Raw"] - v["Conformal"]
               for k, v in pairs.items() if k[0] == "Lag-Llama")
result = {
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(payload).hexdigest(),
    "n_pairs": len(pairs),
    "n_pairs_without_lag_llama": len(without_lag),
    "qs_multiplier": 10000,
    "support": "Existing per-pair posthoc test support; no dates changed",
    "interpretation": "Descriptive arithmetic only; no causal or inferential claim",
    "mean_qs_by_method": overall,
    "mean_qs_by_model_and_method": by_model,
    "mean_qs_without_lag_llama": means(without_lag),
    "lag_llama_share_of_summed_raw_minus_static_gain": lag_gain / total_gain,
    "relative_static_gain_over_raw": (overall["Raw"] - overall["Conformal"]) / overall["Raw"],
    "relative_static_gain_over_scale": (overall["Scale"] - overall["Conformal"]) / overall["Scale"],
    "analytical_illustration_only": {
        "n": 4000, "alpha": 0.01, "iid_K": 0.25, "beta_gap": 0,
        "intermediate_proof_floor": 0.99 - 1 / math.sqrt(4000),
        "displayed_envelope_floor": 0.99 - math.sqrt(math.log(4000) / (4000 * math.log(2))),
        "note": "Evaluates the paper's loose bounds, not actual iid conformal coverage",
    },
}
out = Path(__file__).with_name("review_arithmetic.json")
out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: result[k] for k in (
    "source_sha256", "n_pairs", "n_pairs_without_lag_llama",
    "lag_llama_share_of_summed_raw_minus_static_gain",
    "relative_static_gain_over_raw", "relative_static_gain_over_scale")}, indent=2))
print("Without Lag-Llama:", result["mean_qs_without_lag_llama"])
