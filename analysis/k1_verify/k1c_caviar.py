"""K1c. CAViaR passes Kupiec on 15 of 24 assets: the pass/fail vector, not the count.

A second implementation already exists in analysis/phase3_dynamic (Engle--Manganelli
recursions written directly, Powell optimiser, different starting values). The
pre-registered comparison is the per-asset vector, because two implementations can
agree on 15 and disagree on which fifteen.
"""
import numpy as np, pandas as pd, json
from scipy import stats

orig = pd.read_csv("analysis/phase3_dynamic/dynamic_var.csv")
ver = pd.read_csv("analysis/phase3_dynamic/caviar_verification.csv")
out = {}
for model in ["CAViaR-AS", "CAViaR-SAV"]:
    o = orig[orig.model == model].set_index("asset")[["pihat_raw", "p_kup_raw", "viol_raw", "n_test"]]
    v = ver[ver.model == model].set_index("asset")[["orig_pihat", "orig_p", "orig_n", "comm_pihat", "comm_p"]]
    j = o.join(v, how="inner")
    j["pass_orig"] = j.p_kup_raw > 0.05
    j["pass_ver"] = j.orig_p > 0.05
    j["pass_ver_common"] = j.comm_p > 0.05
    agree = int((j.pass_orig == j.pass_ver).sum())
    agree_c = int((j.pass_orig == j.pass_ver_common).sum())
    print(f"{model}: original {int(j.pass_orig.sum())}/24, second implementation {int(j.pass_ver.sum())}/24 "
          f"(common window {int(j.pass_ver_common.sum())}/24); "
          f"per-asset vector agrees on {agree}/24 (common window {agree_c}/24)")
    dis = j[j.pass_orig != j.pass_ver_common]
    if len(dis):
        print("   assets where the common-window vector differs:")
        print(dis[["p_kup_raw", "comm_p"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"   max |pihat difference| across assets: {float(np.max(np.abs(j.pihat_raw - j.orig_pihat))):.2e}")
    out[model] = {"pass_original": int(j.pass_orig.sum()), "pass_second_impl": int(j.pass_ver.sum()),
                  "pass_second_impl_common_window": int(j.pass_ver_common.sum()),
                  "vector_agreement": agree, "vector_agreement_common": agree_c,
                  "max_abs_pihat_diff": float(np.max(np.abs(j.pihat_raw - j.orig_pihat))),
                  "mean_pihat": float(j.pihat_raw.mean())}
    j.to_csv(f"analysis/k1_verify/k1c_{model}.csv")

print("\n--- NEGATIVE CONTROL: Kupiec on a series violating at 5% against nominal 1% ---")
def kupiec_p(nv, n, a):
    pi = nv/n
    lr = -2*((n-nv)*np.log(1-a) + nv*np.log(a) - (n-nv)*np.log(1-pi) - nv*np.log(pi))
    return float(stats.chi2.sf(max(lr,0.0), 1))
rng = np.random.default_rng(0)
rej = sum(kupiec_p(int(rng.binomial(n, 0.05)), n, 0.01) <= 0.05 for n in orig[orig.model=="CAViaR-AS"].n_test)
print(f"   rejects on {rej}/24 constructed series -> {'control fires as required' if rej == 24 else '!! CONTROL FAILED'}")
out["negative_control_rejections"] = rej
json.dump(out, open("analysis/k1_verify/k1c_result.json", "w"), indent=2)
