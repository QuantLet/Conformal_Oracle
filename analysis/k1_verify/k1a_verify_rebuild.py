"""K1a, part 5. Does the regenerated analytic panel match what R14 predicted?

Unit of analysis: one asset-date-checkpoint. Expected 121,923 rows per
checkpoint over 24 files, on exactly the dates the shipped panels carry.

The comparison is against the *object* — the per-date quantile — not against a
summary of it, which is the corollary R14 added to Rule 2. Three checks that
must agree and three negative controls that must not.
"""
import json
import numpy as np
import pandas as pd
import pathlib

BINW, CONTEXT, LEVELS = 30.0 / 4092.0, 512, [0.01, 0.025, 0.05, 0.1]
ASSETS = sorted(p.stem for p in pathlib.Path("cfp_ijf_data/returns").glob("*.csv"))
PANELS = [("small", "chronos_small_analytic"), ("mini", "chronos_mini_analytic")]

def scale_series(asset, index):
    r = pd.read_csv(f"cfp_ijf_data/returns/{asset}.csv", parse_dates=["date"]).set_index("date")["log_return"]
    r = r[r.abs() <= 0.50]
    pos = r.index.get_indexer(index)
    assert (pos >= CONTEXT).all(), f"{asset}: a panel date has no 512-day context"
    return np.array([np.abs(r.iloc[p - CONTEXT:p].to_numpy()).mean() for p in pos]), r

report = {}
for tag, sub in PANELS:
    old_d, new_d = pathlib.Path("cfp_ijf_data", sub), pathlib.Path("cfp_ijf_data", sub + "_r14")
    rel_all, off_all, ctrl_all = {a: [] for a in LEVELS}, {a: [] for a in LEVELS}, []
    n_rows, n_files, sd_rel, mu_off = 0, 0, [], []
    for asset in ASSETS:
        old = pd.read_parquet(old_d / f"{asset}.parquet")
        new = pd.read_parquet(new_d / f"{asset}.parquet")
        assert new.index.equals(old.index), f"{asset}: date index moved"
        assert not new.isna().any().any(), f"{asset}: NaN in rebuilt panel"
        n_rows += len(new); n_files += 1
        sc, _ = scale_series(asset, new.index)
        for a in LEVELS:                                   # quantiles must stay ordered
            if a != LEVELS[0]:
                assert (new[f"VaR_{a:g}"].to_numpy() >= new[f"VaR_{prev:g}"].to_numpy() - 1e-12).all(), \
                    f"{asset}: VaR not monotone in alpha at {a}"
            prev = a
        for a in LEVELS:
            pred = old[f"VaR_{a:g}"].to_numpy() - BINW * sc
            got = new[f"VaR_{a:g}"].to_numpy()
            rel_all[a].append(np.abs(got - pred) / np.abs(pred))
            off_all[a].append((old[f"VaR_{a:g}"].to_numpy() - got) / (BINW * sc))
        # negative control: the rebuilt series must NOT match the unshifted stored one
        ctrl_all.append(np.abs(new["VaR_0.01"].to_numpy() - old["VaR_0.01"].to_numpy())
                        / np.abs(old["VaR_0.01"].to_numpy()))
        sd_rel.append(np.abs(new["std"].to_numpy() - old["std"].to_numpy()) / old["std"].to_numpy())
        mu_off.append((old["mean"].to_numpy() - new["mean"].to_numpy()) / (BINW * sc))
    print(f"\n=== {sub} ===  {n_files} files, {n_rows} rows")
    assert n_files == 24 and n_rows == 121923, "row/file count does not match the shipped panel"
    r = {"n_files": n_files, "n_rows": n_rows, "levels": {}}
    for a in LEVELS:
        rel, off = np.concatenate(rel_all[a]), np.concatenate(off_all[a])
        one_bin = float(np.mean(np.abs(off - 1) < 1e-3))
        print(f"  a={a:<6} vs stored-minus-one-bin: median|rel| {np.median(rel):.2e}  max {np.max(rel):.2e}"
              f" | shift {off.mean():.5f} bins (sd {off.std():.1e}), exactly one bin on {one_bin:.4%} of dates")
        r["levels"][str(a)] = {"median_rel": float(np.median(rel)), "max_rel": float(np.max(rel)),
                               "mean_shift_bins": float(off.mean()), "frac_exactly_one_bin": one_bin}
    ctrl = np.concatenate(ctrl_all)
    sdr, mo = np.concatenate(sd_rel), np.concatenate(mu_off)
    print(f"  NEGATIVE CONTROL, rebuilt vs stored unshifted: median|rel| {np.median(ctrl):.2e} "
          f"-> {'DISAGREE, as required' if np.median(ctrl) > 1e-3 else '!! AGREED — check is blind'}")
    print(f"  columns the arithmetic rule could not touch: std median|rel| {np.median(sdr):.2e} "
          f"max {np.max(sdr):.2e} | mean shift {mo.mean():.4f} bins")
    r["negative_control_median_rel"] = float(np.median(ctrl))
    r["negative_control_disagrees"] = bool(np.median(ctrl) > 1e-3)
    r["std_median_rel"] = float(np.median(sdr)); r["std_max_rel"] = float(np.max(sdr))
    r["mean_shift_bins"] = float(mo.mean())
    report[sub] = r

json.dump(report, open("analysis/k1_verify/k1a_verify_rebuild.json", "w"), indent=2)
print("\nwrote analysis/k1_verify/k1a_verify_rebuild.json")
