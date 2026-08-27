"""Does a Basel zone upgrade mean the forecast got closer to its nominal level?"""
import numpy as np, pandas as pd, json
ALPHA, TAU = 0.01, 4/250
RANK = {"Green": 0, "Yellow": 1, "Red": 2}
d = pd.read_csv("analysis/ae_point4/pairs_long.csv")
d = d[np.isclose(d.alpha, ALPHA)].copy()
print(f"{len(d)} pairs, {d.model.nunique()} forecasters x {d.asset.nunique()} assets\n")

out = {}
for est, tl, pi, dqs in (("static","TL_static","pihat_static","dQS_static"),
                         ("rolling","TL_roll","pihat_roll","dQS_roll")):
    x = d.copy()
    x["U"] = x.TL_raw.map(RANK) > x[tl].map(RANK)
    x["C"] = (x[pi] - ALPHA).abs() < (x.pihat_raw - ALPHA).abs()
    x["S"] = x[dqs] > 0
    u = x[x.U]
    away = u[~u.C]
    print(f"=== {est} ===")
    print(f"  zone upgrades U:                       {int(x.U.sum())} of {len(x)}")
    print(f"  of those, coverage also improved (C):  {int(u.C.sum())}  ({100*u.C.mean():.1f}%)")
    print(f"  of those, score also improved (S):     {int(u.S.sum())}  ({100*u.S.mean():.1f}%)")
    print(f"  upgraded but coverage moved AWAY from nominal: {len(away)}")
    print(f"  upgraded but score got worse:                  {int((~u.S).sum())}")
    # the reverse direction
    noU = x[~x.U]
    print(f"  NOT upgraded yet coverage improved:    {int(noU.C.sum())} of {len(noU)}")
    if len(away):
        a = away.assign(d_raw=(away.pihat_raw-ALPHA).abs(), d_cor=(away[pi]-ALPHA).abs())
        print(f"  worst over-corrections rewarded by the zone:")
        for _, r in a.nlargest(5, "d_cor")[["model","asset","pihat_raw",pi,"TL_raw",tl,dqs]].iterrows():
            print(f"      {r.model:16s} {r.asset:8s} pi {r.pihat_raw:.4f} -> {r[pi]:.4f}   "
                  f"{r.TL_raw} -> {r[tl]}   dQS {r[dqs]:+.2e}")
    out[est] = {"n_upgrades": int(x.U.sum()), "upgrade_and_coverage": int(u.C.sum()),
                "upgrade_and_score": int(u.S.sum()), "upgrade_coverage_away": len(away),
                "upgrade_score_worse": int((~u.S).sum()),
                "no_upgrade_coverage_improved": int(noU.C.sum()),
                "pct_U_implies_C": float(100*u.C.mean()), "pct_U_implies_S": float(100*u.S.mean())}
    x.to_csv(f"analysis/k2_indication/zone_vs_coverage_{est}.csv", index=False)
    print()

print("--- NEGATIVE CONTROL ---")
ctrl = pd.DataFrame({"pihat_raw":[0.017,0.017], "pi":[0.0155,0.0010],
                     "TL_raw":["Yellow","Yellow"], "tl":["Green","Green"]})
ctrl["U"] = ctrl.TL_raw.map(RANK) > ctrl.tl.map(RANK)
ctrl["C"] = (ctrl.pi-ALPHA).abs() < (ctrl.pihat_raw-ALPHA).abs()
for i,r in ctrl.iterrows():
    print(f"  pi {r.pihat_raw:.4f} -> {r.pi:.4f}  U={r.U}  C={r.C}   "
          f"{'genuine crossing' if r.C else 'crossing with coverage moved away'}")
ok = bool(ctrl.U.all() and ctrl.C.iloc[0] and not ctrl.C.iloc[1])
print(f"  control separates the two cases: {ok}")
out["negative_control_passes"] = ok
json.dump(out, open("analysis/k2_indication/zone_vs_coverage.json","w"), indent=2)

# --------------------------------------------------------------------------- #
# The gated rule's ledger, with the overlap the counts hide.
# The rule applies the correction when the raw series fails on the gating
# window: TL != Green or Kupiec p <= 0.05. `cal` is deployable, `test` is the
# oracle.
# --------------------------------------------------------------------------- #
print("\n=== the gated rule, and what an 'upgrade lost' actually costs ===")
RANK2 = {"Green": 0, "Yellow": 1, "Red": 2}
led = {}
for est, tl, pi, dqs in (("static","TL_static","pihat_static","dQS_static"),
                         ("roll","TL_roll","pihat_roll","dQS_roll")):
    for signal in ("cal","test"):
        a = d.copy()
        a["raw_fails"] = ((a.TL_cal != "Green") | (a.p_kup_cal <= 0.05)) if signal=="cal" \
                         else ((a.TL_raw != "Green") | (a.p_kup_raw <= 0.05))
        a["U"] = a.TL_raw.map(RANK2) > a[tl].map(RANK2)
        up = a[a.U]
        lost = up[~up.raw_fails]
        lost_worse = lost[lost[dqs] < 0]
        both = a[a.U & (a[dqs] < 0)]
        print(f"  {est:7s} signal={signal:5s}  upgrades {len(up):3d}  kept {int(up.raw_fails.sum()):3d}  "
              f"lost {len(lost):2d}  of which the score was ALSO worse: {len(lost_worse):2d}")
        if signal == "cal":
            print(f"            pairs that are simultaneously an upgrade and a "
                  f"score deterioration: {len(both)} of {len(up)}")
        led[f"{est}_{signal}"] = {"upgrades": len(up), "kept": int(up.raw_fails.sum()),
                                  "lost": len(lost), "lost_but_score_worse": len(lost_worse),
                                  "upgrade_and_deterioration": len(both)}
json.dump(led, open("analysis/k2_indication/gate_ledger_overlap.json","w"), indent=2)
