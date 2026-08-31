#!/usr/bin/env python3
"""The gap panel, computed by the driver that produces all_results.csv.

The first version of this panel re-split the scores itself and reported 297
green against the pipeline's 278: right differences, its own levels. It is
therefore built here by importing run_full_evaluation and calling it twice,
once per arm, so both columns come from the function that produced the
manuscript's tables. Nothing in this file implements a split.

    python analysis/convention/build_gap_panel.py --write
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "analysis" / "convention" / "gap_panel.csv"
sys.path.insert(0, str(BASE / "Quantlets"))
spec = importlib.util.spec_from_file_location(
    "rfe", BASE / "Quantlets" / "CO_full_evaluation" / "run_full_evaluation.py")
rfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfe)

KEY = ["model", "symbol", "alpha"]
CARRY = ["qV", "n_test", "viol_cp", "pihat_cp", "p_kup_cp", "TL_cp"]


def control(a0: pd.DataFrame, a1: pd.DataFrame) -> None:
    """The two arms must differ, and differ only where the gap can reach.

    A panel whose arms are identical would report "the gap changes nothing"
    while measuring one estimator twice, which is the failure this file was
    rewritten to avoid.
    """
    assert (a0["gap"] == 0).all(), "the contiguous arm carries a gap"
    assert (a1["gap"] > 0).all(), "the gapped arm carries no gap"
    assert not a0["n_test"].equals(a1["n_test"]), \
        "the arms have the same test length; the gap was not applied"
    j = a0.merge(a1, on=KEY, suffixes=("_0", "_g"))
    assert np.allclose(j["qV_0"], j["qV_g"]), \
        "the shift moved; the gap must come out of the test block, not the calibration block"
    print(f"  ctrl   arms differ in test length, agree in shift ({len(j)} cells)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    models = list(rfe.MODELS)
    a0 = rfe.compute(models, gap=False)
    a1 = rfe.compute(models, gap=True)
    control(a0, a1)
    j = a0[KEY + ["gap"] + CARRY].merge(
        a1[KEY + ["gap"] + CARRY], on=KEY, suffixes=("_0", "_g"))
    j = j.rename(columns={"gap_g": "gap"}).drop(columns=["gap_0"])
    j = j.rename(columns={f"{c}_0": f"g0_{c}" for c in CARRY})
    j = j.rename(columns={f"{c}_g": f"gn_{c}" for c in CARRY})
    j = j.rename(columns={"g0_p_kup_cp": "g0_p_kupiec",
                          "gn_p_kup_cp": "gn_p_kupiec",
                          "g0_TL_cp": "g0_TL", "gn_TL_cp": "gn_TL",
                          "g0_pihat_cp": "g0_pi_hat", "gn_pihat_cp": "gn_pi_hat",
                          "g0_qV": "g0_qV", "gn_qV": "gn_qV",
                          "g0_viol_cp": "g0_viol", "gn_viol_cp": "gn_viol"})
    j["dpi"] = (j["gn_pi_hat"] - j["g0_pi_hat"]).abs()
    one = j[j["alpha"] == 0.01].drop(columns=["alpha"])
    print(f"  {len(one)} cells at alpha = 0.01; gap {int(one['gap'].min())}-"
          f"{int(one['gap'].max())}, median {int(one['gap'].median())}")
    print(f"  zone changes {int((one['g0_TL'] != one['gn_TL']).sum())}, "
          f"Kupiec flips {int(((one['g0_p_kupiec'] > 0.05) != (one['gn_p_kupiec'] > 0.05)).sum())}, "
          f"max |dpi| {one['dpi'].max():.6f}")
    if a.write:
        one.to_csv(OUT, index=False)
        print(f"  wrote {OUT.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
