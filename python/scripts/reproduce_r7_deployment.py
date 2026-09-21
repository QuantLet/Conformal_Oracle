#!/usr/bin/env python3
"""Read-only R7 deployment integration check using stored forecasts, not inference.

Example (from the manuscript root, with the package installed):
    python source/python/scripts/reproduce_r7_deployment.py \
        --data-root cfp_ijf_data --artifact-root .

The artifact root must contain Quantlets/cfp_config.py and
analysis/ae_point4/pairs_long.csv. The former defines the complete stored-series
panel; the latter is checked cell by cell, not used as decision evidence.
Nothing is written. Decisions use CAL only; all TEST diagnostics are ex-post.
The public API rejects at p < .05; the legacy artifact used p <= .05. This
check explicitly reports exact-threshold cases rather than hiding that boundary.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_oracle import recalibration_indication, selectively_recalibrate
from conformal_oracle.conformal.quantile import conformal_quantile
from conformal_oracle.conformal.rolling import compute_qv_roll_from_scores
from conformal_oracle.diagnostics.kupiec import kupiec_pof_pvalue
from conformal_oracle.diagnostics.scoring import quantile_score


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def forecast_path(data_root: Path, asset: str, spec: tuple) -> Path:
    directory, suffix = spec
    name = asset if suffix is None else f"{asset}_{suffix}"
    return data_root / directory / f"{name}.parquet"


def load_pair(returns_path: Path, quantiles_path: Path, alpha: float):
    """Match the original date intersection/forecast warm-up convention strictly."""
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    forecasts = pd.read_parquet(quantiles_path)
    if returns.shape[1] != 1:
        raise ValueError(f"{returns_path}: expected exactly one return column")
    column = f"VaR_{alpha:g}"
    if column not in forecasts:
        raise ValueError(f"{quantiles_path}: missing column {column}")
    for path, frame in ((returns_path, returns), (quantiles_path, forecasts)):
        idx = frame.index
        if not isinstance(idx, pd.DatetimeIndex) or (
            idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing
        ):
            raise ValueError(f"{path}: dates must be unique and increasing")
        # CSV dates are ns; some analytic-forecast Parquets store the same
        # daily timestamps as us. Normalise the in-memory representation only.
        frame.index = idx.as_unit("ns")
    common = returns.index.intersection(forecasts.index)
    r, q = returns.iloc[:, 0].loc[common], forecasts.loc[common, column]
    warmup = int(q.isna().sum())
    r, q = r[q.notna()], q[q.notna()]
    if len(r) == 0 or not np.isfinite(r).all() or not np.isfinite(q).all():
        raise ValueError(f"{quantiles_path}: empty or nonfinite aligned series")
    return r, q, {
        "return_dates_without_forecasts": len(returns) - len(common),
        "forecast_dates_without_returns": len(forecasts) - len(common),
        "missing_forecasts_removed": warmup,
    }


def checked_reference(path: Path, models: dict, assets: list, alpha: float):
    reference = pd.read_csv(path)
    required = {
        "model", "asset", "alpha", "n_test", "qV", "pihat_cal", "p_kup_cal",
        "TL_cal", "QS_raw", "QS_static", "QS_roll", "pihat_raw",
        "pihat_static", "pihat_roll", "TL_raw", "TL_static", "TL_roll",
    }
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    reference = reference[reference.alpha == alpha].copy()
    if reference.duplicated(["model", "asset"]).any():
        raise ValueError(f"{path}: duplicate model/asset keys at alpha={alpha}")
    expected = {(model, asset) for model in models for asset in assets}
    observed = set(zip(reference.model, reference.asset))
    if observed != expected:
        raise ValueError(
            f"{path}: incomplete panel; missing={sorted(expected - observed)}; "
            f"extra={sorted(observed - expected)}"
        )
    return reference.set_index(["model", "asset"])


def zone(returns, quantiles) -> str:
    # The stored manuscript convention uses all observations, not last 250.
    scaled = int(np.sum(returns < quantiles)) * (250.0 / len(returns))
    return "Green" if scaled <= 4 else "Yellow" if scaled <= 9 else "Red"


def score_pair(r, q, alpha: float, fraction: float, window: int) -> dict:
    n_cal = int(len(r) * fraction)
    if n_cal < window or len(r) - n_cal < 50:
        raise ValueError("insufficient observations for the R7 CAL/TEST protocol")
    rc, qc, rt, qt = r.iloc[:n_cal], q.iloc[:n_cal], r.iloc[n_cal:], q.iloc[n_cal:]
    decision = recalibration_indication(
        calibration_returns=rc, calibration_quantiles=qc, alpha=alpha,
    )
    # Outcomes below this line are evaluation only, never decision inputs.
    shift = conformal_quantile(np.asarray(qc - rc), alpha)
    static = np.asarray(qt) - shift
    scores = np.concatenate((np.asarray(qc - rc)[-window:], np.asarray(qt - rt)))
    rolling = np.asarray(qt) - compute_qv_roll_from_scores(scores, alpha, window)
    row = {
        "n_test": len(rt), "qV": shift, "apply": decision.apply,
        "pihat_cal": decision.n_violations / n_cal,
        "p_kup_cal": decision.kupiec_pvalue,
        "TL_cal": decision.basel_zone.title(),
        "exact_kupiec_boundary": decision.kupiec_pvalue == decision.kupiec_level,
        "n_calibration": n_cal,
    }
    for name, forecasts in (("raw", np.asarray(qt)), ("static", static),
                            ("roll", rolling)):
        row[f"QS_{name}"] = quantile_score(np.asarray(rt), forecasts, alpha)
        row[f"pihat_{name}"] = float(np.mean(np.asarray(rt) < forecasts))
        row[f"TL_{name}"] = zone(np.asarray(rt), forecasts)
    row["p_kup_raw"] = kupiec_pof_pvalue(np.asarray(rt < qt), alpha)
    for method, forecasts in (("static", static), ("rolling", rolling)):
        selective = selectively_recalibrate(
            qt, calibration_returns=rc, calibration_quantiles=qc,
            decision=decision, method=method, window=window,
            evaluation_returns=rt if method == "rolling" else None,
        )
        expected = forecasts if decision.apply else np.asarray(qt)
        np.testing.assert_array_equal(selective.final_quantiles, expected)
    return row


def check_cell(key, row: dict, reference: pd.Series) -> None:
    for name in ("n_test", "qV", "pihat_cal", "p_kup_cal", "QS_raw",
                 "QS_static", "QS_roll", "pihat_raw", "pihat_static", "pihat_roll"):
        if not np.isclose(row[name], reference[name], rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"{key}: artifact mismatch {name}: "
                f"recomputed={row[name]!r}, stored={reference[name]!r}"
            )
    for name in ("TL_cal", "TL_raw", "TL_static", "TL_roll"):
        if row[name] != reference[name]:
            raise ValueError(f"{key}: artifact mismatch {name}")
    legacy_apply = reference.TL_cal != "Green" or reference.p_kup_cal <= 0.05
    if row["apply"] != legacy_apply:
        raise ValueError(f"{key}: API decision differs from legacy artifact")


def ledger(panel: pd.DataFrame, estimator: str) -> dict:
    ranks = {"Green": 0, "Yellow": 1, "Red": 2}
    deteriorated = panel[f"QS_{estimator}"] > panel.QS_raw
    upgraded = panel.TL_raw.map(ranks) > panel[f"TL_{estimator}"].map(ranks)
    skipped = ~panel["apply"]
    lost = upgraded & skipped
    lost_worse = lost & deteriorated
    return {
        "deteriorations_total": int(deteriorated.sum()),
        "deteriorations_avoided": int((deteriorated & skipped).sum()),
        "zone_upgrades_total": int(upgraded.sum()),
        "zone_upgrades_retained": int((upgraded & ~skipped).sum()),
        "zone_upgrades_forgone": int(lost.sum()),
        "forgone_upgrades_with_worse_score": int(lost_worse.sum()),
        "net_cost_pairs": int(lost.sum() - lost_worse.sum()),
    }


def reproduce(data_root: Path, artifact_root: Path) -> dict:
    config_path = artifact_root / "Quantlets" / "cfp_config.py"
    reference_path = artifact_root / "analysis" / "ae_point4" / "pairs_long.csv"
    require_files([config_path, reference_path])
    config = runpy.run_path(str(config_path))
    models, assets = config["MODELS"], config["SYMBOLS"]
    alpha, fraction, window = config["ALPHA"], config["F_CAL"], config["W_ROLL"]
    reference = checked_reference(reference_path, models, assets, alpha)
    require_files(
        [data_root / "returns" / f"{asset}.csv" for asset in assets]
        + [forecast_path(data_root, asset, spec)
           for spec in models.values() for asset in assets]
    )
    rows = []
    for model, spec in models.items():
        for asset in assets:
            r, q, alignment = load_pair(
                data_root / "returns" / f"{asset}.csv",
                forecast_path(data_root, asset, spec), alpha,
            )
            try:
                row = score_pair(r, q, alpha, fraction, window)
            except (ValueError, AssertionError) as exc:
                raise ValueError(f"{model}/{asset}: {exc}") from exc
            check_cell((model, asset), row, reference.loc[(model, asset)])
            rows.append({"model": model, "asset": asset, **row, **alignment})
    panel = pd.DataFrame(rows)
    return {
        "data_root": str(data_root), "artifact_root": str(artifact_root),
        "models": len(models), "assets": len(assets), "pairs": len(panel),
        "alpha": alpha, "calibration_fraction": fraction, "rolling_window": window,
        "decision_information": "CAL only; TEST used exclusively for ex-post scoring",
        "basel_convention": "full CAL count scaled to 250, no rounding",
        "api_kupiec_rejection": "p < 0.05; legacy artifact uses p <= 0.05",
        "exact_kupiec_boundary_pairs": int(panel.exact_kupiec_boundary.sum()),
        "applied": int(panel["apply"].sum()),
        "skipped": int((~panel["apply"]).sum()),
        "per_pair_artifacts_match": True, "selective_quantiles_match": True,
        "alignment_counts_over_model_asset_pairs": {
            name: int(panel[name].sum()) for name in (
                "return_dates_without_forecasts", "forecast_dates_without_returns",
                "missing_forecasts_removed",
            )
        },
        "static": ledger(panel, "static"), "rolling": ledger(panel, "roll"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory with returns/ and stored forecast folders")
    parser.add_argument("--artifact-root", type=Path, required=True,
                        help="manuscript root containing Quantlets/ and analysis/")
    args = parser.parse_args(argv)
    try:
        result = reproduce(args.data_root.resolve(), args.artifact_root.resolve())
    except (OSError, ValueError, KeyError, AssertionError, ImportError) as exc:
        print(f"R7 deployment reproduction FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
