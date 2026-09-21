"""Loader and completeness contracts of the optional stored-panel reproduction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/reproduce_r7_deployment.py"
SPEC = importlib.util.spec_from_file_location("r7_deployment_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
r7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r7)


def test_missing_files_report_every_exact_path(tmp_path):
    first, second = tmp_path / "returns.csv", tmp_path / "forecasts.parquet"
    with pytest.raises(FileNotFoundError) as error:
        r7.require_files([first, second])
    assert str(first) in str(error.value)
    assert str(second) in str(error.value)


def reference_frame():
    names = (
        "n_test", "qV", "pihat_cal", "p_kup_cal", "QS_raw", "QS_static",
        "QS_roll", "pihat_raw", "pihat_static", "pihat_roll",
    )
    row = {name: 1.0 for name in names}
    row.update({name: "Green" for name in (
        "TL_cal", "TL_raw", "TL_static", "TL_roll",
    )})
    return pd.DataFrame([{"model": "M", "asset": "A", "alpha": .01, **row}])


def test_reference_requires_exact_cartesian_support(monkeypatch):
    monkeypatch.setattr(r7.pd, "read_csv", lambda *args, **kwargs: reference_frame())
    with pytest.raises(ValueError, match="incomplete panel.*B"):
        r7.checked_reference(Path("panel.csv"), {"M": None}, ["A", "B"], .01)


def test_reference_rejects_duplicate_pairs(monkeypatch):
    monkeypatch.setattr(r7.pd, "read_csv", lambda *args, **kwargs:
                        pd.concat([reference_frame(), reference_frame()]))
    with pytest.raises(ValueError, match="duplicate"):
        r7.checked_reference(Path("panel.csv"), {"M": None}, ["A"], .01)


@pytest.mark.parametrize("bad_index", [
    pd.to_datetime(["2020-01-02", "2020-01-01"]),
    pd.to_datetime(["2020-01-01", "2020-01-01"]),
])
def test_loader_rejects_nonchronological_or_duplicate_dates(monkeypatch, bad_index):
    returns = pd.DataFrame({"r": [.1, .2]}, index=bad_index)
    forecasts = pd.DataFrame({"VaR_0.01": [-.1, -.1]}, index=bad_index)
    monkeypatch.setattr(r7.pd, "read_csv", lambda *args, **kwargs: returns)
    monkeypatch.setattr(r7.pd, "read_parquet", lambda *args, **kwargs: forecasts)
    with pytest.raises(ValueError, match="dates must be unique and increasing"):
        r7.load_pair(Path("returns.csv"), Path("forecasts.parquet"), .01)


def test_loader_reports_alignment_and_missing_forecasts(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=4)
    returns = pd.DataFrame({"r": [.1, .2, .3, .4]}, index=dates)
    forecasts = pd.DataFrame({"VaR_0.01": [np.nan, -.1, -.2]}, index=dates[1:])
    monkeypatch.setattr(r7.pd, "read_csv", lambda *args, **kwargs: returns)
    monkeypatch.setattr(r7.pd, "read_parquet", lambda *args, **kwargs: forecasts)
    r, q, info = r7.load_pair(Path("returns.csv"), Path("forecasts.parquet"), .01)
    assert r.index.equals(q.index) and r.index.equals(dates[2:])
    assert info == {"return_dates_without_forecasts": 1,
                    "forecast_dates_without_returns": 0,
                    "missing_forecasts_removed": 1}


def test_missing_quantile_column_cannot_silently_drop_pair(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"r": [.1, .2]}, index=dates)
    monkeypatch.setattr(r7.pd, "read_csv", lambda *args, **kwargs: returns)
    monkeypatch.setattr(r7.pd, "read_parquet", lambda *args, **kwargs: returns)
    with pytest.raises(ValueError, match="missing column VaR_0.01"):
        r7.load_pair(Path("returns.csv"), Path("forecasts.parquet"), .01)


def test_loader_aligns_identical_dates_with_different_units(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"r": [.1, .2]}, index=dates.as_unit("ns"))
    forecasts = pd.DataFrame({"VaR_0.01": [-.1, -.2]}, index=dates.as_unit("us"))
    monkeypatch.setattr(r7.pd, "read_csv", lambda *args, **kwargs: returns)
    monkeypatch.setattr(r7.pd, "read_parquet", lambda *args, **kwargs: forecasts)
    r, q, _ = r7.load_pair(Path("returns.csv"), Path("forecasts.parquet"), .01)
    assert r.index.equals(q.index)
    np.testing.assert_array_equal(q, [-.1, -.2])


def test_future_outcomes_do_not_enter_decision():
    dates = pd.date_range("2020-01-01", periods=400)
    r = pd.Series(np.linspace(.1, 2, len(dates)), index=dates)
    q = pd.Series(np.zeros(len(dates)), index=dates)
    original = r7.score_pair(r, q, .01, .70, 250)
    r.iloc[280:] = -100
    changed = r7.score_pair(r, q, .01, .70, 250)
    for name in ("apply", "pihat_cal", "p_kup_cal", "TL_cal", "n_calibration"):
        assert original[name] == changed[name]
    assert original["QS_raw"] != changed["QS_raw"]


def test_artifact_numeric_drift_fails():
    stored = reference_frame().iloc[0]
    row = stored.to_dict()
    row["apply"] = False
    row["QS_raw"] += .001
    with pytest.raises(ValueError, match="artifact mismatch QS_raw"):
        r7.check_cell(("M", "A"), row, stored)


def test_main_missing_inputs_exits_nonzero_without_output(tmp_path, capsys):
    assert r7.main(["--data-root", str(tmp_path),
                    "--artifact-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Missing required paths:" in captured.err
    assert str(tmp_path / "Quantlets/cfp_config.py") in captured.err
    assert str(tmp_path / "analysis/ae_point4/pairs_long.csv") in captured.err
