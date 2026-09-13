"""Tests for the R8 additions: one-coefficient corrections, optimism estimators,
paired calendar bootstrap and past-loss selection.

Artifact-backed tests reproduce published numbers from the research archive when
it is present (environment variable ``CONFORMAL_ORACLE_ARCHIVE`` or the default
project folder); they are skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformal_oracle import (
    block_bootstrap_optimism,
    blocked_cv_optimism,
    first_order_shrinkage,
    fit_one_coefficient_corrections,
    paired_calendar_bootstrap,
    past_loss_selection,
)
from conformal_oracle.conformal.quantile import conformal_quantile
from conformal_oracle.diagnostics.optimism import pinball_loss, training_loss_change
from conformal_oracle.recalibration.one_coefficient import (
    shift_cp,
    shift_erm,
    vol_cp,
    vol_erm,
    weighted_quantile,
)

_DEFAULT_ARCHIVE = (
    "/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/"
    "2026 CFP LLM VaR"
)
ARCHIVE = Path(os.environ.get("CONFORMAL_ORACLE_ARCHIVE", _DEFAULT_ARCHIVE))
ALPHA = 0.01


def _synthetic(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    sigma = np.exp(0.2 * rng.standard_normal(n))
    r = sigma * rng.standard_t(5, size=n) * 0.01
    q = -2.0 * sigma * 0.01  # a raw lower quantile that is too narrow
    return q, r, sigma


def test_shift_erm_minimises_pinball_loss_over_constants():
    q, r, _ = _synthetic()
    c = shift_erm(q, r, ALPHA)
    s = q - r
    best = pinball_loss(c - s, ALPHA).mean()
    grid = np.linspace(c - 0.01, c + 0.01, 401)
    losses = [pinball_loss(g - s, ALPHA).mean() for g in grid]
    assert best <= min(losses) + 1e-15


def test_shift_cp_matches_conformal_quantile_and_vol_definitions():
    q, r, sigma = _synthetic()
    s = q - r
    assert shift_cp(q, r, ALPHA) == conformal_quantile(s, ALPHA)
    assert vol_cp(q, r, sigma, ALPHA) == conformal_quantile(s / sigma, ALPHA)
    assert vol_erm(q, r, sigma, ALPHA) == weighted_quantile(s / sigma, sigma, 1 - ALPHA)
    fit = fit_one_coefficient_corrections(q, r, sigma, ALPHA)
    out = fit.apply(q[:5], sigma[:5])
    assert np.allclose(out["Vol-ERM"], q[:5] - fit.vol_erm * sigma[:5])
    assert np.allclose(out["Shift-CP"], q[:5] - fit.shift_cp)


def test_vol_erm_minimises_return_unit_pinball_loss_over_scaled_shifts():
    q, r, sigma = _synthetic()
    s = q - r
    b = vol_erm(q, r, sigma, ALPHA)
    best = pinball_loss(b * sigma - s, ALPHA).mean()
    grid = np.linspace(b - 0.5, b + 0.5, 801)
    losses = [pinball_loss(g * sigma - s, ALPHA).mean() for g in grid]
    assert best <= min(losses) + 1e-15


def test_weighted_quantile_reduces_to_inverted_cdf_with_equal_weights():
    x = np.random.default_rng(1).standard_normal(500)
    w = np.ones_like(x)
    assert weighted_quantile(x, w, 0.99) == np.quantile(x, 0.99, method="inverted_cdf")


def test_blocked_cv_factor_and_bootstrap_are_finite_and_positive_on_iid_scores():
    s = np.random.default_rng(2).standard_normal(1500)
    cv = blocked_cv_optimism(s, ALPHA)
    assert cv.factor == pytest.approx(8 / 9)
    boot = block_bootstrap_optimism(s, ALPHA, resamples=200)
    assert boot.block_length == 12  # ceil(1500 ** (1/3))
    assert cv.penalty > 0 and np.isfinite(boot.penalty)
    assert training_loss_change(s, ALPHA) == cv.training_loss_change


def test_first_order_shrinkage_is_flagged_unvalidated_and_clipped():
    s = np.random.default_rng(3).standard_normal(800) + 0.05
    d = first_order_shrinkage(blocked_cv_optimism(s, ALPHA))
    assert d.validated is False and 0.0 <= d.factor <= 1.0


def test_past_loss_selection_gate_falls_back_to_raw_without_evidence():
    rng = np.random.default_rng(4)
    raw = rng.exponential(1.0, 300)
    losses = {"Raw": raw, "A": raw + rng.normal(0, 0.5, 300), "B": raw - 0.3}
    sel = past_loss_selection(losses, key="unit-test")
    assert sel.candidates == ("Raw", "A", "B")
    assert sel.past_minimum == "B" and sel.cautious_gate == "B"
    noise = {"Raw": raw, "A": raw + rng.normal(0, 0.5, 300)}
    assert past_loss_selection(noise, key="unit-test").cautious_gate == "Raw"


def test_paired_calendar_bootstrap_shapes_and_sign():
    rng = np.random.default_rng(5)
    pairs = []
    for j in range(4):
        idx = pd.bdate_range("2021-01-01", periods=400)
        base = rng.exponential(1.0, 400)
        pairs.append(pd.DataFrame({"Raw": base, "Static": base - 0.2}, index=idx))
    out = paired_calendar_bootstrap(pairs, [("Static", "Raw")], draws=99)
    assert set(out.block_calendar_days) == {20, 60}
    assert np.allclose(out.difference, -0.2)
    assert (out.simultaneous_upper < 0).all()


# ---------------------------------------------------------------- artifact-backed
SYN = ARCHIVE / "results/theory_loop/synthetic"
OPT = ARCHIVE / "artifacts/r8_optimism_estimator/histories.csv"
PAIRS = ARCHIVE / "artifacts/r8_ten_comparators/pairs"
POWER = ARCHIVE / "artifacts/r8_power_analysis/contrasts.csv"


@pytest.mark.skipif(
    not (SYN.exists() and OPT.exists()), reason="research archive not present"
)
def test_optimism_estimators_reproduce_stored_synthetic_histories():
    hist = pd.read_csv(OPT)
    for law, law_index in (("normal", 0), ("t5", 1)):
        scores = np.load(SYN / f"calibration_{law}.npz")["scores"]
        for n in (700, 1000):
            for rep in (0, 7):
                s = scores[rep, :n]
                cell = (hist.law == law) & (hist.n == n) & (hist.rep == rep)
                cell &= hist.bias == 0
                row = hist[cell & (hist.estimator == "E1_blocked_cv")].iloc[0]
                cv = blocked_cv_optimism(s, ALPHA).penalty
                assert cv == pytest.approx(row.O_hat, rel=1e-10, abs=1e-14)
                row = hist[cell & (hist.estimator == "E2_block_bootstrap")].iloc[0]
                seed = [20260913, law_index, n, rep]
                est = block_bootstrap_optimism(s, ALPHA, resamples=200, seed=seed)
                assert est.penalty == pytest.approx(row.O_hat, rel=1e-10, abs=1e-14)


@pytest.mark.skipif(
    not (PAIRS.exists() and POWER.exists()), reason="research archive not present"
)
def test_paired_bootstrap_reproduces_published_static_minus_raw_contrast():
    frames = []
    for folder in sorted(PAIRS.iterdir()):
        f = pd.read_parquet(folder / "daily.parquet")
        r = f["r"].to_numpy()
        loss = {
            m: pinball_loss(r - f[m].to_numpy(), ALPHA) for m in ("Raw", "Shift-CP")
        }
        frames.append(pd.DataFrame(loss, index=f.index))
    assert len(frames) == 240
    out = paired_calendar_bootstrap(frames, [("Shift-CP", "Raw")], scale=1e4)
    published = pd.read_csv(POWER)
    for block in (20, 60):
        mine = out[out.block_calendar_days == block].iloc[0]
        pick = published.contrast == "1_all_pairs"
        pick &= published.block_calendar_days == block
        ref = published[pick].iloc[0]
        assert mine.difference == pytest.approx(ref.point, abs=1e-9)
        assert mine.lower == pytest.approx(ref.pointwise_lower, abs=1e-9)
        assert mine.upper == pytest.approx(ref.pointwise_upper, abs=1e-9)
