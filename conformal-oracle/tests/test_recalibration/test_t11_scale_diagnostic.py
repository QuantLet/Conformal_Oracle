"""T11: diagnose_scale() location-scale diagnostic tests.

The diagnostic relaxes the one-parameter conformal shift to the two-parameter
map VaR_cp = a_hat + b_hat * VaR_raw and reports the share of the correction
taken by the multiplicative term. It is diagnostic-only (not a corrector).
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_oracle.recalibration import (
    ConformalShift,
    ScaleDiagnostic,
    diagnose_scale,
)


def _hetero_returns(n, rng, sigma_lo=0.005, sigma_hi=0.03):
    """Heteroskedastic Normal returns with a known, varying scale sigma_t."""
    sigma = rng.uniform(sigma_lo, sigma_hi, size=n)
    z = rng.standard_normal(n)
    r = sigma * z
    return r, sigma


def test_returns_dataclass():
    rng = np.random.default_rng(0)
    r, sigma = _hetero_returns(3000, rng)
    from scipy.stats import norm

    var_raw = -sigma * norm.ppf(0.05)  # well-calibrated oracle VaR
    out = diagnose_scale(var_raw, r, alpha=0.05)
    assert isinstance(out, ScaleDiagnostic)
    assert 0.0 <= out.scale_share <= 1.0


def test_q_v_stat_matches_conformal_shift():
    """The reported q_v_stat must equal the one-parameter ConformalShift."""
    rng = np.random.default_rng(1)
    r, sigma = _hetero_returns(2000, rng)
    var_raw = np.abs(rng.standard_normal(2000)) * 0.02

    shift = ConformalShift()
    shift.fit(var_raw, r, alpha=0.01)
    out = diagnose_scale(var_raw, r, alpha=0.01)
    assert out.q_v_stat == pytest.approx(shift.q_v_stat, rel=1e-12)


def test_well_calibrated_needs_negligible_correction():
    """A well-calibrated oracle needs essentially no correction: b_hat near 1
    and a one-parameter shift that is tiny relative to the mean VaR. (The
    scale share is not asserted here: with no correction to decompose it is a
    ratio of two near-zero noise terms and is not meaningful.)"""
    from scipy.stats import norm

    rng = np.random.default_rng(2)
    r, sigma = _hetero_returns(6000, rng)
    var_raw = -sigma * norm.ppf(0.05)  # correct alpha-quantile magnitude
    out = diagnose_scale(var_raw, r, alpha=0.05)
    assert out.b_hat == pytest.approx(1.0, abs=0.25)
    assert abs(out.q_v_stat) < 0.1 * float(np.mean(var_raw))


def test_under_scaled_forecast_flags_scale():
    """When VaR_raw = b * oracle with b < 1, the two-parameter fit recovers
    b_hat ~ 1/b > 1 and attributes most of the correction to the scale term."""
    from scipy.stats import norm

    rng = np.random.default_rng(3)
    r, sigma = _hetero_returns(6000, rng)
    b = 0.5
    var_raw = b * (-sigma * norm.ppf(0.05))
    out = diagnose_scale(var_raw, r, alpha=0.05)
    assert out.b_hat > 1.3  # recovers ~1/b = 2
    assert out.scale_share > 0.6  # correction is mostly multiplicative


def test_pure_bias_low_scale_share():
    """A constant additive bias on a varying oracle keeps b_hat near 1 and the
    scale share low."""
    from scipy.stats import norm

    rng = np.random.default_rng(4)
    r, sigma = _hetero_returns(6000, rng)
    oracle = -sigma * norm.ppf(0.05)
    var_raw = oracle + 0.5 * oracle.std()  # additive location shift, b = 1
    out = diagnose_scale(var_raw, r, alpha=0.05)
    assert out.b_hat == pytest.approx(1.0, abs=0.3)
    assert out.scale_share < 0.5


def test_input_validation():
    with pytest.raises(ValueError):
        diagnose_scale(np.array([0.02]), np.array([0.01]), alpha=0.05)
    with pytest.raises(ValueError):
        diagnose_scale(np.zeros(10), np.zeros(9), alpha=0.05)


def test_frozen_dataclass():
    rng = np.random.default_rng(5)
    r, sigma = _hetero_returns(1000, rng)
    out = diagnose_scale(np.abs(r) + 0.02, r, alpha=0.05)
    with pytest.raises(Exception):
        out.b_hat = 2.0  # type: ignore[misc]
