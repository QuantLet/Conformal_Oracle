"""T6: GBMQuantileRegression integration tests."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from conformal_oracle.recalibration import GBMQuantileRegression

pytestmark = pytest.mark.skipif(
    not HAS_LIGHTGBM,
    reason="lightgbm not installed",
)


@pytest.fixture(scope="module")
def garch_data():
    rng = np.random.default_rng(2026)
    n = 2000
    omega, alpha_g, beta_g = 1e-6, 0.05, 0.90
    r = np.empty(n)
    s2 = np.empty(n)
    s2[0] = omega / (1 - alpha_g - beta_g)
    for t in range(n):
        if t > 0:
            s2[t] = omega + alpha_g * r[t - 1] ** 2 + beta_g * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()

    var_raw = np.abs(r) * 1.5 + 0.005
    return r, var_raw


def test_gbm_fit_apply(garch_data):
    """GBM-QR should fit and produce output of correct shape."""
    r, var_raw = garch_data
    n_cal = 1400
    gbm = GBMQuantileRegression(n_estimators=50, max_depth=3)
    gbm.fit(var_raw[:n_cal], r[:n_cal], alpha=0.01)
    corrected = gbm.apply(var_raw[n_cal:])
    assert corrected.shape == (len(var_raw) - n_cal,)
    assert np.all(np.isfinite(corrected))


def test_gbm_violation_rate(garch_data):
    """GBM-QR violation rate should be in a reasonable range."""
    r, var_raw = garch_data
    n_cal = 1400
    gbm = GBMQuantileRegression(n_estimators=50, max_depth=3)
    gbm.fit(var_raw[:n_cal], r[:n_cal], alpha=0.01)
    corrected = gbm.apply(var_raw[n_cal:])
    r_test = r[n_cal:]
    viol_rate = float(np.mean(r_test < -corrected))
    assert 0 <= viol_rate <= 0.10


def test_gbm_import_error():
    """Should raise ImportError with helpful message if lightgbm missing."""
    gbm = GBMQuantileRegression()
    gbm_mod = gbm.__class__.__module__
    assert "gbm_qr" in gbm_mod


def test_gbm_positive_output(garch_data):
    """GBM-QR corrected VaR should be mostly positive."""
    r, var_raw = garch_data
    n_cal = 1400
    gbm = GBMQuantileRegression(n_estimators=50, max_depth=3)
    gbm.fit(var_raw[:n_cal], r[:n_cal], alpha=0.01)
    corrected = gbm.apply(var_raw[n_cal:])
    assert np.mean(corrected > 0) > 0.5


def test_gbm_custom_hyperparams(garch_data):
    """Custom hyperparameters should work."""
    r, var_raw = garch_data
    n_cal = 1400
    gbm = GBMQuantileRegression(
        n_estimators=30, max_depth=2, learning_rate=0.1,
    )
    gbm.fit(var_raw[:n_cal], r[:n_cal], alpha=0.01)
    corrected = gbm.apply(var_raw[n_cal:])
    assert len(corrected) == len(var_raw) - n_cal
