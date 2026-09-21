"""Intensity of the static conformal shift, added in 0.5.0.

The deployed correction is ``intensity * c_hat``. The default 1.0 reproduces the
full conformal shift, which is the correction the primary comparisons of the
paper study. Intensity 0.5 is the averaged correction the paper recommends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.contrib.benchmarks import HistoricalSimulationForecaster
from conformal_oracle.recalibration import ConformalShift


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(20260921)
    n = 1500
    omega, a, b = 1e-6, 0.05, 0.90
    r = np.empty(n)
    s2 = omega / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + a * r[t] ** 2 + b * s2
    return pd.Series(r, index=pd.bdate_range("2018-01-02", periods=n), name="sim")


@pytest.fixture(scope="module")
def calibration():
    rng = np.random.default_rng(7)
    realised = rng.standard_normal(1000) * 0.01
    raw = np.full(1000, 0.023)
    return raw, realised


def test_default_is_one():
    """The default must stay 1.0, so existing results do not move."""
    assert ConformalShift().intensity == 1.0


def test_intensity_zero_returns_the_raw_forecast(calibration):
    raw, realised = calibration
    m = ConformalShift(intensity=0.0)
    m.fit(raw, realised, alpha=0.01)
    test_raw = np.array([0.02, 0.031, 0.017])
    assert np.array_equal(m.apply(test_raw), test_raw)


def test_intensity_one_reproduces_the_package_default(returns):
    """Intensity 1.0 equals the built-in static correction, unchanged by 0.5.0."""
    fc = HistoricalSimulationForecaster(window=250)
    default = audit_static(returns, fc, alpha=0.01)
    full = audit_static(returns, fc, alpha=0.01, recalibration=ConformalShift())
    explicit = audit_static(
        returns, fc, alpha=0.01, recalibration=ConformalShift(intensity=1.0),
    )
    assert default.q_v_stat == pytest.approx(full.q_v_stat, rel=0, abs=0)
    assert default.q_v_stat == pytest.approx(explicit.q_v_stat, rel=0, abs=0)


def test_half_is_the_average_of_raw_and_fully_corrected(calibration):
    raw, realised = calibration
    full = ConformalShift()
    half = ConformalShift(intensity=0.5)
    for m in (full, half):
        m.fit(raw, realised, alpha=0.01)
    test_raw = np.array([0.02, 0.031, 0.017])
    average = 0.5 * (test_raw + full.apply(test_raw))
    np.testing.assert_array_equal(half.apply(test_raw), average)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf")])
def test_intensity_outside_the_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError):
        ConformalShift(intensity=bad)


def test_short_window_warns_that_the_shift_is_the_largest_score():
    """At n = 125 and alpha = 0.01 the conformal rank is the sample size."""
    rng = np.random.default_rng(3)
    realised = rng.standard_normal(125) * 0.01
    raw = np.full(125, 0.023)
    m = ConformalShift()
    with pytest.warns(UserWarning, match="largest calibration score"):
        m.fit(raw, realised, alpha=0.01)
    assert m.q_v_stat == pytest.approx(float(np.max(-raw - realised)), rel=0, abs=0)


def test_long_window_does_not_warn(calibration):
    raw, realised = calibration
    with warnings_as_errors():
        ConformalShift().fit(raw, realised, alpha=0.01)


def warnings_as_errors():
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            yield

    return ctx()


def test_version_is_declared_once():
    """pyproject.toml and __init__ must not drift apart again."""
    import re
    from pathlib import Path

    import conformal_oracle

    root = Path(__file__).resolve().parents[2]
    declared = re.search(
        r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M,
    ).group(1)
    assert conformal_oracle.__version__ == declared
