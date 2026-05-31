"""T_MOIRAI_WITHIN_FAMILY: Within-family contrast test.

Verifies that Moirai 1.1 (sample-based) and Moirai 2.0
(quantile-grid) produce different conformal correction regimes
when run through audit_panel on the same data.

This is the load-bearing methodological test: the within-family
contrast is the cleanest causal evidence for predictive-interface
effects on tail calibration.

Skipped if uni2ts is not installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.forecasters.tsfm.moirai import MoiraiForecaster


def _moirai_available() -> bool:
    return (
        importlib.util.find_spec("uni2ts") is not None
        and importlib.util.find_spec("gluonts") is not None
        and importlib.util.find_spec("torch") is not None
    )


pytestmark = pytest.mark.skipif(
    not _moirai_available(),
    reason="uni2ts or its dependencies not installed",
)


@pytest.fixture(scope="module")
def synthetic_panel():
    rng = np.random.default_rng(2026)
    n = 800
    assets = ["A", "B", "C", "D", "E"]
    panel = {}
    for asset in assets:
        omega, alpha_g, beta_g = 1e-6, 0.05, 0.90
        r = np.empty(n)
        s2 = np.empty(n)
        s2[0] = omega / (1 - alpha_g - beta_g)
        for t in range(n):
            if t > 0:
                s2[t] = omega + alpha_g * r[t - 1] ** 2 + beta_g * s2[t - 1]
            r[t] = np.sqrt(s2[t]) * rng.standard_normal()
        dates = pd.bdate_range("2020-01-02", periods=n)
        panel[asset] = pd.Series(r, index=dates, name=asset)
    return pd.DataFrame(panel)


def test_within_family_output_types(synthetic_panel):
    """Moirai 1.1 should produce SampleDistribution,
    Moirai 2.0 should produce QuantileGridDistribution.
    """
    from conformal_oracle._types import QuantileGridDistribution, SampleDistribution

    fc11 = MoiraiForecaster(version="1.1", n_samples=100, device="cpu")
    fc20 = MoiraiForecaster(version="2.0", device="cpu")

    ret = synthetic_panel["A"]
    d11 = fc11.forecast(ret, t=600)
    d20 = fc20.forecast(ret, t=600)

    assert isinstance(d11, SampleDistribution)
    assert isinstance(d20, QuantileGridDistribution)


def test_within_family_regime_contrast(synthetic_panel):
    """Run audit_static with both Moirai versions.

    The within-family contrast should appear as a difference in
    the magnitude of conformal correction (qV_stat). This test
    verifies the pipeline produces distinct results for the two
    versions, confirming the wrapper distinguishes them correctly.
    """
    from conformal_oracle.audit import audit_static

    ret = synthetic_panel["A"]

    fc11 = MoiraiForecaster(version="1.1", n_samples=100, device="cpu")
    fc20 = MoiraiForecaster(version="2.0", device="cpu")

    r11 = audit_static(ret, fc11, alpha=0.01)
    r20 = audit_static(ret, fc20, alpha=0.01)

    differs = (
        r11.q_v_stat != r20.q_v_stat
        or r11.violation_rate_raw != r20.violation_rate_raw
    )
    assert differs, "Moirai 1.1 and 2.0 should differ on at least one metric"
