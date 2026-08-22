"""Verify that GJR-GARCH on GJR-GARCH data produces qV_stat within ~1.5x CI of zero.

This is the methodological soundness test: a correctly-specified model
should need negligible conformal correction.
"""

from conformal_oracle import audit_static
from conformal_oracle.forecasters import GJRGARCHForecaster


def test_qv_stat_within_ci_of_zero(synthetic_returns):
    """qV_stat should be within 1.5x bootstrap CI width of zero."""
    fc = GJRGARCHForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01, warmup=250)

    ci_lo, ci_hi = result.q_v_stat_ci
    ci_width = ci_hi - ci_lo

    assert abs(result.q_v_stat) < 1.5 * ci_width, (
        f"qV_stat={result.q_v_stat:.6f} is more than 1.5x the CI width "
        f"({ci_width:.6f}) away from zero. CI=[{ci_lo:.6f}, {ci_hi:.6f}]"
    )


def test_correct_spec_signal_preserving(synthetic_returns):
    """Correctly-specified model must be classified signal-preserving."""
    fc = GJRGARCHForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01, warmup=250)
    assert result.regime == "signal-preserving"
