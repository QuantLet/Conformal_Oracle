"""Calibration-only gating, exact skip semantics, and causal correction tests."""

from dataclasses import FrozenInstanceError, replace
from inspect import Parameter, signature

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from conformal_oracle.conformal.quantile import conformal_quantile
from conformal_oracle.deployment import (
    recalibration_indication,
    selectively_recalibrate,
)


def calibration(n=250, violations=2):
    returns = np.ones(n)
    returns[:violations] = -1
    return returns, np.zeros(n)


def decide(returns, quantiles, **kwargs):
    return recalibration_indication(
        calibration_returns=returns, calibration_quantiles=quantiles, **kwargs
    )


@pytest.mark.parametrize(
    "n,x,zone,kupiec,apply",
    [
        (250, 2, "green", False, False),
        (250, 5, "yellow", False, True),
        (500, 0, "green", True, True),
        (250, 20, "red", True, True),
    ],
)
def test_or_rule_all_four_cases(n, x, zone, kupiec, apply):
    r, q = calibration(n, x)
    decision = decide(r, q)
    assert decision.apply is apply
    assert decision.basel_zone == zone
    assert (decision.kupiec_pvalue < decision.kupiec_level) is kupiec
    assert ("basel_not_green" in decision.reasons) is (zone != "green")
    assert ("kupiec_rejection" in decision.reasons) is kupiec
    assert decision.kupiec_pvalue == pytest.approx(
        chi2.sf(decision.kupiec_statistic, 1)
    )
    assert decision.n_calibration == n
    assert decision.n_violations == x
    assert decision.scaled_violations_250 == x * 250 / n
    assert decision.information_window == "calibration"
    assert decision.level == decision.kupiec_level == 0.05


@pytest.mark.parametrize(
    "x,zone", [(16, "green"), (17, "yellow"), (36, "yellow"), (37, "red")]
)
def test_basel_full_window_unrounded_thresholds(x, zone):
    r, q = calibration(1000, x)
    assert decide(r, q).basel_zone == zone


def test_basel_uses_all_calibration_not_last_250():
    r, q = calibration(500, 0)
    r[-8:] = -1
    assert decide(r, q).basel_zone == "green"  # 8 * 250 / 500 = 4, not 8


def test_kupiec_threshold_equality_does_not_reject():
    r, q = calibration(250, 2)
    threshold = decide(r, q).kupiec_pvalue
    assert not decide(r, q, kupiec_level=threshold).apply
    assert decide(r, q, kupiec_level=np.nextafter(threshold, 1)).apply


def test_indication_has_calibration_only_keyword_arguments_and_is_immutable():
    params = signature(recalibration_indication).parameters
    assert all(p.kind is Parameter.KEYWORD_ONLY for p in params.values())
    assert not any("evaluation" in name or "test" in name for name in params)
    r, q = calibration()
    with pytest.raises(TypeError):
        recalibration_indication(
            calibration_returns=r,
            calibration_quantiles=q,
            evaluation_returns=np.zeros(20),
        )
    decision = decide(r, q)
    with pytest.raises(FrozenInstanceError):
        decision.apply = True
    assert decide(r.copy(), q.copy()) == decision


@pytest.mark.parametrize("method", ["static", "rolling"])
def test_skip_exact_bitwise_copy_and_ignores_evaluation_outcomes(method):
    r, q = calibration()
    raw = np.array([-0.0, 0.0, -1.125, -2.5], dtype=np.float32)
    before = raw.tobytes()
    result = selectively_recalibrate(
        raw,
        calibration_returns=r,
        calibration_quantiles=q,
        decision=decide(r, q),
        method=method,
        evaluation_returns=object(),
        window=9999,
    )
    assert not result.applied
    assert result.final_quantiles.dtype == raw.dtype
    assert result.final_quantiles.tobytes() == result.raw_quantiles.tobytes() == before
    assert raw.tobytes() == before
    assert not np.shares_memory(result.final_quantiles, raw)
    assert not np.shares_memory(result.final_quantiles, result.raw_quantiles)
    assert not result.final_quantiles.flags.writeable
    np.testing.assert_array_equal(result.corrections, np.zeros(len(raw)))
    assert (
        result.window is result.finite_sample_rank is result.maximum_score_proxy is None
    )


def test_static_matches_existing_order_statistic_without_evaluation_data():
    r, q = calibration(100, 20)
    raw = np.array([-1.0, -2.0, -3.0])
    result = selectively_recalibrate(
        raw,
        calibration_returns=r,
        calibration_quantiles=q,
        decision=decide(r, q),
        evaluation_returns=object(),
    )
    shift = conformal_quantile(q - r, 0.01)
    np.testing.assert_array_equal(result.corrections, np.repeat(shift, len(raw)))
    np.testing.assert_array_equal(result.final_quantiles, raw - shift)
    assert result.applied
    assert result.alpha == 0.01
    assert result.finite_sample_rank == 100
    assert result.maximum_score_proxy is False


def test_maximum_score_proxy_metadata():
    r, q = calibration(10, 10)
    result = selectively_recalibrate(
        [-1.0],
        calibration_returns=r,
        calibration_quantiles=q,
        decision=decide(r, q),
    )
    assert result.finite_sample_rank == 11
    assert result.maximum_score_proxy is True


def test_rolling_matches_past_only_order_statistics_and_future_independence():
    r, q = calibration(20, 20)
    raw = np.zeros(8)
    outcomes = np.array([-2.0, -3.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    decision = decide(r, q)
    kwargs = dict(
        calibration_returns=r,
        calibration_quantiles=q,
        decision=decision,
        method="rolling",
        window=4,
    )
    result = selectively_recalibrate(raw, evaluation_returns=outcomes, **kwargs)
    all_scores = np.concatenate([q - r, raw - outcomes])
    expected = [
        conformal_quantile(all_scores[20 + t - 4 : 20 + t], 0.01)
        for t in range(len(raw))
    ]
    np.testing.assert_array_equal(result.corrections, expected)
    changed = outcomes.copy()
    changed[3:] = -1e5
    future = selectively_recalibrate(raw, evaluation_returns=changed, **kwargs)
    np.testing.assert_array_equal(
        result.final_quantiles[:4], future.final_quantiles[:4]
    )
    assert result.final_quantiles[4] != future.final_quantiles[4]
    assert result.decision is future.decision is decision
    assert result.window == 4


@pytest.mark.parametrize("which", ["return", "quantile", "length"])
def test_rejects_reuse_on_different_calibration(which):
    r, q = calibration()
    decision = decide(r, q)
    if which == "return":
        r[5] = 2  # same violations, but a different score history
    elif which == "quantile":
        q[5] = 0.25
    else:
        r, q = r[:-1], q[:-1]
    with pytest.raises(ValueError, match="does not match"):
        selectively_recalibrate(
            [0.0], calibration_returns=r, calibration_quantiles=q, decision=decision
        )


@pytest.mark.parametrize(
    "kw",
    [
        dict(alpha=0),
        dict(alpha=1),
        dict(alpha=np.nan),
        dict(kupiec_level=0),
        dict(kupiec_level=np.inf),
        dict(rule="oracle"),
    ],
)
def test_invalid_decision_configuration(kw):
    r, q = calibration()
    with pytest.raises(ValueError):
        decide(r, q, **kw)


@pytest.mark.parametrize(
    "bad", [[], [[0, 1]], [0, np.nan], [0, np.inf], [0, 1j], [False, True], ["0", "1"]]
)
def test_invalid_calibration_inputs(bad):
    with pytest.raises(ValueError):
        decide(bad, np.zeros(2))


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_invalid_rolling_window(window):
    r, q = calibration()
    with pytest.raises(ValueError, match="positive integer"):
        selectively_recalibrate(
            [0.0],
            calibration_returns=r,
            calibration_quantiles=q,
            decision=decide(r, q),
            method="rolling",
            window=window,
        )


def test_rolling_requires_sufficient_history_and_aligned_outcomes():
    r, q = calibration(20, 20)
    kwargs = dict(
        calibration_returns=r,
        calibration_quantiles=q,
        decision=decide(r, q),
        method="rolling",
    )
    with pytest.raises(ValueError, match="exceeds calibration"):
        selectively_recalibrate([0.0, 0.0], window=21, **kwargs)
    with pytest.raises(ValueError, match="requires evaluation_returns"):
        selectively_recalibrate([0.0, 0.0], window=20, **kwargs)
    with pytest.raises(ValueError, match="lengths must match"):
        selectively_recalibrate(
            [0.0, 0.0], window=20, evaluation_returns=[0.0], **kwargs
        )


def test_index_alignment_and_chronological_boundary():
    r, q = calibration(20, 20)
    dates = pd.date_range("2020-01-01", periods=22)
    r, q = pd.Series(r, index=dates[:20]), pd.Series(q, index=dates[:20])
    decision = decide(r, q)
    with pytest.raises(ValueError, match="indices must match"):
        decide(r, q.set_axis(dates[1:21]))
    with pytest.raises(ValueError, match="index must be unique"):
        decide(r.iloc[::-1], q.iloc[::-1])
    kwargs = dict(calibration_returns=r, calibration_quantiles=q, decision=decision)
    with pytest.raises(ValueError, match="strictly after calibration"):
        selectively_recalibrate(pd.Series([0.0, 0.0], index=dates[19:21]), **kwargs)
    raw = pd.Series([0.0, 0.0], index=dates[20:])
    result = selectively_recalibrate(raw, **kwargs)
    assert result.evaluation_index == tuple(dates[20:])
    with pytest.raises(ValueError, match="indices must match"):
        selectively_recalibrate(
            raw,
            method="rolling",
            window=20,
            evaluation_returns=pd.Series([0.0, 0.0], index=dates[19:21]),
            **kwargs,
        )


def test_inconsistent_decision_cannot_override_indication():
    r, q = calibration()
    decision = replace(decide(r, q), apply=True)
    with pytest.raises(ValueError, match="diagnostics, reasons, and apply"):
        selectively_recalibrate(
            [0.0], calibration_returns=r, calibration_quantiles=q, decision=decision
        )


@pytest.mark.parametrize("changes", [dict(alpha=0.02), dict(kupiec_level=0.04)])
def test_decision_configuration_cannot_change_after_gating(changes):
    r, q = calibration()
    decision = replace(decide(r, q), **changes)
    with pytest.raises(ValueError, match="does not match"):
        selectively_recalibrate(
            [0.0], calibration_returns=r, calibration_quantiles=q, decision=decision
        )


def test_score_overflow_is_reported():
    r = np.full(20, -1e308)
    q = np.full(20, 1e308)
    with pytest.raises(ValueError, match="exceeds finite range"):
        selectively_recalibrate(
            [0.0], calibration_returns=r, calibration_quantiles=q, decision=decide(r, q)
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(apply=True, basel_zone="yellow", reasons=("basel_not_green",)),
        dict(n_violations=9999),
        dict(scaled_violations_250=-100),
        dict(kupiec_statistic=123),
    ],
)
def test_coherent_forged_decision_metadata_is_rejected(changes):
    r, q = calibration()
    decision = replace(decide(r, q), **changes)
    with pytest.raises(ValueError, match="calibration-only diagnostics"):
        selectively_recalibrate(
            [0.0], calibration_returns=r, calibration_quantiles=q, decision=decision
        )


def test_decision_cannot_be_reused_after_relabelling_calibration_window():
    r, q = calibration()
    index = pd.date_range("2020-01-01", periods=len(r))
    r, q = pd.Series(r, index=index), pd.Series(q, index=index)
    decision = decide(r, q)
    old_index = pd.date_range("2010-01-01", periods=len(r))
    with pytest.raises(ValueError, match="does not match"):
        selectively_recalibrate(
            pd.Series([0.0], index=pd.date_range("2011-01-01", periods=1)),
            calibration_returns=r.set_axis(old_index),
            calibration_quantiles=q.set_axis(old_index),
            decision=decision,
        )


def test_equivalent_datetime_storage_units_preserve_decision_provenance():
    r, q = calibration()
    index = pd.date_range("2020-01-01", periods=len(r))
    r, q = pd.Series(r, index=index), pd.Series(q, index=index)
    decision = decide(r, q)
    equivalent = index.as_unit("us")
    result = selectively_recalibrate(
        pd.Series([0.0], index=pd.date_range("2021-01-01", periods=1)),
        calibration_returns=r.set_axis(equivalent),
        calibration_quantiles=q.set_axis(equivalent),
        decision=decision,
    )
    assert result.decision is decision
    assert not result.applied
