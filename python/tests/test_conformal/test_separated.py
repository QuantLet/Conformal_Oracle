"""Separated splits preserve the shift and keep proxy provenance explicit."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.conformal.quantile import conformal_quantile
from conformal_oracle.conformal.separated import (
    SeparatedSplitConformalVaR,
    proxy_separation_gap,
)


def test_exact_split_positions_and_lower_quantile_sign():
    returns = np.linspace(-0.03, 0.03, 20)
    quantiles = np.linspace(-0.05, -0.01, 20)
    before_returns, before_quantiles = returns.copy(), quantiles.copy()
    result = SeparatedSplitConformalVaR(
        gap=3, alpha=0.2, calibration_fraction=0.5,
    ).split(returns, quantiles)
    expected = conformal_quantile(quantiles[:10] - returns[:10], 0.2)
    np.testing.assert_array_equal(result.calibration_indices, np.arange(10))
    np.testing.assert_array_equal(result.gap_indices, np.arange(10, 13))
    np.testing.assert_array_equal(result.evaluation_indices, np.arange(13, 20))
    np.testing.assert_array_equal(result.raw_quantiles, quantiles[13:])
    np.testing.assert_array_equal(result.corrected_quantiles, quantiles[13:] - expected)
    assert result.q_v_stat == expected
    assert result.n_calibration == 10 and result.n_evaluation == 7
    assert result.conformal_rank == 9
    assert not result.maximum_score_fallback and not result.certified
    assert result.evaluation_index is None
    np.testing.assert_array_equal(returns, before_returns)
    np.testing.assert_array_equal(quantiles, before_quantiles)
    assert not np.shares_memory(result.raw_quantiles, quantiles)


def test_gap_changes_only_evaluation_start_not_shift_or_calibration():
    returns = np.linspace(-0.02, 0.03, 40)
    quantiles = -np.abs(returns) - 0.01
    contiguous = SeparatedSplitConformalVaR(gap=0).split(returns, quantiles)
    separated = SeparatedSplitConformalVaR(gap=5).split(returns, quantiles)
    assert contiguous.q_v_stat == separated.q_v_stat
    assert contiguous.gap_indices.size == 0
    np.testing.assert_array_equal(
        contiguous.calibration_indices, separated.calibration_indices,
    )
    np.testing.assert_array_equal(
        contiguous.corrected_quantiles[5:], separated.corrected_quantiles,
    )
    assert not contiguous.certified


def test_held_out_outcomes_and_gap_scores_cannot_change_shift():
    returns = np.linspace(-0.02, 0.03, 30)
    quantiles = np.full(30, -0.025)
    estimator = SeparatedSplitConformalVaR(gap=3, calibration_fraction=0.5)
    first = estimator.split(returns, quantiles)
    changed_returns, changed_quantiles = returns.copy(), quantiles.copy()
    changed_returns[15:] = np.arange(15) * 1000.0
    changed_quantiles[15:18] = 9000.0
    second = estimator.split(changed_returns, changed_quantiles)
    assert first.q_v_stat == second.q_v_stat
    np.testing.assert_array_equal(first.corrected_quantiles, second.corrected_quantiles)


def test_series_labels_are_preserved_separately_from_positions():
    index = pd.date_range("2024-01-01", periods=10)
    returns = pd.Series(np.arange(10) / 100, index=index)
    quantiles = pd.Series(np.full(10, -0.02), index=index)
    result = SeparatedSplitConformalVaR(
        gap=np.int64(2), calibration_fraction=0.5,
    ).split(returns, quantiles)
    assert result.evaluation_index.equals(index[7:])
    np.testing.assert_array_equal(result.evaluation_indices, [7, 8, 9])


def test_short_calibration_uses_existing_maximum_proxy_and_reports_it():
    returns = np.zeros(6)
    quantiles = np.array([-0.04, -0.02, -0.03, -0.06, -0.05, -0.04])
    result = SeparatedSplitConformalVaR(
        gap=1, alpha=0.01, calibration_fraction=0.5,
    ).split(returns, quantiles)
    assert result.q_v_stat == -0.02
    assert result.conformal_rank == 4
    assert result.maximum_score_fallback
    assert not result.certified


def test_rank_equal_to_calibration_size_is_not_overflow_fallback():
    result = SeparatedSplitConformalVaR(
        gap=0, alpha=0.25, calibration_fraction=0.5,
    ).split(np.zeros(6), np.arange(6, dtype=float))
    assert result.conformal_rank == 3
    assert not result.maximum_score_fallback


def test_minimum_evaluation_size_and_noninteger_fraction_boundary():
    estimator = SeparatedSplitConformalVaR(
        gap=2, calibration_fraction=0.55, minimum_evaluation_size=3,
    )
    result = estimator.split(np.zeros(10), np.ones(10))
    assert result.n_calibration == 5
    assert result.n_evaluation == 3
    with pytest.raises(ValueError, match="insufficient evaluation"):
        estimator.split(np.zeros(8), np.ones(8))


@pytest.mark.parametrize("key,value", [
    ("gap", -1), ("gap", 1.0), ("gap", True), ("gap", np.bool_(False)),
    ("gap", np.nan), ("gap", "1"),
    ("minimum_evaluation_size", 0), ("minimum_evaluation_size", 1.5),
    ("minimum_evaluation_size", True),
    ("alpha", 0), ("alpha", 1), ("alpha", -0.1), ("alpha", np.inf),
    ("alpha", np.nan), ("alpha", True), ("alpha", "0.01"),
    ("alpha", 10**400),
    ("calibration_fraction", 0), ("calibration_fraction", 1),
    ("calibration_fraction", np.nan), ("calibration_fraction", True),
])
def test_invalid_configuration_is_rejected(key, value):
    options = {"gap": 0, key: value}
    with pytest.raises(ValueError):
        SeparatedSplitConformalVaR(**options)


@pytest.mark.parametrize("returns,quantiles", [
    ([], []), ([1], [1]), ([1, 2], [1]),
    ([[1, 2]], [[1, 2]]), ([1, np.nan, 2], [1, 2, 3]),
    ([1, 2, 3], [1, np.inf, 3]), ([True, False], [1, 2]),
    (["1", "2"], [1, 2]), ([1j, 2j], [1, 2]),
])
def test_invalid_input_arrays_are_rejected(returns, quantiles):
    with pytest.raises(ValueError):
        SeparatedSplitConformalVaR(gap=0).split(returns, quantiles)


@pytest.mark.parametrize("gap", [3, 4, 9999999999999999999999999])
def test_empty_or_negative_remaining_window_is_rejected(gap):
    with pytest.raises(ValueError, match="insufficient evaluation"):
        SeparatedSplitConformalVaR(gap=gap).split(np.zeros(10), np.ones(10))


def test_empty_calibration_partition_is_rejected():
    with pytest.raises(ValueError, match="calibration observation"):
        SeparatedSplitConformalVaR(gap=0, calibration_fraction=0.01).split(
            np.zeros(10), np.ones(10),
        )


@pytest.mark.parametrize("index", [[1, 2, 4], [1, 1, 2], [3, 2, 1]])
def test_series_alignment_and_order_are_not_silently_repaired(index):
    returns = pd.Series([0.0, 0.1, 0.2], index=[1, 2, 3])
    quantiles = pd.Series([-0.1, -0.2, -0.3], index=index)
    if index != [1, 2, 4]:
        returns.index = index
    with pytest.raises(ValueError):
        SeparatedSplitConformalVaR(gap=0).split(returns, quantiles)


def test_mixed_series_and_positional_arrays_are_rejected():
    with pytest.raises(ValueError, match="two Series or two positional"):
        SeparatedSplitConformalVaR(gap=0).split(pd.Series([1, 2, 3]), [1, 2, 3])


def test_finite_inputs_with_overflowing_scores_are_rejected():
    with pytest.raises(ValueError, match="calibration scores must be finite"):
        SeparatedSplitConformalVaR(gap=0).split(
            [-1e308, -1e308, 0.0], [1e308, 1e308, 0.0],
        )


def test_configuration_is_immutable():
    estimator = SeparatedSplitConformalVaR(gap=0)
    with pytest.raises(FrozenInstanceError):
        estimator.gap = -1


def test_proxy_gap_matches_pandas_and_absolute_persistence():
    positive = np.array([0.0, 1, 2, 0, -2, -1, 0])
    negative = positive * (-1.0) ** np.arange(len(positive))
    a = proxy_separation_gap(positive, context_length=512)
    b = proxy_separation_gap(negative, context_length=512)
    assert a.signed_autocorrelation > 0 > b.signed_autocorrelation
    assert a.persistence_proxy == pytest.approx(pd.Series(positive).autocorr())
    assert b.persistence_proxy == pytest.approx(abs(pd.Series(negative).autocorr()))
    assert a.rho_tilde == pytest.approx(b.rho_tilde)
    expected = int(np.ceil(1.1 * np.log(len(positive)) / abs(np.log(a.rho_tilde))))
    assert a.log_gap == b.log_gap == expected
    assert a.gap == b.gap == 512 + expected
    assert a.context_length == 512 and a.n_calibration == 7
    assert a.safety_factor == 1.1 and a.minimum_log_gap == 5
    assert a.numerical_zero_threshold == 1e-12
    assert a.proxy_based and not a.certified and not a.near_zero


def test_proxy_zero_floor_is_not_a_general_minimum_gap():
    zero = np.array([0.0, 1, 0, -1, 0])
    a = proxy_separation_gap(zero, context_length=0)
    assert a.near_zero and a.rho_tilde == 0.0 and a.log_gap == 5
    small = zero.copy()
    small[-1] = 4e-12
    b = proxy_separation_gap(small, context_length=0)
    assert b.rho_tilde > 1e-12 and not b.near_zero
    assert b.log_gap == 1 < b.minimum_log_gap


def test_proxy_threshold_boundary_and_custom_floor():
    scores = [0.0, 1, 2, 0, -2, -1, 0]
    rho = abs(pd.Series(scores).autocorr())
    result = proxy_separation_gap(
        scores, context_length=10, minimum_log_gap=2,
        numerical_zero_threshold=rho,
    )
    assert result.near_zero and result.log_gap == 2 and result.gap == 12
    assert not proxy_separation_gap(
        scores, context_length=10, numerical_zero_threshold=np.nextafter(rho, 0),
    ).near_zero


def test_near_unit_persistence_has_no_extra_rejection_buffer():
    scores = np.array([0.0, 1, 2, 3, 4, 5])
    scores[2] += 3e-8
    result = proxy_separation_gap(scores, context_length=0)
    rho = abs(pd.Series(scores).autocorr())
    assert 1 - np.finfo(float).eps <= rho < 1
    assert result.persistence_proxy == rho
    expected = int(np.ceil(1.1 * np.log(len(scores)) / abs(np.log(rho))))
    assert result.log_gap == expected
    assert result.log_gap > 10**15
    with pytest.raises(ValueError, match="insufficient evaluation"):
        SeparatedSplitConformalVaR(gap=result.gap).split(np.zeros(20), np.ones(20))


def test_negative_near_unit_proxy_follows_computed_pandas_magnitude():
    scores = [1, -1, 1, -1]
    rho = abs(pd.Series(scores).autocorr())
    assert rho < 1
    result = proxy_separation_gap(scores, context_length=0)
    assert result.persistence_proxy == rho
    assert result.log_gap == int(np.ceil(1.1 * np.log(4) / abs(np.log(rho))))


def test_numerical_gap_overflow_is_rejected():
    with pytest.raises(ValueError, match="logarithmic gap is not finite"):
        proxy_separation_gap(
            [0, 1, 2, 0, -2, -1, 0], context_length=0, safety_factor=1e308,
        )


@pytest.mark.parametrize("scores", [
    [], [1], [1, 2], [1, 1, 1, 1], [1, 1, 1, 2],
    [0, 1, 2, 3], [1, -1, 1, -1, 1], [0, np.nan, 1],
    [0, 1, np.inf], [[0, 1, 2]], [True, False, True], [1j, 2j, 3j],
])
def test_invalid_or_undefined_proxy_is_not_silently_floored(scores):
    with pytest.raises(ValueError):
        proxy_separation_gap(scores, context_length=0)


@pytest.mark.parametrize("key,value", [
    ("context_length", -1), ("context_length", True), ("context_length", 2.0),
    ("safety_factor", 1), ("safety_factor", 0), ("safety_factor", True),
    ("safety_factor", np.inf), ("safety_factor", np.nan),
    ("minimum_log_gap", -1), ("minimum_log_gap", 2.0),
    ("minimum_log_gap", True), ("numerical_zero_threshold", -1),
    ("numerical_zero_threshold", 1), ("numerical_zero_threshold", np.nan),
    ("numerical_zero_threshold", True),
])
def test_invalid_proxy_configuration_is_rejected(key, value):
    with pytest.raises(ValueError):
        proxy_separation_gap(
            [0, 1, 2, 0, -2, -1, 0], **{"context_length": 0, key: value},
        )
