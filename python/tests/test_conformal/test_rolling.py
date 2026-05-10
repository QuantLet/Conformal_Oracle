"""Tests for compute_qv_roll and compute_drift_diagnostic."""

import numpy as np
import pytest

from conformal_oracle._types import SampleDistribution
from conformal_oracle.conformal.rolling import (
    compute_qv_roll,
    compute_qv_roll_from_scores,
    compute_drift_diagnostic,
)


def test_qv_roll_output_length():
    """Output length should be len(forecasts) - window."""
    rng = np.random.default_rng(42)
    n = 500
    window = 100
    alpha = 0.05

    forecasts = [
        SampleDistribution(samples=rng.standard_normal(100))
        for _ in range(n)
    ]
    realised = rng.standard_normal(n)

    qv = compute_qv_roll(forecasts, realised, alpha, window)
    assert len(qv) == n - window


def test_qv_roll_from_scores_length():
    """Same length check on precomputed scores."""
    scores = np.random.default_rng(42).standard_normal(300)
    qv = compute_qv_roll_from_scores(scores, 0.01, window=100)
    assert len(qv) == 200


def test_drift_diagnostic_stationary():
    """On stationary iid scores, TV distance should be small."""
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(500)
    delta = compute_drift_diagnostic(scores, window=200)
    assert len(delta) == 300
    assert np.mean(delta) < 0.3


def test_drift_diagnostic_shifted():
    """When scores shift halfway, TV distance should spike."""
    rng = np.random.default_rng(42)
    scores = np.concatenate([
        rng.standard_normal(300),
        rng.standard_normal(300) + 5.0,
    ])
    delta = compute_drift_diagnostic(scores, window=200)
    max_delta = np.max(delta)
    assert max_delta > 0.3
