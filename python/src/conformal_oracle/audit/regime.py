"""Regime classification: signal-preserving vs replacement."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def classify_regime_static(
    q_v_stat: float,
    var_raw: np.ndarray,
    threshold: float = 1.0,
) -> tuple[Literal["signal-preserving", "replacement"], float]:
    """Static-mode regime classification.

    R = |q_v_stat| / mean(|var_raw|).
    Returns (regime_label, R).
    """
    mean_var = float(np.mean(np.abs(var_raw)))
    if mean_var < 1e-12:
        r = float("inf")
    else:
        r = abs(q_v_stat) / mean_var

    regime: Literal["signal-preserving", "replacement"]
    if r > threshold:
        regime = "replacement"
    else:
        regime = "signal-preserving"
    return regime, r


def classify_regime_rolling(
    replacement_ratio: pd.Series,
    threshold: float = 1.0,
    persistence: int = 20,
) -> Literal["signal-preserving", "replacement"]:
    """Rolling-mode regime classification with persistence rule.

    A model is classified as replacement if R_t > threshold for at
    least `persistence` consecutive trading days.
    """
    above = (replacement_ratio > threshold).astype(int).values
    max_run = _max_consecutive(above)
    if max_run >= persistence:
        return "replacement"
    return "signal-preserving"


def _max_consecutive(arr: np.ndarray) -> int:
    """Length of longest consecutive run of 1s."""
    if len(arr) == 0:
        return 0
    max_run = 0
    current = 0
    for v in arr:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
