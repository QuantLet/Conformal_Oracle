"""Basel traffic light classifier."""

from __future__ import annotations

from typing import Literal

import numpy as np


def basel_traffic_light(
    violations: np.ndarray,
    window: int = 250,
) -> Literal["green", "yellow", "red"]:
    """Basel classification scaled to a 250-day window.

    Green: <= 4 violations per 250 days.
    Yellow: 5-9 violations per 250 days.
    Red: >= 10 violations per 250 days.
    """
    n = len(violations)
    if n == 0:
        return "green"

    count = int(np.sum(violations[-window:]))

    if n != window:
        count = int(round(count * 250 / min(n, window)))

    if count <= 4:
        return "green"
    elif count <= 9:
        return "yellow"
    else:
        return "red"
