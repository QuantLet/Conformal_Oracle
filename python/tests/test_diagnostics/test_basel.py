"""Tests for Basel traffic light classifier."""

import numpy as np

from conformal_oracle.diagnostics.basel import basel_traffic_light


def test_green_boundary():
    """4 violations in 250 days = green."""
    violations = np.zeros(250)
    violations[:4] = 1
    assert basel_traffic_light(violations) == "green"


def test_yellow_boundary_low():
    """5 violations in 250 days = yellow."""
    violations = np.zeros(250)
    violations[:5] = 1
    assert basel_traffic_light(violations) == "yellow"


def test_yellow_boundary_high():
    """9 violations in 250 days = yellow."""
    violations = np.zeros(250)
    violations[:9] = 1
    assert basel_traffic_light(violations) == "yellow"


def test_red_boundary():
    """10 violations in 250 days = red."""
    violations = np.zeros(250)
    violations[:10] = 1
    assert basel_traffic_light(violations) == "red"


def test_empty_violations():
    assert basel_traffic_light(np.array([])) == "green"
