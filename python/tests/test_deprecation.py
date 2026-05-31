"""Tests for deprecation warnings on old import paths."""

from __future__ import annotations

import warnings

import pytest

_has_arch = pytest.importorskip is not None
try:
    import arch  # noqa: F401

    _has_arch = True
except ImportError:
    _has_arch = False


def test_forecasters_import_warns():
    """Importing from conformal_oracle.forecasters emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from conformal_oracle.forecasters import (
            HistoricalSimulationForecaster,  # noqa: F401
        )

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "deprecated" in str(dep_warnings[0].message).lower()


@pytest.mark.skipif(not _has_arch, reason="arch not installed")
def test_forecasters_gjr_warns():
    """Importing GJR from the old path emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from conformal_oracle.forecasters import GJRGARCHForecaster  # noqa: F401

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1


def test_contrib_benchmarks_no_warning():
    """Importing from contrib.benchmarks does NOT emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from conformal_oracle.contrib.benchmarks import (  # noqa: F401
            HistoricalSimulationForecaster,
        )

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0


@pytest.mark.skipif(not _has_arch, reason="arch not installed")
def test_contrib_benchmarks_gjr_no_warning():
    """Importing GJR from contrib.benchmarks does NOT emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from conformal_oracle.contrib.benchmarks import (  # noqa: F401
            GJRGARCHForecaster,
        )

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0
