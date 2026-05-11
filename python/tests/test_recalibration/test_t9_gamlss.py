"""T9: GAMLSS-SST tests (optional, skipped if rpy2 unavailable)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import rpy2.robjects  # noqa: F401
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

from conformal_oracle.recalibration.gamlss_sst import GAMLSSSST


@pytest.mark.skipif(not HAS_RPY2, reason="rpy2 not installed")
def test_gamlss_placeholder():
    """GAMLSS-SST raises NotImplementedError (deferred to v0.3)."""
    gamlss = GAMLSSSST()
    rng = np.random.default_rng(42)
    with pytest.raises(NotImplementedError):
        gamlss.fit(rng.random(100), rng.random(100), 0.01)


def test_gamlss_not_available_raises():
    """GAMLSS-SST raises NotImplementedError without rpy2."""
    gamlss = GAMLSSSST()
    rng = np.random.default_rng(42)
    with pytest.raises(NotImplementedError):
        gamlss.fit(rng.random(100), rng.random(100), 0.01)
