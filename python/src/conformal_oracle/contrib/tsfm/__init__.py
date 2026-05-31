"""TSFM forecaster wrappers (optional dependencies)."""

from __future__ import annotations

from conformal_oracle.contrib.tsfm._base import BaseTSFMForecaster
from conformal_oracle.contrib.tsfm._cache import TSFMPredictionCache

__all__ = ["BaseTSFMForecaster", "TSFMPredictionCache"]

try:
    from conformal_oracle.contrib.tsfm.chronos import ChronosForecaster

    __all__ += ["ChronosForecaster"]
except ImportError:
    pass

try:
    from conformal_oracle.contrib.tsfm.lag_llama import LagLlamaForecaster

    __all__ += ["LagLlamaForecaster"]
except ImportError:
    pass

try:
    from conformal_oracle.contrib.tsfm.timesfm import TimesFM25Forecaster

    __all__ += ["TimesFM25Forecaster"]
except ImportError:
    pass

try:
    from conformal_oracle.contrib.tsfm.moirai import MoiraiForecaster

    __all__ += ["MoiraiForecaster"]
except ImportError:
    pass


def clear_cache() -> None:
    """Remove all cached TSFM predictions."""
    from pathlib import Path

    cache_root = Path.home() / ".cache" / "conformal-oracle"
    if cache_root.exists():
        for pkl in cache_root.rglob("*.pkl"):
            pkl.unlink(missing_ok=True)


def set_cache_limit(max_gb: float = 5.0) -> None:
    """Set cache size limit (applied at next write)."""
    from conformal_oracle.contrib.tsfm import _cache

    _cache._DEFAULT_MAX_BYTES = int(max_gb * 1024 * 1024 * 1024)
