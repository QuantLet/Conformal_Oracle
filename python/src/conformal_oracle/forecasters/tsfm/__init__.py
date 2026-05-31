"""TSFM forecaster wrappers — compatibility shim.

.. deprecated:: 0.3.0
    Import from ``conformal_oracle.contrib.tsfm`` instead.
"""

from __future__ import annotations

import warnings as _warnings


def __getattr__(name: str) -> object:
    _NAMES = {
        "BaseTSFMForecaster",
        "TSFMPredictionCache",
        "ChronosForecaster",
        "LagLlamaForecaster",
        "TimesFM25Forecaster",
        "MoiraiForecaster",
        "clear_cache",
        "set_cache_limit",
    }
    if name in _NAMES:
        _warnings.warn(
            f"Importing {name} from conformal_oracle.forecasters.tsfm is "
            f"deprecated since v0.3.0. Use "
            f"conformal_oracle.contrib.tsfm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from conformal_oracle.contrib import tsfm

        return getattr(tsfm, name)
    raise AttributeError(
        f"module 'conformal_oracle.forecasters.tsfm' has no attribute {name!r}"
    )


__all__ = [
    "BaseTSFMForecaster",
    "TSFMPredictionCache",
]
