"""Reference forecasters — compatibility shim.

.. deprecated:: 0.3.0
    Import from ``conformal_oracle.contrib.benchmarks`` or
    ``conformal_oracle.contrib.tsfm`` instead.
"""

from __future__ import annotations

import warnings as _warnings


def __getattr__(name: str) -> object:
    _BENCHMARKS = {
        "GJRGARCHForecaster",
        "GARCHNormalForecaster",
        "HistoricalSimulationForecaster",
    }
    _TSFM = {
        "BaseTSFMForecaster",
        "ChronosForecaster",
        "LagLlamaForecaster",
        "TimesFM25Forecaster",
        "MoiraiForecaster",
    }
    if name in _BENCHMARKS:
        _warnings.warn(
            f"Importing {name} from conformal_oracle.forecasters is "
            f"deprecated since v0.3.0. Use "
            f"conformal_oracle.contrib.benchmarks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from conformal_oracle.contrib import benchmarks

        return getattr(benchmarks, name)
    if name in _TSFM:
        _warnings.warn(
            f"Importing {name} from conformal_oracle.forecasters is "
            f"deprecated since v0.3.0. Use "
            f"conformal_oracle.contrib.tsfm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from conformal_oracle.contrib import tsfm

        return getattr(tsfm, name)
    msg = f"module 'conformal_oracle.forecasters' has no attribute {name!r}"
    raise AttributeError(msg)


# Eagerly populate __all__ so `from conformal_oracle.forecasters import *`
# still works (with deprecation warnings emitted on actual use).
__all__ = [
    "GJRGARCHForecaster",
    "GARCHNormalForecaster",
    "HistoricalSimulationForecaster",
]
