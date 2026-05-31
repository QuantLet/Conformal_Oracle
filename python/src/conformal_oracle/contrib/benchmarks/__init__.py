"""Benchmark forecasters (GJR-GARCH, GARCH-Normal, Historical Simulation).

These require ``arch>=6.0`` for the GARCH variants.  Install with::

    pip install conformal-oracle[benchmarks]
"""

from conformal_oracle.contrib.benchmarks.hist_sim import (
    HistoricalSimulationForecaster,
)

__all__ = [
    "HistoricalSimulationForecaster",
]

try:
    from conformal_oracle.contrib.benchmarks.garch_normal import (
        GARCHNormalForecaster,
    )
    from conformal_oracle.contrib.benchmarks.gjr_garch import (
        GJRGARCHForecaster,
    )

    __all__ += ["GJRGARCHForecaster", "GARCHNormalForecaster"]
except ImportError:
    pass
