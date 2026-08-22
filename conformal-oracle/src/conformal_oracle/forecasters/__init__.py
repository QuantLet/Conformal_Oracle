"""Reference forecasters (GJR-GARCH, GARCH-Normal, Historical Simulation)."""

from conformal_oracle.forecasters.garch_normal import GARCHNormalForecaster
from conformal_oracle.forecasters.gjr_garch import GJRGARCHForecaster
from conformal_oracle.forecasters.hist_sim import HistoricalSimulationForecaster

__all__ = [
    "GJRGARCHForecaster",
    "GARCHNormalForecaster",
    "HistoricalSimulationForecaster",
]

try:
    from conformal_oracle.forecasters.tsfm import BaseTSFMForecaster

    __all__ += ["BaseTSFMForecaster"]
except ImportError:
    pass

try:
    from conformal_oracle.forecasters.tsfm.chronos import ChronosForecaster

    __all__ += ["ChronosForecaster"]
except ImportError:
    pass

try:
    from conformal_oracle.forecasters.tsfm.lag_llama import LagLlamaForecaster

    __all__ += ["LagLlamaForecaster"]
except ImportError:
    pass

try:
    from conformal_oracle.forecasters.tsfm.timesfm import TimesFM25Forecaster

    __all__ += ["TimesFM25Forecaster"]
except ImportError:
    pass

try:
    from conformal_oracle.forecasters.tsfm.moirai import MoiraiForecaster

    __all__ += ["MoiraiForecaster"]
except ImportError:
    pass
