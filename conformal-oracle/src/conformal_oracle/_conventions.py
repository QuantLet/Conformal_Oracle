"""Documented constants and conventions.

Returns: pandas Series of log returns, decimal form (not percent),
         negative = loss.

VaR sign: VaR is reported as a positive number representing the loss
          threshold. A violation occurs when r_t < -VaR_t.

Quantile level alpha: small (e.g. 0.01) means the lower-tail probability.
                      The forecast is the alpha-quantile of the predictive
                      distribution.
"""

DEFAULT_ALPHA: float = 0.01
DEFAULT_CALIBRATION_SPLIT: float = 0.70
DEFAULT_ROLLING_WINDOW: int = 250
DEFAULT_WARMUP: int = 250
DEFAULT_SAMPLE_SIZE: int = 1000
DEFAULT_BLOCK_LENGTH: int = 20
DEFAULT_N_BOOT: int = 999
DEFAULT_SEED: int = 2026
DEFAULT_PERSISTENCE: int = 20
REPLACEMENT_THRESHOLD: float = 1.0
