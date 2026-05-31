"""GAMLSS skewed-t conditional density recalibration (optional)."""

from __future__ import annotations

import numpy as np


class GAMLSSSST:
    """Skewed-t conditional density via the GAMLSS framework.

    mu_t = beta_0 + beta_1 * q_hat_lo_t
    log sigma_t = gamma_0 + gamma_1 * r_{t-1}^2
    r_t | F_{t-1} ~ SST(mu_t, sigma_t, nu, tau)

    This baseline requires rpy2 and the R gamlss package.
    Raises NotImplementedError if dependencies are unavailable.
    """

    def __init__(self) -> None:
        self._available = False
        try:
            import rpy2.robjects  # noqa: F401
            self._available = True
        except ImportError:
            pass

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        if not self._available:
            raise NotImplementedError(
                "GAMLSS-SST requires rpy2 and the R gamlss package. "
                "Install with: pip install rpy2 && "
                "R -e 'install.packages(\"gamlss\")'"
            )
        raise NotImplementedError(
            "GAMLSS-SST implementation deferred to v0.3"
        )

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError(
            "GAMLSS-SST implementation deferred to v0.3"
        )
