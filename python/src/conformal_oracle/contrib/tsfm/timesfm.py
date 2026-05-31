"""TimesFM25Forecaster wrapper for Google TimesFM 2.5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from conformal_oracle._types import (
    PredictiveDistribution,
    QuantileGridDistribution,
)
from conformal_oracle.contrib.tsfm._base import BaseTSFMForecaster
from conformal_oracle.contrib.tsfm._cache import TSFMPredictionCache

PAPER_MODEL = "google/timesfm-2.5-200m-pytorch"
QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


class TimesFM25Forecaster(BaseTSFMForecaster):
    """Wrapper for Google TimesFM 2.5.

    Produces QuantileGridDistribution: TimesFM 2.5 emits a fixed
    9-decile grid (0.1 through 0.9) natively. The
    QuantileGridDistribution handles tail completion to extract
    the 1% quantile via Student-t closure.

    Paper model:
      google/timesfm-2.5-200m-pytorch
    """

    def __init__(
        self,
        model_revision: str | None = None,
        context_length: int = 512,
        seed: int = 2026,
        cache_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(
            model_id=PAPER_MODEL,
            model_revision=model_revision,
            context_length=context_length,
            n_samples=0,
            seed=seed,
            cache_dir=cache_dir,
            device=device,
        )
        self._cache_obj: TSFMPredictionCache | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import timesfm
        except ImportError as e:
            raise ImportError(
                "timesfm is not installed. Install with: "
                "pip install conformal-oracle[timesfm]"
            ) from e

        try:
            from timesfm import ForecastConfig
        except ImportError as e:
            raise ImportError(
                "timesfm is installed but lacks the 2.5 API "
                "(ForecastConfig not found). Install from GitHub: "
                "pip install git+https://github.com/google-research/timesfm.git"
            ) from e

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            PAPER_MODEL,
        )
        self._model.compile(
            ForecastConfig(
                max_horizon=1,
                max_context=self.context_length,
                per_core_batch_size=1,
            )
        )

        self._cache_obj = TSFMPredictionCache(
            cache_dir=self.cache_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
        )

    def forecast(
        self, returns: pd.Series, t: int
    ) -> PredictiveDistribution:
        self._ensure_model()

        context = self._get_context(returns, t)

        if self._cache_obj is not None:
            cached = self._cache_obj.get(context, t, 0, self.seed)
            if cached is not None:
                return cached

        _point_fc, quantile_fc = self._model.forecast(
            horizon=1, inputs=[context]
        )

        # quantile_fc shape: (1, 1, num_quantiles)
        # Index 0 is the point forecast; indices 1: are the 9 deciles
        q_raw = np.asarray(quantile_fc[0, 0, 1:], dtype=np.float64)
        q_vals = q_raw[: len(QUANTILE_LEVELS)]

        dist = QuantileGridDistribution(
            levels=QUANTILE_LEVELS.copy(),
            quantiles=q_vals,
        )

        if self._cache_obj is not None:
            self._cache_obj.put(context, t, 0, self.seed, dist)

        return dist
