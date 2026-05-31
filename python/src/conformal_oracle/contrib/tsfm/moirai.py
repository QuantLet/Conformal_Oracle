"""MoiraiForecaster wrapper for Salesforce Moirai 1.1 and 2.0."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from conformal_oracle._types import (
    PredictiveDistribution,
    QuantileGridDistribution,
    SampleDistribution,
)
from conformal_oracle.contrib.tsfm._base import BaseTSFMForecaster
from conformal_oracle.contrib.tsfm._cache import TSFMPredictionCache

PAPER_MODELS = {
    "1.1": {
        "small": "Salesforce/moirai-1.1-R-small",
        "base": "Salesforce/moirai-1.1-R-base",
        "large": "Salesforce/moirai-1.1-R-large",
    },
    "2.0": {
        "small": "Salesforce/moirai-2.0-R-small",
        "base": "Salesforce/moirai-2.0-R-base",
        "large": "Salesforce/moirai-2.0-R-large",
    },
}


class MoiraiForecaster(BaseTSFMForecaster):
    """Wrapper for Salesforce Moirai family.

    Output format depends on version:
      - Moirai 1.1: SampleDistribution (N MC samples)
      - Moirai 2.0: QuantileGridDistribution (quantile grid)

    The within-family contrast — same architecture family,
    different output interface — is the cleanest causal evidence
    in the paper for predictive-interface effects on tail
    calibration.
    """

    def __init__(
        self,
        version: Literal["1.1", "2.0"] = "1.1",
        size: Literal["small", "base", "large"] = "small",
        model_revision: str | None = None,
        context_length: int = 512,
        n_samples: int = 1000,
        seed: int = 2026,
        cache_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        model_id = PAPER_MODELS[version][size]
        super().__init__(
            model_id=model_id,
            model_revision=model_revision,
            context_length=context_length,
            n_samples=n_samples,
            seed=seed,
            cache_dir=cache_dir,
            device=device,
        )
        self.version = version
        self.size = size
        self._predictor: object | None = None
        self._quantile_probs: np.ndarray | None = None
        self._cache_obj: TSFMPredictionCache | None = None

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return

        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is not installed. Install with: "
                "pip install conformal-oracle[moirai]"
            ) from e

        device_str = self._resolve_device()
        device = torch.device(device_str)

        if self.version == "1.1":
            self._load_v11(device)
        else:
            self._load_v20(device)

        self._cache_obj = TSFMPredictionCache(
            cache_dir=self.cache_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
        )

    def _load_v11(self, device: object) -> None:
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as e:
            raise ImportError(
                "uni2ts is not installed. Install with: "
                "pip install conformal-oracle[moirai]"
            ) from e

        module = MoiraiModule.from_pretrained(self.model_id)
        forecast_obj = MoiraiForecast(
            module=module,
            prediction_length=1,
            context_length=self.context_length,
            patch_size="auto",
            num_samples=self.n_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        forecast_obj = forecast_obj.to(device).eval()
        self._predictor = forecast_obj.create_predictor(batch_size=1)

    def _load_v20(self, device: object) -> None:
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as e:
            raise ImportError(
                "uni2ts >= 1.2 is required for Moirai 2.0. Install with: "
                "pip install conformal-oracle[moirai]"
            ) from e

        module = Moirai2Module.from_pretrained(self.model_id)
        forecast_obj = Moirai2Forecast(
            module=module,
            prediction_length=1,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        device_str = str(device)
        forecast_obj = forecast_obj.to(device).eval()
        self._predictor = forecast_obj.create_predictor(
            batch_size=1, device=device_str,
        )

    def forecast(
        self, returns: pd.Series, t: int
    ) -> PredictiveDistribution:
        self._ensure_model()

        import torch
        from gluonts.dataset.common import ListDataset

        context = self._get_context(returns, t)

        n_s = self.n_samples if self.version == "1.1" else 0
        if self._cache_obj is not None:
            cached = self._cache_obj.get(context, t, n_s, self.seed)
            if cached is not None:
                return cached

        call_seed = self._call_seed(t)
        torch.manual_seed(call_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(call_seed)

        start_date = returns.index[max(0, t - self.context_length)]
        ds = ListDataset(
            [{"target": context, "start": pd.Period(start_date, freq="D")}],
            freq="D",
        )

        with torch.no_grad():
            forecasts = list(self._predictor.predict(ds))

        fc = forecasts[0]

        if self.version == "1.1":
            samples = fc.samples.flatten().astype(np.float64)
            dist: PredictiveDistribution = SampleDistribution(samples=samples)
        else:
            quantile_probs = np.array(
                [float(k) for k in fc.forecast_keys]
            )
            quantile_vals = fc.forecast_array.flatten().astype(np.float64)
            dist = QuantileGridDistribution(
                levels=quantile_probs, quantiles=quantile_vals,
            )

        if self._cache_obj is not None:
            self._cache_obj.put(context, t, n_s, self.seed, dist)

        return dist
