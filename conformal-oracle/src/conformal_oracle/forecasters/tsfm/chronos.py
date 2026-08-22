"""ChronosForecaster wrapper for Amazon Chronos-T5."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from conformal_oracle._types import PredictiveDistribution, SampleDistribution
from conformal_oracle.forecasters.tsfm._base import BaseTSFMForecaster
from conformal_oracle.forecasters.tsfm._cache import TSFMPredictionCache

PAPER_MODELS = {
    "small": "amazon/chronos-t5-small",
    "mini": "amazon/chronos-t5-mini",
}


class ChronosForecaster(BaseTSFMForecaster):
    """Wrapper for Amazon Chronos time-series foundation model.

    Produces SampleDistribution (N Monte Carlo samples from the
    predictive distribution at each forecast step).

    Paper models:
      - Chronos-Small: amazon/chronos-t5-small
      - Chronos-Mini:  amazon/chronos-t5-mini
    """

    def __init__(
        self,
        size: Literal["small", "mini"] = "small",
        model_revision: str | None = None,
        context_length: int = 512,
        n_samples: int = 1000,
        seed: int = 2026,
        cache_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        model_id = PAPER_MODELS[size]
        super().__init__(
            model_id=model_id,
            model_revision=model_revision,
            context_length=context_length,
            n_samples=n_samples,
            seed=seed,
            cache_dir=cache_dir,
            device=device,
        )
        self.size = size
        self._cache: TSFMPredictionCache | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            from chronos import ChronosPipeline
        except ImportError as e:
            raise ImportError(
                "Chronos is not installed. Install with: "
                "pip install conformal-oracle[chronos]"
            ) from e

        import torch

        device = self._resolve_device()
        kwargs: dict = {
            "dtype": torch.float32,
            "device_map": device,
        }
        if self.model_revision is not None:
            kwargs["revision"] = self.model_revision

        self._model = ChronosPipeline.from_pretrained(self.model_id, **kwargs)
        self._cache = TSFMPredictionCache(
            cache_dir=self.cache_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
        )

    def forecast(
        self, returns: pd.Series, t: int
    ) -> PredictiveDistribution:
        self._ensure_model()

        import torch

        context = self._get_context(returns, t)

        if self._cache is not None:
            cached = self._cache.get(context, t, self.n_samples, self.seed)
            if cached is not None:
                return cached

        call_seed = self._call_seed(t)
        torch.manual_seed(call_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(call_seed)

        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw_samples = self._model.predict(
                context_tensor,
                prediction_length=1,
                num_samples=self.n_samples,
            )

        samples = raw_samples[0, :, 0].cpu().numpy().astype(np.float64)
        dist = SampleDistribution(samples=samples)

        if self._cache is not None:
            self._cache.put(context, t, self.n_samples, self.seed, dist)

        return dist
