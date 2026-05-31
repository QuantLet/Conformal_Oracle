"""LagLlamaForecaster wrapper for Lag-Llama (Rasul et al. 2024)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from conformal_oracle._types import PredictiveDistribution, SampleDistribution
from conformal_oracle.contrib.tsfm._base import BaseTSFMForecaster
from conformal_oracle.contrib.tsfm._cache import TSFMPredictionCache

PAPER_MODEL = "time-series-foundation-models/Lag-Llama"
PAPER_CKPT = "lag-llama.ckpt"


class LagLlamaForecaster(BaseTSFMForecaster):
    """Wrapper for Lag-Llama time-series foundation model.

    Produces SampleDistribution. Uses Student-t prediction heads
    internally; the wrapper draws N samples from the predicted
    distribution.

    Paper model:
      time-series-foundation-models/Lag-Llama @ main
    """

    def __init__(
        self,
        model_revision: str | None = None,
        context_length: int = 512,
        n_samples: int = 1000,
        seed: int = 2026,
        cache_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(
            model_id=PAPER_MODEL,
            model_revision=model_revision,
            context_length=context_length,
            n_samples=n_samples,
            seed=seed,
            cache_dir=cache_dir,
            device=device,
        )
        self._predictor: object | None = None
        self._cache_obj: TSFMPredictionCache | None = None

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface-hub is not installed. Install with: "
                "pip install conformal-oracle[lag_llama]"
            ) from e

        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError as e:
            raise ImportError(
                "lag-llama is not installed. Install with: "
                "pip install conformal-oracle[lag_llama]"
            ) from e

        import torch

        device_str = self._resolve_device()
        device = torch.device(device_str)

        kwargs = {"repo_id": PAPER_MODEL, "filename": PAPER_CKPT}
        if self.model_revision is not None:
            kwargs["revision"] = self.model_revision
        ckpt = hf_hub_download(**kwargs)

        # PyTorch 2.6+ defaults to weights_only=True; the Lag-Llama
        # checkpoint contains gluonts distribution and loss classes
        # that aren't in the safe globals whitelist. Temporarily
        # override torch.load to allow full deserialization of this
        # trusted HuggingFace checkpoint.
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(
            *a, **{**kw, "weights_only": False}
        )
        try:
            rope_factor = max(1.0, self.context_length / 32)
            estimator = LagLlamaEstimator(
                prediction_length=1,
                context_length=self.context_length,
                input_size=1,
                n_layer=8,
                n_embd_per_head=36,
                n_head=4,
                num_parallel_samples=self.n_samples,
                batch_size=1,
                device=device,
                rope_scaling={"type": "linear", "factor": rope_factor},
                ckpt_path=ckpt,
                time_feat=True,
            )

            transformation = estimator.create_transformation()
            module = estimator.create_lightning_module()
            self._predictor = estimator.create_predictor(
                transformation, module
            )
        finally:
            torch.load = _orig_load

        self._cache_obj = TSFMPredictionCache(
            cache_dir=self.cache_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
        )

    def forecast(
        self, returns: pd.Series, t: int
    ) -> PredictiveDistribution:
        self._ensure_model()

        import torch
        from gluonts.dataset.common import ListDataset

        context = self._get_context(returns, t)

        if self._cache_obj is not None:
            cached = self._cache_obj.get(context, t, self.n_samples, self.seed)
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
            forecasts = list(
                self._predictor.predict(ds, num_samples=self.n_samples)
            )

        samples = forecasts[0].samples.flatten().astype(np.float64)
        dist = SampleDistribution(samples=samples)

        if self._cache_obj is not None:
            self._cache_obj.put(context, t, self.n_samples, self.seed, dist)

        return dist
