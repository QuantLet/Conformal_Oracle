"""Base class for TSFM forecaster wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_oracle._types import PredictiveDistribution


class BaseTSFMForecaster(ABC):
    """Abstract base for Time Series Foundation Model wrappers.

    Provides shared machinery: context-length management, model
    loading, caching, and deterministic seeding for sample-based
    outputs.
    """

    def __init__(
        self,
        model_id: str,
        model_revision: str | None = None,
        context_length: int = 512,
        n_samples: int = 1000,
        seed: int = 2026,
        cache_dir: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.context_length = context_length
        self.n_samples = n_samples
        self.seed = seed
        self.cache_dir = cache_dir or Path.home() / ".cache" / "conformal-oracle"
        self.device = device
        self._model: object | None = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _call_seed(self, t: int) -> int:
        return (
            self.seed + hash((self.__class__.__name__, self.model_id, t))
        ) & 0x7FFFFFFF

    def _get_context(self, returns: pd.Series, t: int) -> np.ndarray:
        start = max(0, t - self.context_length)
        return returns.iloc[start:t].values.astype(np.float32)

    def fit(self, returns: pd.Series) -> None:
        pass

    @abstractmethod
    def forecast(
        self, returns: pd.Series, t: int
    ) -> PredictiveDistribution: ...
