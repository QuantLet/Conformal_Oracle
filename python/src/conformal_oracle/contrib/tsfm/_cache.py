"""Disk-based prediction cache for TSFM forecasters."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from conformal_oracle._types import PredictiveDistribution

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


class TSFMPredictionCache:
    """Disk cache for TSFM predictions.

    Key: hash of (model_id, model_revision, context_hash,
         context_length, n_samples, seed, t)
    Value: serialised PredictiveDistribution
    """

    def __init__(
        self,
        cache_dir: Path,
        model_id: str,
        model_revision: str | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._dir = cache_dir / _safe_name(model_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model_id = model_id
        self._model_revision = model_revision
        self._max_bytes = max_bytes

    def get(
        self,
        context: np.ndarray,
        t: int,
        n_samples: int,
        seed: int,
    ) -> PredictiveDistribution | None:
        path = self._key_path(context, t, n_samples, seed)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def put(
        self,
        context: np.ndarray,
        t: int,
        n_samples: int,
        seed: int,
        dist: PredictiveDistribution,
    ) -> None:
        self._evict_if_needed()
        path = self._key_path(context, t, n_samples, seed)
        with open(path, "wb") as f:
            pickle.dump(dist, f, protocol=pickle.HIGHEST_PROTOCOL)

    def clear(self) -> None:
        for p in self._dir.glob("*.pkl"):
            p.unlink(missing_ok=True)

    def _key_path(
        self,
        context: np.ndarray,
        t: int,
        n_samples: int,
        seed: int,
    ) -> Path:
        ctx_hash = hashlib.sha256(context.tobytes()).hexdigest()[:16]
        key_str = json.dumps(
            [self._model_id, self._model_revision, ctx_hash, t, n_samples, seed],
            sort_keys=True,
        )
        digest = hashlib.sha256(key_str.encode()).hexdigest()[:24]
        return self._dir / f"{digest}.pkl"

    def _evict_if_needed(self) -> None:
        files = sorted(self._dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while total > self._max_bytes and files:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")
