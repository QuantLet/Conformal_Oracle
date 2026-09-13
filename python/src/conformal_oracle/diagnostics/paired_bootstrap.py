"""Paired loss comparisons with a common-calendar circular block bootstrap.

Each pair (forecaster, series) contributes a daily loss series per method on
its own trading dates. Resampling draws circular blocks of calendar days,
shared by every pair, so that observed date alignment and missingness are
preserved; each pair keeps equal weight. Bands condition on the stored
forecasts and corrections and exclude fitting uncertainty. Simultaneous bands
use the 95th percentile of the maximum absolute standardised deviation over
the contrasts of one family. Seeds follow the manuscript convention
``int.from_bytes(sha256('20260909/{key}/{block}')[:4], 'little')``.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["calendar_seed", "paired_calendar_bootstrap"]


def calendar_seed(key: str, block: int, prefix: str = "20260909") -> int:
    """Deterministic seed for one block length of one bootstrap family."""
    digest = sha256(f"{prefix}/{key}/{block}".encode()).digest()[:4]
    return int.from_bytes(digest, "little")


def paired_calendar_bootstrap(
    pair_losses: Sequence[pd.DataFrame],
    contrasts: Sequence[tuple[str, str]],
    block_days: Sequence[int] = (20, 60),
    draws: int = 999,
    seed_key: str = "panel-calendar",
    simultaneous: bool = True,
    scale: float = 1.0,
    pair_weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Mean loss differences with pointwise and simultaneous bootstrap bands.

    ``pair_losses``: one DataFrame per pair, indexed by date, one column of
    daily losses per method. ``contrasts``: ``(method, reference)`` pairs; the
    difference is method minus reference. ``pair_weights`` optionally divides
    each pair's losses (for example by its calibration-return standard
    deviation) before averaging. All contrasts form one simultaneous family.
    """
    if not pair_losses:
        raise ValueError("pair_losses must contain at least one pair")
    methods = sorted({m for c in contrasts for m in c})
    for f in pair_losses:
        missing = [m for m in methods if m not in f.columns]
        if missing:
            raise ValueError(f"pair is missing loss columns {missing}")
        if not isinstance(f.index, pd.DatetimeIndex):
            raise ValueError("each pair must be indexed by a DatetimeIndex")
    weights = (
        np.ones(len(pair_losses))
        if pair_weights is None
        else np.asarray(pair_weights, float)
    )
    if weights.shape != (len(pair_losses),) or not (weights > 0).all():
        raise ValueError("pair_weights must be positive, one per pair")
    dates = pd.date_range(
        min(f.index[0] for f in pair_losses),
        max(f.index[-1] for f in pair_losses),
        freq="D",
    )
    T, P, M = len(dates), len(pair_losses), len(methods)
    valid = np.zeros((T, P))
    values = np.zeros((T, P, M))
    for j, f in enumerate(pair_losses):
        ix = dates.get_indexer(f.index)
        if (ix < 0).any():
            raise ValueError("pair dates must lie inside the common calendar")
        valid[ix, j] = 1.0
        values[ix, j, :] = f[methods].to_numpy(dtype=float) / weights[j]
    per_pair = values.sum(axis=0) / valid.sum(axis=0)[:, None]
    point = per_pair.mean(axis=0)
    col = {m: i for i, m in enumerate(methods)}
    rows = []
    for block in block_days:
        rng = np.random.default_rng(calendar_seed(seed_key, int(block)))
        output = []
        for start in range(0, draws, 25):
            B = min(25, draws - start)
            counts = np.empty((B, T))
            for b in range(B):
                begins = rng.integers(0, T, size=int(np.ceil(T / block)))
                idx = ((begins[:, None] + np.arange(block)) % T).ravel()[:T]
                counts[b] = np.bincount(idx, minlength=T)
            denom = counts @ valid
            if not (denom > 0).all():
                raise ValueError(
                    "a resample left a pair without dates; use longer histories"
                )
            numerator = (counts @ values.reshape(T, -1)).reshape(B, P, M)
            output.append((numerator / denom[:, :, None]).mean(axis=1))
        draw = np.concatenate(output)
        delta = np.column_stack(
            [draw[:, col[m]] - draw[:, col[r]] for m, r in contrasts]
        )
        center = np.array([point[col[m]] - point[col[r]] for m, r in contrasts])
        sd = delta.std(axis=0, ddof=1)
        critical = (
            float(
                np.quantile(
                    np.max(
                        np.abs((delta - center) / np.where(sd > 0, sd, 1.0)), axis=1
                    ),
                    0.95,
                )
            )
            if simultaneous
            else float("nan")
        )
        for i, (m, r) in enumerate(contrasts):
            rows.append(
                dict(
                    method=m,
                    reference=r,
                    block_calendar_days=int(block),
                    draws=int(draws),
                    pairs=P,
                    difference=float(center[i] * scale),
                    lower=float(np.quantile(delta[:, i], 0.025) * scale),
                    upper=float(np.quantile(delta[:, i], 0.975) * scale),
                    simultaneous_lower=float((center[i] - critical * sd[i]) * scale)
                    if simultaneous
                    else float("nan"),
                    simultaneous_upper=float((center[i] + critical * sd[i]) * scale)
                    if simultaneous
                    else float("nan"),
                    simultaneous_family_size=len(contrasts) if simultaneous else 0,
                    critical_value=critical,
                )
            )
    return pd.DataFrame(rows)
