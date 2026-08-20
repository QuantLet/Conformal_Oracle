# REPRO NOTES — N8: Cross-Sectional Correlations (six TSFMs)

**Quantlet:** `CO_cross_sectional`
**Script:** `run_cross_sectional.py`
**Date:** 2026-05-08

## What changed

Added Moirai-1.1 to the model list (loaded from `moirai11_results.csv`).
The table now includes all six TSFMs plus four classical benchmarks,
with two summary-mean rows at the bottom.

## Per-TSFM Pearson correlations (qV vs asset characteristic, 24 assets)

| Model | Ann. Volatility | Excess Kurtosis | Tail Frequency |
|-------|-----------------|-----------------|----------------|
| Chronos-Small | 0.9556 | -0.2741 | 0.5078 |
| Chronos-Mini | 0.9614 | -0.2720 | 0.5023 |
| TimesFM 2.5 | 0.9571 | -0.3189 | 0.5276 |
| Moirai 2.0 | 0.9647 | -0.3109 | 0.5158 |
| Lag-Llama | 0.9073 | -0.2476 | 0.5311 |
| Moirai 1.1 | 0.6737 | -0.1969 | 0.5237 |

## Summary means

| Group | Ann. Volatility | Tail Frequency |
|-------|-----------------|----------------|
| TSFMs (six) | 0.9033 | 0.5180 |
| Replacement-regime TSFMs (four) | 0.9597 | 0.5134 |

## B4 flag: Moirai-1.1 volatility correlation

ρ(vol) = 0.674 for Moirai-1.1. This is OUTSIDE the [-0.30, +0.30]
range specified in the B4 guard. The "near-zero" framing proposed in
the issue text does not hold. Moirai-1.1 shows a positive but
attenuated volatility correlation relative to the four replacement-
regime TSFMs (0.67 vs 0.96). Body text adapted to say "positive but
attenuated" instead of "near-zero."

## Rationale

Six-TSFM completeness: all TSFMs in the study are now represented.
Replacement-regime conditional mean: isolates the four TSFMs where
the conformal correction replaces the base forecast's tail signal,
yielding the strongest volatility dependence.

## Data sources

- `cfp_ijf_data/paper_outputs/tables/all_results.csv` (9 models, α=0.01 → 216 rows)
- `cfp_ijf_data/paper_outputs/tables/moirai11_results.csv` (Moirai-1.1, 24 rows)
- Asset returns: `cfp_ijf_data/returns/{ASSET}.csv`
