# Why the main-panel bands include zero: results

Exploratory. The protocol (`research/r8_power_analysis/PROTOCOL.md`, fixed 13 September 2026)
was written after the 240-pair panel and its static-minus-raw contrast had been inspected. All
nine contrasts listed there are reported; none was added or removed. No forecasts, fits or
corrections were recomputed. Inputs are the daily losses in
`artifacts/r8_ten_comparators/pairs/*/daily.parquet` and the per-pair `metrics.csv`. Static is
the method `Shift-CP`; raw is `Raw`; the loss is the 1% pinball loss; d_it is the static loss
minus the raw loss for pair i on date t. Losses are reported multiplied by 10^4 except where
stated.

## Reproduction of the published contrast 1 (measured)

The resampling was reimplemented from scratch (`research/r8_power_analysis/analysis.py`): 999
circular block-bootstrap draws on the common calendar of 2707 days (2019-04-04 to 2026-08-31),
blocks of 20 and 60 calendar days, `numpy.random.default_rng` seeded with
`int.from_bytes(sha256('20260909/panel-calendar/{block}')[:4], 'little')`, giving seeds
4034980058 (20 days), 196994084 (60 days), 77764878 (5 days) and 3075887739 (10 days). Pairs
are weighted equally; within a pair the resampled loss is the count-weighted sum divided by
the count-weighted number of trading days.

- Point estimate static minus raw: -0.159292 x 10^-4, equal to the published value (-0.1593)
  to 4 decimals and to 1e-9 in absolute terms.
- The 999 reimplemented draws of (Shift-CP minus Raw) equal the published
  `results/bootstrap_20.npz` and `bootstrap_60.npz` draws to a maximum absolute difference of
  1.1e-18 (20 days) and 9.8e-19 (60 days).
- The published pointwise 95% band of contrast 1, [-0.3072, -0.0242] at 20 days and
  [-0.3249, -0.0239] at 60 days, is reproduced. The published simultaneous band
  [-0.3470, 0.0284] belongs to the eight-comparison family of the main table and is not
  recomputed here; the simultaneous bands below are over the nine contrasts of the protocol.

## Which contrasts exclude zero (measured; `contrasts.csv`)

Simultaneous 95% bands over the nine contrasts (max-standardised-deviation rule; critical
values 2.4546 at 20 days, 2.5181 at 60 days):

- No contrast excludes zero at both block lengths.
- At 20 days, two contrasts exclude zero: contrast 3 (foundation models only, upper edge
  -0.0016) and contrast 8 (pair-normalised, upper edge -0.00004 in loss / calibration SD).
- At 60 days, no contrast excludes zero. Contrast 3 has upper edge +0.0225; contrast 8 has
  upper edge +0.00011.

Pointwise 95% percentile bands:

- Exclude zero at both block lengths: contrasts 1 (all pairs), 3 (foundation only),
  5 (equities only), 6 (FX, bonds, commodities), 8 (pair-normalised).
- Contrast 2 (classical only) excludes zero at 60 days (upper -0.0019) and has upper edge
  +0.00003 at 20 days.
- Include zero at both block lengths: contrast 4 (excluding Lag-Llama, upper +0.009 at both),
  contrast 7 (crypto only, 20 pairs, band [-1.38, +0.58] at 20 days) and contrast 9
  (date-weighted, band [-0.27, +0.16] at 20 days).

Nine contrasts, point and simultaneous band, x 10^4 unless stated:

| # | contrast | pairs | point | 20-day simultaneous | 60-day simultaneous |
|---|---|---|---|---|---|
| 1 | all pairs | 240 | -0.1593 | [-0.3372, +0.0186] | [-0.3537, +0.0352] |
| 2 | classical forecasters only | 120 | -0.1386 | [-0.3208, +0.0436] | [-0.3306, +0.0534] |
| 3 | foundation models only | 120 | -0.1800 | [-0.3584, -0.0016] | [-0.3825, +0.0225] |
| 4 | excluding Lag-Llama | 216 | -0.1099 | [-0.2667, +0.0469] | [-0.2774, +0.0576] |
| 5 | equities only | 110 | -0.1752 | [-0.3615, +0.0110] | [-0.3965, +0.0461] |
| 6 | FX, bonds, commodities | 110 | -0.1124 | [-0.2330, +0.0082] | [-0.2473, +0.0226] |
| 7 | crypto only | 20 | -0.3298 | [-1.6294, +0.9698] | [-1.6331, +0.9734] |
| 8 | pair-normalised (loss / calibration SD, not x 10^4) | 240 | -0.00110 | [-0.00215, -0.00004] | [-0.00230, +0.00011] |
| 9 | date-weighted | 240 | -0.0320 | [-0.3064, +0.2424] | [-0.3544, +0.2904] |

Contrast 9 definition: the mean over the 2264 trading dates of the cross-sectional mean of
d_it over the pairs present on that date; resampled as the count-weighted mean of the date
means. Its point (-0.032) is one fifth of the pair-weighted point (-0.159): dates early in
the test window (10 pairs present on the first date) and the pairs that dominate the gain
(Lag-Llama, 38% of the pair-weighted sum) are weighted differently.

## Sensitivity of contrast 1 to block length (measured; `sensitivity.csv`)

Contrast 1 alone, not in the family: at 5 calendar days the pointwise band is
[-0.2997, -0.0455] and the single-contrast max-|z| band is [-0.2903, -0.0283]; at 10 days
[-0.3080, -0.0366] and [-0.2952, -0.0234]. Shorter blocks give narrower bands. The
bootstrap SD of contrast 1 rises from 0.0725 at 20 days to 0.0772 at 60 days (x 10^4).

## Decomposition (measured; `decomposition.csv`)

- Pooled mean of d_it over all 362,520 pair-date observations: -0.1565 x 10^-4. Pair-weighted
  mean (contrast 1): -0.1593 x 10^-4.
- Between-pair SD of the 240 pair means: 0.2997 x 10^-4.
- Average within-pair SD of d_it: 5.264 x 10^-4 (median 3.528 x 10^-4). The within-pair SD
  is 33 times the pair mean; the daily difference is dominated by tail days on which one of
  the two forecasts is breached.
- Average lag-1 autocorrelation of d_it: 0.0335; lag 2: 0.0250; lag 3: 0.0251; lags 4 to 14
  between 0.007 and 0.020; lags 15 to 20 between -0.008 and +0.004. Mean over lags 1 to 20:
  0.0095.
- Distinct calendar dates with at least one pair: 2264 (of 2707 calendar days). Pairs present
  per trading date range from 10 to 240.
- Share of the pair-weighted gain by model: Lag-Llama 37.9%, EWMA 14.0%, GARCH-N 13.5%,
  PatchTST-FM 12.7%, GJR-GARCH 10.1%, Moirai-1.1 5.2%, Hist-Sim 3.8%, GJR-GARCH-t 2.0%,
  Chronos-2 1.3%, TS-ICL -0.6%. By asset class: Equity 50.4%, Commodity 25.1%, Crypto 17.3%,
  FX 5.2%, Bond ETF 2.0%. Observation-pooled shares are in the table (Lag-Llama 38.2%,
  Equity 61.1%, Crypto 9.1%).
- Ten largest dates. Ranked by the absolute cross-sectional sum of d_it (the date's
  contribution to the pooled numerator), the ten dates are 2025-10-10, 2026-02-05,
  2025-04-04, 2021-11-26, 2025-04-03, 2025-03-03, 2024-08-02, 2020-03-09, 2026-06-05 and
  2025-04-06. They hold 60.4% of the pooled sum of d_it. Removing them from every pair moves
  the contrast-1 point from -0.1593 to -0.0364 x 10^-4 and removes 63.0% (20-day blocks)
  and 60.1% (60-day blocks) of the bootstrap variance of contrast 1. Ranked instead by the
  largest single |d_it| on the date, the ten dates hold 44.2% of the pooled sum and 41.5%
  (20-day) and 42.8% (60-day) of the bootstrap variance; the point without them is -0.0520.

## Sign statistic (measured; `sign.csv`)

188 of 240 pairs (78.3%) have lower static than raw test loss, matching the published count.
The circular block bootstrap of the fraction (same draws) gives 95% percentile intervals
[0.504, 0.858] at 20 days and [0.508, 0.854] at 60 days, with bootstrap SD 0.090 and 0.095.
dates is listed in `sign.csv` for reference only.

## Reading

The static-minus-raw gain is concentrated: ten calendar dates carry 60% of the pooled sum
and 60-63% of the bootstrap variance, Lag-Llama carries 38% of the pair-weighted gain, and
the within-pair daily SD is 33 times the pair mean. A calendar block bootstrap that
resamples those dates with the rest produces a band whose half-width (0.18 at 20 days, 0.19
at 60 days after the family adjustment) exceeds the point (0.16). Contrast 4, which drops
Lag-Llama, has a point of -0.110 with pointwise bands that include zero at both block lengths.
The date-weighted contrast 9 has point -0.032. These are descriptive statements about this
panel; the protocol fixed the contrasts after the panel had been seen, and no band here is a
confirmatory test.

## Runtime and provenance

- Producer: `research/r8_power_analysis/analysis.py`; replay with `--check` recomputes every
  table and asserts equality with the saved CSVs (atol 1e-10) and with the published contrast 1
  and published draws. The replay passed.
- Runtime: 0.9 s per run (Python 3.13.9, numpy 2.3.5, pandas 2.3.3, two BLAS threads,
  `/private/tmp/irfa-r8-conda-clean/bin/python`).
- Seeds and hashes: `run.json` (protocol SHA-256, producer SHA-256, seeds per block length).
- Files: `decomposition.csv`, `contrasts.csv`, `sensitivity.csv`, `sign.csv`,
  `pair_means.csv` (per-pair static and raw QS with class labels), `run.json`.


## Corrections of 13 September 2026 (after the independent review)

The printed seed key had an extra `/0`; the actual key is `20260909/panel-calendar/{block}` and the random streams are unchanged. The binomial reference p-value was removed from `sign.csv` because pairs share dates; the calendar-bootstrap interval is the evidence. The leave-ten-dates-out variance change is an influence sensitivity, not an additive variance attribution.
