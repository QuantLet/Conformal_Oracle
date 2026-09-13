# Why the main-panel bands include zero — prespecified exploratory analysis

Fixed 13 September 2026 before computation. Declared exploratory: the
240-pair panel and its static-minus-raw contrast have been inspected. No
new forecasts, fits or corrections. Inputs are read-only:
`artifacts/r8_ten_comparators/pairs/*/daily.parquet` (daily losses of
every method per pair) and `metrics.csv`; the resampling convention of
`research/r8_ten_comparators/aggregate.py` (999 common-calendar circular
draws, blocks of 20 and 60 calendar days, seed as in that producer).

## Decomposition (descriptive)

For the static-minus-raw daily loss difference $d_{it}$ (pair $i$, date $t$),
report: the pooled mean; the between-pair standard deviation of pair means;
the average within-pair standard deviation; the average lag-1 to lag-20
autocorrelation of $d_{it}$; the number of distinct calendar dates; the
share of the pooled mean contributed by each model and each asset class;
the fraction of the bootstrap variance attributable to the ten largest
$|d_{it}|$ dates (leave-those-dates-out variance).

## Contrasts (all listed here; all reported; one simultaneous family of nine)

1. Static minus raw, all 240 pairs (reference; already published).
2. Static minus raw, five classical forecasters only (120 pairs).
3. Static minus raw, five foundation models only (120 pairs).
4. Static minus raw, excluding Lag-Llama (216 pairs).
5. Static minus raw, equities and equity ETF only.
6. Static minus raw, FX, bonds and commodities only.
7. Static minus raw, cryptocurrencies only.
8. Static minus raw, pair-normalised loss (divided by calibration-return
   standard deviation), all pairs.
9. Static minus raw, date-weighted (each date weighted equally across pairs
   present) rather than pair-weighted.

Bands: simultaneous 95% over the nine contrasts at 20 and 60 calendar days,
max-standardised-deviation rule; pointwise 95% also reported. Sensitivity:
block lengths 5 and 10 calendar days for contrast 1 only, reported
separately, not in the family.

## Sign statistics

Fraction of pairs with lower static than raw test QS (published: 188 of 240),
with a circular block bootstrap of the fraction (same draws) rather than a
binomial test, because pairs share dates.

## Outputs

`artifacts/r8_power_analysis/`: decomposition table, contrasts table with
means and both band types, sensitivity table, sign statistic, and
`RESULTS.md` stating plainly which contrasts exclude zero, with the
exploratory status repeated. No R8 file is modified.
