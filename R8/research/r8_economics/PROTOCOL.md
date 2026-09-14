# Width, breach severity and capital arithmetic — protocol

Fixed 14 September 2026, before computation. Descriptive: the 240-pair panel and
its loss results have been inspected. No forecast, correction or bootstrap is
recomputed. Inputs are read-only: `artifacts/r8_ten_comparators/pairs/*/daily.parquet`
(column `r` and one lower-quantile column per method) and, for a cross-check,
`artifacts/r8_model_extension/ten_common_evaluation/summary.csv` (column `width`).

Quantities, per method.
1. VaR level: the mean of $-q_t$ over a pair's test dates, in percent of
   notional (log-return units multiplied by 100); reported as the pair-equal
   mean over the 240 pairs and as the pooled mean over all pair-days.
2. Width ratio: each pair's mean VaR level under the method divided by the
   same pair's mean raw VaR level; reported as mean, median, maximum, minimum
   and the number of pairs with ratio above one.
3. Breach severity, for Raw and Shift-CP: on dates with $r_t<q_t$, the mean
   exceedance $q_t-r_t$ and the mean loss $-r_t$, both in percent, pooled over
   pair-days and pair-equal.
4. Calendar-year widening of Shift-CP relative to Raw (pooled ratio by year).

Cross-check: the pair-equal mean VaR level for Raw, Shift-CP and Rolling250
must equal the `width` column of `summary.csv` averaged over the ten models,
to within 1e-12, or the run stops.

Outputs: `artifacts/r8_economics/{widths.csv, pair_ratios.csv, severity.csv,
by_year.csv, run.json, RESULTS.md}`; displays via `displays.py` with `--check`.
Capital arithmetic in the manuscript uses only the displayed macros, the
declared FRTB multipliers (MAR32 Table 1, MAR33.42) and one stated assumption.
