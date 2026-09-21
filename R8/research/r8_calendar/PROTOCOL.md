# Calendar heterogeneity and external return autocorrelation — protocol

Fixed 14 September 2026, before computation. Descriptive facts about the
sampling calendars of the evaluated series and the serial correlation of the
external portfolio returns. No forecast, correction or bootstrap is recomputed.

Inputs (read-only):
- `artifacts/r8_commodity_etp/panel/base/data/returns/{BTC,ETH,SP500,GDAXI}.csv`
  (August-2026 return histories): weekday counts, calendar-day gaps between
  consecutive observations, weekend observations.
- `artifacts/r8_ten_comparators/pairs/*/daily.parquet`: number of pairs
  present on each of the ten stress dates listed in
  `artifacts/r8_power_analysis/decomposition.csv` (row `top10_dates_abs_date_sum`)
  and the weekday of each; the smallest and largest number of pairs present on
  any evaluated date.
- `artifacts/r8_external2/devexus/pairs/HS__*/daily.parquet` and
  `artifacts/r8_external/july2026/pairs/HS__*/daily.parquet` (column `r`, one
  folder per portfolio): sample lag-1 autocorrelation of test-sample daily
  returns per portfolio; mean, minimum and maximum over portfolios; weekday
  counts of the ex-US test calendar.

Outputs: `artifacts/r8_calendar/{calendar.csv, stress_dates.csv, external_acf.csv,
run.json, RESULTS.md}`; displays via `displays.py` (`numbers_calendar.tex`) with `--check`.
