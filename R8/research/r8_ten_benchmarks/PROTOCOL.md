# CAViaR, GAS and dedicated tail benchmarks on the ten-model support — protocol

Fixed 15 September 2026, before computation. Descriptive: the five benchmarks
already appear on the earlier reference-panel support (`supp-tab:forecasters`),
and the ten-model panel's own results have been inspected. This study evaluates
the same stored forecasts on the ten-model panel's calibration and test dates,
so that the two forecaster tables share one support. No forecast is refitted.

Inputs (read-only; every file is bound by SHA-256 in `run.json`):
- `artifacts/r8_commodity_etp/panel/base/data/returns/{asset}.csv`, 24 series,
  whose digests must equal `input_sha256` in
  `artifacts/r8_model_extension/full_preflight/support.csv`.
- `artifacts/r8_commodity_etp/panel/base/data/dynamic/{asset}_{CAViaR-SAV,CAViaR-AS,GAS-t}.parquet`
  (column `VaR_0.01`, one row per return date; fitted on the first 70% of each
  full return series by `source/scripts/extension_20260831/dynamic.py` for the
  21 shared series and by its root-retargeted copy
  `research/r8_commodity_etp/dynamic.py` for GLD, USO and UNG).
- `artifacts/r8_commodity_etp/panel/base/evt_fhs/{asset}.parquet` (columns
  `FHS`, `EVT_POT`; one row per date from the 70% point of the full series,
  produced by `source/scripts/extension_20260831/evt_fhs.py`).
- `artifacts/r8_model_extension/ten_common_evaluation/daily/{asset}.parquet`
  (test-date index of the ten-model panel, used only as a cross-check).

Procedure, per asset.
1. Eligible dates are the return dates after the first 512 observations;
   `n_cal = floor(0.70 * eligible)`; test dates are the remainder. These must
   equal `support.csv` (`n_cal`, `n_test`, `first_test`, `last_date`) and the
   index of the ten-model daily file.
2. CAViaR-SAV, CAViaR-AS, GAS-t: read the 1% quantile on the eligible dates;
   residual `s_t = q_t - y_t`; static shift `qshift(s[:n_cal])`
   (`panel_statistics.qshift`); rolling shift from the preceding 250 residuals
   (order statistic 249 of 250, as in `evaluate_ten.py`). Score Raw, Static and
   Rolling250 on the test dates with `panel_statistics.scores`; score the
   calibration window for the indication flag.
3. EVT-POT and FHS: read the stored daily forecasts on the test dates; score
   Raw only. Record how many stored forecasts precede the first test date
   (fewer than 250, so neither correction is defined under the panel's rules).
4. Record, per asset, the last date of the CAViaR/GAS fitting sample
   (`int(0.7 * len(returns))`), the number of ten-model calibration dates that
   lie inside that sample, and confirm every test date lies after it.
5. Aggregate as `evaluate_ten.py`: asset-equal means of QS, violation rate and
   width; counts of Kupiec, independence and conditional rejections at 5%;
   scaled traffic-light zones.

Outputs: `artifacts/r8_ten_benchmarks/{metrics.csv, summary.csv, overlap.csv,
run.json, RESULTS.md}`; displays via `displays.py`
(`numbers_ten_benchmarks.tex`, `tab_ten_benchmarks.tex`) with `--check`.
