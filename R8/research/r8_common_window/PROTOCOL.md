# Static-minus-raw contrast on the common test window — protocol

Fixed 14 September 2026, before computation. Descriptive and exploratory: the
240-pair panel and its published contrast have been inspected. The per-asset
70/30 split makes the first test date range from April 2019 to June 2024. This
study restricts every pair to the calendar window on which all 24 assets are
under test, so that the crash of 2020 lies in every calibration sample and no
pair is absent on any evaluated date.

Inputs (read-only): `artifacts/r8_model_extension/ten_common_evaluation/boundaries.csv`
(column `common_test_first`), `artifacts/r8_ten_comparators/pairs/*/daily.parquet`
(columns `r`, `Raw`, `Shift-CP`, `Rolling250`).

Procedure.
1. Common start = the latest `common_test_first` over the 24 assets; it must be
   the same for all ten models of an asset.
2. Restrict each pair's daily losses to dates on or after the common start.
3. Contrast: pair-equal mean of the pair mean of $d_{it}=\ell_{it}(\text{Shift-CP})-\ell_{it}(\text{Raw})$,
   in loss units times $10^4$; also mean raw and static QS, violation rates and
   the number of pairs improved, on the same window.
4. Uncertainty: the common-calendar circular block bootstrap of
   `research/r8_power_analysis/analysis.py` reimplemented on the restricted
   calendar (999 draws, blocks of 20 and 60 calendar days, seed convention
   `sha256('20260909/panel-calendar/{block}')[:4]`); percentile 95% interval and
   the family-of-one studentised 95% band (max-|z| rule) for the contrast.
5. Table of test-start dates, calibration and test lengths per asset from
   `boundaries.csv`.

Outputs: `artifacts/r8_common_window/{contrast.csv, pairs.csv, split_dates.csv,
run.json, RESULTS.md}`; displays via `displays.py` (`numbers_common.tex`,
`tab_common.tex`, `tab_split_dates.tex`) with `--check`.
