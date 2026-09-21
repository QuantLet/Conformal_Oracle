# Expected Shortfall and the FZ0 joint score on the ten-model support — protocol

Fixed 15 September 2026, before computation. Descriptive: the ten-model panel's
VaR results have been inspected; no ES forecast has been computed before. The
study scores VaR and ES jointly for the seven main-panel forecasters whose
stored output defines a predictive law below the 1% quantile. Chronos-2,
PatchTST-FM and TS-ICL store 99 fixed quantile levels 0.01–0.99 and have no
output below the 1% level, so they have no ES forecast and are excluded.

Inputs (read-only; every file is bound by SHA-256 in `run.json`):
- returns: `artifacts/r8_commodity_etp/panel/base/data/returns/{asset}.csv`,
  digests equal to `artifacts/r8_model_extension/full_preflight/support.csv`;
- VaR forecasts: `artifacts/r8_commodity_etp/panel/base/data/{moirai,lagllama,benchmarks}/…`
  (columns `mean`, `std`, `VaR_0.01`);
- native draws: `artifacts/extension_20260831/native/{moirai,lagllama}/{asset}/*.npz`
  (21 series) and `artifacts/r8_commodity_etp/native/{moirai,lagllama}/{asset}/*.npz`
  (GLD, USO, UNG), 1,000 draws per date, chunk digests checked against each
  folder's `complete.json`;
- GJR-GARCH-t degrees of freedom: `.../parameters/gjr_t/{asset}.parquet`
  (column `nu_used`) from the same two roots;
- ten-model results for cross-checks: `artifacts/r8_model_extension/ten_common_evaluation/{calibration,metrics}.csv`.

ES forecasts at α = 0.01, one per date, in log-return units (negative for a loss):
1. Moirai-1.1, Lag-Llama: the mean of the ⌊1000α⌋ = 10 smallest draws
   (Acerbi and Tasche 2002, Proposition 4.1). Guard: `np.percentile(draws, 1)`
   equals the stored `VaR_0.01`.
2. GARCH-N, GJR-GARCH, EWMA: Normal law with the stored `mean` and `std`,
   ES = mean − std·φ(z)/α, z = Φ⁻¹(α). Guard: mean + std·z equals the stored VaR.
3. GJR-GARCH-t: unit-variance Student-t with the stored `nu_used` (the arch
   convention, quantile mean + std·√((ν−2)/ν)·t_ν⁻¹(α)); ES = mean −
   std·√((ν−2)/ν)·f_ν(t_α)(ν + t_α²)/((ν−1)α), with the Normal formula on dates
   whose `nu_used` is undefined. Guard: the recomputed quantile equals the stored VaR.
4. Hist-Sim: the mean of the ⌊250α⌋ = 2 smallest returns of the 250-day window
   that produced the stored VaR. Guard: `np.percentile(window, 1)` equals the stored VaR.

Support and correction, per asset, as in the ten-model panel: eligible dates
after 512 observations, `n_cal = floor(0.70 · eligible)`, test dates the rest.
Static shift `qV = qshift(VaR − y)` on the calibration dates (must equal
`calibration.csv`); static forecasts are VaR − qV and ES − qV (one location
shift of the predictive law). Raw and static QS on the test dates must equal
`metrics.csv`.

Score: FZ0 of Patton, Ziegel and Chen (2019, eq. 6),
L(y, v, e) = −1{y ≤ v}(v − y)/(αe) + v/e + log(−e) − 1, defined for e < 0.
Test days with e ≥ 0 (raw or static) are counted and excluded from both
versions of that pair; days with v ≥ 0 are counted.

Aggregation: asset-equal means per model of FZ0 (raw, static), mean ES and
mean VaR (raw, static), violation rate, and the mean of y − e on violation
days. Uncertainty: pair-equal static-minus-raw FZ0 contrast for each model
and for all 168 pairs, with the common-calendar circular block bootstrap
(999 draws, blocks of 20 and 60 calendar days, seed
`sha256('20260915/es-fz0/{block}')[:4]`), percentile 95% intervals and a
studentised simultaneous 95% band over the eight comparisons.

Outputs: `artifacts/r8_es_fz0/{metrics.csv, summary.csv, contrasts.csv,
undefined.csv, run.json, RESULTS.md}`; displays via `displays.py`
(`numbers_es.tex`, `tab_es.tex`) with `--check`.
