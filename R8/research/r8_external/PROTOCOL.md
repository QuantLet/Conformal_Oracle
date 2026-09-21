# External industry-portfolio test — protocol before outcome inspection

Declared 9 September 2026, after development on the 24-asset R8 panel.
This is an external-universe retrospective test, not a prospective live
forecasting trial: US industry portfolios share market shocks with the
development panel. No external forecast outcomes have been computed.

## Source, endpoint and admission

Use all 12 value-weighted daily industry portfolios, including dividends,
from Kenneth French's current CRSP-based Data Library. Do not substitute
equal weights, industry ETFs, the ex-dividend file or a selected subset.

- Details: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html
- Data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip

The official details page advertises daily data through 31 July 2026.
The required endpoint is now 31 July 2026 for every portfolio, following
the author-authorised amendment below. Fail closed if the actual data do
not reach this date. Do not shorten the endpoint further. Store the original response,
retrieval time, source URL, SHA-256 and format notes; CRSP revisions can change
historical returns. The source's current CIZ convention is part of the input
definition. No assertion of vintage-real-time availability is made.

After admission, parse only the value-weighted daily table. Convert percent
simple returns R to log1p(R/100); treat the documented -99.99/-999 sentinels
as missing, never as crashes. Fail on duplicate dates, nonfinite values,
returns at or below -100%, a missing portfolio, or an incomplete common
2000–July 2026 calendar. Diagnose before changing any exclusion rule.
Use pre-2000 history solely for initial contexts. Retain every valid large
return. Archive raw and transformed data, including any source revision.

## Forecast and correction policies fixed before external results

The four base forecasters are HS, GJR-GARCH-t, CAViaR-AS and GAS-t. Use the
existing numerical fitting definitions in `source/scripts/extension_20260831`
and `source/analysis/phase3_dynamic/run_dynamic_var.py`: HS/GJR use the
preceding 250 observations; CAViaR/GAS refit at the start of each calendar
year on the preceding 1,250 observations, then update their states from
observed returns. Produce past-only forecasts from January 2000 onward.
Record every context, parameter, attempt and fallback. No TSFM inference.
The annual dynamic refit is an explicit external-design change from R8's
fixed 70% fit; it must pass prefix-invariance and independent recursion
replay before evaluation. If dynamic fitting fails, stop that producer for
diagnosis instead of dropping the model or selecting a new optimiser from
its test performance.

Use forecast–return pairs from 2000–2014 for correction development and
calibration, with the last 30% as chronological inner validation. Freeze
correction hyperparameters and policy choices before evaluating 2015–31
August 2026. Static corrections refit on full calibration after tuning,
except gate candidates, which retain their inner-fit coefficients. Base
model yearly refits and rolling corrections continue using past data.

Carry the completed comparator definitions unchanged into this test:
Raw, Shift-CP, Vol-ERM, State-L1, POT-Shift, POT-Vol, projected DtACI,
rolling 500, past-selected rolling, coverage-gated selected rolling,
loss gate and past-loss minimum. Keep separate clipped-L1 and seeded-DtACI
sensitivities. POT selects 90%/95% on inner validation; no adjustment of
thresholds, penalty grids or gate confidence levels after external results.
The entire 48-pair universe is required, with common test support.

## Evaluation and decision criteria

Primary endpoint: mean per-pair pinball loss divided by that pair's
2000–2014 calibration-return standard deviation. Also report return-unit
QS, violation rates, Kupiec/conditional diagnostics, threshold magnitudes,
fitting diagnostics, deterioration counts and every pair's results. No
mean-threshold substitution for DtACI mixture loss and no ordinary Kupiec
count for fractional expected hits.

Use 999 common-calendar circular bootstrap draws at 20 and 60 calendar
days, with deterministic seed 20260909. Report simultaneous two-sided 95%
bands separately for (i) State-L1, POT-Shift, POT-Vol and projected DtACI
against Shift-CP, and (ii) the two gates and past-loss minimum against
past-selected rolling. The claimed transfer of a loss advantage requires
the primary normalised difference to remain negative with its simultaneous
upper endpoint below zero at both block lengths. Report results that fail
this criterion as unresolved, not as confirmation. Loss success does not
waive poor coverage. Intervals condition on fitted paths and choices.

Report 2015–2019, 2020–2021 and 2022–July 2026 descriptively, without
choosing the headline period by its ranking. These historical periods are
not a controlled structural-break experiment. A separately specified
regime-change simulation remains a distinct planned study.

## State and reproducibility

The July endpoint is available in source metadata. Actual data admission,
the external model adapter, fresh fitting and full numerical replay must
pass separately. This document does not claim a completed external
pipeline or validation. Hash this protocol and the reused numerical
producers before ingesting external return values. Changes needed for a
documented defect must preserve the previous version and be reported
before aggregate comparisons are inspected.

## Author-authorised endpoint amendment — 10 September 2026

Before downloading external return values or calculating external forecast
outcomes, the author explicitly approved using the available data through
31 July 2026: “Da, testul extern până în iulie”. The reason is publication
lag at the source, not an observed loss ranking. The original August
protocol is preserved at `artifacts/r8_external/july2026/protocol_original_august.md`.
All twelve portfolios, four models, calibration/test split, correction
selection rules, bootstrap settings and decision criteria are unchanged.
The market-panel endpoint remains August 2026. This amendment and its
timing must be stated when reporting the external test.

Implementation clarifications fixed before external outcomes: yearly dynamic
models are fitted to the preceding 1,250 returns and filtered through that
context to initialise the first forecast of the new year. The GAS density
uses Student-t scale, so its quantile is scale times the Student-t quantile,
consistent with the corrected existing `extension_20260831/dynamic.py`.
There is no extra variance standardisation. Retain optimiser attempts and
flags under the existing selection rule; abort on absent, invalid or
nonfinite parameters/forecasts. Convergence flags alone do not redefine
the existing minimum-objective selection rule.
