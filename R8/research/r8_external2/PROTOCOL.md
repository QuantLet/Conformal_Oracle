# Second external universe — Developed ex-US 25 size/value portfolios

Fixed 13 September 2026 before download. Purpose: test transfer of the
correction and indication results to a universe that excludes the US
market of the development panel and the first external test.

## Source and admission

- Details page: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_25_port_form_sz_bm_daily_dev_ex_us.html (verify the exact page name from the library index; the data file below is authoritative)
- Data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_25_Portfolios_ME_BE-ME_daily_CSV.zip

Use all 25 value-weighted daily portfolios (USD, dividends included).
Store the raw response, retrieval time, SHA-256 and the documented
conventions. Convert percent simple returns to $\log(1+R/100)$; treat the
documented missing sentinels as missing. Fail on duplicate dates,
nonfinite values, returns at or below $-100\%$, a missing portfolio, or an
incomplete common calendar. International daily data begin in November
1990; use pre-2000 history only for initial contexts. The endpoint is the
last date in the file; record it. If the file ends before 31 December
2025, stop and report.

## Forecasts, corrections, evaluation

Identical to `research/r8_external/PROTOCOL.md` and its 10 September
amendment: HS and GJR-GARCH-t on the preceding 250 observations;
CAViaR-AS and GAS-t refitted each January on the preceding 1,250
returns; correction development on 2000–2014 with the last 30% as inner
validation; test from 2015 to the endpoint; the twelve comparator and
policy definitions carried over unchanged; primary endpoint the mean
per-pair pinball loss divided by the pair's 2000–2014 calibration-return
standard deviation; 999 common-calendar circular bootstrap draws at 20
and 60 calendar days, seed 20260913; the same two simultaneous families
(four correction contrasts against Shift-CP; three indication contrasts
against past-selected rolling) plus, new and prespecified here, a third
family of one contrast, Shift-CP against Raw. Transfer of a loss
advantage is claimed only if the normalised difference is negative with
simultaneous upper endpoint below zero at both block lengths. All 100
pairs are required.

## Implementation

Reuse `research/r8_external/*.py` by parameterising the root
(`artifacts/r8_external2/devexus`) and the parser for the 25-portfolio
file. Do not change fitting, correction or aggregation definitions; any
defect fix must preserve the previous version and be reported. Validate
with the existing validators (`validate_base.py`, `validate_corrections.py`,
`validate_aggregate.py`) adapted only in paths. Record runtime and
environment. No foundation-model inference.

## Outputs

`artifacts/r8_external2/devexus/results/`: per-pair metrics, the
aggregate table in the format of `source/sections_r8/tab_external.tex`,
`intervals.csv` with the three families, and `RESULTS.md` with every
number, failures and the transfer decision. No R8 file is modified.
