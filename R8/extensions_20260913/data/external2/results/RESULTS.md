# Developed ex-US 25 size/value portfolios: external test results

Protocol: `research/r8_external2/PROTOCOL.md` (SHA-256 1c0d56b4d935a08169398ef68939f1d0676d8fa15d79d73f8ea6bcdf78998bda), fixed 13 September 2026 before download. Universe: 25 value-weighted Developed ex-US ME x BE-ME daily portfolios (USD, dividends included). Models: HS, GJR-GARCH-t, CAViaR-AS, GAS-t. Pairs: 100 (all completed).

## Source and admission

- Data file: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_25_Portfolios_ME_BE-ME_daily_CSV.zip
- Retrieved (UTC): 2026-09-13T07:13:21.616566+00:00
- SHA-256 of the archived zip: a5415042de0491bcfa9406dcdc01446b865a536402b26eb3cfb425d94340097d
- Zip member: Developed_ex_US_25_Portfolios_ME_BE-ME_Daily.csv; header: "This file was created using the 202607 Bloomberg database. It contains"
- Details page used: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_developed.html (SHA-256 1b25b822caf1534e7383a69e3c1cd4d90763f52128c98f9bcd83f27b973faaf2); advertised daily returns July 1, 1990 - July 31, 2026. The protocol's guessed details page https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_25_port_form_sz_bm_daily_dev_ex_us.html returned HTTP 404; the library index links the page above as "Details" for this file.
- Value-weighted table: "Average Value Weighted Returns -- Daily", file lines 19-9434; the equal-weighted table (['Average Equal Weighted Returns -- Daily']) is excluded.
- Sentinels: file header states "Missing data are indicated by -99.99."; -99.99 is mapped to missing; -999 (documented only in the CRSP industry file) is counted and never mapped. Counts in the value-weighted table: {'-99.99': 0, '-999.0': 0}. Missing cells in the admitted window: 0.
- Rows in the file: 9415, 1990-07-02 to 2026-07-31. The file begins 2 July 1990; the protocol text said November 1990. Pre-2000 history enters only as the 1,250-observation initial context (first context date 1995-03-20).
- Endpoint: last date in the file = 2026-07-31 (protocol minimum 2025-12-31). First forecast 2000-01-03; calibration 2000-2014 = 3913 dates; test 2015-01-01 to 2026-07-31 = 3022 dates.
- Calendar: file weekday calendar (Monday-Friday, complete between first and last admitted date). Every admitted date carries all 25 portfolios. Compared with the archived S&P 500 calendar over 2000-07/2026: 6935 dates here versus 6683; 252 weekday dates here are US holidays absent from the S&P calendar; 0 S&P dates are absent here. The first test date is 2015-01-01 (a weekday in this file).
- Transformation: log(1+R/100). Checks passed: unique increasing dates, finite values, all returns above -100%, 25 named columns, complete weekday calendar.

## Base forecasts and validation

- HS and GJR-GARCH-t: 250-observation daily fits (`source/scripts/extension_20260831/classical.py`, unchanged). Classical forecast rows reconstructed independently: 396750; fixed-parameter GJR reconstructions: 198375; window-sd fallbacks: 0.
- CAViaR-AS and GAS-t: refitted each January 2000-2026 on the preceding 1,250 returns. Yearly fits replayed: 1350; selected fits with a non-success optimiser flag: 0 (retained under the minimum-objective rule).
- Base series: 100; full fresh replays: 100; base validation status: passed.

## Corrections and validation

- Pairs: 100; fresh replays: 100; independent scalar checks: 21000; exact array values compared: 24966000; inner gate bootstrap draws recomputed: 99800; LP objectives: 1300; POT inversions: 600; future-outcome perturbation pairs: ['HS__SMALL_LoBM', 'GJR-GARCH-t__SMALL_LoBM', 'CAViaR-AS__SMALL_LoBM', 'GAS-t__SMALL_LoBM']; status: passed.
- Loss-gate selections over 100 pairs: {'Raw': 100}; past-loss minimum selections: {'Raw': 52, 'Inner-Vol': 32, 'Rolling500': 14, 'Inner-Shift': 2}; selected rolling windows: {500: 76, 250: 13, 125: 11}; coverage gate active in 88 pairs; POT final fallbacks: 0.

## Aggregate table (100 pairs, test 2015-01-01 to 2026-07-31)

QS in return units x 10^4; normalised QS = per-pair pinball loss divided by the pair's 2000-2014 calibration-return standard deviation, averaged over pairs; Viol. = mean violation rate in percent; UC = number of pairs with Kupiec p < 0.05 (-- when the mixture loss has no integer hit count); Worse = pairs with QS above Raw.

| Method | QS x1e4 | Normalised QS | Viol. (%) | UC | Worse |
|---|---|---|---|---|---|
| Raw | 3.2043 | 0.032995 | 1.526 | 74 | 0 |
| Shift-CP | 3.1611 | 0.032547 | 0.878 | 11 | 21 |
| Vol-ERM | 3.1416 | 0.032337 | 0.847 | 17 | 16 |
| Regularised state | 3.1657 | 0.032554 | 1.183 | 24 | 20 |
| POT-Shift | 3.1580 | 0.032511 | 0.909 | 7 | 17 |
| POT-Vol | 3.1367 | 0.032285 | 0.990 | 3 | 9 |
| Projected DtACI | 3.2483 | 0.033443 | 1.390 | -- | 78 |
| Rolling 500 | 3.2528 | 0.033497 | 1.145 | 6 | 84 |
| Selected rolling | 3.3017 | 0.034010 | 1.098 | 6 | 86 |
| Coverage-gated rolling | 3.2945 | 0.033939 | 1.116 | 9 | 75 |
| Loss gate | 3.2043 | 0.032995 | 1.526 | 74 | 0 |
| Past-loss minimum | 3.1825 | 0.032768 | 1.226 | 42 | 16 |
| Regularised state (clipped) | 3.1472 | 0.032388 | 1.161 | 23 | 16 |
| Projected DtACI (seeded path) | 3.2482 | 0.033447 | 1.394 | 45 | 74 |

The last two rows are the clipped-L1 and seeded-DtACI sensitivities. LaTeX version: `results/tab_external.tex`.

## Simultaneous 95% bands (999 calendar circular-bootstrap draws, seed prefix 20260913)

Normalised units; difference = method minus reference in mean normalised QS; simultaneous bands are sup-t over the family. Return-unit bands (x 10^4) are in `intervals.csv`.

| Family | Method | Reference | Block | Difference | Pointwise 95% | Simultaneous 95% |
|---|---|---|---|---|---|---|
| correction | Regularised state | Shift-CP | 20 | 0.000006 | [-0.000709, 0.000583] | [-0.000779, 0.000792] |
| correction | POT-Shift | Shift-CP | 20 | -0.000036 | [-0.000098, 0.000031] | [-0.000113, 0.000041] |
| correction | POT-Vol | Shift-CP | 20 | -0.000262 | [-0.000542, -0.000023] | [-0.000586, 0.000062] |
| correction | Projected DtACI | Shift-CP | 20 | 0.000896 | [-0.000093, 0.001702] | [-0.000183, 0.001974] |
| indication | Coverage-gated rolling | Selected rolling | 20 | -0.000071 | [-0.000138, 0.000004] | [-0.000155, 0.000012] |
| indication | Loss gate | Selected rolling | 20 | -0.001015 | [-0.002004, 0.000291] | [-0.002424, 0.000393] |
| indication | Past-loss minimum | Selected rolling | 20 | -0.001242 | [-0.001844, -0.000477] | [-0.002083, -0.000401] |
| reference | Shift-CP | Raw | 20 | -0.000448 | [-0.001541, 0.000494] | [-0.001466, 0.000570] |
| correction | Regularised state | Shift-CP | 60 | 0.000006 | [-0.000639, 0.000595] | [-0.000755, 0.000768] |
| correction | POT-Shift | Shift-CP | 60 | -0.000036 | [-0.000100, 0.000042] | [-0.000123, 0.000052] |
| correction | POT-Vol | Shift-CP | 60 | -0.000262 | [-0.000504, -0.000011] | [-0.000554, 0.000029] |
| correction | Projected DtACI | Shift-CP | 60 | 0.000896 | [0.000040, 0.001704] | [-0.000128, 0.001919] |
| indication | Coverage-gated rolling | Selected rolling | 60 | -0.000071 | [-0.000140, -0.000001] | [-0.000151, 0.000008] |
| indication | Loss gate | Selected rolling | 60 | -0.001015 | [-0.002088, 0.000435] | [-0.002517, 0.000486] |
| indication | Past-loss minimum | Selected rolling | 60 | -0.001242 | [-0.001907, -0.000420] | [-0.002146, -0.000338] |
| reference | Shift-CP | Raw | 60 | -0.000448 | [-0.001642, 0.000554] | [-0.001546, 0.000651] |

## Transfer decision

Criterion (protocol): a loss advantage transfers only if the normalised difference is negative and its simultaneous upper endpoint is below zero at both block lengths (20 and 60 calendar days).

- correction: Projected DtACI vs Shift-CP: difference 0.000896; simultaneous upper endpoints 0.001974 (20 d), 0.001919 (60 d): **unresolved**.
- correction: POT-Shift vs Shift-CP: difference -0.000036; simultaneous upper endpoints 0.000041 (20 d), 0.000052 (60 d): **unresolved**.
- correction: POT-Vol vs Shift-CP: difference -0.000262; simultaneous upper endpoints 0.000062 (20 d), 0.000029 (60 d): **unresolved**.
- correction: Regularised state vs Shift-CP: difference 0.000006; simultaneous upper endpoints 0.000792 (20 d), 0.000768 (60 d): **unresolved**.
- indication: Coverage-gated rolling vs Selected rolling: difference -0.000071; simultaneous upper endpoints 0.000012 (20 d), 0.000008 (60 d): **unresolved**.
- indication: Loss gate vs Selected rolling: difference -0.001015; simultaneous upper endpoints 0.000393 (20 d), 0.000486 (60 d): **unresolved**.
- indication: Past-loss minimum vs Selected rolling: difference -0.001242; simultaneous upper endpoints -0.000401 (20 d), -0.000338 (60 d): **TRANSFERS**.
- reference: Shift-CP vs Raw: difference -0.000448; simultaneous upper endpoints 0.000570 (20 d), 0.000651 (60 d): **unresolved**.

## By model

| Model | Method | Pairs | QS | Normalised QS | Violation rate |
|---|---|---|---|---|---|
| CAViaR-AS | Projected DtACI | 25 | 0.00030885 | 0.031716 | 0.01341 |
| CAViaR-AS | Projected DtACI (seeded path) | 25 | 0.00030960 | 0.031813 | 0.01340 |
| CAViaR-AS | Coverage-gated rolling | 25 | 0.00030385 | 0.031224 | 0.01092 |
| CAViaR-AS | Loss gate | 25 | 0.00029804 | 0.030609 | 0.01211 |
| CAViaR-AS | POT-Shift | 25 | 0.00029887 | 0.030682 | 0.00854 |
| CAViaR-AS | POT-Vol | 25 | 0.00029694 | 0.030477 | 0.00919 |
| CAViaR-AS | Past-loss minimum | 25 | 0.00030120 | 0.030927 | 0.01166 |
| CAViaR-AS | Raw | 25 | 0.00029804 | 0.030609 | 0.01211 |
| CAViaR-AS | Rolling 500 | 25 | 0.00030502 | 0.031330 | 0.01121 |
| CAViaR-AS | Selected rolling | 25 | 0.00030665 | 0.031501 | 0.01051 |
| CAViaR-AS | Shift-CP | 25 | 0.00029945 | 0.030743 | 0.00827 |
| CAViaR-AS | Regularised state | 25 | 0.00029963 | 0.030753 | 0.01153 |
| CAViaR-AS | Regularised state (clipped) | 25 | 0.00029834 | 0.030612 | 0.01124 |
| CAViaR-AS | Vol-ERM | 25 | 0.00029888 | 0.030670 | 0.00743 |
| GAS-t | Projected DtACI | 25 | 0.00032366 | 0.033352 | 0.01288 |
| GAS-t | Projected DtACI (seeded path) | 25 | 0.00032365 | 0.033356 | 0.01298 |
| GAS-t | Coverage-gated rolling | 25 | 0.00032562 | 0.033620 | 0.01062 |
| GAS-t | Loss gate | 25 | 0.00031429 | 0.032377 | 0.01465 |
| GAS-t | POT-Shift | 25 | 0.00031141 | 0.032078 | 0.00879 |
| GAS-t | POT-Vol | 25 | 0.00031026 | 0.031952 | 0.01007 |
| GAS-t | Past-loss minimum | 25 | 0.00031449 | 0.032397 | 0.01443 |
| GAS-t | Raw | 25 | 0.00031429 | 0.032377 | 0.01465 |
| GAS-t | Rolling 500 | 25 | 0.00031785 | 0.032758 | 0.01093 |
| GAS-t | Selected rolling | 25 | 0.00032570 | 0.033628 | 0.01028 |
| GAS-t | Shift-CP | 25 | 0.00031159 | 0.032107 | 0.00864 |
| GAS-t | Regularised state | 25 | 0.00031126 | 0.032064 | 0.01048 |
| GAS-t | Regularised state (clipped) | 25 | 0.00031103 | 0.032040 | 0.01038 |
| GAS-t | Vol-ERM | 25 | 0.00031086 | 0.032018 | 0.00883 |
| GJR-GARCH-t | Projected DtACI | 25 | 0.00030102 | 0.030977 | 0.01295 |
| GJR-GARCH-t | Projected DtACI (seeded path) | 25 | 0.00030054 | 0.030927 | 0.01289 |
| GJR-GARCH-t | Coverage-gated rolling | 25 | 0.00029746 | 0.030619 | 0.01046 |
| GJR-GARCH-t | Loss gate | 25 | 0.00029659 | 0.030531 | 0.01768 |
| GJR-GARCH-t | POT-Shift | 25 | 0.00028910 | 0.029751 | 0.00887 |
| GJR-GARCH-t | POT-Vol | 25 | 0.00028914 | 0.029743 | 0.00977 |
| GJR-GARCH-t | Past-loss minimum | 25 | 0.00029035 | 0.029866 | 0.00973 |
| GJR-GARCH-t | Raw | 25 | 0.00029659 | 0.030531 | 0.01768 |
| GJR-GARCH-t | Rolling 500 | 25 | 0.00029596 | 0.030455 | 0.01073 |
| GJR-GARCH-t | Selected rolling | 25 | 0.00029746 | 0.030619 | 0.01046 |
| GJR-GARCH-t | Shift-CP | 25 | 0.00028941 | 0.029781 | 0.00854 |
| GJR-GARCH-t | Regularised state | 25 | 0.00028960 | 0.029799 | 0.01024 |
| GJR-GARCH-t | Regularised state (clipped) | 25 | 0.00028955 | 0.029795 | 0.01022 |
| GJR-GARCH-t | Vol-ERM | 25 | 0.00028930 | 0.029759 | 0.00966 |
| HS | Projected DtACI | 25 | 0.00036579 | 0.037726 | 0.01636 |
| HS | Projected DtACI (seeded path) | 25 | 0.00036549 | 0.037691 | 0.01648 |
| HS | Coverage-gated rolling | 25 | 0.00039086 | 0.040293 | 0.01267 |
| HS | Loss gate | 25 | 0.00037279 | 0.038462 | 0.01661 |
| HS | POT-Shift | 25 | 0.00036381 | 0.037534 | 0.01018 |
| HS | POT-Vol | 25 | 0.00035835 | 0.036968 | 0.01058 |
| HS | Past-loss minimum | 25 | 0.00036696 | 0.037884 | 0.01324 |
| HS | Raw | 25 | 0.00037279 | 0.038462 | 0.01661 |
| HS | Rolling 500 | 25 | 0.00038231 | 0.039443 | 0.01291 |
| HS | Selected rolling | 25 | 0.00039086 | 0.040293 | 0.01267 |
| HS | Shift-CP | 25 | 0.00036400 | 0.037558 | 0.00965 |
| HS | Regularised state | 25 | 0.00036580 | 0.037598 | 0.01506 |
| HS | Regularised state (clipped) | 25 | 0.00035994 | 0.037106 | 0.01461 |
| HS | Vol-ERM | 25 | 0.00035760 | 0.036901 | 0.00797 |

## Descriptive periods (mean normalised QS over 100 pairs)

| Period | Method | QS | Normalised QS |
|---|---|---|---|
| 2015-2019 | Projected DtACI | 0.00026174 | 0.027036 |
| 2015-2019 | Projected DtACI (seeded path) | 0.00026118 | 0.026978 |
| 2015-2019 | Coverage-gated rolling | 0.00026686 | 0.027548 |
| 2015-2019 | Loss gate | 0.00025556 | 0.026374 |
| 2015-2019 | POT-Shift | 0.00025419 | 0.026227 |
| 2015-2019 | POT-Vol | 0.00025089 | 0.025883 |
| 2015-2019 | Past-loss minimum | 0.00025407 | 0.026210 |
| 2015-2019 | Raw | 0.00025556 | 0.026374 |
| 2015-2019 | Rolling 500 | 0.00026288 | 0.027133 |
| 2015-2019 | Selected rolling | 0.00026801 | 0.027657 |
| 2015-2019 | Shift-CP | 0.00025490 | 0.026301 |
| 2015-2019 | Regularised state | 0.00025564 | 0.026329 |
| 2015-2019 | Regularised state (clipped) | 0.00025282 | 0.026070 |
| 2015-2019 | Vol-ERM | 0.00025243 | 0.026048 |
| 2020-2021 | Projected DtACI | 0.00048721 | 0.049755 |
| 2020-2021 | Projected DtACI (seeded path) | 0.00048618 | 0.049654 |
| 2020-2021 | Coverage-gated rolling | 0.00049680 | 0.050886 |
| 2020-2021 | Loss gate | 0.00049780 | 0.050929 |
| 2020-2021 | POT-Shift | 0.00047352 | 0.048404 |
| 2020-2021 | POT-Vol | 0.00047102 | 0.048119 |
| 2020-2021 | Past-loss minimum | 0.00048593 | 0.049694 |
| 2020-2021 | Raw | 0.00049780 | 0.050929 |
| 2020-2021 | Rolling 500 | 0.00049348 | 0.050524 |
| 2020-2021 | Selected rolling | 0.00049640 | 0.050854 |
| 2020-2021 | Shift-CP | 0.00047198 | 0.048263 |
| 2020-2021 | Regularised state | 0.00047245 | 0.048160 |
| 2020-2021 | Regularised state (clipped) | 0.00046875 | 0.047848 |
| 2020-2021 | Vol-ERM | 0.00046353 | 0.047314 |
| 2022-July2026 | Projected DtACI | 0.00032261 | 0.033295 |
| 2022-July2026 | Projected DtACI (seeded path) | 0.00032365 | 0.033412 |
| 2022-July2026 | Coverage-gated rolling | 0.00032450 | 0.033495 |
| 2022-July2026 | Loss gate | 0.00031358 | 0.032371 |
| 2022-July2026 | POT-Shift | 0.00031400 | 0.032413 |
| 2022-July2026 | POT-Vol | 0.00031332 | 0.032341 |
| 2022-July2026 | Past-loss minimum | 0.00031490 | 0.032517 |
| 2022-July2026 | Raw | 0.00031358 | 0.032371 |
| 2022-July2026 | Rolling 500 | 0.00031977 | 0.032988 |
| 2022-July2026 | Selected rolling | 0.00032524 | 0.033571 |
| 2022-July2026 | Shift-CP | 0.00031470 | 0.032485 |
| 2022-July2026 | Regularised state | 0.00031484 | 0.032516 |
| 2022-July2026 | Regularised state (clipped) | 0.00031484 | 0.032516 |
| 2022-July2026 | Vol-ERM | 0.00031614 | 0.032645 |

## Aggregate validation

Status passed; daily losses rebuilt: 4230800; independent calendar draws: 1998; bootstrap means verified: 55944; period rows: 4200; interval rows: 32.

## Implementation notes and departures

- Scripts: `research/r8_external2/` copies of `research/r8_external/*.py`, changed only in root path, source URLs, the 25-column parser (whitespace-padded fields, one documented sentinel), the endpoint constant, the 2015 boundary assertion (first test date is 2015-01-01 here, 2015-01-02 in the industry file), the pair count, the calendar-bootstrap seed prefix (20260913) and the added third family (Shift-CP against Raw). Fitting, correction, selection and aggregation definitions are unchanged. No file outside `research/r8_external2/` and `artifacts/r8_external2/` was modified.
- The inner loss-gate bootstrap (499 draws) and the seeded DtACI paths use the seed construction inside `research/r8_decision/methods.py` (prefix 20260909 with the pair key), carried over unchanged as part of the correction definitions. The protocol seed 20260913 applies to the 999-draw calendar bootstrap.
- One parser check added relative to the industry parser: all 25 columns must parse as numeric and are cast to float. The archived file is all-float; the check changes no value.
- No foundation-model inference was run. The manuscript was not edited.

## Runtime and environment

- Python 3.13.9 at `/private/tmp/irfa-r8-conda-clean/bin/python`; macOS-26.6.2-arm64-arm-64bit-Mach-O; libraries: {'numpy': '2.3.5', 'pandas': '2.3.3', 'scipy': '1.16.3', 'numba': '0.62.1', 'arch': '8.0.0', 'pyarrow': '21.0.0', 'pytest': '8.4.2'}; 18 cores available, 12 workers.
- classical: 267 s (2026-09-13T07:17:01Z to 2026-09-13T07:21:28Z)
- dynamic: 20 s (2026-09-13T07:21:28Z to 2026-09-13T07:21:48Z)
- classical_replay: 263 s (2026-09-13T07:21:48Z to 2026-09-13T07:26:11Z)
- dynamic_replay: 20 s (2026-09-13T07:26:11Z to 2026-09-13T07:26:31Z)
- validate_base: 136 s (2026-09-13T07:26:31Z to 2026-09-13T07:28:47Z)
- corrections: 13 s (2026-09-13T07:28:59Z to 2026-09-13T07:29:12Z)
- corrections_replay: 14 s (2026-09-13T07:29:12Z to 2026-09-13T07:29:26Z)
- validate_corrections: 24 s (2026-09-13T07:29:26Z to 2026-09-13T07:29:50Z)
- aggregate: 2 s (2026-09-13T07:29:50Z to 2026-09-13T07:29:52Z)
- validate_aggregate: 2 s (2026-09-13T07:29:52Z to 2026-09-13T07:29:54Z)
- Total of staged wall-clock seconds: 761 s.
