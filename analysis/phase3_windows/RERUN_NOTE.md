# AE point 7 — the window sweep did not need re-running, and here is the test

`REFEREE_RESPONSE.md` marked this analysis **MUST BE RE-RUN for GJR-GARCH**, on
the grounds that "the window sweep was run against the old quantile map, so its
GJR rows describe a model that used a raw t5 multiplier with hard-coded df".

That was a precaution written without checking the script, and it is wrong.

## Why the sweep was never exposed to the defect

1. `run_window_sensitivity.py` does not consume the stored GJR-GARCH series. It
   re-estimates the model itself, per rolling window, in `fit_garch_var`.
2. Its quantile map is `sigma * stats.norm.ppf(alpha)` — the Gaussian innovation
   the manuscript describes. The defect was `t.ppf(alpha, 5)` on an
   unstandardised distribution, and it lived in the series-generation pipeline,
   not here.
3. The signature confirms it. The stored sweep reports GJR-GARCH on SP500 at
   w = 250 with pihat_raw = 0.0216. The defective series ran at 0.0042; the
   corrected panel runs at 0.0200. The sweep was always in the corrected regime.
4. Its inputs are unchanged: the `returns/` CSVs are untouched since April and
   clean in git, and the four helpers it imports from `run_ae_point4`
   (`SYMBOLS`, `kupiec_p`, `qhat_ceil`, `quantile_score`, `traffic_light`) are
   untouched by the 17 August edit, which changed only the model dictionary and
   one line in `regression()`.

## The test

`verify_window_sp500.py` re-runs the full sweep for SP500 — four models, three
window lengths, ~18k GARCH refits — and diffs against the stored CSV.

| quantity | max abs difference |
|---|---|
| pihat_raw, pihat_cp | 9.4e-17, 6.4e-17 |
| p_kupiec raw, corrected | 9.0e-17, 5.6e-17 |
| qV | 1.2e-10 |
| QS_raw, QS_cp | 5.7e-9 |

Basel zones identical on every row, raw and corrected. Violation rates and
Kupiec p-values reproduce to floating point. The residual on qV and QS is
optimiser noise in `arch`'s GARCH fits across library versions — it is 1.4e-5 in
relative terms and cannot move a reported figure.

**Verdict: the stored `window_sensitivity.csv` and `MEMO.md` stand as computed.
No refit of the remaining 23 assets is required.** Full log in
`verify_window_sp500.log`.
