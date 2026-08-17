# Pre-registration — corrected TSFM VaR recompute

Written and committed **before** the corrected numbers were computed, so the
reading of each outcome cannot be chosen after seeing it. The project's subject
is calibration failure; steering a recompute toward the prior conclusion would
be the same class of error as the one being corrected.

## The defect

`pipeline/CFP_Moirai_Forecasts.ipynb` and `CFP_TimesFM_Forecasts.ipynb` write

    row[f"VaR_{alpha}"] = -student_t.ppf(alpha, df_fit, loc=mu_fit, scale=sigma_fit)

The negation is wrong for a left-tail quantile. Confirmed by three independent
tests: stored VaR₀.₀₁ positive on 100% of observations against 1.6% elsewhere;
monotonicity across α reversed on 100% of days against 0% elsewhere; and the
empirical CDF of returns at the stored threshold reproducing the published
98.8% / 99.0% to four decimals with nothing unexplained.

Affected: TimesFM 2.5 and Moirai 2.0 only — 48 of 240 model–asset pairs at
α = 0.01, and the corresponding cells at 0.025, 0.05, 0.10.

## The correction

`VaR_α = student_t.ppf(α, df_student, loc=mean, scale=std)`, recomputed from the
fitted parameters already stored in each parquet. No re-inference is required
for this step: `mean`, `std` and `df_student` are present and finite for 100% of
observations in both models.

## What each outcome means — decided now

**Outcome A — both grid models come back near-calibrated (π̂ within, say, a
factor of ~3 of nominal).** The 99% rates were entirely the sign error. Panel B
reduces to Chronos-Small and Chronos-Mini, both token-categorical, both
sample-based, both correctly signed. *The interface thesis inverts*: the two
worst-calibrated forecasters would be sample-based and the two grid models
adequate. The paper's lead claim after Phase 5 must be withdrawn, not softened,
and the honest headline becomes the Chronos failure, which is real and unexplained.

**Outcome B — the grid models remain badly miscalibrated after correction, and
worse than the sample-based models.** The interface thesis survives the bug, but
every number supporting it changes and the within-family Moirai contrast must be
recomputed before it can be quoted. It would then need the mechanism test below
to distinguish "the grid cannot reach 1%" from "the closure fit is poor".

**Outcome C — mixed: one grid model recovers, the other does not.** No interface
claim is supportable from two models split one-one. The paper reports the audit
without the mechanism, and the interface question moves to future work.

In all three outcomes the Chronos result (38% and 42% raw violations, correctly
signed, monotone on 88% of days) stands and requires its own explanation. It is
currently unexplained by anything in the manuscript.

## The second defect, not fixed by the recompute

Both notebooks write **every** level, including α = 0.10, from the Student-t fit,
discarding the model's native quantile grid (`quantiles_pred`) immediately after
fitting. The manuscript's stated mechanism — a nine-decile grid has no 1%
quantile, so external completion supplies it — is therefore not what the code
does: completion supplies all four levels.

The test that separates "the fit is bad" from "the grid cannot reach" requires
comparing, at α = 0.10, the model's native grid point against the Student-t fit
at the same level. **The native grids were never stored**, so this requires
re-running inference for both models and is not possible from disk. Until it is
run, the interface claim is untested — it may be true; nobody has looked.

## Commitments

- No table, figure or claim is regenerated until the corrected series exist.
- The corrected and superseded series are both retained.
- If Outcome A obtains, the abstract and Section 4.2.1 written during Phase 5
  are withdrawn in full rather than edited.
