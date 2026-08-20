# main_R2 rebuild log

Decision of 20 August 2026 (`drafts/MAIN_R2_DECISION.md`, option A): rebuild the
manuscript as one paper around configuration failure and structural validation,
rather than correct the existing one sentence by sentence.

## Done

| part | what changed |
|---|---|
| title | ``What Backtests Cannot Detect: Structural Validation of Tail-Risk Pipelines'' |
| abstract | rewritten; the interface claim, the within-family contrast and the audit-statistic framing are gone |
| §1 Introduction | rewritten around the five traced defects, the blindness of the instruments, four contributions, and an explicit ``what this paper does not claim'' |
| §2 Related literature | the audit-statistic paragraph replaced; adds what the sampling-parameter survey does and does not establish |
| §3.3 Models | the two Chronos configurations named as the paper's controlled comparison |
| §4 (new) The Sampling Mechanism | default, design, dose--response table, analytic estimator, validation, cost, consequences |
| §5.1 Raw diagnostics | rewritten on the corrected panel; R withdrawn as an instrument, with the Spearman 0.9912 evidence and the uMCB share of 27% |
| §5.1.2 | ``the 99% violation rates'' rewritten as what they were: our sign defect, with the arithmetic that establishes it |
| §5.2 Static recalibration | 335/384 green, the R range restated, the Christoffersen degeneracy stated as 204 of 384 undefined |
| §5.3 When the correction helps | re-run numbers: 66/312 static, 174/312 rolling, the 94 Kupiec-passing pairs, the gate rule at 218/94 |
| §5.x Within-family contrast | replaced by an explicit withdrawal, with both reasons stated |
| §6 (new) What the diagnostics cannot detect | degeneracy, Kupiec's discrimination, recalibration as concealment, and the retrospective |
| §7 (new) Structural validation | the ten checks, their scoping, the 4-of-13 result, and the in-sample caveat |
| §8 Conclusion | rewritten |
| Appendix validation gate | tenth check (extremes) added; scoping restated by estimator class |
| Table 1 caption, traffic-light caption | 16 forecasters; heatmap regenerated on the corrected panel (13 x 24) |
| bibliography | `lei2018distribution` added |

Compiles at 81 pages with no undefined reference or citation.

## Removed pending regeneration

Three passages were deleted rather than updated, because every number in them
comes from a table computed on the defective series. They are recorded here so
they can be reinstated deliberately or dropped deliberately, not lost:

1. **The location--scale diagnostic** (quantile regression of corrected on raw
   VaR; slope inversion as a replacement signature, 51 of 52 pairs, 79%
   agreement, mean scale share 0.459). Its content was tied to the
   signal-preserving/replacement taxonomy, which is gone.
2. **The diagnostic regression** ($R^2 = 0.782$, partial $R^2$ 53.4% for
   $\qVstat$) --- `CO_diagnostic_regression`, last built 8 May.
3. **The cross-sectional correlations** ($\bar\rho = 0.96$ with volatility among
   high-$R$ TSFMs, $-0.786$ for GJR-GARCH) --- `CO_cross_sectional`, last built
   8 May. The GJR figure survives in §6 as evidence about the fifth defect,
   which is what it now is.

## Blocked on the table rebuild

`analysis/provenance/STALE_TABLES.md` lists 20 panel-dependent artefacts still
built on the defective series, and the missing producer for the QS and violation
sequences that four of them consume. Sections still carrying numbers from those
tables, and therefore not yet rewritten:

- §5.4--§5.6 (rolling recalibration, static versus rolling, multi-quantile and
  Diebold--Mariano)
- §5.7 Baseline comparison
- §5.8 Simulation and robustness
- §6 Discussion: the COVID response figure, the FRTB capital arithmetic
- Appendices C--I

The theory (§3.2, Appendix on proofs) is untouched by all of this and stands.
