# Verification-claim inventory — main_R2.tex

Every sentence asserting a verification or a model property. Each gets a
primary-source citation, a script reference, or deletion. No third option.

Appendix J ("Forensic Validation of Quantile-Grid TSFM Failures") has been
**deleted**, not repaired, and replaced by the scripted validation gate
(Appendix `app:validation_gate`, `scripts/promotion_gate.py`).

## Why Appendix J was deleted rather than corrected

It asserted, as a table row:

> Quantile-level monotonicity: q(0.01) ≤ q(0.025) ≤ q(0.05) ≤ q(0.10) ≤ q(0.50)
> holds for all (asset, date) combinations.

For TimesFM 2.5 and Moirai 2.0 the ordering was **reversed on 100% of days** —
measured, `analysis/interface/sign_diagnostic.py`. The appendix asserted the
exact negation of the fact, on the one check that would have caught the sign
error.

Its other rows could not have caught it. The "central-quantile sanity check"
reported violation rates at the mean of 49.7% and 50.2% as confirming correct
interpretation — but `mean` is the one column the generation code did **not**
negate, so that check was structurally blind to the defect. "Return scaling …
verified by checking that predicted standard deviation matches realised
volatility" is the check that Chronos fails at 0.117×, and it was not applied to
Chronos.

Provenance stops at a squashed commit: the claims cannot be substantiated and
the author cannot be established. A repaired appendix would carry the same
unverifiable authority.

## Inventory

| # | Location | Claim | Verdict |
|---|---|---|---|
| 1 | Abstract, L95 | "Moirai … two versions share an architecture and a closely related pretraining design" | **DELETE.** Contradicted by the primary source: arXiv:2511.11698 states Moirai 2.0 "replaces masked-encoder training, multi-patch inputs, and mixture-distribution outputs with a simpler decoder-only architecture, single patch, and quantile loss", trained on a new 36M-series corpus. Architecture, corpus and output parameterisation all differ. |
| 2 | Intro, L139 | Same claim, contribution 1 | **DELETE**, same source. |
| 3 | §4.2, L577 | "Both models are masked-encoder TSFMs trained under closely related pretraining designs" | **DELETE.** Moirai 1.x is masked-encoder (arXiv:2402.02592); Moirai 2.0 is decoder-only. |
| 4 | §4.2.1, L408 | "Both models emit a nine-decile grid whose lowest point is the 10% quantile; the 1% quantile is … extrapolated from that grid" | **WITHDRAWN pending test.** The grid property is citable (arXiv:2511.11698: quantile forecasting). The *mechanism* claim is not: `CFP_Moirai_Forecasts.ipynb` fits the Student-t and writes **every** level from it, including 0.10, discarding the native grid. The paper describes a method the code does not run. Untested — the native grids were never stored. |
| 5 | §4.2.1, L408 and Conclusion, L690 | "A practitioner can see this before deployment, from the output format alone" | **DELETE.** Rests on #4. |
| 6 | §3.3, L378 | "TimesFM 2.5 and Moirai 2.0 return a fixed nine-decile grid" | **KEEP with citation.** Verifiable from arXiv:2511.11698 and the TimesFM model card; cite both at this sentence. |
| 7 | Appendix, L1786, L1799 | Tail-completion robustness; inner-7-decile refit | **KEEP, script-referenced** (`Quantlets/CO_robustness_inner7/`). But note the deciles are *reconstructed* from fitted parameters, not native — state that. |
| 8 | Conclusion, L692 | "where the Christoffersen test is defined it rejects on 62% of pairs" | **KEEP, script-referenced** (`analysis/cc_column/`). |
| 9 | Conclusion, L692 | "duration-based tests flag clustering on a further 47 pairs" | **KEEP, script-referenced** (`analysis/duration_tests/`). |
| 10 | Appendix, L1898 | Historical Simulation "takes few distinct values by construction" | **KEEP.** Arithmetic property of an order statistic on a 250-day window; state as a limitation, not a defect. |
| 11 | Appendix, L1819 | Gap-parameter ablation | **KEEP, script-referenced** (`scripts/gap_ablation.py`, verified against `gap_ablation.csv`). |
| 12 | §3.3 / §4 passim | Raw violation rates for TimesFM 2.5 and Moirai 2.0 (~99%) | **DELETE.** Sign defect; corrected series give 1.41% and 1.66%. |
| 13 | §4 passim | Chronos raw violation rates (38%, 42%) | **HOLD.** Traced to the checkpoint default `top_k = 50` truncating support to 50 of 4094 bins. Corrected estimate on SP500 is 1.75%, but that is one asset and it still fails Kupiec (p = 0.018). Not restatable until the 24-asset run completes. |
| 14 | §3.2 | Student-t tail closure described as applying below the grid | **CORRECT.** It is applied at every level. Restate to match the code, or change the code and re-run. |

## Standing requirement

Any sentence asserting what a model *is* — architecture, training corpus, output
format, sampling behaviour — carries a primary-source citation at drafting time.
Claim #1 was drafted, asserted an architectural fact, and cited a paper
describing a different model (Moirai-MoE, arXiv:2410.10469, under the key
`liu2024moirai2`). That is the same pattern as the fabricated bibliography
entries, escaped from the bibliography into a claim.
