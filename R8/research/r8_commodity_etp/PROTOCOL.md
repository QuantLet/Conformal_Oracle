# Observable commodity exposures — 10 September 2026

The author approved the proposed USO replacement and a consistent audit of
the other commodity futures after the negative-WTI data issue. This is a
retrospective instrument-design repair, not a prospectively chosen universe.
Fix the choices below before forecasting or comparing new Quantile Scores.

## Selection and target

Replace CL=F/WTI with USO, GC=F/GOLD with GLD and NG=F/NATGAS with UNG if
the existing continuous series do not identify same-contract holding returns
and the replacement histories pass the data checks. These named US-listed,
unlevered commodity exposures are selected for identifiable traded shares,
the same three underlying commodity themes and histories covering 2020.
Do not search alternative funds based on forecast losses or significance.
Use the explicit fund tickers everywhere; they are different exposures from
individual futures. The issuer's roll policy, costs, collateral and tracking
effects enter the observed share return. GLD holds gold rather than a CL-like
futures position. No claim is made that USO owns the May-2020 expiry loss.

Keep the other 21 assets and their inputs unchanged, including the explicitly
identified DJCI index. Its index-return interpretation must remain explicit;
it is not being called a self-financing futures account.

Use daily log differences of adjusted share closing prices, from each
fund's first available trading date through 31 August 2026. No backfill to
2000, price shift, interpolation, clipping, stress-day removal or silent
endpoint reduction. Correct vendor corporate-action treatment must be
verified, including USO's April-2020 and UNG's reverse splits. A change in
sample inception is part of this replacement and must be disclosed.

## Sources and admission

The EODHD authenticated standard catalog did not supply individual CL
settlements. Use the existing project's public Yahoo chart interface for
the named ETPs, preserving the complete raw JSON and retrieval hashes;
verify identity and corporate actions against issuer documentation.
An independent vendor daily-price download, when available, is a cross-check
and never silently spliced into the primary series. A discrepancy remains
open until its adjustment/date basis is explained.

Require positive finite prices, unique increasing dates, exchange-local date
conversion, the exact August endpoint, the expected trading calendar and
coverage across the 2020 stress episode. Save all corporate actions, price
adjustment factors, gaps and large moves. A large genuine move is retained.
Audit the three old raw futures responses for contract identification and
roll metadata; missing identification is not evidence that every observed
jump is a roll artifact. Admission must precede all new model scoring.

## Forecast and evaluation design

Reuse the existing pinned checkpoints and configurations. Recompute five
classical models, Moirai-1.1 and the corrected actual-calendar Lag-Llama on
the new histories. Preserve 1,000 native predictive draws; no added tail
completion. The already evaluated native Chronos-2 and PatchTST-FM receive
the same three new histories in their separate comparison. No additional
model selection or tuning is introduced.

Keep alpha levels 0.01/0.025/0.05/0.10 for the seven-model core; native-grid
extensions retain their established 0.01 comparison. Keep the existing
past-only contexts, chronological 70/30 calibration/test split, conformal
order statistic and trailing-window definitions. Recompute the entire
affected correction/baseline/policy chain; never relabel old WTI forecasts
as USO. Use unchanged inputs and configurations for unaffected assets.

Archive originals and bind new forecasts, fits, native outputs and resulting
metrics to the new return files. Require independent return/result checks,
fresh-process replay and regenerated tables before replacing canonical
paper outputs. Changing exposure does not extend the separated coverage
theorem to contiguous or rolling recalibration.

Primary product references:
- https://www.uscfinvestments.com/uso
- https://www.uscfinvestments.com/ung
- https://www.spdrgoldshares.com/usa/
