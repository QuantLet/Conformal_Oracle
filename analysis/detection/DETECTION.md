# Detection: does q̂_V flag what the backtests miss?

Rules and branch readings fixed in advance in `PREREGISTRATION.md`. Labelled forecasters are scored on their **superseded** series — the counterfactual is whether a detector would have fired when the defect was live.

Labelled (defective): **5** forecasters. Unlabelled: **8** — *not known to be defective*, not established as clean.

## Forecaster-level verdicts

| forecaster | defect | family | R̄ | q̂_V<0 | Kupiec fail | CC fail | CC undef | q̂_V magnitude, R̄ > 1 | q̂_V sign, negative on a majority of assets | Kupiec, p < 0.05 on a majority of assets | Christoffersen, p < 0.05 on a majority of assets |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GJR-GARCH | gjr_quantile_map | classical | 0.155 | 23/24 | 17/24 | 2/24 | 22/24 | — | **FLAG** | **FLAG** | — |
| Chronos-Mini-A | none | foundation | 0.161 | 0/24 | 16/24 | 6/24 | 7/24 | — | — | **FLAG** | — |
| Chronos-Small-A | none | foundation | 0.145 | 0/24 | 16/24 | 5/24 | 7/24 | — | — | **FLAG** | — |
| EWMA | none | classical | 0.184 | 0/24 | 19/24 | 5/24 | 11/24 | — | — | **FLAG** | — |
| GARCH-N | none | classical | 0.171 | 0/24 | 16/24 | 6/24 | 8/24 | — | — | **FLAG** | — |
| GJR-GARCH-t | none | classical | 0.101 | 1/24 | 11/24 | 1/24 | 12/24 | — | — | — | — |
| Hist-Sim | none | classical | 0.109 | 0/24 | 13/24 | 11/24 | 7/24 | — | — | **FLAG** | — |
| Lag-Llama | none | foundation | 0.357 | 0/24 | 24/24 | 0/24 | 14/24 | — | — | **FLAG** | — |
| Moirai-1.1 | none | foundation | 0.106 | 1/24 | 11/24 | 0/24 | 16/24 | — | — | — | — |
| Moirai-2.0 | sign_flip | foundation | 3.177 | 0/24 | 24/24 | 2/24 | 13/24 | **FLAG** | — | **FLAG** | — |
| TimesFM-2.5 | sign_flip | foundation | 3.183 | 0/24 | 24/24 | 1/24 | 13/24 | **FLAG** | — | **FLAG** | — |
| Chronos-Mini | top_k_truncation | foundation | 23.539 | 0/24 | 24/24 | 5/24 | 0/24 | **FLAG** | — | **FLAG** | — |
| Chronos-Small | top_k_truncation | foundation | 17.260 | 0/24 | 24/24 | 3/24 | 0/24 | **FLAG** | — | **FLAG** | — |

## Sensitivity and specificity

Fisher exact, two-sided, on the 2x2 of flag against label. Specificity is an **upper bound**: the unlabelled set has not been audited to the depth the labelled set has.

| detector | sensitivity | specificity | Fisher p | foundation sens. | classical sens. |
|---|---|---|---|---|---|
| q̂_V magnitude, R̄ > 1 | 4/5 | 8/8 | 0.0070 | 4/4 | 0/1 |
| q̂_V sign, negative on a majority of assets | 1/5 | 8/8 | 0.3846 | 0/4 | 1/1 |
| Kupiec, p < 0.05 on a majority of assets | 5/5 | 2/8 | 0.4872 | 4/4 | 1/1 |
| Christoffersen, p < 0.05 on a majority of assets | 0/5 | 8/8 | 1.0000 | 0/4 | 0/1 |

## Flagged but unlabelled — to be traced, not counted

- **Chronos-Mini-A** (R̄ = 0.161, π̂ = 0.0178) flagged by: Kupiec, p < 0.05 on a majority of assets
- **Chronos-Small-A** (R̄ = 0.145, π̂ = 0.0175) flagged by: Kupiec, p < 0.05 on a majority of assets
- **EWMA** (R̄ = 0.184, π̂ = 0.0208) flagged by: Kupiec, p < 0.05 on a majority of assets
- **GARCH-N** (R̄ = 0.171, π̂ = 0.0193) flagged by: Kupiec, p < 0.05 on a majority of assets
- **Hist-Sim** (R̄ = 0.109, π̂ = 0.0158) flagged by: Kupiec, p < 0.05 on a majority of assets
- **Lag-Llama** (R̄ = 0.357, π̂ = 0.0294) flagged by: Kupiec, p < 0.05 on a majority of assets

Each requires a verdict. An untraceable flag is a false positive; a traceable one is the next defect.

