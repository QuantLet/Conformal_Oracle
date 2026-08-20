# Promotion gate

Series tree: `/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/2026 CFP LLM VaR/cfp_ijf_data`. A failing series blocks promotion and gets a written diagnosis; no tolerance is widened to accommodate a series.

| Check | Condition |
|---|---|
| `sign` | median VaR_a < 0 at every alpha |
| `monotonicity` | VaR_.01 < VaR_.025 < VaR_.05 < VaR_.10 on >= 99.9% of days |
| `scale` | VaR_0.01 / realised sigma in [-3.5, -1.8] |
| `alpha_response` | pihat(0.10)/pihat(0.01) >= 3 |
| `coverage` | pihat in [0.2a, 5a] at every alpha |
| `alignment` | forecast for t uses data through t-1 only |
| `dispersion` | predictive std / realised sigma in [0.5, 2.0] |
| `cardinality` | distinct VaR_0.01 values > 5% of observations |
| `tail_reach` | alpha-quantile strictly above the support minimum, with >= 5 distinct sampled values below it (needs sample paths; n/a for series that stored quantiles only) |
| `extremes` | max |VaR_0.01| <= 50x the asset's own median |VaR_0.01| |

| Model | n | `sign` | `monotonicity` | `scale` | `alpha_response` | `coverage` | `alignment` | `dispersion` | `cardinality` | `tail_reach` | `extremes` | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Small | 24 | 24/24 | **0/24** | **0/24** | **0/24** | **0/24** | 24/24 | **0/24** | 24/24 | n/a | **19/24** | **BLOCK** |
| Chronos-Mini | 24 | 24/24 | **0/24** | **0/24** | **0/24** | **0/24** | 24/24 | **0/24** | 24/24 | n/a | **15/24** | **BLOCK** |
| TimesFM-2.5 | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| Moirai-2.0 | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| Moirai-1.1 | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| Lag-Llama | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| GJR-GARCH | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| GJR-GARCH-t | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| GARCH-N | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| Hist-Sim | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | n/a | 24/24 | PASS |
| EWMA | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | 24/24 | PASS |
| Chronos-Small-A | 24 | 24/24 | 24/24 | 24/24 | 24/24 | **23/24** | 24/24 | 24/24 | 24/24 | n/a | 24/24 | **BLOCK** |
| Chronos-Mini-A | 24 | 24/24 | 24/24 | 24/24 | **23/24** | **23/24** | 24/24 | 24/24 | 24/24 | n/a | 24/24 | **BLOCK** |

**4 of 13 series block.**

