# Promotion gate

Series tree: `/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/2026 CFP LLM VaR/analysis/gjr_quantile/_stage`. A failing series blocks promotion and gets a written diagnosis; no tolerance is widened to accommodate a series.

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

| Model | n | `sign` | `monotonicity` | `scale` | `alpha_response` | `coverage` | `alignment` | `dispersion` | `cardinality` | `tail_reach` | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-Small | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| Chronos-Mini | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| TimesFM-2.5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| Moirai-2.0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| Moirai-1.1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| Lag-Llama | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| GJR-GARCH | 24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | n/a | PASS |
| GARCH-N | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| Hist-Sim | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |
| EWMA | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASS |

**0 of 10 series block.**

