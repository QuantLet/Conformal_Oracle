# Sign diagnostic for the TSFM VaR construction

All figures from stored forecast files; no grids or samples needed.

| Model | code path | Panel | median VaR₀.₀₁ | % positive | monotone correct | monotone reversed | π̂ |
|---|---|---|---|---|---|---|---|
| TimesFM-2.5 | `student_t_negated` | B | +0.02350 | 100.0% | 0.0% | **100.0%** | 0.9900 |
| Moirai-2.0 | `student_t_negated` | B | +0.02154 | 100.0% | 0.0% | **100.0%** | 0.9880 |
| Chronos-Mini | `percentile_raw` | B | -0.00115 | 10.0% | 88.3% | **0.0%** | 0.4167 |
| Chronos-Small | `percentile_raw` | B | -0.00191 | 2.9% | 88.6% | **0.0%** | 0.3831 |
| Lag-Llama | `percentile_raw` | A | -0.02069 | 0.0% | 100.0% | **0.0%** | 0.0297 |
| EWMA | `benchmark` | A | -0.02026 | 0.0% | 100.0% | **0.0%** | 0.0197 |
| GARCH-N | `benchmark` | A | -0.02092 | 0.0% | 100.0% | **0.0%** | 0.0186 |
| Hist-Sim | `benchmark` | A | -0.02608 | 0.0% | 99.9% | **0.0%** | 0.0152 |
| Moirai-1.1 | `percentile_raw` | A | -0.02465 | 0.0% | 100.0% | **0.0%** | 0.0151 |
| GJR-GARCH | `benchmark` | A | -0.02962 | 0.0% | 100.0% | **0.0%** | 0.0041 |

## 1. Sign

- `student_t_negated` path: stored VaR₀.₀₁ positive on **100.0%** of observations.
- every other path: positive on 1.6%.

## 2. Monotonicity

- `student_t_negated`: ordering across α is **reversed** on **100.0%** of days, correct on 0.0%.
- every other path: correct on 97.1%, reversed on 0.0%.

## 3. Predicted versus published violation rate

| Model | π̂ predicted by the stored threshold | published |
|---|---|---|
| Moirai-2.0 | **0.9880** | 0.988 | 
| TimesFM-2.5 | **0.9900** | 0.990 | 

## 4. Is Panel B collinear with the code path?

| Model | Panel | code path |
|---|---|---|
| EWMA | A | `benchmark` |
| GARCH-N | A | `benchmark` |
| GJR-GARCH | A | `benchmark` |
| Hist-Sim | A | `benchmark` |
| Lag-Llama | A | `percentile_raw` |
| Moirai-1.1 | A | `percentile_raw` |
| Chronos-Mini | B | `percentile_raw` |
| Chronos-Small | B | `percentile_raw` |
| Moirai-2.0 | B | `student_t_negated` |
| TimesFM-2.5 | B | `student_t_negated` |

Panel B = ['Chronos-Mini', 'Chronos-Small', 'Moirai-2.0', 'TimesFM-2.5']
Student-t path = ['Moirai-2.0', 'TimesFM-2.5']

**Not collinear**: Panel B contains ['Chronos-Mini', 'Chronos-Small'] which do not use the Student-t path. The partition is therefore not a pure code-path artefact, though ['Moirai-2.0', 'TimesFM-2.5'] are affected.

