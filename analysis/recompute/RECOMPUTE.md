# Corrected TSFM VaR series

Sign error corrected from stored Student-$t$ parameters; no re-inference. Readings fixed in advance in `PREREGISTRATION.md`.

| Model | series | mean π̂ | median VaR₀.₀₁ | monotone | Kupiec pass | Green |
|---|---|---|---|---|---|---|
| Moirai-2.0 | corrected | **0.0166** | -0.02154 | 100.0% | 4/24 | 12/24 |
| Moirai-2.0 | stored | **0.9880** | +0.02154 | 0.0% | 0/24 | 0/24 |
| TimesFM-2.5 | corrected | **0.0141** | -0.02350 | 100.0% | 8/24 | 16/24 |
| TimesFM-2.5 | stored | **0.9900** | +0.02350 | 0.0% | 0/24 | 0/24 |

For comparison, unaffected models at α = 0.01 (from the audit): Chronos-Mini 0.4188, Chronos-Small 0.3884, Lag-Llama 0.0294, Moirai-1.1 0.0154, Hist-Sim 0.0158, GARCH-N 0.0193, EWMA 0.0208, GJR-GARCH 0.0042.

