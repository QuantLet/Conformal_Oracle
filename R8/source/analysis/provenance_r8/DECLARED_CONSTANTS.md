# R8 structural constants

Empirical results are generated macros or tables. The following are design choices, identifiers or exact rank arithmetic, checked against the producers.

| Constants | Role |
|---|---|
| 0.01, 0.025, 0.05, 0.10, 2.5 | Specified tail levels (2.5 is a percentage); 0.05 is also the significance level, GBM learning rate and mixture weight. |
| 0.99 | Complement of the primary tail level. |
| 0.996, 0.992, 0.9943 | Deterministic ceil((n+1)*.99)/n at n=250,500,350 (last rounded to four decimals). |
| 0.94 | RiskMetrics EWMA decay. |
| 0.85 | Simulation GARCH beta1; alpha1 is 0.10. |
| 0.5 | Magnitude of simulation skew, POT admissible shape endpoints, and the predetermined bias in units of baseline sigma in `research/r8_regime/PROTOCOL.md`. |
| 0.95, 2.2 | Mixture weight and population variance 0.95+0.05*25. |
| 1.6, 0.016 | Scaled descriptive Green boundary 4/250, percentage/fraction. |
| 2.10, 2.01 | GJR-t degrees-of-freedom acceptance and grid-fit lower bounds. |
| 0.999 | Dynamic-model stability screen. |
| 0.20 | Second native decile used for linear closure. |
| 0.001, 0.005 | Specified ACI learning rates and clipping endpoint. |
| 0.9, 0.8 | Specified LightGBM feature and bagging fractions. |
| 0.25, 0.75 | Prespecified constant and log-volatility distortion coefficients in `research/r8_review/complexity_simulation.py`. |
| 1.96 | Normal approximation multiplier for displayed Monte Carlo standard errors. |
| 0.7 | Inner fitting fraction fixed in the stronger-comparator development protocol. |
| 0.1 | L1 penalty candidate, fixed before the extension runs. |
| 0.0099 | Exact iid target-hit variance 0.01*(1-0.01) in the archived AR optimism control. |
