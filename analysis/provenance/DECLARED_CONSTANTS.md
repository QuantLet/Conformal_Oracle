# Declared constants

Numeric literals the manuscript may carry without being emitted by
`paper_numbers.py`. Every entry is a *choice* or a *definition*, never a
measured result. `scripts/build_guards.py` parses the `value` column and allows
those literals; anything else in the prose must be a generated macro.

A constant enters this file only with a stated role. "It looked fine" is not a
role.

| value | role | why it is not a result |
|---|---|---|
| 0.01, 0.025, 0.05, 0.10 | nominal VaR levels | the design grid |
| 0.0100 | the 1% level written to four places | same level, matched to the precision of the figure beside it |
| 0.005 | lower arm of the severity cut in the detection exercise | a threshold chosen after seeing the data; the manuscript says so at the point of use |
| 0.70 | calibration fraction `f_c` | design choice |
| 0.50, 0.80 | endpoints of the `f_c` robustness range | design choice |
| 0.94 | RiskMetrics EWMA decay | the convention |
| 0.9, 0.99 | nucleus-sampling settings in the dose-response | the swept grid |
| 0.5, 2.0 | temperature settings in the dose-response | the swept grid |
| 1.0 | dispersion target (predictive sd equal to realised) | a definition |
| 0.30 | alignment-proxy correlation bound | declared gate band |
| 0.95 | score-persistence threshold flagged as operationally risky | declared judgement |
| 0.95, 0.05 | mixture weights of Monte Carlo DGP (v) | design of the simulation |
| 0.2, 0.8 | inner-decile refit range | design of the closure check |
| 97.5 | FRTB Expected Shortfall level | the regulation |
| 3.00, 3.65, 1.22 | Basel Green and Yellow multipliers, and their ratio | the regulation, and one division |
| 0.0005 | weighting parameter in the regime-similarity diagnostic | design choice |
| -3.5, -1.8 | scale band edges | declared gate band, chosen not derived |
| 0.5, 2.0 | dispersion band edges | declared gate band |
| 3 | alpha-response band threshold | declared gate band |
| 5 | coverage-plausibility band multiplier | declared gate band |

## Not admitted here

Typesetting lengths (`\includegraphics[width=0.85\textwidth]`,
`\renewcommand{\arraystretch}{1.15}`), citation locators
(`\citet[Theorem 2.1]{...}`) and grant contract numbers are stripped by the
guard before literals are extracted. They are not claims and do not need a
declaration.
