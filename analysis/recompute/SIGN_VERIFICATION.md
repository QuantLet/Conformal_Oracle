# Is the sign defect real?

Two readings predict a positive stored VaR: a sign defect, and a deliberate positive-loss convention that the evaluation code misread. Positivity and reversed ordering are consistent with **both**, so neither is evidence. Checks 3 and 4 are what decide it.

## Checks 1, 2, 3, 5, 6 — the accused series

| model | assets | frac > 0 | increasing in α | decreasing in α | max&#124;stored −(−ppf)&#124; | max&#124;stored − ppf&#124; | π̂ stored | π̂ corrected | corrected VaR/σ |
|---|---|---|---|---|---|---|---|---|---|
| Moirai-2.0 | 24 | 1.000 | 0.000 | 1.000 | **4.44e-10** | 9.16e-01 | 0.9880 | **0.0166** | -2.331 |
| TimesFM-2.5 | 24 | 1.000 | 0.000 | 1.000 | **5.57e-10** | 6.54e-01 | 0.9900 | **0.0141** | -2.431 |

Published violation rates for comparison: Moirai-2.0 0.988, TimesFM-2.5 0.99.

## Check 4 — what convention does the rest of the dataset use?

| model | assets | positive median | convention |
|---|---|---|---|
| Chronos-Small | 24 | 0 | negative |
| Chronos-Mini | 24 | 0 | negative |
| TimesFM-2.5 | 24 | 24 | **POSITIVE** |
| Moirai-2.0 | 24 | 24 | **POSITIVE** |
| Moirai-1.1 | 24 | 0 | negative |
| Lag-Llama | 24 | 0 | negative |
| GJR-GARCH | 24 | 0 | negative |
| GARCH-N | 24 | 0 | negative |
| Hist-Sim | 24 | 0 | negative |
| EWMA | 24 | 0 | negative |

## Reading

Check 3 is arithmetic, not inference: the stored column is reproduced by `-student_t.ppf(alpha, df, loc=mu, scale=sigma)` to the precision shown, from the degrees of freedom, location and scale stored in the same file. The negation is in the data, not in an interpretation of it.

Check 4 removes the convention defence. A positive-loss convention shared by two of the ten forecast directories, absent from the other eight, and not implemented by the single evaluation routine that consumes all ten, is not a convention. Whatever the intent, the files are inconsistent with their only consumer, and the published violation rates are what that inconsistency produced.

