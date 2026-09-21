# Panel-level violation rates: results

Descriptive; protocol fixed before computation (`PROTOCOL.md`). Pair-equal mean violation
rate over 240 pairs, percentile 95% intervals from 999 common-calendar circular block draws
(seeds {'20': 4034980058, '60': 196994084}, calendar of 2707 days). Backtest rejections at 5% from `pairs.csv`.

| method | rate % | 20-day 95% | 60-day 95% | 20-day excludes 1% | 60-day excludes 1% | Kupiec | Indep. | Joint | undefined indep. |
|---|---|---|---|---|---|---|---|---|---|
| Raw | 1.7345 | [1.4459, 2.0389] | [1.4523, 2.0736] | True | True | 133 | 51 | 134 | 0 |
| Shift-CP | 0.9576 | [0.7489, 1.1982] | [0.7439, 1.2266] | False | False | 29 | 42 | 51 | 0 |
| Rolling250 | 0.9023 | [0.7175, 1.1074] | [0.7084, 1.1052] | False | False | 6 | 20 | 18 | 0 |
| Rolling500 | 1.0634 | [0.8495, 1.3119] | [0.8323, 1.3205] | False | False | 3 | 44 | 34 | 0 |
| Selected-rolling | 1.0295 | [0.8341, 1.2637] | [0.8149, 1.2745] | False | False | 3 | 36 | 32 | 0 |
| Gate-selected-rolling | 1.0617 | [0.8560, 1.2982] | [0.8481, 1.3113] | False | False | 6 | 36 | 34 | 0 |
| Loss-gate | 1.4841 | [1.2233, 1.7684] | [1.2277, 1.7884] | True | True | 90 | 50 | 100 | 0 |
| Past-minimum | 1.0934 | [0.8795, 1.3265] | [0.8839, 1.3464] | False | False | 46 | 48 | 69 | 0 |
