# GJR-GARCH with a Student-$t$ innovation

Added because correcting GJR to the Gaussian innovation the manuscript describes leaves it under-covering (π̂ = 0.0194 at a nominal 0.01, Kupiec 0/24). This is also the model the shipped series was attempting: it used `stats.t.ppf(alpha, 5)` with the degrees of freedom hard-coded and the *raw* rather than standardised quantile, so its innovation had variance 5/3 and every VaR was 1.29× too wide on top of that.

ν is estimated per window. Windows where the optimiser pins ν at its lower bound (ν ≤ 2.1, where the variance ceases to exist and the standardised quantile diverges) are counted as degenerate and the last non-degenerate ν is carried forward. The rule was fixed before seeing which assets it touches.

| asset | n | ν median | degenerate | failed fits | π̂(0.01) | Kupiec p | TL | mean width | max abs VaR |
|---|---|---|---|---|---|---|---|---|---|
| ASX200 | 6371 | 15.05 | 0 (0.0%) | 0 | 0.0206 | 0.0000 | Yellow | 0.02090 | 0.1770 |
| AUDUSD | 4909 | 14.99 | 0 (0.0%) | 0 | 0.0153 | 0.0006 | Green | 0.01701 | 0.1388 |
| BOVESPA | 6242 | 20.62 | 0 (0.0%) | 0 | 0.0144 | 0.0010 | Green | 0.03639 | 0.3276 |
| BTC | 3950 | 3.25 | 11 (0.3%) | 2 | 0.0147 | 0.0057 | Green | 0.08828 | 0.7808 |
| CBU0 | 3515 | 8.19 | **214 (6.1%)** | 190 | 0.0139 | 0.0267 | Green | 0.00902 | 0.0412 |
| DJCI | 2572 | 8.91 | 9 (0.3%) | 9 | 0.0156 | 0.0089 | Green | 0.02249 | 0.0944 |
| ETH | 2800 | 3.61 | 0 (0.0%) | 0 | 0.0114 | 0.4576 | Green | 0.11442 | 0.4001 |
| EURUSD | 5533 | 11.08 | 4 (0.1%) | 4 | 0.0117 | 0.2036 | Green | 0.01422 | 0.3511 |
| FCHI | 6448 | 11.11 | 0 (0.0%) | 0 | 0.0154 | 0.0001 | Green | 0.02998 | 0.2635 |
| FTSE100 | 6370 | 13.15 | 0 (0.0%) | 0 | 0.0185 | 0.0000 | Yellow | 0.02373 | 0.1950 |
| GBPUSD | 5545 | 10.69 | 2 (0.0%) | 2 | 0.0146 | 0.0013 | Green | 0.01369 | 0.0603 |
| GDAXI | 6404 | 9.34 | 0 (0.0%) | 0 | 0.0166 | 0.0000 | Yellow | 0.03078 | 0.2333 |
| GOLD | 6159 | 5.67 | 4 (0.1%) | 1 | 0.0156 | 0.0000 | Green | 0.02687 | 0.1468 |
| HSI | 6206 | 9.94 | 0 (0.0%) | 0 | 0.0143 | 0.0013 | Green | 0.03191 | 0.2433 |
| IBGL | 4348 | 7.32 | **109 (2.5%)** | 2 | 0.0154 | 0.0009 | Green | 0.02031 | 0.1479 |
| ICLN | 4209 | 10.11 | 0 (0.0%) | 0 | 0.0152 | 0.0016 | Green | 0.03935 | 0.2720 |
| NATGAS | 6164 | 8.24 | 0 (0.0%) | 0 | 0.0094 | 0.6379 | Green | 0.08681 | 0.3801 |
| NIFTY | 4288 | 11.50 | 0 (0.0%) | 0 | 0.0152 | 0.0016 | Green | 0.02572 | 0.2223 |
| NIKKEI | 6168 | 10.64 | 0 (0.0%) | 0 | 0.0167 | 0.0000 | Yellow | 0.03296 | 0.1878 |
| SP500 | 6340 | 9.66 | 0 (0.0%) | 0 | 0.0181 | 0.0000 | Yellow | 0.02544 | 0.3070 |
| STOXX | 5255 | 8.96 | 0 (0.0%) | 0 | 0.0167 | 0.0000 | Yellow | 0.02473 | 0.2301 |
| TLT | 5696 | 44.96 | 2 (0.0%) | 2 | 0.0112 | 0.3579 | Green | 0.02025 | 0.1107 |
| USDJPY | 6553 | 7.10 | 1 (0.0%) | 1 | 0.0116 | 0.2049 | Green | 0.01587 | 0.3935 |
| WTI | 6166 | 13.24 | 0 (0.0%) | 0 | 0.0172 | 0.0000 | Yellow | 0.05545 | 0.4067 |

**Panel:** mean π̂(0.01) = 0.0150, Kupiec pass 5/24, Green 17/24, mean width 0.03444, median implied z(1%) -2.472, 356 degenerate windows in total.

Not promoted by this script. `max abs VaR` is reported per asset because a single diverging window is invisible in a mean and inflates any width the benchmark is compared on.

