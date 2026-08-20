# GJR-GARCH: the fit/quantile mismatch, and both repairs

`pipeline/CFP_Parametric_Benchmarks.ipynb` fits GJR with `dist='skewt'` and takes its quantile from `stats.norm.ppf`. GARCH-N shares that line and is correct only because its `dist` is `'normal'`. Neither candidate below is promoted.

| series | assets | implied z(1%) | VaR₀.₀₁/σ | mean width | monotone | π̂(0.01) | π̂(0.025) | π̂(0.05) | π̂(0.1) | Kupiec | Green |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped | 24 | -3.365 | -3.152 | 0.04725 | 100.0% | **0.0041** | **0.0119** | **0.0290** | **0.0678** | 1/24 | 24/24 |
| normal | 24 | -2.305 | -2.137 | 0.03152 | 100.0% | **0.0194** | **0.0346** | **0.0563** | **0.0974** | 0/24 | 3/24 |
| skewt | 24 | -2.540 | -2.333 | 61.07806 | 100.0% | **0.0136** | **0.0308** | **0.0559** | **0.1043** | 9/24 | 23/24 |

Nominal is the α in each π̂ column. GARCH-N's implied z is −2.326 by construction; the shipped GJR series sits at −3.365.

`candidate_skewt` stores the fitted `eta` and `lambda`, so the quantile map can be revisited without refitting.

