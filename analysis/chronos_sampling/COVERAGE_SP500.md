# Chronos coverage: default configuration vs the model's own distribution

Asset SP500, 1200 dates, backend mps (published run: CUDA/A30 — indicative).
 Realised sigma 0.00991.

| estimator | dispersion | distinct values | π̂(0.01) | π̂(0.025) | π̂(0.05) | π̂(0.1) |
|---|---|---|---|---|---|---|
| analytic (all 4093 bins) | 1.041 | 4093 (exact) | **0.0175** | **0.0367** | **0.0575** | **0.1067** |
| sampled, top_k=4094 | 1.038 | 493 | **0.0175** | **0.0358** | **0.0575** | **0.1067** |
| sampled, top_k=50 (default) | 0.128 | 50 | **0.3942** | **0.4033** | **0.4100** | **0.4192** |

Nominal is the α in each column header.

