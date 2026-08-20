# Chronos sampling dose-response

Asset SP500, 20 dates, 1000 samples, seed 42, backend **mps** (published run: CUDA/A30 — indicative only).

Realised sigma (250d median): 0.00991. Target dispersion is ~1.0; the shipped series sit at 0.117 / 0.109.

| Model | cell | top_k | top_p | temp | pred_std/sigma | distinct values |
|---|---|---|---|---|---|---|
| Chronos-Mini | temp=0.5 @ k=50 | 50 | 1.0 | 0.5 | **0.110** | 50 |
| Chronos-Mini | temp=2.0 @ k=50 | 50 | 1.0 | 2.0 | **0.110** | 50 |
| Chronos-Mini | top_k=1000 | 1000 | 1.0 | 1.0 | **0.880** | 476 |
| Chronos-Mini | top_k=200 | 200 | 1.0 | 1.0 | **0.336** | 198 |
| Chronos-Mini | top_k=4094 (full vocab) | 4094 | 1.0 | 1.0 | **0.925** | 482 |
| Chronos-Mini | top_k=50 (default) | 50 | 1.0 | 1.0 | **0.110** | 50 |
| Chronos-Mini | top_p=0.9 @ k=50 | 50 | 0.9 | 1.0 | **0.105** | 45 |
| Chronos-Mini | top_p=0.99 @ k=50 | 50 | 0.99 | 1.0 | **0.110** | 50 |
| Chronos-Small | temp=0.5 @ k=50 | 50 | 1.0 | 0.5 | **0.115** | 50 |
| Chronos-Small | temp=2.0 @ k=50 | 50 | 1.0 | 2.0 | **0.117** | 50 |
| Chronos-Small | top_k=1000 | 1000 | 1.0 | 1.0 | **0.902** | 481 |
| Chronos-Small | top_k=200 | 200 | 1.0 | 1.0 | **0.338** | 198 |
| Chronos-Small | top_k=4094 (full vocab) | 4094 | 1.0 | 1.0 | **0.953** | 488 |
| Chronos-Small | top_k=50 (default) | 50 | 1.0 | 1.0 | **0.116** | 50 |
| Chronos-Small | top_p=0.9 @ k=50 | 50 | 0.9 | 1.0 | **0.110** | 45 |
| Chronos-Small | top_p=0.99 @ k=50 | 50 | 0.99 | 1.0 | **0.116** | 50 |

