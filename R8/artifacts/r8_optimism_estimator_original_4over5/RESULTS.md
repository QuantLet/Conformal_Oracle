# Nuisance-free optimism estimators and feasible shrinkage: results

Protocol: `research/r8_optimism_estimator/PROTOCOL.md` (fixed 13 September 2026 before any calculation). Engine: `research/r8_optimism_estimator/engine.py`. All numbers below are read from the CSV/JSON files in this folder.

## Admission decision

- E1 (blocked cross-validation, K=5): criterion 1 (penalty accuracy, |ratio-1| <= 0.15 at n in 700, 1000, 2000, both laws, both b): **PASS**; criterion 2 (rule value at n >= 700): **FAIL**; criterion 3 (simultaneous family): VALID, family size 20, critical value 2.8734, cells whose band includes 1: 10/20; estimator admitted: **NO**.
- E2 (circular block bootstrap, R=200, b=ceil(n^(1/3))): criterion 1 (penalty accuracy, |ratio-1| <= 0.15 at n in 700, 1000, 2000, both laws, both b): **PASS**; criterion 2 (rule value at n >= 700): **FAIL**; criterion 3 (simultaneous family): VALID, family size 20, critical value 2.9910, cells whose band includes 1: 18/20; estimator admitted: **NO**.

Passing estimators: none. Financial application: **NOT_RUN**.

### Criterion 2 detail (n >= 700; expected loss R(lambda C_n) - R(0) from the reference risk, mean over 500 histories)

| estimator | law | n | b | raw | full (lambda=1) | shrunken (lambda_hat) | shrunken < full | shrunken < raw | pass |
|---|---|---|---|---|---|---|---|---|---|
| E1_blocked_cv | normal | 700 | 0.000000 | 0.000e+00 | 3.509e-06 | 1.261e-06 | True | False | PASS |
| E1_blocked_cv | normal | 700 | 0.003536 | 0.000e+00 | -1.204e-05 | -1.041e-05 | False | True | FAIL |
| E1_blocked_cv | normal | 1000 | 0.000000 | 0.000e+00 | 2.508e-06 | 9.879e-07 | True | False | PASS |
| E1_blocked_cv | normal | 1000 | 0.003536 | 0.000e+00 | -1.304e-05 | -1.211e-05 | False | True | FAIL |
| E1_blocked_cv | normal | 2000 | 0.000000 | 0.000e+00 | 1.196e-06 | 4.837e-07 | True | False | PASS |
| E1_blocked_cv | normal | 2000 | 0.003536 | 0.000e+00 | -1.436e-05 | -1.402e-05 | False | True | FAIL |
| E1_blocked_cv | t5 | 700 | 0.000000 | 0.000e+00 | 6.937e-06 | 2.488e-06 | True | False | PASS |
| E1_blocked_cv | t5 | 700 | 0.003536 | 0.000e+00 | -1.287e-06 | -1.746e-06 | True | True | PASS |
| E1_blocked_cv | t5 | 1000 | 0.000000 | 0.000e+00 | 4.764e-06 | 1.735e-06 | True | False | PASS |
| E1_blocked_cv | t5 | 1000 | 0.003536 | 0.000e+00 | -3.459e-06 | -3.349e-06 | False | True | FAIL |
| E1_blocked_cv | t5 | 2000 | 0.000000 | 0.000e+00 | 2.442e-06 | 9.723e-07 | True | False | PASS |
| E1_blocked_cv | t5 | 2000 | 0.003536 | 0.000e+00 | -5.782e-06 | -5.039e-06 | False | True | FAIL |
| E2_block_bootstrap | normal | 700 | 0.000000 | 0.000e+00 | 3.509e-06 | 1.059e-06 | True | False | PASS |
| E2_block_bootstrap | normal | 700 | 0.003536 | 0.000e+00 | -1.204e-05 | -1.063e-05 | False | True | FAIL |
| E2_block_bootstrap | normal | 1000 | 0.000000 | 0.000e+00 | 2.508e-06 | 7.266e-07 | True | False | PASS |
| E2_block_bootstrap | normal | 1000 | 0.003536 | 0.000e+00 | -1.304e-05 | -1.206e-05 | False | True | FAIL |
| E2_block_bootstrap | normal | 2000 | 0.000000 | 0.000e+00 | 1.196e-06 | 3.742e-07 | True | False | PASS |
| E2_block_bootstrap | normal | 2000 | 0.003536 | 0.000e+00 | -1.436e-05 | -1.405e-05 | False | True | FAIL |
| E2_block_bootstrap | t5 | 700 | 0.000000 | 0.000e+00 | 6.937e-06 | 2.023e-06 | True | False | PASS |
| E2_block_bootstrap | t5 | 700 | 0.003536 | 0.000e+00 | -1.287e-06 | -1.807e-06 | True | True | PASS |
| E2_block_bootstrap | t5 | 1000 | 0.000000 | 0.000e+00 | 4.764e-06 | 1.387e-06 | True | False | PASS |
| E2_block_bootstrap | t5 | 1000 | 0.003536 | 0.000e+00 | -3.459e-06 | -3.131e-06 | False | True | FAIL |
| E2_block_bootstrap | t5 | 2000 | 0.000000 | 0.000e+00 | 2.442e-06 | 6.945e-07 | True | False | PASS |
| E2_block_bootstrap | t5 | 2000 | 0.003536 | 0.000e+00 | -5.782e-06 | -4.909e-06 | False | True | FAIL |

Failing cells (both estimators): every b > 0 cell at n in 700, 1000, 2000 except t5 at n = 700. In those cells the shrunken policy has a higher expected loss than full correction. All b = 0 cells pass (shrunken below full).

## Synthetic tables

Columns: mean and MCSE (sd/sqrt(500)) of O_hat; ratio = mean O_hat / (2A_0n) with 2A_0n = Omega/(n f*) from truth.csv; band = simultaneous 95% band on the ratio (one family of 20 cells per estimator, saved 999 history-bootstrap rows, max-standardised-deviation rule, 95th percentile with NumPy method="higher"); v3 ref = the known-truth independent-marginal optimism ratio from results/theory_loop_v3/diagnostic/simultaneous_bands.csv (oracle, for orientation only); lambda_hat mean and sd over histories; share of histories with lambda_hat = 0; expected losses R(lambda C_n) - R(0) for raw (lambda = 0, identically zero), full (lambda = 1), shrunken (lambda_hat); shrunken - full with its MCSE.

### E1 (blocked cross-validation, K=5)

| law | n | b | mean O_hat | MCSE | 2A_0n | ratio | band lower | band upper | incl. 1 | v3 ref | mean lambda_hat | sd | share lambda=0 | loss raw | loss full | loss shrunken | shrunken - full | MCSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 250 | 0.000000 | 1.449e-05 | 6.429e-07 | 1.978e-05 | 0.7326 | 0.6392 | 0.8260 | False | 0.9486 | 0.2921 | 0.3924 | 0.600 | 0.000e+00 | 1.335e-05 | 3.687e-06 | -9.663e-06 | 8.152e-07 |
| normal | 250 | 0.003536 | 1.449e-05 | 6.429e-07 | 1.978e-05 | 0.7326 | 0.6392 | 0.8260 | False | 0.9486 | 0.4671 | 0.4146 | 0.380 | 0.000e+00 | -2.203e-06 | -3.932e-06 | -1.729e-06 | 7.658e-07 |
| normal | 500 | 0.000000 | 8.828e-06 | 3.746e-07 | 9.892e-06 | 0.8925 | 0.7837 | 1.0013 | True | 1.0163 | 0.2793 | 0.3731 | 0.588 | 0.000e+00 | 5.113e-06 | 2.093e-06 | -3.020e-06 | 1.999e-07 |
| normal | 500 | 0.003536 | 8.828e-06 | 3.746e-07 | 9.892e-06 | 0.8925 | 0.7837 | 1.0013 | True | 1.0163 | 0.6242 | 0.3593 | 0.186 | 0.000e+00 | -1.044e-05 | -9.574e-06 | 8.656e-07 | 2.541e-07 |
| normal | 700 | 0.000000 | 7.243e-06 | 2.931e-07 | 7.066e-06 | 1.0251 | 0.9059 | 1.1443 | True | 0.9705 | 0.2313 | 0.3338 | 0.612 | 0.000e+00 | 3.509e-06 | 1.261e-06 | -2.248e-06 | 1.435e-07 |
| normal | 700 | 0.003536 | 7.243e-06 | 2.931e-07 | 7.066e-06 | 1.0251 | 0.9059 | 1.1443 | True | 0.9705 | 0.6605 | 0.3415 | 0.136 | 0.000e+00 | -1.204e-05 | -1.041e-05 | 1.631e-06 | 2.312e-07 |
| normal | 1000 | 0.000000 | 4.332e-06 | 1.713e-07 | 4.946e-06 | 0.8758 | 0.7763 | 0.9753 | False | 0.9902 | 0.2637 | 0.3435 | 0.556 | 0.000e+00 | 2.508e-06 | 9.879e-07 | -1.520e-06 | 9.303e-08 |
| normal | 1000 | 0.003536 | 4.332e-06 | 1.713e-07 | 4.946e-06 | 0.8758 | 0.7763 | 0.9753 | False | 0.9902 | 0.7828 | 0.2674 | 0.064 | 0.000e+00 | -1.304e-05 | -1.211e-05 | 9.392e-07 | 1.513e-07 |
| normal | 2000 | 0.000000 | 2.144e-06 | 8.098e-08 | 2.473e-06 | 0.8668 | 0.7727 | 0.9609 | False | 0.9835 | 0.2615 | 0.3446 | 0.544 | 0.000e+00 | 1.196e-06 | 4.837e-07 | -7.128e-07 | 4.370e-08 |
| normal | 2000 | 0.003536 | 2.144e-06 | 8.098e-08 | 2.473e-06 | 0.8668 | 0.7727 | 0.9609 | False | 0.9835 | 0.9008 | 0.1434 | 0.008 | 0.000e+00 | -1.436e-05 | -1.402e-05 | 3.345e-07 | 6.513e-08 |
| t5 | 250 | 0.000000 | 2.499e-05 | 1.205e-06 | 3.548e-05 | 0.7045 | 0.6069 | 0.8021 | False | 0.9964 | 0.3069 | 0.3947 | 0.578 | 0.000e+00 | 2.589e-05 | 6.819e-06 | -1.907e-05 | 1.510e-06 |
| t5 | 250 | 0.003536 | 2.499e-05 | 1.205e-06 | 3.548e-05 | 0.7045 | 0.6069 | 0.8021 | False | 0.9964 | 0.3374 | 0.3960 | 0.530 | 0.000e+00 | 1.767e-05 | 4.971e-06 | -1.270e-05 | 1.377e-06 |
| t5 | 500 | 0.000000 | 1.694e-05 | 6.905e-07 | 1.774e-05 | 0.9552 | 0.8434 | 1.0671 | True | 0.9666 | 0.2342 | 0.3487 | 0.616 | 0.000e+00 | 8.915e-06 | 3.082e-06 | -5.833e-06 | 4.567e-07 |
| t5 | 500 | 0.003536 | 1.694e-05 | 6.905e-07 | 1.774e-05 | 0.9552 | 0.8434 | 1.0671 | True | 0.9666 | 0.3427 | 0.3657 | 0.446 | 0.000e+00 | 6.911e-07 | -1.021e-06 | -1.712e-06 | 4.779e-07 |
| t5 | 700 | 0.000000 | 1.298e-05 | 4.777e-07 | 1.267e-05 | 1.0245 | 0.9161 | 1.1328 | True | 1.0514 | 0.2370 | 0.3379 | 0.600 | 0.000e+00 | 6.937e-06 | 2.488e-06 | -4.449e-06 | 3.241e-07 |
| t5 | 700 | 0.003536 | 1.298e-05 | 4.777e-07 | 1.267e-05 | 1.0245 | 0.9161 | 1.1328 | True | 1.0514 | 0.3837 | 0.3760 | 0.416 | 0.000e+00 | -1.287e-06 | -1.746e-06 | -4.589e-07 | 3.489e-07 |
| t5 | 1000 | 0.000000 | 8.389e-06 | 3.101e-07 | 8.869e-06 | 0.9459 | 0.8454 | 1.0463 | True | 1.0261 | 0.2408 | 0.3417 | 0.596 | 0.000e+00 | 4.764e-06 | 1.735e-06 | -3.030e-06 | 2.086e-07 |
| t5 | 1000 | 0.003536 | 8.389e-06 | 3.101e-07 | 8.869e-06 | 0.9459 | 0.8454 | 1.0463 | True | 1.0261 | 0.4727 | 0.3717 | 0.286 | 0.000e+00 | -3.459e-06 | -3.349e-06 | 1.101e-07 | 2.364e-07 |
| t5 | 2000 | 0.000000 | 3.878e-06 | 1.471e-07 | 4.435e-06 | 0.8744 | 0.7791 | 0.9697 | False | 1.0612 | 0.2651 | 0.3542 | 0.568 | 0.000e+00 | 2.442e-06 | 9.723e-07 | -1.469e-06 | 9.131e-08 |
| t5 | 2000 | 0.003536 | 3.878e-06 | 1.471e-07 | 4.435e-06 | 0.8744 | 0.7791 | 0.9697 | False | 1.0612 | 0.6366 | 0.3491 | 0.162 | 0.000e+00 | -5.782e-06 | -5.039e-06 | 7.430e-07 | 1.291e-07 |

### E2 (circular block bootstrap, R=200, b=ceil(n^(1/3)))

| law | n | b | mean O_hat | MCSE | 2A_0n | ratio | band lower | band upper | incl. 1 | v3 ref | mean lambda_hat | sd | share lambda=0 | loss raw | loss full | loss shrunken | shrunken - full | MCSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 250 | 0.000000 | 1.715e-05 | 4.176e-07 | 1.978e-05 | 0.8666 | 0.8035 | 0.9297 | False | 0.9486 | 0.1754 | 0.2917 | 0.670 | 0.000e+00 | 1.335e-05 | 2.505e-06 | -1.084e-05 | 8.189e-07 |
| normal | 250 | 0.003536 | 1.711e-05 | 4.364e-07 | 1.978e-05 | 0.8648 | 0.7988 | 0.9307 | False | 0.9486 | 0.3939 | 0.3815 | 0.410 | 0.000e+00 | -2.203e-06 | -3.771e-06 | -1.567e-06 | 7.186e-07 |
| normal | 500 | 0.000000 | 9.455e-06 | 1.700e-07 | 9.892e-06 | 0.9559 | 0.9045 | 1.0072 | True | 1.0163 | 0.1893 | 0.3001 | 0.650 | 0.000e+00 | 5.113e-06 | 1.621e-06 | -3.492e-06 | 2.008e-07 |
| normal | 500 | 0.003536 | 9.459e-06 | 1.764e-07 | 9.892e-06 | 0.9562 | 0.9029 | 1.0095 | True | 1.0163 | 0.5731 | 0.3349 | 0.186 | 0.000e+00 | -1.044e-05 | -9.623e-06 | 8.163e-07 | 2.602e-07 |
| normal | 700 | 0.000000 | 6.885e-06 | 1.204e-07 | 7.066e-06 | 0.9744 | 0.9234 | 1.0254 | True | 0.9705 | 0.1701 | 0.2823 | 0.666 | 0.000e+00 | 3.509e-06 | 1.059e-06 | -2.450e-06 | 1.475e-07 |
| normal | 700 | 0.003536 | 6.904e-06 | 1.284e-07 | 7.066e-06 | 0.9771 | 0.9227 | 1.0314 | True | 0.9705 | 0.6490 | 0.3186 | 0.132 | 0.000e+00 | -1.204e-05 | -1.063e-05 | 1.417e-06 | 2.271e-07 |
| normal | 1000 | 0.000000 | 4.785e-06 | 7.627e-08 | 4.946e-06 | 0.9674 | 0.9213 | 1.0135 | True | 0.9902 | 0.1732 | 0.2785 | 0.650 | 0.000e+00 | 2.508e-06 | 7.266e-07 | -1.782e-06 | 9.997e-08 |
| normal | 1000 | 0.003536 | 4.766e-06 | 8.776e-08 | 4.946e-06 | 0.9636 | 0.9105 | 1.0166 | True | 0.9902 | 0.7491 | 0.2613 | 0.064 | 0.000e+00 | -1.304e-05 | -1.206e-05 | 9.827e-07 | 1.461e-07 |
| normal | 2000 | 0.000000 | 2.398e-06 | 3.308e-08 | 2.473e-06 | 0.9696 | 0.9296 | 1.0096 | True | 0.9835 | 0.1722 | 0.2782 | 0.632 | 0.000e+00 | 1.196e-06 | 3.742e-07 | -8.222e-07 | 4.482e-08 |
| normal | 2000 | 0.003536 | 2.378e-06 | 4.253e-08 | 2.473e-06 | 0.9616 | 0.9102 | 1.0131 | True | 0.9835 | 0.8929 | 0.1120 | 0.004 | 0.000e+00 | -1.436e-05 | -1.405e-05 | 3.043e-07 | 5.061e-08 |
| t5 | 250 | 0.000000 | 3.574e-05 | 1.239e-06 | 3.548e-05 | 1.0074 | 0.9030 | 1.1118 | True | 0.9964 | 0.1809 | 0.3084 | 0.692 | 0.000e+00 | 2.589e-05 | 4.769e-06 | -2.112e-05 | 1.542e-06 |
| t5 | 250 | 0.003536 | 3.576e-05 | 1.252e-06 | 3.548e-05 | 1.0079 | 0.9024 | 1.1134 | True | 0.9964 | 0.2064 | 0.3119 | 0.628 | 0.000e+00 | 1.767e-05 | 3.501e-06 | -1.417e-05 | 1.459e-06 |
| t5 | 500 | 0.000000 | 1.807e-05 | 3.717e-07 | 1.774e-05 | 1.0189 | 0.9563 | 1.0816 | True | 0.9666 | 0.1687 | 0.2861 | 0.668 | 0.000e+00 | 8.915e-06 | 2.449e-06 | -6.466e-06 | 4.356e-07 |
| t5 | 500 | 0.003536 | 1.794e-05 | 3.795e-07 | 1.774e-05 | 1.0111 | 0.9471 | 1.0751 | True | 0.9666 | 0.2751 | 0.3277 | 0.530 | 0.000e+00 | 6.911e-07 | -1.046e-06 | -1.737e-06 | 4.672e-07 |
| t5 | 700 | 0.000000 | 1.301e-05 | 2.432e-07 | 1.267e-05 | 1.0268 | 0.9694 | 1.0842 | True | 1.0514 | 0.1861 | 0.2917 | 0.644 | 0.000e+00 | 6.937e-06 | 2.023e-06 | -4.914e-06 | 3.435e-07 |
| t5 | 700 | 0.003536 | 1.297e-05 | 2.492e-07 | 1.267e-05 | 1.0235 | 0.9646 | 1.0823 | True | 1.0514 | 0.3188 | 0.3411 | 0.452 | 0.000e+00 | -1.287e-06 | -1.807e-06 | -5.199e-07 | 3.698e-07 |
| t5 | 1000 | 0.000000 | 9.186e-06 | 1.670e-07 | 8.869e-06 | 1.0357 | 0.9794 | 1.0920 | True | 1.0261 | 0.1692 | 0.2821 | 0.674 | 0.000e+00 | 4.764e-06 | 1.387e-06 | -3.378e-06 | 2.044e-07 |
| t5 | 1000 | 0.003536 | 9.219e-06 | 1.746e-07 | 8.869e-06 | 1.0395 | 0.9806 | 1.0984 | True | 1.0261 | 0.3972 | 0.3450 | 0.338 | 0.000e+00 | -3.459e-06 | -3.131e-06 | 3.282e-07 | 2.458e-07 |
| t5 | 2000 | 0.000000 | 4.526e-06 | 6.837e-08 | 4.435e-06 | 1.0206 | 0.9745 | 1.0667 | True | 1.0612 | 0.1749 | 0.2770 | 0.654 | 0.000e+00 | 2.442e-06 | 6.945e-07 | -1.747e-06 | 9.525e-08 |
| t5 | 2000 | 0.003536 | 4.509e-06 | 7.258e-08 | 4.435e-06 | 1.0168 | 0.9678 | 1.0657 | True | 1.0612 | 0.5723 | 0.3324 | 0.176 | 0.000e+00 | -5.782e-06 | -4.909e-06 | 8.735e-07 | 1.313e-07 |

E2 block lengths ceil(n^(1/3)): n=250: 7, n=500: 8, n=700: 9, n=1000: 10, n=2000: 13.
E1 is invariant to the bias translation b by construction (the baseline term rho(-s) cancels between L_cv and I_n), so its b = 0 and b > 0 rows coincide; E2 depends on b through the bootstrap-sample baseline and differs slightly between the two b rows.

## Supplementary simultaneous bands on expected-loss differences (not a protocol criterion)

One family per estimator with 30 cells: shrunken - full for all 20 (law, n, b) cells and shrunken - raw for the 10 b > 0 cells; same 999 history-bootstrap rows and band rule as above.

| estimator | law | n | b | contrast | mean | SE | lower | upper | critical value |
|---|---|---|---|---|---|---|---|---|---|
| E1_blocked_cv | normal | 250 | 0.000000 | shrunk_minus_full | -9.663e-06 | 8.152e-07 | -1.227e-05 | -7.059e-06 | 3.1935 |
| E1_blocked_cv | normal | 250 | 0.003536 | shrunk_minus_full | -1.729e-06 | 7.658e-07 | -4.174e-06 | 7.170e-07 | 3.1935 |
| E1_blocked_cv | normal | 250 | 0.003536 | shrunk_minus_raw | -3.932e-06 | 5.230e-07 | -5.602e-06 | -2.262e-06 | 3.1935 |
| E1_blocked_cv | normal | 500 | 0.000000 | shrunk_minus_full | -3.020e-06 | 1.999e-07 | -3.659e-06 | -2.382e-06 | 3.1935 |
| E1_blocked_cv | normal | 500 | 0.003536 | shrunk_minus_full | 8.656e-07 | 2.541e-07 | 5.410e-08 | 1.677e-06 | 3.1935 |
| E1_blocked_cv | normal | 500 | 0.003536 | shrunk_minus_raw | -9.574e-06 | 2.849e-07 | -1.048e-05 | -8.664e-06 | 3.1935 |
| E1_blocked_cv | normal | 700 | 0.000000 | shrunk_minus_full | -2.248e-06 | 1.435e-07 | -2.706e-06 | -1.790e-06 | 3.1935 |
| E1_blocked_cv | normal | 700 | 0.003536 | shrunk_minus_full | 1.631e-06 | 2.312e-07 | 8.923e-07 | 2.369e-06 | 3.1935 |
| E1_blocked_cv | normal | 700 | 0.003536 | shrunk_minus_raw | -1.041e-05 | 2.648e-07 | -1.126e-05 | -9.567e-06 | 3.1935 |
| E1_blocked_cv | normal | 1000 | 0.000000 | shrunk_minus_full | -1.520e-06 | 9.303e-08 | -1.817e-06 | -1.223e-06 | 3.1935 |
| E1_blocked_cv | normal | 1000 | 0.003536 | shrunk_minus_full | 9.392e-07 | 1.513e-07 | 4.559e-07 | 1.422e-06 | 3.1935 |
| E1_blocked_cv | normal | 1000 | 0.003536 | shrunk_minus_raw | -1.211e-05 | 2.036e-07 | -1.276e-05 | -1.145e-05 | 3.1935 |
| E1_blocked_cv | normal | 2000 | 0.000000 | shrunk_minus_full | -7.128e-07 | 4.370e-08 | -8.524e-07 | -5.732e-07 | 3.1935 |
| E1_blocked_cv | normal | 2000 | 0.003536 | shrunk_minus_full | 3.345e-07 | 6.513e-08 | 1.266e-07 | 5.425e-07 | 3.1935 |
| E1_blocked_cv | normal | 2000 | 0.003536 | shrunk_minus_raw | -1.402e-05 | 1.095e-07 | -1.437e-05 | -1.367e-05 | 3.1935 |
| E1_blocked_cv | t5 | 250 | 0.000000 | shrunk_minus_full | -1.907e-05 | 1.510e-06 | -2.390e-05 | -1.425e-05 | 3.1935 |
| E1_blocked_cv | t5 | 250 | 0.003536 | shrunk_minus_full | -1.270e-05 | 1.377e-06 | -1.710e-05 | -8.302e-06 | 3.1935 |
| E1_blocked_cv | t5 | 250 | 0.003536 | shrunk_minus_raw | 4.971e-06 | 8.412e-07 | 2.285e-06 | 7.657e-06 | 3.1935 |
| E1_blocked_cv | t5 | 500 | 0.000000 | shrunk_minus_full | -5.833e-06 | 4.567e-07 | -7.292e-06 | -4.375e-06 | 3.1935 |
| E1_blocked_cv | t5 | 500 | 0.003536 | shrunk_minus_full | -1.712e-06 | 4.779e-07 | -3.238e-06 | -1.860e-07 | 3.1935 |
| E1_blocked_cv | t5 | 500 | 0.003536 | shrunk_minus_raw | -1.021e-06 | 3.160e-07 | -2.030e-06 | -1.176e-08 | 3.1935 |
| E1_blocked_cv | t5 | 700 | 0.000000 | shrunk_minus_full | -4.449e-06 | 3.241e-07 | -5.484e-06 | -3.414e-06 | 3.1935 |
| E1_blocked_cv | t5 | 700 | 0.003536 | shrunk_minus_full | -4.589e-07 | 3.489e-07 | -1.573e-06 | 6.552e-07 | 3.1935 |
| E1_blocked_cv | t5 | 700 | 0.003536 | shrunk_minus_raw | -1.746e-06 | 3.223e-07 | -2.775e-06 | -7.166e-07 | 3.1935 |
| E1_blocked_cv | t5 | 1000 | 0.000000 | shrunk_minus_full | -3.030e-06 | 2.086e-07 | -3.696e-06 | -2.364e-06 | 3.1935 |
| E1_blocked_cv | t5 | 1000 | 0.003536 | shrunk_minus_full | 1.101e-07 | 2.364e-07 | -6.448e-07 | 8.650e-07 | 3.1935 |
| E1_blocked_cv | t5 | 1000 | 0.003536 | shrunk_minus_raw | -3.349e-06 | 2.039e-07 | -4.000e-06 | -2.698e-06 | 3.1935 |
| E1_blocked_cv | t5 | 2000 | 0.000000 | shrunk_minus_full | -1.469e-06 | 9.131e-08 | -1.761e-06 | -1.178e-06 | 3.1935 |
| E1_blocked_cv | t5 | 2000 | 0.003536 | shrunk_minus_full | 7.430e-07 | 1.291e-07 | 3.308e-07 | 1.155e-06 | 3.1935 |
| E1_blocked_cv | t5 | 2000 | 0.003536 | shrunk_minus_raw | -5.039e-06 | 1.507e-07 | -5.520e-06 | -4.558e-06 | 3.1935 |
| E2_block_bootstrap | normal | 250 | 0.000000 | shrunk_minus_full | -1.084e-05 | 8.189e-07 | -1.344e-05 | -8.248e-06 | 3.1699 |
| E2_block_bootstrap | normal | 250 | 0.003536 | shrunk_minus_full | -1.567e-06 | 7.186e-07 | -3.845e-06 | 7.105e-07 | 3.1699 |
| E2_block_bootstrap | normal | 250 | 0.003536 | shrunk_minus_raw | -3.771e-06 | 5.049e-07 | -5.371e-06 | -2.170e-06 | 3.1699 |
| E2_block_bootstrap | normal | 500 | 0.000000 | shrunk_minus_full | -3.492e-06 | 2.008e-07 | -4.129e-06 | -2.856e-06 | 3.1699 |
| E2_block_bootstrap | normal | 500 | 0.003536 | shrunk_minus_full | 8.163e-07 | 2.602e-07 | -8.538e-09 | 1.641e-06 | 3.1699 |
| E2_block_bootstrap | normal | 500 | 0.003536 | shrunk_minus_raw | -9.623e-06 | 2.751e-07 | -1.049e-05 | -8.751e-06 | 3.1699 |
| E2_block_bootstrap | normal | 700 | 0.000000 | shrunk_minus_full | -2.450e-06 | 1.475e-07 | -2.917e-06 | -1.982e-06 | 3.1699 |
| E2_block_bootstrap | normal | 700 | 0.003536 | shrunk_minus_full | 1.417e-06 | 2.271e-07 | 6.975e-07 | 2.137e-06 | 3.1699 |
| E2_block_bootstrap | normal | 700 | 0.003536 | shrunk_minus_raw | -1.063e-05 | 2.568e-07 | -1.144e-05 | -9.812e-06 | 3.1699 |
| E2_block_bootstrap | normal | 1000 | 0.000000 | shrunk_minus_full | -1.782e-06 | 9.997e-08 | -2.098e-06 | -1.465e-06 | 3.1699 |
| E2_block_bootstrap | normal | 1000 | 0.003536 | shrunk_minus_full | 9.827e-07 | 1.461e-07 | 5.196e-07 | 1.446e-06 | 3.1699 |
| E2_block_bootstrap | normal | 1000 | 0.003536 | shrunk_minus_raw | -1.206e-05 | 2.008e-07 | -1.270e-05 | -1.143e-05 | 3.1699 |
| E2_block_bootstrap | normal | 2000 | 0.000000 | shrunk_minus_full | -8.222e-07 | 4.482e-08 | -9.643e-07 | -6.801e-07 | 3.1699 |
| E2_block_bootstrap | normal | 2000 | 0.003536 | shrunk_minus_full | 3.043e-07 | 5.061e-08 | 1.439e-07 | 4.647e-07 | 3.1699 |
| E2_block_bootstrap | normal | 2000 | 0.003536 | shrunk_minus_raw | -1.405e-05 | 9.997e-08 | -1.437e-05 | -1.373e-05 | 3.1699 |
| E2_block_bootstrap | t5 | 250 | 0.000000 | shrunk_minus_full | -2.112e-05 | 1.542e-06 | -2.601e-05 | -1.624e-05 | 3.1699 |
| E2_block_bootstrap | t5 | 250 | 0.003536 | shrunk_minus_full | -1.417e-05 | 1.459e-06 | -1.880e-05 | -9.542e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 250 | 0.003536 | shrunk_minus_raw | 3.501e-06 | 7.495e-07 | 1.125e-06 | 5.877e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 500 | 0.000000 | shrunk_minus_full | -6.466e-06 | 4.356e-07 | -7.847e-06 | -5.085e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 500 | 0.003536 | shrunk_minus_full | -1.737e-06 | 4.672e-07 | -3.218e-06 | -2.557e-07 | 3.1699 |
| E2_block_bootstrap | t5 | 500 | 0.003536 | shrunk_minus_raw | -1.046e-06 | 2.977e-07 | -1.989e-06 | -1.020e-07 | 3.1699 |
| E2_block_bootstrap | t5 | 700 | 0.000000 | shrunk_minus_full | -4.914e-06 | 3.435e-07 | -6.003e-06 | -3.825e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 700 | 0.003536 | shrunk_minus_full | -5.199e-07 | 3.698e-07 | -1.692e-06 | 6.523e-07 | 3.1699 |
| E2_block_bootstrap | t5 | 700 | 0.003536 | shrunk_minus_raw | -1.807e-06 | 2.734e-07 | -2.674e-06 | -9.401e-07 | 3.1699 |
| E2_block_bootstrap | t5 | 1000 | 0.000000 | shrunk_minus_full | -3.378e-06 | 2.044e-07 | -4.026e-06 | -2.730e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 1000 | 0.003536 | shrunk_minus_full | 3.282e-07 | 2.458e-07 | -4.511e-07 | 1.107e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 1000 | 0.003536 | shrunk_minus_raw | -3.131e-06 | 1.962e-07 | -3.753e-06 | -2.509e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 2000 | 0.000000 | shrunk_minus_full | -1.747e-06 | 9.525e-08 | -2.049e-06 | -1.445e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 2000 | 0.003536 | shrunk_minus_full | 8.735e-07 | 1.313e-07 | 4.572e-07 | 1.290e-06 | 3.1699 |
| E2_block_bootstrap | t5 | 2000 | 0.003536 | shrunk_minus_raw | -4.909e-06 | 1.469e-07 | -5.374e-06 | -4.443e-06 | 3.1699 |

## Supplementary oracle diagnostic (not a protocol criterion)

Expected loss of the oracle factor lambda* = B/(B + A_0) with the true B = R(0) - R(q) and A_0 = Omega/(2 n f*), applied to the same C_n, from the reference risk; also the best fixed lambda on a 1001-point grid. Computed by `research/r8_optimism_estimator/oracle_lambda.py`, saved in `oracle_lambda.csv`. It separates the value of the shrinkage rule from the estimation error in lambda_hat.

| law | n | b | B | A_0 | lambda* | loss full | loss oracle lambda* | oracle - full | MCSE | best grid lambda | its loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 250 | 0.000000 | 0.000e+00 | 9.892e-06 | 0.0000 | 1.335e-05 | 0.000e+00 | -1.335e-05 | 9.303e-07 | 0.000 | 0.000e+00 |
| normal | 250 | 0.003536 | 1.555e-05 | 9.892e-06 | 0.6112 | -2.203e-06 | -1.042e-05 | -8.221e-06 | 6.354e-07 | 0.460 | -1.128e-05 |
| normal | 500 | 0.000000 | 0.000e+00 | 4.946e-06 | 0.0000 | 5.113e-06 | 0.000e+00 | -5.113e-06 | 3.088e-07 | 0.000 | 0.000e+00 |
| normal | 500 | 0.003536 | 1.555e-05 | 4.946e-06 | 0.7587 | -1.044e-05 | -1.186e-05 | -1.426e-06 | 1.706e-07 | 0.705 | -1.192e-05 |
| normal | 700 | 0.000000 | 0.000e+00 | 3.533e-06 | 0.0000 | 3.509e-06 | 0.000e+00 | -3.509e-06 | 2.283e-07 | 0.000 | 0.000e+00 |
| normal | 700 | 0.003536 | 1.555e-05 | 3.533e-06 | 0.8149 | -1.204e-05 | -1.282e-05 | -7.766e-07 | 1.186e-07 | 0.773 | -1.285e-05 |
| normal | 1000 | 0.000000 | 0.000e+00 | 2.473e-06 | 0.0000 | 2.508e-06 | 0.000e+00 | -2.508e-06 | 1.561e-07 | 0.000 | 0.000e+00 |
| normal | 1000 | 0.003536 | 1.555e-05 | 2.473e-06 | 0.8628 | -1.304e-05 | -1.342e-05 | -3.733e-07 | 7.063e-08 | 0.836 | -1.343e-05 |
| normal | 2000 | 0.000000 | 0.000e+00 | 1.236e-06 | 0.0000 | 1.196e-06 | 0.000e+00 | -1.196e-06 | 7.625e-08 | 0.000 | 0.000e+00 |
| normal | 2000 | 0.003536 | 1.555e-05 | 1.236e-06 | 0.9264 | -1.436e-05 | -1.442e-05 | -6.717e-08 | 2.492e-08 | 0.928 | -1.442e-05 |
| t5 | 250 | 0.000000 | 0.000e+00 | 1.774e-05 | 0.0000 | 2.589e-05 | 0.000e+00 | -2.589e-05 | 1.731e-06 | 0.000 | 0.000e+00 |
| t5 | 250 | 0.003536 | 8.224e-06 | 1.774e-05 | 0.3168 | 1.767e-05 | -4.328e-06 | -2.200e-05 | 1.636e-06 | 0.250 | -4.587e-06 |
| t5 | 500 | 0.000000 | 0.000e+00 | 8.869e-06 | 0.0000 | 8.915e-06 | 0.000e+00 | -8.915e-06 | 5.969e-07 | 0.000 | 0.000e+00 |
| t5 | 500 | 0.003536 | 8.224e-06 | 8.869e-06 | 0.4811 | 6.911e-07 | -4.384e-06 | -5.076e-06 | 5.102e-07 | 0.435 | -4.425e-06 |
| t5 | 700 | 0.000000 | 0.000e+00 | 6.335e-06 | 0.0000 | 6.937e-06 | 0.000e+00 | -6.937e-06 | 4.923e-07 | 0.000 | 0.000e+00 |
| t5 | 700 | 0.003536 | 8.224e-06 | 6.335e-06 | 0.5649 | -1.287e-06 | -4.714e-06 | -3.427e-06 | 3.856e-07 | 0.495 | -4.791e-06 |
| t5 | 1000 | 0.000000 | 0.000e+00 | 4.435e-06 | 0.0000 | 4.764e-06 | 0.000e+00 | -4.764e-06 | 3.065e-07 | 0.000 | 0.000e+00 |
| t5 | 1000 | 0.003536 | 8.224e-06 | 4.435e-06 | 0.6497 | -3.459e-06 | -5.480e-06 | -2.021e-06 | 2.250e-07 | 0.583 | -5.538e-06 |
| t5 | 2000 | 0.000000 | 0.000e+00 | 2.217e-06 | 0.0000 | 2.442e-06 | 0.000e+00 | -2.442e-06 | 1.463e-07 | 0.000 | 0.000e+00 |
| t5 | 2000 | 0.003536 | 8.224e-06 | 2.217e-06 | 0.7876 | -5.782e-06 | -6.360e-06 | -5.781e-07 | 8.909e-08 | 0.743 | -6.379e-06 |

## What is established and what is not

1. Measured. Both estimators recover the optimism penalty 2A_0n within the 15% criterion at n in 700, 1000, 2000 on both laws: E1 ratios range 0.8668 to 1.0251, E2 ratios range 0.9616 to 1.0395. E1 undershoots at n = 250 (ratio 0.73 normal, 0.70 t5) and n = 1000, 2000 (0.87 to 0.95); E2 stays within 0.96 to 1.04 for n >= 500. Criterion 1 holds for both. Simultaneous bands: E1_blocked_cv: 10 of 20 bands include 1; E2_block_bootstrap: 18 of 20 bands include 1.
2. Measured. The feasible shrinkage lambda_hat = clip(B_hat/(B_hat + A_hat), 0, 1) fails criterion 2 for both estimators: with b > 0 and n >= 700 its expected loss exceeds full correction in 5 of 6 cells per estimator (all three normal cells, t5 at n = 1000 and 2000); the supplementary bands on shrunken - full exclude zero on the harmful side in the three normal cells and in t5 at n = 2000 for both estimators. With b = 0 the shrunken policy beats full correction in every cell (it is between raw and full, and raw is optimal there).
3. Measured (supplementary). The oracle factor lambda* with the true B and A_0 beats full correction in every b > 0 cell (oracle - full negative, at least 2.7 MCSE from zero in all cells), so the rule has value when its inputs are known; the failure lies in the sampling error of B_hat = -I_n(C_n) - A_hat, which enters lambda_hat through a ratio and, for b > 0, produces a mean lambda_hat below lambda* (e.g. normal n = 1000: mean lambda_hat 0.78 (E1) and 0.75 (E2) against lambda* = 0.86) with history-to-history spread (sd 0.27 (E1) and 0.26 (E2) at that cell) whose cost is not offset.
4. Not established. No feasible shrinkage rule with demonstrated value on these synthetic laws; no financial result. The financial application is NOT_RUN by the protocol's admission rule. Nothing here transfers to a rolling rule or to an individual conditional date.
5. Scope. The evidence is confined to the two stored synthetic laws (500 histories each, GARCH-dependent scores with iid true hits), the two bias translations b in {0, 0.003536}, calibration prefixes of length n in [250, 500, 700, 1000, 2000], and the reference risk from the saved Chebyshev interpolants (error budget 4e-11).

## Financial application

**NOT_RUN.** No estimator passed criteria 1 and 2. `financial.py` was not written. For the record, an inspection of one pair folder (`artifacts/r8_ten_comparators/pairs/Chronos-2__ASX200/`) found that `daily.parquet` holds the 1868 test dates only (columns r, sigma, the 19 method paths and 4 DtACI-expected columns); the calibration scores would have to be rebuilt from the returns file and the raw forecast path as in `research/r8_ten_comparators/run.py` (`y = returns[512:]`, `nc = int(0.7 * len(y))`, score `q - y`, Shift-CP = `q - qshift(score[:nc])` with k = ceil((nc+1)(1-alpha))).

## Checks and replay

- Negative-control checks in `run.json`: 111, all rejected the defective input and accepted the valid one (wrong-tail loss, rank rule against the v3 rank at n and 4n/5, block length, empirical vs conformal rank, empirical integral against the v3 path, dropped (K-1)/K factor, circular block indices, shrinkage clip and zero rule, band rule centring, finite saved scores, 999x500 bootstrap rows, and per (law, n, b) replay of C_n, I_n(C_n) against estimators.csv (abs 1e-12, rel 1e-10) and of R(C_n) - R(0) against loss_histories.csv (abs and rel 4e-11)).
- Fresh-process replay `engine.py --check`: **PASS**; max absolute difference over all saved CSV columns 1.78e-15; admission.json pass/fail identical.
- Input binding: SHA-256 and mtime of the eight synthetic inputs equal the v3 lock records (`run.json`, field `inputs`); v3 lock entries changed since the v3 run: ['source/sections_r8/risk.tex', 'source/sections_r8/risk_proofs.tex'] (manuscript files, not inputs here).

## Runtime, seeds, environment

- Runtime: E2 3.49 s, E1 and evaluation 0.45 s, total 4.13 s with 16 worker processes; replay 20000 history rows.
- Seeds: E2 block bootstrap seed 20260913, stream numpy default_rng([20260913, law_index, n, history]); history bootstrap: saved results/theory_loop/synthetic/history_bootstrap_indices.npy (999 x 500).
- Constants: K = 5, R = 200, block lengths {'250': 7, '500': 8, '700': 9, '1000': 10, '2000': 13}, p = 0.99, Omega = 0.0099.
- Environment: Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, macOS-26.6.2-arm64-arm-64bit-Mach-O, 18 cores; executable /private/tmp/irfa-r8-conda-clean/bin/python.

## Files

- `histories.csv`: 20,000 rows (estimator, law, n, b, history): C_n, I_n, O_hat, ratio, A_hat, B_hat, lambda_hat, expected losses.
- `summary.csv`, `simultaneous_bands.csv`, `supplementary_loss_bands.csv`, `bootstrap_maxima.csv`, `e2_bootstrap_diagnostics.csv`, `oracle_lambda.csv`.
- `admission.json`: every criterion cell with pass/fail. `run.json`: seeds, constants, environment, timing, input hashes, checks, output hashes. `check.json`: replay result.
