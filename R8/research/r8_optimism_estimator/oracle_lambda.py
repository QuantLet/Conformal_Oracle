"""Supplementary diagnostic (not a protocol criterion): expected loss of the oracle factor
lambda* = B/(B+A0) with the true B = R(0)-R(q) and A0 = Omega/(2 n f*), per cell, from the
reference risk. Separates the shrinkage rule's own value from the estimation error of lambda_hat.
Reads only stored inputs and artifacts/r8_optimism_estimator/histories.csv; writes oracle_lambda.csv.
"""
import numpy as np
import pandas as pd
from engine import BIASES, LAWS, OMEGA, OUT, SIZES, SYN, expected_loss, reference_risk

truth = pd.read_csv(SYN / 'truth.csv').set_index('law')
hist = pd.read_csv(OUT / 'histories.csv')
hist = hist[hist.estimator == 'E1_blocked_cv']  # C_shifted, loss_raw, loss_full are estimator-independent
rows = []
for law in LAWS:
    poly, domain = reference_risk(law)
    f_ref = float(truth.loc[law, 'f_true'])
    for n in SIZES:
        A0 = OMEGA / (2 * n * f_ref)
        for b in BIASES:
            g = hist[(hist.law == law) & (hist.n == n) & np.isclose(hist.bias, b, atol=1e-15, rtol=0)].sort_values('rep')
            assert len(g) == 500
            B = float(poly(b) - poly(0.))  # R(0)-R(q) on shifted scores: q_shifted = b
            lam = B / (B + A0) if B + A0 > 0 else 0.
            lam = min(max(lam, 0.), 1.)
            cs = g.C_shifted.to_numpy()
            loss_oracle = expected_loss(poly, domain, b, cs, lam)
            loss_full = expected_loss(poly, domain, b, cs, 1.)
            assert np.allclose(loss_full, g.loss_full, atol=1e-15)
            grid = np.linspace(0, 1, 1001)
            curve = np.array([expected_loss(poly, domain, b, cs, l).mean() for l in grid])
            rows.append(dict(law=law, n=n, bias=b, B_true=B, A0=A0, lambda_star=lam, loss_raw=0., loss_full=loss_full.mean(),
                             loss_oracle_lambda=loss_oracle.mean(), oracle_minus_full=loss_oracle.mean() - loss_full.mean(),
                             oracle_minus_full_mcse=(loss_oracle - loss_full).std(ddof=1) / np.sqrt(500),
                             best_fixed_lambda_on_grid=float(grid[curve.argmin()]), best_fixed_lambda_loss=float(curve.min())))
table = pd.DataFrame(rows)
table.to_csv(OUT / 'oracle_lambda.csv', index=False)
pd.set_option('display.width', 250)
print(table.to_string(float_format=lambda v: f'{v:.4g}'))
