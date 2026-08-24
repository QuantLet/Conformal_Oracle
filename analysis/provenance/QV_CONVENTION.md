# The conformal shift: four conventions, one definition, and a check

The shift is defined once, by equation (8): $S_{(k)}$ with
$k = \lceil (n+1)(1-\alpha)\rceil$. The manuscript asserted in two places that
every result uses it. Both assertions were false, and this is the fourth time a
variant of this convention has been found in this project — R1 in the retraction
register was the first, and it was recorded as a DEFINITION_MISMATCH and closed.
It reappeared because the convention was documented in a docstring and
implemented nowhere.

## What the scan found

`scripts/audit_qv_convention.py` enumerates every site in the repository that
takes a quantile of nonconformity scores. There were **52**, in four conventions:

| convention | sites | what it is |
|---|---|---|
| `ORDER_STATISTIC` | 23 | `np.sort(s)[k-1]`, equation (8) |
| `NOT_THE_SHIFT` | 13 | a quantile of something else — ACI's adaptive level, bootstrap CI bounds, the forecast's own quantile |
| `LEVEL_K_OVER_N` | 8 | `np.quantile(s, k/n)`, which interpolates **above** $S_{(k)}$ |
| `PLAIN_QUANTILE` | 5 | `np.quantile(s, 1-\alpha)`, which interpolates **below** it |
| `CANONICAL` | 3 | the definitions themselves |

## What each costs, measured on the panel

312 pairs, calibration samples of 1,617 to 4,587, against equation (8):

| convention | median relative gap in $\hat q_V$ | max | pairs whose violation count changes | cell-mean corrected $\hat\pi$ |
|---|---|---|---|---|
| equation (8) | — | — | — | 0.010617 |
| `LEVEL_K_OVER_N` | 3.1e-04 | 3.4e-01 | **0 of 312** | 0.010617 |
| `PLAIN_QUANTILE` | 5.1e-02 | 8.9e+00 | **120 of 312** | 0.010989 |

`LEVEL_K_OVER_N` is invisible at every printed precision and changes no
classification anywhere. `PLAIN_QUANTILE` is a different estimator: it moves the
shift by a median 5% and changes 120 violation counts by up to 4.

## What was done about it

**Migrated to `cfp_config.conformal_quantile`:**

- `scripts/gap_ablation.py` — Table S.10. Re-run. **The two printed conclusions
  are identical under both conventions**: max $|\Delta\hat\pi|$ is 0.0005 over the
  full sample and 0.0058 within COVID, before and after. The ablation's argument
  never depended on the convention; the intermediate $\hat q_V$ did, by up to 919%
  on the short crisis sub-windows, because at $n_{\mathrm{cal}}\approx70$ the
  conformal index exceeds $n$ and equation (8) returns the sample maximum.
- `Quantlets/CO_robustness/run_robustness_mc.py` — Tables S.5–S.8, both the
  static and rolling arms. Re-run.
- `python/src/conformal_oracle/conformal/bootstrap.py` — Table S.9. Through
  version 0.3.1 of the published package the bootstrap replicates used the plain
  empirical quantile while the point estimate used the order statistic, so the
  interval was centred on a different estimator from the value it was reported
  around. Corrected.

**Declared and left, with the measurement:** the eight `LEVEL_K_OVER_N` sites.
Two of them are deliberate (the convention comparison and the Monte Carlo
reproduction gate). The other six produce Tables S.1, S.20, S.23, the
identification exhibit and the superseded Table S.26, and the measurement above
shows they are indistinguishable from equation (8) on this panel.

## A property of equation (8) that the scan surfaced

$k \ge n$ whenever $n < 2/\alpha - 1$. At $\alpha = 0.01$ that is
$n \le 198$: **for any calibration sample of 198 observations or fewer the
conformal shift is the sample maximum.** Verified against the closed form at
$\alpha \in \{0.01, 0.025, 0.05, 0.10\}$, where the bounds are $n \le 198, 78,
38, 18$.

A short calibration window therefore does not give a noisy estimate of the
conformal quantile — it gives an extreme-value statistic with no stable variance.
This is the exact answer to R2-1's request for $w \in \{125, 250, 500\}$: at
$w = 125$ the rolling shift **is** the window maximum, at $w = 250$ it is the
second largest of 250, and at $w = 500$ the fifth largest of 500. The three
windows are not three points on a variance curve; the first is a different
estimator.

## The check

`scripts/audit_qv_convention.py` fails the build when a site takes a quantile of
nonconformity scores and is neither a call to `cfp_config.conformal_quantile` nor
an entry in `QV_CONVENTION_SITES.tsv`. It also fails when a registered site's
code changes, so a silent change of convention at an existing site is caught too.

Negative controls, both run: a synthetic registry key is confirmed absent, and a
planted file computing `np.quantile(cal_scores, 1 - alpha, method="nearest")` was
flagged and the audit exited 1; removing it returned the audit to 0.
