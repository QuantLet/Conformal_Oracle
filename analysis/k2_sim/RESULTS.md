# K2 §5 — the Monte Carlo convergence grid

5 DGPs × 5 sample sizes × 500 replications = 12,500 replications. GARCH(1,1)
with $\omega = 10^{-5}$, $\alpha_1 = 0.10$, $\beta_1 = 0.85$; the forecaster uses
the true GARCH parameters and assumes Normal innovations, so the only thing that
varies across DGPs is the innovation law. $\alpha = 0.01$, $f_c = 0.70$,
equation (8) throughout. Gates and their two revisions: `GATE_REVISION.md`.

All four pre-registered predictions hold.

## 1. The correction converges to nominal

Corrected $\hat\pi$, mean across replications:

| DGP | $T{=}500$ | 1,000 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|
| Normal (correct) | 0.0076 | 0.0101 | 0.0100 | 0.0100 | 0.0100 |
| Student-$t$(5) | 0.0085 | 0.0100 | 0.0100 | 0.0102 | 0.0098 |
| Student-$t$(3) | 0.0084 | 0.0095 | 0.0099 | 0.0098 | 0.0101 |
| Skewed-$t$(3) | 0.0083 | 0.0104 | 0.0105 | 0.0101 | 0.0101 |
| Mixture | 0.0084 | 0.0094 | 0.0098 | 0.0100 | 0.0099 |

From $T = 1{,}000$ every DGP is within 0.0006 of nominal and the residual falls
with $T$. The $T = 500$ column is uniformly conservative, at 0.0076–0.0085, and
carries the small-sample bias documented in `GATE_REVISION.md`.

## 2. The raw traffic light does not converge — it diverges, and the direction is set by a threshold

Basel green share of raw forecasts:

| DGP | population $\hat\pi$ | $T{=}500$ | 1,000 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|
| Normal (correct) | 0.0100 | 83.4 | 80.6 | 92.6 | 98.8 | **100.0** |
| Student-$t$(5) | 0.0150 | 61.6 | 54.2 | 57.8 | 65.0 | **74.2** |
| Student-$t$(3) | 0.0136 | 69.6 | 61.4 | 67.4 | 81.6 | **85.6** |
| Skewed-$t$(3) | 0.0233 | 36.4 | 18.0 | 10.8 | 3.2 | **0.4** |
| Mixture | 0.0125 | 68.4 | 68.2 | 79.8 | 90.8 | **96.4** |

The green zone is $\hat\pi \le 4/250 = 0.016$, and the green share is *exactly*
the share of replications on the correct side of it — verified cell by cell
against the replication data, 10 of 10 to the decimal. As $T$ grows the sampling
distribution of $\hat\pi$ concentrates on its population value, so the green share
runs to 100% or to 0% according to which side of 0.016 that value sits.

**The Student-$t$(5) row is the result.** A forecaster miscovering by 51% —
$\hat\pi = 0.0150$ against a nominal 0.0100 — is classified green **74.2%** of the
time at $T = 10{,}000$, against 61.6% at $T = 500$. More data does not expose it.
More data makes the traffic light *more confidently green* about it, because its
miscoverage is real but lands inside the zone boundary. Only the skewed-$t$ DGP,
whose miscoverage clears the boundary, is driven out by sample size.

This is the identification argument in a controlled setting: the raw diagnostic
converges to a verdict that is a function of where the true miscoverage sits
relative to a fixed threshold, not to the miscoverage itself. The correction
converges to the miscoverage.

## 3. $\hat q_V$ converges to the population miscalibration, which does not vanish

Mean $\hat q_V$:

| DGP | $T{=}500$ | 1,000 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|
| Normal (correct) | 0.00165 | 0.00030 | 0.00015 | 0.00008 | **0.00003** |
| Student-$t$(5) | 0.00758 | 0.00459 | 0.00405 | 0.00369 | **0.00372** |
| Student-$t$(3) | 0.00868 | 0.00493 | 0.00424 | 0.00365 | **0.00360** |
| Skewed-$t$(3) | 0.02112 | 0.01383 | 0.01249 | 0.01220 | **0.01214** |
| Mixture | 0.01451 | 0.00863 | 0.00727 | 0.00631 | **0.00601** |

Under correct specification $\hat q_V \to 0$; under each misspecification it
converges to a non-zero limit ordered by the severity of the tail departure. The
inflation at small $T$ is the conformal convention's overshoot, $k/n - (1-\alpha)$,
which is 0.0043 at $n_{\mathrm{cal}} = 350$ and 0.00014 at 7,000 — visible in the
Normal row, where there is nothing else for it to be.

## 4. Its variance falls at the parametric rate

sd of $\hat q_V$ across replications, Normal DGP: 0.00307 → 0.00064 over a
twentyfold increase in $T$, a factor of 4.80 against $\sqrt{20} = 4.47$. Under
Lemma S.9.3 the rate carries an extra $\sqrt{\log n}$; over this range the two are
not separable and no attempt is made to separate them.

## Figure

`figures/fig_mc_convergence.{pdf,png}` — three panels: the correction converging
to nominal, the raw traffic light diverging, and the $T^{-1/2}$ variance decline.
