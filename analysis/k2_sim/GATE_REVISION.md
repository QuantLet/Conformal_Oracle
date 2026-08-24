# The two gates failed. What was wrong, and what was changed

Both gates in `PREREGISTRATION.md` fired. Neither was loosened until its cause
was identified, and neither cause was the extension.

---

## Gate A — `Std_qV` outside 3 SE on 2 of 10 cells

**Declared tolerance:** each cell within 3 Monte Carlo standard errors, with the
SE of a standard deviation taken as $\mathrm{sd}/\sqrt{2N}$.

**That formula assumes a normal replication distribution.** $\hat q_V$ is a 1%
order statistic of heavy-tailed scores and its replication distribution is
right-skewed. Measured on the committed replications:

| cell | skew | excess kurtosis | declared SE | bootstrap SE | ratio |
|---|---|---|---|---|---|
| normal, $T=1000$ | +0.53 | +0.70 | 6.1e-05 | 7.0e-05 | 1.15× |
| $t_5$, $T=5000$ | +0.66 | +1.22 | 5.3e-05 | 6.7e-05 | 1.27× |
| skewed-$t_3$, $T=1000$ | +1.51 | +4.16 | 2.49e-04 | 4.38e-04 | **1.76×** |

The declared SE understates the true sampling variability of the statistic by up
to 1.76×, worst exactly where the replication distribution is most skewed — which
is the skewed-$t$ cell, one of the two that failed.

**Revision.** The SE for `Std_qV` is taken from a 4,000-draw bootstrap of the
committed replications rather than from the normal formula. Every other tolerance
is unchanged. Under the corrected gate **all 10 cells reproduce, on all six
quantities**; the two failures were 0.000160 against a corrected threshold of
0.000205, and 0.001082 against 0.001331.

**What the corrected gate still cannot see** (PROTOCOL Rule 2): a systematic bias
below 3 bootstrap SE — up to 2.1e-04 on mean $\hat q_V$ for the Normal cell and
1.3e-03 on `Std_qV` for the skewed-$t$ cell. At 500 replications this check is not
evidence that the two implementations agree exactly, only that they agree to
Monte Carlo resolution. The exact comparison is Gate 1, below, which carries no
Monte Carlo at all.

---

## Gate B — the negative control failed at $T = 1{,}000$

A forecaster given the *correct* innovation law returned mean $\hat q_V$ of
+0.000806 ($t_5$) and +0.001161 ($t_3$) at $T = 1{,}000$, against a 3 SE band of
0.00048 and 0.00065. At $T = 5{,}000$ both were indistinguishable from zero.

**This is not a harness fault. It is the conformal convention.** The index
$k = \lceil (n+1)(1-\alpha)\rceil$ sits above the $(1-\alpha)$ sample percentile
by $k/n - (1-\alpha)$, and that overshoot shrinks with $n$:

| $T$ | $n$ | $k$ | $k/n$ | overshoot |
|---|---|---|---|---|
| 1,000 | 700 | 694 | 0.991429 | 0.001429 |
| 5,000 | 3,500 | 3,466 | 0.990286 | 0.000286 |

Predicted bias ratio 5.00×. Observed 6.1× ($t_5$) and 6.6× ($t_3$) — same order,
same direction, and vanishing in $n$ as predicted.

**Confirmed by removing the convention.** Re-running the same control with the
plain empirical quantile at exactly $1-\alpha$:

| DGP | $T$ | exact $(1-\alpha)$ quantile | conformal, eq. (8) |
|---|---|---|---|
| $t_5$ | 1,000 | −0.000395 (3 SE 0.000438) — **zero** | +0.000806 — nonzero |
| $t_5$ | 5,000 | −0.000117 (3 SE 0.000215) — **zero** | +0.000132 — zero |
| $t_3$ | 1,000 | −0.000531 (3 SE 0.000553) — **zero** | +0.001161 — nonzero |
| $t_3$ | 5,000 | −0.000131 (3 SE 0.000236) — **zero** | +0.000177 — zero |

The harness returns zero for a correctly specified forecaster at every sample
size, as the control demands. The conformal estimator does not, at small $n$.

**Revision.** The control is restated to test what it was meant to test — the
harness — using the exact $(1-\alpha)$ quantile. What it detected instead is
promoted from a failed control to a **§5 result**:

> The one-parameter conformal correction is **upward-biased at small calibration
> samples even when the forecaster is correctly specified**, by construction and
> not by misspecification. The bias is $O(1/n)$ — it is the gap between
> $\lceil (n+1)(1-\alpha)\rceil/n$ and $1-\alpha$ — and at $\alpha = 0.01$ with
> $n = 700$ it is large enough to be detected against 500 replications.

That is the finite-sample cost of the finite-sample guarantee, and it is the
mechanism behind the panel result that correcting an already-calibrated pair
degrades it (§7, 66 of 312 pairs static). It also sharpens the answer to R2-1
(K4b): the calibration-window length $w$ enters the bias through $k/w$, not only
the variance.

---

## Gate 1 — the convention, checked exactly, with no Monte Carlo

The committed simulation script computes $\hat q_V$ as
`np.quantile(s, ceil((n+1)(1-alpha))/n)` with linear interpolation. That is a
**third convention**, distinct from both equation (8) and the plain empirical
quantile: at $n = 700$, $k = 694$, it takes 0-based index 693.009 and interpolates,
where equation (8) takes index 693 exactly.

| $T$ | $n$ | $k$ | eq. (8) index | committed index | gap |
|---|---|---|---|---|---|
| 1,000 | 700 | 694 | 693 | 693.009 | 2.8e-06 (0.01%) |
| 5,000 | 3,500 | 3,466 | 3,465 | 3,465.010 | 2.4e-06 (0.01%) |

Numerically negligible. But the manuscript states that "every static and rolling
result in this paper uses equation (8)", and the simulation does not. The grid is
run under equation (8); the reproduction of the committed cells is run under the
committed convention, so that the comparison isolates the implementation and not
the convention. Logged for §3, alongside the other two variants
(`prop:rolling_drift` and `lem:quantile_mixing` in the supplement, both of which
write $Q_{1-\alpha}$).

---

## One non-finding, recorded so it is not re-opened

The DGP-5 label differs between artefacts: the published table says
$0.95\mathcal N(0,1)+0.05\mathcal N(0,25)$ and the superseded April CSV says
$0.05\mathcal N(0,5)$. The code draws `np.random.normal(0, 5)`, whose second
argument is the standard deviation. Variance 25, standard deviation 5. The two
labels describe the same law under different conventions. No defect.

---

## Second revision, after the corrected gates were run

Two further defects, both mine, both found by the gates rather than by inspection
of the results.

**1. The seeds were not reproducible.** `np.random.default_rng(abs(hash((kind, T))))`
uses Python's string `hash()`, which is salted per interpreter process. Two runs
of the same script drew different Monte Carlo samples, and the set of cells
flagged changed between them. Replaced by `zlib.crc32` of the joined key, which is
deterministic across processes. This is the defect the project exists to avoid,
in the harness written to check the project.

**2. The declared SEs were for one estimate, not for a difference.** The gate
compares two *independent* Monte Carlo estimates of the same population quantity —
the committed 500-replication run and this one. The variance of the difference is
the sum of the two variances, so every declared tolerance was short by a factor of
$\sqrt2$. Identified from the estimand, not from the outcome.

Rather than widen the tolerance, the replication count for the reproduction gate
was raised to **2,000** — four times the committed run — so that the committed
run's error dominates and the factor is $\sqrt{1 + 500/2000} = 1.12$ rather than
$\sqrt2$. The Green-percentage SE was also corrected from the maximal-variance
$p = 0.5$ form to the observed $p$.

**Under the corrected gate all 10 cells reproduce on all six quantities.**

---

## The negative control's residual failure, reported not repaired

With the harness tested at the exact $(1-\alpha)$ quantile, the control passes at
9 of 10 cells. It fails at $t_5$, $T = 500$: mean $\hat q_V = -0.000663$ against a
3 SE band of 0.000605. The $t_3$ cell at the same size is marginal
($-0.000705$ against $0.000757$).

The sign is systematic, not random: **all ten cells return a negative mean**, and
the magnitude falls monotonically in $T$ — $-0.00066, -0.00040, -0.00016, -0.00012,
-0.000035$ for $t_5$. This is the interpolated empirical quantile's own
small-sample bias in a right-skewed tail, and it runs *opposite* to the conformal
convention's overshoot.

At $\alpha = 0.01$ with $n_{\mathrm{cal}} = 350$, neither convention is unbiased.
The consequence for the grid is stated rather than removed: **the $T = 500$ row
carries a small-sample bias in both directions and is reported with that caveat.**
From $T = 1{,}000$ ($n_{\mathrm{cal}} = 700$) upward the control passes on every
cell.
