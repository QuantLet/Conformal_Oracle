# K2 §5 — extending the Monte Carlo to a convergence grid. Pre-registration

Written before the extension was run.

## Why

Section 5 of the rebuilt manuscript reports convergence to nominal with $T$. The
existing study has two sample sizes, $T \in \{1{,}000, 5{,}000\}$, which is two
points and not a curve. The extension adds sample sizes; it changes no design
choice, no DGP, and no estimator.

## Status of the existing table, settled first

Table S.26 (`tab_simulation_extended`) **reproduces exactly** from
`Quantlets/CO_simulation_study/simulation_study_results.csv`, 5,000
per-replication rows (5 DGPs × 2 sample sizes × 500 replications), on every
printed value. `cfp_ijf_data/paper_outputs/tables/simulation_results.csv`, dated
6 April, disagrees with the published table on every number and is a
**superseded earlier run** that backs nothing. It is an orphan with a confusable
name and is logged as such.

## Unit of analysis

One row = one replication of one DGP at one sample size.
**Row count.** 5 DGPs × 5 sample sizes × 500 replications = **12,500**.
**What varies from row to row.** The DGP (Normal, $t_5$, $t_3$, skewed-$t(3,-0.5)$,
mixture $0.95\mathcal N(0,1)+0.05\mathcal N(0,25)$), the sample size
$T \in \{500, 1000, 2000, 5000, 10000\}$, and the random seed. Held fixed:
GARCH(1,1) with $\omega = 10^{-5}$, $\alpha_1 = 0.10$, $\beta_1 = 0.85$; the
forecaster uses the *true* GARCH parameters and assumes Normal innovations;
$\alpha = 0.01$; $f_c = 0.70$.
**Range.** Raw $\hat\pi$ expected in $[0.009, 0.035]$; $\hat q_V$ in
$[-0.002, 0.025]$.

## Reimplementation, and the check that has to pass first

The recursion and the estimator are written fresh. **The extension reports
nothing until the reimplementation reproduces the two existing sample sizes.**
The comparison is against the 5,000 committed replication rows, on the six
reported quantities, at each of the ten existing cells.

**Tolerance, declared with what it cannot see** (PROTOCOL Rule 2, "The tolerance
is part of the check"). Two independent 500-replication runs of the same design
differ by Monte Carlo error. The relevant scale is the replication standard
deviation divided by $\sqrt{500}$: for mean $\hat q_V$ under the Normal DGP at
$T = 1{,}000$ that is $0.0019/22.4 = 8.5\times10^{-5}$, and for a Green
percentage it is at most $\sqrt{0.25/500} = 2.2$ percentage points.

- **Tolerance:** each recomputed cell within 3 Monte Carlo standard errors of the
  committed value.
- **The smallest defect that survives it:** a systematic bias of up to 3 SE —
  about $2.5\times10^{-4}$ on mean $\hat q_V$, about 6.7 points on a Green
  percentage. A defect smaller than that is invisible to this check at 500
  replications, and this check is therefore not evidence that the two
  implementations agree exactly.
- **Is the compared statistic one the plausible failure modes move?** Yes for
  an error in the score, the split or the quantile convention, all of which shift
  $\hat q_V$ by far more than 3 SE. **No** for an off-by-one in the order
  statistic, which is $O(1/n)$ and below the floor — so that specific failure mode
  is checked separately and exactly, by comparing the conformal index
  $\lceil (n+1)(1-\alpha)\rceil$ against the committed script's, on a fixed
  score vector, with no Monte Carlo in the comparison at all.

If the reproduction fails, the extension is abandoned and §5 is written on the
two existing sample sizes.

## What is expected, written in advance

1. Corrected $\hat\pi \to 0.010$ from below or above, monotonically in $T$, for
   all five DGPs. This is the section's claim and the reason for the grid.
2. $\mathrm{sd}(\hat q_V)$ falls at rate $T^{-1/2}$. Under
   Lemma S.9.3 the rate carries an extra $\sqrt{\log n}$; over a twentyfold range
   of $T$ the two are not separable, and no attempt is made to separate them.
3. Mean $\hat q_V$ converges to a **non-zero** limit for DGPs 2–5 and to zero for
   DGP 1. The limit is the population miscalibration, and it does not vanish with
   more data — that is the point of reporting it as a diagnostic.
4. **Raw Green percentage diverges rather than converging**, in a direction set by
   whether the DGP's population $\hat\pi$ is above or below the Basel green
   threshold $4/250 = 0.016$. Predicted before running: DGPs 1, 3, 5
   ($\hat\pi \approx 0.010, 0.014, 0.013$) rise toward 100%; DGP 2
   ($\hat\pi \approx 0.015$) rises toward 100% more slowly; DGP 4
   ($\hat\pi \approx 0.023$) falls toward 0%. If this holds, it is the section's
   result and not a by-product: **more data makes the Basel light more confident,
   including more confidently green about a forecaster miscovering by 50%.**

## Negative control

The estimator is run on a forecaster given the *correct* innovation law for each
DGP. Mean $\hat q_V$ must then be indistinguishable from zero at every $T$, and
the raw and corrected coverages must coincide. If a misspecification-free
forecaster shows a non-zero shift, the harness is wrong and nothing from it is
reported.
