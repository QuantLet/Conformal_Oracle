# The rolling calibration window — results

Run 2026-08-29 against `PREREG_W_SWEEP.md`. All **312 of 312** cells produce a
defined shift on all three windows, compared on the intersection of dates, so
the falsification condition does not fire.

## The prediction half-holds, and the reason it half-holds is the result

The pre-registration expected $w = 125$ to separate from the other two on three
counts. Two hold and the third does not, and the third is the informative one.

| prediction | outcome |
|---|---|
| the shift is larger at $w = 125$ on most cells | **holds**, 281 of 312 |
| the corrected rate sits further below nominal | **weak**, 169 of 312, barely a majority |
| the dispersion exceeds what halving the window alone gives | **holds**, 1.541 against $\sqrt 2 = 1.414$ |

But the same dispersion test applied to the *other* step is worse:
$\mathrm{sd}(250)/\mathrm{sd}(500) = 1.959$, against the same $\sqrt 2$. The
excess over square-root scaling is therefore **not specific to $w = 125$**, and
the implied exponent in $\mathrm{sd} \propto w^{-b}$ is 0.624 between 125 and 250
and 0.970 between 250 and 500. A variance curve would return one exponent, and
0.5 at that. It returns neither.

## What actually separates the three windows

They do not estimate the same quantity. Equation~(8) puts the index at
$k = \lceil (w+1)(1-\alpha)\rceil$, so the **effective level $k/w$ depends on
$w$**:

| $w$ | $k$ | effective level $k/w$ | overshoot $k/w - (1-\alpha)$ | median shift | median sd | median $\hat\pi$ |
|---|---|---|---|---|---|---|
| 125 | 125 | **1.0000** | 0.0100 | 0.010851 | 0.016994 | 0.00920 |
| 250 | 249 | 0.9960 | 0.0060 | 0.008570 | 0.010667 | 0.00932 |
| 500 | 496 | 0.9920 | 0.0020 | 0.005113 | 0.004952 | 0.01144 |

The overshoot falls by a factor of five across the sweep, and the median shift
and the corrected rate both move monotonically with it: the shift shrinks as the
target level falls toward nominal, and $\hat\pi$ rises toward it from below. A
comparison across $w$ therefore confounds window length with target level, and
the confound is the larger of the two effects.

This is the same quantity Supplement S.4.1 already reports for the Monte Carlo,
where $n_{\mathrm{cal}} = 350$ gives an overshoot of 0.004286. It is a second
instance of one mechanism, not a new one.

## What Section 7 says, and it is stronger than the ablation asked for

R2-1 asked for an ablation over $w \in \{125, 250, 500\}$ read as a variance
curve. The answer is that those three points **do not estimate the same
quantity**, so the curve does not exist to be read:

- at $w = 125$ the effective level is exactly 1 and the shift **is the window
  maximum**, an extreme-value statistic, because $k \ge w$ whenever
  $w < 2/\alpha - 1$, which at $\alpha = 0.01$ is $w \le 198$;
- at $w = 250$ and $w = 500$ it is the second and fifth largest of the window,
  targeting 0.996 and 0.992 against a nominal 0.99;
- the dispersion follows no single power in $w$, so no point on the sweep is a
  point on a variance curve.

The degenerate estimator at $w = 125$ is the extreme case of a confound present
at all three, which is a better answer than the one anticipated in the rebuild
notes --- there the claim was that one of the three points is a different
estimator, and the measurement says all three target different levels.

## What this does not license

The sweep says nothing about which window a practitioner should choose. Choosing
by corrected coverage would pick $w = 500$, whose $\hat\pi$ is closest to nominal
here, but that is the point with the smallest overshoot rather than the point
with the best estimator, and the comparison is confounded exactly as described.
A window recommendation would need the effective level held fixed across $w$,
which equation~(8) does not permit, and it is not made.
