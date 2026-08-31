# The reading, fixed before the census runs

## What this document's timestamp is worth

It is worth less than the one behind the scale bands. There the declaration is
dated 22 August and the estimator family it was tested on ran on 23 August, so
the ordering survives without trusting the author. Here the declaration, the
script and the output are 21 seconds apart inside one commit. That establishes
the order in which three files were written in one working session and nothing
stronger; a reader who does not take the author's word has no independent
evidence, and the paper accordingly says the readings were fixed before the run
and does not call them pre-registered, which is a term with a technical meaning
a referee is entitled to test.

What the document does carry is the second reading, written out in full below.
The claim is not that a rule existed, but that the rule had a branch on which
nothing would have been written, and that branch is on the page.

Written before the census runs. Corollary 4.6 sets $g_n = \lceil c\log n\rceil$
with $c = 1/\lvert\log\hat\rho\rvert$, $\hat\rho$ the lag-1 autocorrelation of
the calibration scores. When $\hat\rho \le 0$ the constant is undefined and the
implementation falls back to a floor. The paper currently states no reason for
where that fallback lands, and a fallback with no stated pattern reads as an
implementation detail escaping into a theorem's neighbourhood.

## Object

`analysis/convention/rho_census.csv`, one row per (model, asset) at
$\alpha = 0.01$, written by `measure_rho_census.py`, which computes $\hat\rho$
from the calibration scores of the pipeline's own `load_pair` --- the same
function that produced `all_results.csv`.

## Grouping

`truncated` is `model in {"Chronos-Small", "Chronos-Mini"}`, the definition
already used by `analysis/k0a_mcb/run_k0a.py:22` and
`analysis/detection/run_detection.py`, where the two series carry the label
`top_k_truncation`. The grouping is not chosen after seeing the census.

## The two readings, written now

**If the $\hat\rho \le 0$ cells concentrate on the truncated series** --- the
mechanism is the truncation itself. `top_k = 50` collapses the predictive law
onto its 50 most probable atoms, leaving a lower quantile that barely moves from
day to day while the return does. The score $S_t = \hat q^{lo}_t - r_t$ is then
dominated by $-r_t$ and inherits the return's lack of persistence, so $\hat\rho$
lands at or below zero. On that reading the degeneracy is not the gap failing;
it is the gap correctly reporting that those series have no dependence to
separate, and the floor gives them more separation than the corollary asks for.

**If the $\hat\rho \le 0$ cells are spread across forecasters** --- there is no
mechanism to state. The sentence is not written, the fallback is reported as a
generic degeneracy affecting $k$ of 312 cells with no forecaster pattern, and
the paper says so rather than leaving it unremarked.

## Decision rule

Let $p$ be the share of $\hat\rho \le 0$ cells on truncated series and $p_0$
their share of the panel ($2/13 = 0.154$). Concentration is claimed only if
$p / p_0 \ge 2$ **and** a Fisher exact test of the $2\times2$ table rejects at
5%. Both must hold; either alone is not enough.

## Negative control

Scores replaced by an AR(1) sequence with $\rho = 0.6$ must yield
$\hat\rho > 0$ and a gap above the floor; scores replaced by i.i.d. draws must
yield the fallback. If the census cannot distinguish these two, it is not
measuring $\hat\rho$ and reports nothing.
