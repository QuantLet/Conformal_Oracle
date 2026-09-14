# Stepwise multiple testing on the stored eight-comparison draws

Fixed 14 September 2026 before computation, in response to the statistics
referee's point that a single-step max-|z| band is not a test of whether any
correction differs from Shift-CP. Declared exploratory: the eight bands have
been inspected. No new forecasts, fits, corrections or bootstrap draws.

## Inputs (read-only)

`artifacts/r8_referee_revision/results/bootstrap_{20,60}.npz` (999 draws of the
pair-equal mean QS of 20 methods on the common calendar, with the point
estimates and method names) and `intervals.csv` (published bands, used as a
check). `artifacts/r8_ten_comparators/results/bootstrap_{20,60}.npz` for the
six-member family's critical value.

## Family

The eight published comparisons with Shift-CP: Raw, Vol-ERM, State-L1,
POT-Shift, POT-Vol, DtACI-projected-expected, Loss-gate, Past-minimum.
Statistic $\hat\theta_j$ = mean QS of method $j$ minus mean QS of Shift-CP
(x 10^4); bootstrap draws $\theta^*_{bj}$; $s_j$ = standard deviation of the
draws (ddof 1); $t_j=\hat\theta_j/s_j$; centred draws $t^*_{bj}=(\theta^*_{bj}-\hat\theta_j)/s_j$.

## Check

The single-step critical value, the 95th percentile (linear interpolation) of
$\max_j|t^*_{bj}|$, and the resulting bands must reproduce `intervals.csv` to
1e-9 at both block lengths.

## Procedure (Romano and Wolf, 2005, stepdown with max-|t|)

Order hypotheses by $|t_j|$ descending. At step $k$ with remaining set $R_k$,
$c_k$ = 95th percentile of $\max_{j\in R_k}|t^*_{bj}|$; reject every $j\in R_k$
with $|t_j|>c_k$; remove them and repeat until no rejection. Level 5%, two-sided
$H_{0j}:\theta_j=0$. Report for each comparison: $t_j$, the step at which it was
rejected (or none), the sign (negative favours the method, positive favours
Shift-CP), and the final critical value.

## Model confidence set (Hansen, Lunde and Nason, 2011; range statistic)

Over the nine methods (Shift-CP and the eight), with the same draws:
eliminate the method with the largest pairwise |t| range when the max
pairwise-|t| statistic exceeds its bootstrap 95th percentile; report the set
that survives at 95%. This uses the stored draws only.

## Outputs

`artifacts/r8_romano_wolf/{results.csv, mcs.csv, critical_values.csv, run.json, RESULTS.md}`;
`displays.py` writes `numbers_rw.tex` (prefix `\nRW`) and `tab_rw.tex`; `--check`
compares regenerated displays with disk.
