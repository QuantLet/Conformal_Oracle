# Native-tail panel: author decision, 10 September 2026

The author requested that models requiring extrapolation of central output
quantiles be removed. This is a retrospective restriction by predictive
interface, after the original nine-model results were known. It is not a
prospectively registered selection or an independent validation sample.

Remove TimesFM-2.5 and Moirai-2.0 from the current empirical panel: their
stored 1% forecasts use Student-t completion of nine native deciles.
Retain Moirai-1.1, Lag-Llama, GJR-GARCH, GJR-GARCH-t, GARCH-N, Historical
Simulation and EWMA: seven forecasters, all 24 assets, 168 pairs. Preserve
all dates, splits, forecasts, fitted corrections, selections and diagnostics.
No pair is screened on coverage, loss or the sign of the conformal shift.

The restriction concerns completion of an absent base-model quantile. It
does not remove models whose predictive law already specifies its tails,
or the explicit POT correction and dedicated VaR benchmarks. Those remain
labelled parametric/tail-model comparisons, not assumption-free forecasts.

Recompute aggregate statistics and the existing 999-draw calendar-block
intervals at both original lengths (20 and 60 days), with the original seeds,
losses, equal-pair weights and simultaneous comparison families. Keep all
24-asset common-support dates unchanged, and verify their intersection is
unchanged after removal. Do not refit corrections or run model inference,
new simulated paths, or select alternatives by the resulting rankings.

Store derived results separately from the nine-model archive. Regenerate
current manuscript/supplement tables and figures from those results. Remove
the grid-completion subsection and its fitting specification. Preserve
the theoretical statements and proofs; change only the empirical panel
denominator in the gap illustration. Historical closure evidence stays in
the previous archive and is not part of the current manuscript panel.

The author subsequently excluded half-normal tail constructions as well.
Do not replace such tails manually to make a checkpoint eligible: that
would define a different predictive model. TabPFN variants using the
FullSupportBarDistribution half-normal construction are therefore excluded
from the planned additions, regardless of whether their ICDF is repaired.
This does not exclude an ordinary full Normal predictive law such as the
classical GARCH-N benchmark, or a native Student-t predictive law.

Future candidates are not yet empirical results. Chronos-2 supplies native
0.01; original Chronos-T5 requires renewed extraction/input QA; Sundial
supplies samples. TabPFN-TS permits requested quantiles, but the inspected
half-normal construction does not meet the author's new criterion.
None is silently inserted into the existing panel by this restriction.
