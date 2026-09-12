# Matched second-order dependence, different calibration-count laws

10 September 2026. Fixed before computing the outputs below. This is a
deterministic mathematical diagnostic, with no model fits or sampled paths.

Construct a stationary regenerative sequence with independent block lengths
one (probability 1-theta) and three (probability theta). A singleton is an
independent Uniform(0,1). A triple is (U,V,(U+V) mod 1), with independent
uniform U,V. Use the equilibrium initial phase. Every pair of observations
is independent, but a whole triple need not be. Transform all observations
by the Normal inverse CDF. The resulting scores have the same Normal
marginal, all pairwise independence, and therefore the same long-run tail
indicator variance as iid Normal scores. Theta<1 gives geometric mixing.

Compute the exact threshold-count law through a renewal recursion. Integrate
the resulting conformal and empirical order-statistic regrets against the
Normal marginal. Use n=125,250,500,1000; alpha=0.01,0.05; theta=0,0.5,0.9;
the exact ranks ceil((n+1)(1-alpha)) and ceil(n(1-alpha)). Retain all 48
configurations. Compare 256/512 quadrature nodes on each half at the target
and at the copula breakpoint 0.5. Truncate the Normal integral at 12 standard
deviations and record the same union-bound tail remainder as the previous
deterministic frontier calculation.

Verify the iid beta distribution, the exact three-observation mixture,
and exact binomial first two count moments at all thresholds on a small
full-count grid. Independently integrate the triple copula probability.
Compare risk values with the archived iid calculation. Report the entire
grid, differences from iid cost, and the resulting positive-bias break-even
points. No parameter or displayed cell may be chosen to maximise an empirical
panel result. These are population mathematical comparisons, not estimated
deployment cutoffs or a new evaluation of any financial forecast.
