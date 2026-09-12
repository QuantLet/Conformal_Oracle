# Authorised shape-cost experiment

Fixed on 11 September 2026, after the author said “go” to the three-specialist
proposal, before generating its new calibration histories or viewing its
new loss comparisons. This is a development-motivated controlled simulation,
not an untouched external financial evaluation. The financial companion has
its own locked protocol and uses already inspected stored forecasts.

## Target

Compare the expected original-return pinball loss of a volatility-scaled ERM
correction with that of a constant ERM correction. Each has one fitted
coefficient and minimises the same calibration objective. Coefficients remain
fixed over the entire contiguous test horizon. All twelve cells are retained.

The score is sigma*(z_alpha-epsilon+h/sqrt(n)), where epsilon has mean zero,
variance one, and either a Normal or Student-t(5) law. The scale is an exogenous
stationary Markov chain on {1,2}, independent of innovations. Its initial state
is uniform and its probability of staying in either state is .95. The
transition eigenvalue is .90. The current scale is known before the innovation.

Alpha=.01; n in {250,1000,4000}; H=floor(3n/7). Put g=f_epsilon(z_alpha),
h_star=sqrt(2*.01*.99)/g and h in {.5*h_star,2*h_star}. Known scale moments
are A=3/2, B=5/3, C=4/3. The leading Vol-ERM-minus-Shift-ERM prediction is

    [.01*.99*(B-C)/g - g*h*h*(A-C)]/(2*n).

The sole primary method comparison covers these twelve cells. The two
registered primary conditions use Normal innovations and n=1000. The low-h
condition predicts a positive loss difference; the high-h condition predicts
a negative one. The other ten cells are fixed length/law sensitivities.

## Estimation and evaluation

Shift-ERM is the leftmost empirical .99 quantile of raw scores; Vol-ERM is the
leftmost sigma-weighted .99 quantile of standardised scores. Compare integer
cumulative weights exactly as 100*cumulative >= 99*total. Empirical ranks are
ceil(99*n/100); conformal ranks are ceil(99*(n+1)/100). All specified ranks
are admissible. No interpolation, random tie breaking, clipping of fitted
coefficients or outcome-dependent fallback is allowed.

Save Raw and three supplementary estimators: unweighted standardised ERM
(Vol-UERM), standardised conformal rank (Vol-CP), and raw-score conformal rank
(Shift-CP). Their differences against Shift-ERM form a SEPARATE 36-contrast
sensitivity family. They do not replace the primary comparison. Their
leading expansions distinguish the scaled empirical weighting penalty B-A
from correction form. Raw losses and method-wise hits are descriptive.

At each future date integrate pinball loss analytically over the innovation
law. Weight states by the exact Markov transition probabilities conditional
on the final calibration state, averaging dates1..H. Save an independent
marginal-state calculation separately. Do not replace the contiguous target
with marginal evaluation. Save conditional violation probabilities, both
state losses, coefficients, population constant optimum and all cell metadata.

## Randomness and precision

Use exactly 5,000 independent history indices0..4999, each with independent
scale/innovation streams. SeedSequence inputs are [20260911,3101,index] and
[20260911,3102,index]. Generate length4000, using prefixes at shorter lengths.
Transform the same innovation uniforms to Normal and variance-one t(5) for
paired comparisons. A representable zero uniform is replaced with the nearest
interior double; endpoints are otherwise unchanged. Save uniforms and states.

Use 9,999 whole-history bootstrap draws from default_rng(2026091131), each
sampling 5,000 indices with replacement. Reuse those indices for every cell
and both families; no day-level or independent-cell resampling. In each family
take the .95 quantile (higher convention) of the maximum absolute centred
bootstrap mean divided by the original Monte Carlo standard error. Intervals
are original mean +/- critical*standard error. These are approximate
simultaneous Monte Carlo intervals, not financial confidence intervals.

The primary crossing is supported only if the low-h simultaneous lower bound
is positive AND the high-h simultaneous upper bound is negative. An interval
wholly on the opposite side contradicts that finite-n prediction. Otherwise
the conclusion is unresolved. A finite-n failure does not refute the
asymptotic theorem. Report the observed-minus-predicted discrepancy and its
n-scaled magnitude even if signs agree. No post-result equivalence tolerance,
precision extension, cell replacement or new significance family is allowed.

## Checks and preservation

Before the run, check analytic loss against independent quadrature, exact
weighted minimiser/tie conventions, and future occupancy against transition
matrix powers. Bind protocol, design, engine, run and aggregation sources
by hashes in lock.json. Source corrections required by a failed check are
documented; do not silently redefine the protocol after outcomes.

After the run, an independent validator checks saved histories, parameters,
conditional losses, paired inference and all decisions. Preserve the known
two adverse earlier quadratic-approximation examples. No base-model fits,
TSFM inference, new market downloads or changes to existing financial results
are part of this study. Current canonical files are bound before execution.
