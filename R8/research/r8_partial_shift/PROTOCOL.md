# Partial correction and the cost of choosing its strength

10 September 2026. Declared before computing this extension, after the
author approved the proposed partial-correction study. This is development
on previously inspected simulation histories. It is not an untouched
financial test or a new claim of conformal coverage.

## Fixed inputs and full design

Use the existing `artifacts/r8_mechanism/paths.npz`, verified against its
stored checksum. Reuse every one of the 500 latent AR histories and both
sets of 500 GARCH histories. Use exactly the previous 144 configurations:
Normal/t(5) margins, AR coefficients 0/.5/.8, n=125/250/500/1000,
alpha=.01/.05, and zero/constant distortion; for GARCH use its two laws,
four lengths, two levels and zero/constant/state distortion. The scale,
raw forecasts, score definitions and fixed 1,024 GARCH test states are
unchanged. No random paths, returns or base forecasts are generated.

Primary evaluation is the previous independent-marginal AR loss and the
conditional expected GARCH loss averaged on its fixed independent test
states. Reconstruct Raw and full Shift-CP and compare all 72,000
configuration-history values with their archived results. The histories
are shared across configurations; they are not 72,000 independent samples.

## One feasible selection rule

For each n, reserve the first m=max(100,floor(.7*n)) observations for
fitting, and the remaining n-m for validation. This gives at least 100
fitting scores, so the uncapped conformal rank is finite at alpha=.01.
No capped maximum is substituted for an infeasible rank. At n=125 only
25 validation observations remain; this limitation is part of the design.

Fit C_inner at rank ceil((m+1)*(1-alpha)) on the first m scores.
Evaluate the five fractions L={0,.25,.5,.75,1} on the following validation
observations using the observed pinball loss of q_raw-lambda*C_inner.
Select minimum mean loss; exact ties prefer the smaller lambda. Freeze
both C_inner and the selected lambda for primary evaluation. The decision
function receives only calibration returns and raw forecasts; it receives
neither the generator label, test states nor population risk.

Report eight methods on the same evaluation object: Raw, Full-CP,
Half-Full, Inner-CP, Half-Inner, Selected-Inner, Oracle-Grid and
Oracle-Continuous. Full/Half-Full use all n observations. The inner
methods isolate the effect of choosing strength on identical candidate
forecasts. Oracle-Grid minimises population evaluation loss over L for
that history's C_inner. Oracle-Continuous uses the best lambda in [0,1].
Both are explicitly infeasible lower references and never enter selection.
Do not refit after selecting lambda or add a second feasible rule after
examining outcomes. All selected fractions and validation losses are saved.

## Exact accounting

Let R(c) be the known evaluation loss of the raw forecast minus a constant
c, and c_star its population best constant correction. For GARCH state
distortion, minimise the same fixed-state mixture risk, not the unattainable
conditional oracle. Save the remaining gap to that conditional oracle.

For each history, decompose the feasible loss change as

    R(C_selected)-R(0)
      = -B + A_inner - G_oracle + S_selection,

where B=R(0)-R(c_star), A_inner=R(C_inner)-R(c_star),
G_oracle=R(C_inner)-min_{lambda in L} R(lambda*C_inner), and
S_selection=R(C_selected)-min_{lambda in L} R(lambda*C_inner).
The last three quantities are nonnegative, but enter with the displayed
signs. Report the grid-discretisation gap separately. Also report
R(C_inner)-R(C_full) as the effect of reserving validation data; its sign
is not imposed. This exact decomposition avoids a quadratic approximation
and separates available shrinkage gains from the cost of estimating them.

## Reporting, inference and scope

Report every configuration, paired differences versus Raw, Full-CP,
Half-Inner and Oracle-Grid, Monte Carlo standard errors across the 500
histories, fraction-selection frequencies and expected violation rates.
Pointwise intervals are descriptive, not simultaneous discovery claims.
Count-based summaries of configurations are descriptive because the
configurations share histories. Summaries use equal configuration weights
and disclose this artificial design weighting. Coverage remains a separate
outcome; a partial shift does not inherit the full conformal guarantee.

For all 48 Normal-AR configurations, add the already specified contiguous
H=floor(3*n/7) evaluation by integrating the known conditional Gaussian
future given the last saved state. Keep exactly the primary chosen
coefficients. Recompute only the infeasible grid oracle for that evaluation
object. This is a labelled boundary sensitivity; it does not change the
primary rule or redefine the primary independent-marginal estimand.

The rule is worth further external development only if its losses and
coverage justify the selection effort relative to fixed half and full
correction. A negative result is retained with the decomposition. There
is no search for a favourable subset or revised grid. The previously
examined French test cannot serve as untouched confirmation of this rule.

## Verification and preservation

Independently reconstruct ranks and choices by explicit sorting/loops;
verify the exact accounting and oracle ordering; integrate Normal and t(5)
loss against their densities; verify GARCH scalar optima by score gradients;
compare all Raw/Full-CP values with the old archive; check future-input
invariance and a deliberately leaking negative control. Recompute all
outputs in a fresh process. Bind protocol, producers, environment, inputs
and outputs by hash. Keep the canonical manuscript, empirical results and
previous validation/release receipts unchanged during this research stage.
