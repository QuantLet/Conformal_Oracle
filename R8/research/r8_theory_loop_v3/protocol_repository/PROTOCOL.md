# Training optimism under dependence — bounded v3 protocol

11 September 2026. Authorisation: the author's “Hai” continues the proposed
derivation and checks on already generated synthetic data. This is a
retrospective mechanism diagnostic, not a new external validation. The original
theory-loop and v2 outputs, their failures and all R8 files remain immutable.
No new histories, forecasts, fitted densities, simulations or financial panels.
Commit this protocol before calculating new diagnostic summaries.

## Mathematical target

Under the current R8 expected-loss assumptions, write p=1-alpha, q=F^{-1}(p),
C=S_(k), |k-np|=O(1), I=R_n(C)-R_n(0), B=R(0)-R(q) and
A0=Omega/(2 n f(q)). Derive E I=-B-A0+o(1/n) and the independent-marginal
expected optimism E[D(C,S*)-I]=2A0+o(1/n). Establish the L1 remainder,
conformal-rank effect and possible dependent ties explicitly. Compare with
the contiguous average using the existing finite transfer bound. Do not
transfer the result to a rolling rule or an individual conditional date.
Distinguish the true finite-sample estimation cost A_n from its leading A0.
Do not assume I+2 Ahat is validated merely because the oracle expansion holds.

## Existing inputs and exact diagnostic targets

Use only results/theory_loop/synthetic: calibration_Normal/t5 arrays (actual
filenames use normal and t5), estimators.csv, loss_histories.csv, expansion.csv,
truth.csv, mixture_integration_Normal/t5.npz, validation_summary.csv and the
saved 999x500 history_bootstrap_indices.npy. Use all 500 original histories,
laws Normal and variance-one t5, and n=250,500,700,1000,2000. The two original
translations are b=0 and b=.25 sqrt(1e-5/(1-.1-.85)). Recompute pinball losses
from saved scores and compare with the original integral and stored outputs.
Use the original density/LRV estimates as diagnostics, without re-estimation.

For each history form J=R_n(C)-R_n(q) and V=R(C)-R(q). V comes from the
original common long-reference marginal integration; its numerical error
budget remains 4e-11. The primary independent-marginal optimism variable is
O=V-J. It has the same expectation as test-minus-training loss change and
removes the zero-mean error of the fixed oracle/raw baseline. This is a
known-truth synthetic diagnostic, not a feasible centring on the financial
panel. Compute I and the uncentred delta_test-I separately for both b values;
their difference from O is the explicit fixed-baseline sampling term.

For an actual contiguous evaluation, the existing 2000-observation histories
allow n=250,500,700,1000 with H=floor(3n/7), using scores [n:n+H] after the
calibration prefix [0:n]. Hold C fixed. Form V_H=mean[rho(C-Sfuture)-
rho(q-Sfuture)] and O_H=V_H-J, with uncentred loss changes separately.
n=2000 has no saved future observations and is NOT_AVAILABLE; generate none.
The sequences have GARCH-dependent scores but iid true target hits, so this
does not test a nonzero hit-autocovariance contribution to Omega.

Exact nuisance: Omega=.0099 and f(q) from the original saved million-scale
reference. Primary ratio is mean O/(2 A0). Report separately mean J/A0,
mean V/A0, the error of I+A0, I+2A0 and I+2 Ahat, and the two hybrid costs
using exact Omega/estimated f and estimated Omega/reference f. Expectations
are the target; no rowwise prediction or sign-gate accuracy is asserted.
Report finite-rank k, |k-np|, true-quantile hit count and estimated-cutoff count.
Keep sample variance of C (ddof=1), squared mean-C bias and MSE distinct.

## Uncertainty fixed before new summaries

One primary simultaneous family contains all 18 centred optimism ratios:
10 independent-marginal cells and 8 contiguous cells. The resampling unit is
the entire original history vector across all cells. Pair the Normal/t5
history indices as in the stored design; no artificial independence across n.
Reuse the saved 999 resampling rows. For each cell fix se=sample_sd/sqrt(500),
and take the 95th percentile (NumPy method='higher') of max_j
|bootstrap_mean_j-original_mean_j|/se_j. Report mean +/- critical*se_j.
No pointwise-only alternatives or selective cell deletion. A zero/nonfinite
standard error, incomplete family or duplicate key is a validation failure.
Bands condition on the saved reference scales; their uncertainty is reported
separately, not silently treated as a certified population quantity.

The point 1 is the first-order reference, not an exact finite-n null. Report
which bands include it and all absolute/relative approximation discrepancies.
No new admission threshold is invented. The original <15% density and LRV
criteria and original failed panel gate remain in force. Descriptive tables
for raw/centred losses and hybrid nuisances carry no extra inferential claim.

## Checks and outputs

Write code/results/reports only to the new r8_theory_loop_v3 directories.
Build defective fixtures before accepting checks: reversed loss sign, wrong
quantile rank, wrong tail for reflection, dropped factor two, shifted future
slice/leakage, incomplete or duplicate family, altered bootstrap indices and
stale input hashes. Use actual implementation alternatives where possible,
and clearly identify simple output-corruption tests. Verify exact Uniform
examples against v2 and check each algebraic decomposition numerically.
Require score/rank/loss replay absolute tolerance 1e-12 with relative 1e-10,
and mixture replay absolute 4e-11 with relative 4e-11. Tolerances never widen.

Independent mathematical and statistical reviews must identify any extra
assumption, unsupported numerical interpretation or novelty claim. Expected
optimism for quantile regression has prior literature: document the relation
to Giessing and He, On the Predictive Risk in Misspecified Quantile Regression
(arXiv:1802.00555), whose framework uses row-wise independent observations.
Do not present the general optimism principle as new.

Commit/hash the protocol before numerical work, bind all inputs and record
source/output SHA-256 plus mtimes. Verify both old study manifests before and
after; fresh-process replay must match all numerical CSVs. No clean-environment
installation claim. Keep current R8 source/PDF unchanged in this stage; report
the exact amendment that would be supported and the unresolved feasible gate.
