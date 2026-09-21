# Decisive archived-path check: clustered target hits

11 September 2026. Continue the author's requested research assessment using
existing synthetic arrays only. No new simulated histories, forecaster fits,
density estimates, financial results or manuscript edits. This is a
retrospective mechanism check, not an untouched external test. The v1-v3
records and original R8 remain immutable. Commit before new summaries.

## Question and smallest informative design

Does the population target-hit long-run variance account for the leading
training optimism when hit autocovariances are nonzero? Use only the archived
Normal-margin stationary AR paths in artifacts/r8_mechanism/paths.npz:
phi=0 and .8, the same 500 latent histories, alpha=.01, zero raw distortion.
Calibration lengths n=500 and1000. Use prefixes [:n] for this check. The
old mechanism's n500 results used trailing windows, so they are not identical
replications; the full n1000 independent losses must reproduce old outputs.

Score S=q_raw-Y with q_raw=V0*Phi^{-1}(.01), V0=sqrt(.0002), has population
optimal shift q*=0 and density f*=phi(Phi^{-1}(.01))/V0. Hold the actual
conformal rank k=ceil((n+1)*.99) fixed. Set J=mean[rho(C-S)-rho(-S)].
Independent-marginal V=R(C)-R(0) is analytic Normal pinball risk. At n500,
also evaluate V_H on saved scores [500:714], H=floor(3*500/7)=214, holding
C fixed. At n1000 no future observations are saved: contiguous evaluation
is NOT_AVAILABLE, and no horizon is clipped or generated.

There are six primary cells: two phi values times two lengths under
independent-marginal evaluation, plus two n500 contiguous cells. O=V-J;
compare E O with 2A0, A0=Omega/(2nf*). Both margins, raw quantile, density,
tail level and number of coefficients are fixed across phi.

## Dependence quantities and tolerances

At lag j, target-hit covariance is the Gaussian integral from0 to phi^j of
exp[-z_alpha^2/(1+u)]/[2pi sqrt(1-u^2)]. Use scipy.quad epsabs1e-14,
epsrel1e-12. Truncate the long-run sum at the first L such that
phi^(L+1)/[pi sqrt(1-phi^2)(1-phi)]<=1e-13. Phi0 has Omega=.0099 exactly.
Also report population finite-n count variance with Bartlett factors; it is
not a second-order accuracy claim. Match stored counts.csv Omega values to
absolute1e-12 and relative1e-10. Validate analytic Normal loss against separate
innovation quadrature at deterministic fixed corrections and full n1000
losses against the stored mechanism outputs to the same tolerances.

## Inference and interpretation

Use the saved v1 999x500 bootstrap index matrix, jointly resampling whole
latent histories across all six cells. Fixed cell standard errors, maximum
absolute centred bootstrap t statistic and higher .95 order quantile give
one simultaneous family for O/(2A0). Preserve all cells and failures.
Zero/nonfinite standard errors are failures. The reference one represents
a first-order limit, not an exact finite-sample null or equivalence threshold.

Report O/(2A_iid) descriptively too, where A_iid=.0099/(2nf*). The iid
penalty corresponds to .0099/Omega on the primary normalised scale, and its
compatibility can be read from the same simultaneous bands; do not create
another selectively reported inferential family. No feasible estimated
penalty, shrinkage rule or panel admission is tested. The previous density
and LRV accuracy failures remain in force.

## Integrity and decision

Build defective fixtures first: wrong rank, wrong loss sign, omission of hit
autocovariances, incorrect future prefix, missing/duplicate family, altered
input or bootstrap hash. Bind paths to their original paths.json checksum,
counts.csv and replications.parquet to their existing completion metadata,
then bind all inputs/producer/protocol with SHA-256 and mtime. Independent
verification and exact fresh-process numerical replay are required.

If oracle optimism is compatible at the checked resolution and the iid
penalty is incompatible for phi=.8, this supports retaining a concise
dependent-optimism explanation. It does not establish quantitative equality,
novelty of optimism, an implementable gate, or a higher publication grade.
If the leading approximation is poor, report its finite-sample limits and
do not grow the design in pursuit of a favourable result. A research decision
memo must rank this supporting result against the paper's existing main claims.
