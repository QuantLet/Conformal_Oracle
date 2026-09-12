# Theory-loop diagnostic v2 — locked before computation, 2026-09-11

Authorisation: the author's “pai hai” continues the specialist recommendation
for a deterministic Gaussian risk map and the twelve-window numerical check.
This is a separate diagnostic. The original theory-loop protocol, failure,
results, archive and R8 files remain unchanged. No new random histories, market
data, forecasts, forecaster fits or panel inference are permitted here.

## 1. Gaussian local experiment

Let X=a+Z, Z standard Normal. The proposed action is d(X)=0 for |X|<=1 and
d(X)=X-1/X otherwise. Evaluate r(a)=E[(d(a+Z)-a)^2], in units
A0=Omega/(2 n f0). Compare Raw=a^2, Full=1, Half=(a^2+1)/4 and the best
fixed population fraction a^2/(1+a^2). The last comparator is an oracle only
within fixed fractions, not an optimality lower bound for all procedures.
Never substitute the random estimated fraction into the fixed-fraction risk.

Primary quadrature integrates squared error against the Normal density over
Z in [-12,12], splitting at -a-1 and -a+1 when interior. epsabs=5e-12 and
epsrel=1e-11, limit=250 per segment. The omitted-tail bound is uniform in a:
2[(12+2)phi(12)+2 Q(12)], since |d(a+Z)-a|<=|Z|+1.
Use a=-8,-7.95,...,8 and +/-16,32,64,128. Save errors, active probability
P(|a+Z|>1), all comparator risks and pairwise differences.
Scan positive a in [0,8] for sign changes against each comparator and refine
bracketed crossings (Brent xtol=2e-13, rtol=1e-13). These are bounded numerical
findings; neither global uniqueness nor dominance beyond the grid is inferred.

Independent validation uses full-real-line Stein integration:
r(a)=1+E[(X^2-2)1{|X|<=1}+3/X^2 1{|X|>1}], tolerance 1e-10 against the
primary calculation. Check symmetry, r(0)=4[phi(1)-Q(1)], active probability,
and a^2[r(a)-1] approaching 3 at a=32,64,128, tolerance .05. Exact comparator
values and actual incorrect implementations are tested: averaging the
fixed-fraction formula after estimating the fraction, reversing the correction
coefficient, and omitting the positive part. Build failing cases first, run
them against each acceptance predicate, then evaluate the real output.

As a separate exact calculation, p=99/100, k=ceil((n+1)p), delta=k-np,
v=p(1-p), n=250,500,700,1000,2000,10000. Compute using rational arithmetic:
E I_n(C)=(delta^2-delta-nv)/(2n(n+1)); independent-test loss change
E(C-p)^2/2=[k(k+1)/((n+1)(n+2))-2pk/(n+1)+p^2]/2; A0=v/(2n).
Compare I+A0 and I+2A0 with the exact test loss. A second derivation must
verify both expressions. This is a Uniform example, not a universal finite
sample correction or an endorsement of estimated 2 Ahat.

## 2. Twelve-window SJ diagnostic

Reuse existing synthetic scores, with laws Normal and variance-one t(5),
n=700,1000 and history indices 0,17,499: twelve windows, selected before
individual diagnostic outcomes. No resampling. Retain the original reference
histories and original 500-history variance calculations.

Use stats::bw.SJ(method="ste") under five prespecified settings:
default nb=1000 with default root tolerance;
tight1000 nb=1000 tol=1e-10*hmax;
tight4096 nb=4096 tol=1e-10*hmax;
tight16384 nb=16384 tol=1e-10*hmax;
solvercheck16384 nb=16384 tol=1e-12*hmax.
Use x, x+b, x-b, 100*x, -x, where b=.25 sqrt(1e-5/(1-.10-.85)).
Evaluate the Gaussian density at mapped original C and mapped population
c*=0; do not take the .99 quantile again after reflection. Save 300 rows,
bandwidths, hmax, actual tolerance, independent root residual and scaled
residual, warnings, exact call arguments, internal binned counts and R version.
There is no adaptive search beyond these five settings.

Report relative bandwidth and density differences for tight4096 versus
tight16384 (convergence criterion <=.001); solvercheck16384 versus tight16384
(criterion <=1e-6). Apply these to both evaluation densities and each of the
60 window/transform combinations. Failure is informative, not grounds for
widening tolerances. Quantify mapped translation, reflection and scale errors;
binned SJ is not assumed exactly invariant. Separately verify fixed-bandwidth
Gaussian-kernel equivariance at relative tolerance 1e-10 (absolute 1e-12).

Population reference density uses the saved million conditional scales:
f_ref(s)=mean[d_epsilon(z_alpha-s/sigma)/sigma], with each law's correct
variance-one density. Decompose fhat(C)/f_ref(0) exactly as
[fhat(C)/f_ref(C)] [f_ref(C)/f_ref(0)], and retain fhat(0). This distinguishes
kernel estimation from evaluating at an estimated tail quantile.

For each selected history report Omega_hat/(n fhat^2), .0099/(n fhat^2),
Omega_hat/(n f_ref(0)^2), .0099/(n f_ref(0)^2), versus the existing across-500
variance of C (sample variance ddof=1). Report mean-C bias and MSE separately.
Numerical settings may be assessed only by numerical residuals and convergence,
never by a more favourable density truth error or Quantile Score.

## 3. Checks, provenance and decision

Before computation commit this file unchanged in the diagnostic's local
protocol_repository (the supplied project has no root Git metadata). Record
the commit and SHA-256, and bind every input with path, bytes, mtime and hash.
Rebind inputs at both entry and completion; reject a stale digest, missing
window, duplicate law/n/history, or missing expected setting/transform. Keep
failing fixtures and identify which real predicate rejects them. Verify old
theory-loop receipt plus 143 protected R8 files before and after.

Save only new files under research/r8_theory_loop_v2,
results/theory_loop_v2 and docs/theory_loop_v2_20260911. Standalone figures
have transparent backgrounds and legends outside below the axes. Record
all source/output hashes and mtimes. Independent verification and a fresh
process replay are required. Do not call this a clean-environment reinstall.

The original density threshold of median relative error <15% remains failed.
Twelve selected windows cannot certify a replacement estimator. This stage
does not admit any financial-panel deliverable or trigger a larger simulation.
Its conclusion must distinguish a mathematical weakness of the decision rule,
numerical bandwidth errors, and statistical scarcity of tail information.
No manuscript claims or grade are automatically upgraded by passing checks.
