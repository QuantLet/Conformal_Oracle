"""Exact fixtures and source-grounded v3 report, without modifying R8."""
from fractions import Fraction as F
import json
import numpy as np
import pandas as pd
from engine import ROOT,BASE,OUT,validate_lock,check,dump,sha,rank

def exact_checks():
    rows=[];checks=[]
    old=pd.read_csv(ROOT/'results/theory_loop_v2/risk/uniform_exact.csv',dtype=str)
    for n in (250,500,700,1000,2000,10000):
        p=F(99,100);a=1-p;k=rank(n);v=p*a
        # Sum expected gaps on either side of the empirical order statistic.
        fitted=(a*k*(k-1)+p*(n-k)*(n-k+1))/(2*n*(n+1))
        train=fitted-v/2
        mean=F(k,n+1)-p;var=F(k*(n+1-k),(n+1)**2*(n+2))
        future=(var+mean*mean)/2;A=v/(2*n)
        previous=old[old.n==str(n)].iloc[0]
        check(checks,'Uniform_training_'+str(n),-train,train,lambda z:z==F(previous.training_delta))
        check(checks,'Uniform_future_'+str(n),future+A,future,lambda z:z==F(previous.test_delta))
        rows.append(dict(n=n,k=k,training_delta=str(train),test_delta=str(future),
            A0=str(A),exact_optimism=str(future-train),optimism_over_2A0=float((future-train)/(2*A))))
    # Finite tied sample: exact rank minimisation need not have Fn(C)=k/n.
    x=np.r_[np.zeros(247),np.ones(3)];n=len(x);pn=rank(n)/n;c=np.sort(x)[rank(n)-1]
    loss=lambda u:np.mean(np.where(u-x>=0,(1-pn)*(u-x),-pn*(u-x)))
    check(checks,'actual_wrong_tied_rank_minimum',loss(0),loss(c),
        lambda z:all(z<=loss(u)+1e-15 for u in (-2,-1,0,.5,1,2)))
    left=np.mean(x<c);right=np.mean(x<=c)
    check(checks,'tied_rank_subgradient_not_CDF_equality',right==pn,left<pn<=right,bool)
    pd.DataFrame(rows).to_csv(OUT/'exact_uniform.csv',index=False)
    dump(OUT/'exact_checks.json',dict(status='PASS',checks=checks,tied_sample=dict(zeros=247,ones=3,k=rank(n),left=left,right=right,p_n=pn)))

def main():
    validate_lock();exact_checks()
    bands=pd.read_csv(OUT/'diagnostic/simultaneous_bands.csv')
    summary=pd.read_csv(OUT/'diagnostic/summary.csv')
    s=summary[summary.bias.eq(0)&summary.evaluation.eq('independent')]
    text='''# V3 — the calibration-to-future loss bridge

**The expectation-level correction is now justified under the existing R8
assumptions. The feasible financial application remains unvalidated.**

The proof and independent reviews are in
`docs/theory_loop_v3_20260911/MATHEMATICAL_REVIEW.md`,
`STATISTICAL_REVIEW.md` and `INTERPRETATION.md`. The current R8 sources and
PDFs are unchanged; so are the original and v2 diagnostic records.

## Mathematical result

Let I=R_n(C_n)-R_n(0) denote empirical calibration loss change, B the population
loss removable by a constant shift, and A0=Omega/(2nf*) the leading estimation
cost. Under the current stationary geometric beta-mixing, fourth-moment,
bounded-density and positive continuous local density assumptions,

    E I = -B - A0 + o(1/n),
    E D_test = -B + A0 + o(1/n),
    E(D_test - I) = 2 A0 + o(1/n).

This holds for the actual admissible conformal rank ceil((n+1)p), including
possible dependent ties. The proof establishes an L2 quantile linearisation
and an L1 integrated empirical-process remainder. No additional smoothness,
conditional density or tie-exclusion assumption was introduced.

The same expected optimism applies when one estimated shift is held fixed
over a contiguous test block with H/sqrt(n) tending to infinity. The 70/30
geometry has H/n tending to 3/7. This is an unconditional mean-loss result,
not a rolling or individual-date guarantee. Finite-count Omega_n may replace
Omega in the leading term, but does not create a second-order error bound.

Adding A0 once to fitted training loss cancels its downward fitting effect
and still omits the cost on future data. Adding 2A0 has the correct leading
expectation. This does not make I+2Ahat a validated predictor for individual
model/asset pairs.

## Checks on existing synthetic histories

No new histories, model fits, density fits or forecasts were produced. We
reused all 500 original histories per law, Normal and variance-one t(5).
Independent-marginal loss uses the saved million-scale mixture integration.
For n=250,500,700,1000, existing suffixes also permit a genuine contiguous
test of length floor(3n/7), with the calibration threshold held fixed.
There are no saved future observations after n=2000, so its contiguous row
is explicitly NOT_AVAILABLE. No shortened horizon or generated replacement.

The centred optimism O=V-J compares future and training losses relative to
the fixed true optimum. It has the same expectation as test-minus-training
loss change, while removing fixed-baseline sampling noise. This known-truth
centring is available only in the synthetic check. The uncentred quantities
for both original translations are retained in histories.csv and summary.csv.

All 18 simultaneous 95% bands include the first-order reference 1 for the
ratio E[O]/(2A0_ref). A0_ref uses exact target-hit variance .0099 and the saved
reference density. These bands are compatible with the leading approximation;
they are not equivalence tests, a finite-sample theorem, or new panel admission.

| Law | n | Evaluation | Ratio | Simultaneous 95% band |
|---|---:|---|---:|---|
'''
    for r in bands.itertuples():
        text+=f'| {r.law} | {r.n} | {r.evaluation} | {r.mean_optimism_over_2A0:.4f} | [{r.lower:.4f}, {r.upper:.4f}] |\n'
    text+='''
At n=250 the separate terms are less accurate: J/A0_ref is approximately
-0.548 (Normal) and -0.533 (t5), while independent V/A0_ref is 1.349 and 1.460.
Their deviations partially cancel in V-J. A good optimism ratio therefore
does not establish an equally accurate approximation for each separate loss
term or identify conformal rank as the sole source of finite-sample error.

The complete 18-cell family was fixed before these summaries. Resampling used
the 999 stored whole-history index vectors and a fixed-standard-error max-T
critical value, preserving shared histories across sample sizes. Bands condition
on the saved reference scales; the original relative MC standard errors of
the two reference densities are approximately 0.101% and 0.110%.
The design is retrospective, using previously inspected synthetic data.

Although the GARCH scores are dependent, their oracle target-hit indicators
are iid. The nonzero-hit-autocovariance component of the theorem is established
by the proof and is not empirically isolated by these GARCH checks.

## Why the feasible penalty remains blocked

The original density/LRV estimates are reused unchanged. Their mean cost ratio
shows how much of the leading penalty an estimated implementation actually adds:

| Law | n | Mean Ahat/A0_ref | Exact Omega, estimated density | Estimated Omega, reference density |
|---|---:|---:|---:|---:|
'''
    for r in s.itertuples():
        text+=f'| {r.law} | {r.n} | {r.mean_Ahat_over_A0:.4f} | {r.mean_exact_omega_cost_over_A0:.4f} | {r.mean_reference_density_cost_over_A0:.4f} |\n'
    text+='''
At n=1000 the available cost estimator is low in mean by 22.02% for Normal
and 11.93% for t5. The two hybrid columns isolate components without fitting
anything new. Their errors interact multiplicatively; summary.csv records
that interaction rather than attributing the full discrepancy to one nuisance.
Mean accuracy is different from the previously specified median relative-error
admission gate. Its failed density criterion remains failed.

The factor-two correction repairs the theoretical target. It does not repair
the estimated cost's precision or convert the dropped shrinkage rule into a
validated method. The financial-panel deliverables remain NOT_RUN.

## Finite-rank and implementation checks

The six Uniform examples in exact_uniform.csv use exact rational expected-gap
and variance-plus-bias formulas, matching v2 exactly. They retain the actual
conformal rank, including its pronounced n=250 effect. No claim that 2A0 is
an exact finite-n optimism identity is made. A tied finite sample also verifies
that the empirical quantile minimises the p_n=k/n objective even when its
empirical CDF jumps past p_n.

Negative controls include actual wrong-tail loss, empirical instead of
conformal rank, an incorrect reflected quantile, a reversed training-loss
sign, a leaking future prefix, omission of the factor two, and incorrect
bootstrap centring. Other checks deliberately corrupt inputs/output values;
these are recorded separately from implementation mutants. The independent
statistical verifier replays losses and the simultaneous family from source
arrays rather than accepting producer summaries.

## Literature and contribution

Expected optimism in quantile regression is established prior work.
[Giessing and He](https://arxiv.org/abs/1802.00555) derive covariance/trace
representations and risk estimators; their stated framework uses row-wise
independent observations. The present derivation supplies the dependent
scalar conformal-rank version and its static contiguous transfer under R8's
assumptions. This is a supporting link for the paper's existing argument,
not a claim to have discovered optimism or a priority claim for all dependent
quantile risk estimation.

## Exact amendment supported; not yet inserted

One short paragraph after the existing expected-cost discussion could state:

> Estimating the benefit of correction from its calibration loss introduces
> a second cost. Under the assumptions above, the fitted calibration loss
> change has expectation -B-A0+o(1/n), while the corresponding future change
> has expectation -B+A0+o(1/n). The leading optimism correction is therefore
> 2A0. This relation retains the conformal rank and applies to a static
> contiguous average when H/sqrt(n) grows without bound. Estimating the
> penalty at the 1% tail remains a separate accuracy requirement.

The complete proof is prepared separately; no existing population-risk
equation in R8 needs withdrawal. A feasible panel prediction or a new gate
must not be claimed from these expectation results. No publication grade is
certified by this diagnostic.

## Reproduction

Protocol commit: 973132284a16235c780162406cf393031e96e036. The lock binds the
actual input arrays, original results, bootstrap indices and producer code.
Fresh-process replay reproduces the numerical CSVs. Completion and all new
artifact SHA-256/mtime bindings are in completion.json and manifest.json.
The original study and v2 verifiers preserve all recorded protected artifacts,
including 143 R8 files. No clean-environment installation is claimed.

| Numerical source | SHA-256 |
|---|---|
'''
    for name in ('histories.csv','summary.csv','simultaneous_bands.csv','rank_diagnostics.csv','bootstrap_maxima.csv','unavailable.csv'):
        text+=f'| `diagnostic/{name}` | `{sha(OUT/"diagnostic"/name)}` |\n'
    text+=f'| `exact_uniform.csv` | `{sha(OUT/"exact_uniform.csv")}` |\n'
    (OUT/'RESULTS.md').write_text(text)
    validate_lock();print('Exact Uniform/tie checks and source-bound report complete.')

if __name__=='__main__':main()
