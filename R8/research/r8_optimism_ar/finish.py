"""Research decision, replay receipt and immutable artifact bindings."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
import scipy
from run import ROOT,BASE,OUT,LOCK,validate,bind,sha,dump

MANIFEST=OUT/'manifest.json';DOC=ROOT/'docs/optimism_ar_20260911'

def write_report():
    bands=pd.read_csv(OUT/'primary/bands.csv');dep=pd.read_csv(OUT/'primary/dependence.csv')
    d=dep[(dep.phi==.8)&(dep.n==1000)].iloc[0]
    text=f'''# Research decision: make dependence carry an observable implication

11 September 2026. This is a completed six-cell check on archived paths and a
research assessment. It is not a manuscript edit, a new forecasting method,
new simulated data or financial validation.

## Finding

Holding the Normal score margin, raw marginal quantile, density, sample length
and one-coefficient correction fixed, changing AR persistence from 0 to .8
changes the target-hit long-run variance from .0099 to {d.omega:.12f}.
The leading optimism penalty increases by a factor {d.long_run_inflation:.6f}.
The density is {d.f_true:.12f} in the same return units for both processes.

All three dependent cells exclude the iid penalty from their simultaneous
bands and include the dependence-aware first-order reference. This is a
discriminating check of which variance enters the penalty. It supports the
mechanism without certifying an exact finite-n formula or a feasible estimator.

The single simultaneous family contains all six declared cells:

| AR coefficient | n | Evaluation | Mean O/(2A0) | Simultaneous 95% band | iid reference on this scale |
|---:|---:|---|---:|---|---:|
'''
    for r in bands.itertuples():
        text+=f'| {r.phi:g} | {r.n} | {r.evaluation} | {r.mean_ratio:.4f} | [{r.lower:.4f}, {r.upper:.4f}] | {r.iid_reference:.4f} |\n'
    text+='''
For phi=.8 the iid reference is about .3376, outside all three bands.
The leading reference one remains inside all six bands. Inclusion is not
equivalence: the contiguous phi=.8 estimate is .8395 with band [.6331,1.0460],
so appreciable finite-sample discrepancies are compatible with this check.

## Why this check was necessary

V3 established expected calibration loss -B-A0 and future loss -B+A0 under
R8's assumptions, hence optimism 2A0. Its GARCH controls had dependent scores
but iid true target-hit indicators. They could not empirically distinguish
Omega from alpha(1-alpha). Here the target hits genuinely cluster, while
the score marginal and density stay fixed. The iid simplification is now
directly contradicted at the resolution of the declared simultaneous family.

This result belongs to the existing argument about estimation cost. Expected
optimism itself has prior literature: [Giessing and He](https://arxiv.org/abs/1802.00555)
derive quantile-risk optimism under row-wise independent observations. The
dependent conformal-rank proof and contiguous transfer are the relevant
specialisation here. No general first-ever claim is made.

## Exact design and limits

The 500 latent Normal AR histories were already stored and inspected in the
original mechanism study. The same latent innovations are shared across phi,
and the same histories across n. This is retrospective reuse, not six
independent experiments or new out-of-sample evidence. Calibration uses
prefixes of 500 or 1000 points. At 500, the contiguous test is the next 214
saved points, with the correction held fixed. At 1000 there is no saved future,
and the two contiguous rows are NOT_AVAILABLE. No replacement was generated.

The original n500 mechanism used trailing windows; those results are not
claimed as identical prefix replications. The full n1000 independent losses
reproduce their archived values. With phi>0 the raw quantile is marginally
correct, not the conditional AR quantile. The design isolates the cost of
estimating a marginal correction; it is not a comparison of financial AR
forecasters or a recommendation to ignore conditional information.

Population covariance uses the declared Gaussian integral. At phi=.8 the
long-run sum stops at 138 lags, with omitted contribution below 8.98e-14.
Finite-n count variance is reported separately. Covariance quadrature and
analytic loss have independent checks. The same saved 999 resampling vectors
yield one six-cell max-T family; no row was selected after seeing outcomes.

## Research decision for the paper

1. Keep the main thesis on the cost of learning a tail correction. The strongest
   argument joins removable loss, dependence in tail counts, correction form,
   and the cost of deciding whether to act. It does not need a claim that one
   method wins universally.
2. Retain the v3 optimism result as a short supporting corollary or paragraph,
   with a compact proof that uses existing moment lemmas. This AR check closes
   its missing nonzero-hit-dependence control. It does not merit another broad
   benchmark section or a fourth contribution.
3. Stop development of the tested positive-part shrinkage rule. Its known-
   nuisance risk failure is separate and is not repaired by this result.
4. Keep I+2Ahat off the financial panel under the failed original admission
   protocol. Population-input verification cannot compensate for insufficient
   accuracy in estimated density/LRV. No extra model or dataset is justified
   merely as a way to obtain a more favourable ranking.
5. Keep the current financial conclusion exact: mean QS is lower for the
   static shift, but the main simultaneous comparison does not establish a
   clear improvement; predeclared external comparisons establish no richer-
   correction advantage. This research check changes none of those bands.

The supporting mathematics is now more tightly connected to a mechanism
that actually separates dependent from iid calibration. The unresolved
claim is a reliably estimable pre-deployment benefit on financial series.
That is the scientific boundary of the current paper. A publication grade
cannot be raised by counting technical checks or adding this control alone.

## Proposed concise explanatory sentence

> Dependence in tail hits affects both the cost of estimating a correction
> and the optimism of its calibration loss: at first order, expected future
> loss exceeds fitted calibration loss by Omega/(nf*). In the controlled
> comparison, fixed margins and density allow this dependence penalty to be
> distinguished from the iid approximation.

The expectation scope, conformal rank and static contiguous conditions belong
in the accompanying formal statement. No new source or bibliography text was
inserted into R8 during this check.

## Evidence and reproduction

Protocol commit: e96ee479bb9455d94aeb90a95c3cc86f45d264df. Original paths and
outputs were verified against their own metadata before the new lock. The
source-bound independent review is docs/optimism_ar_20260911/STATISTICAL_REVIEW.md.
completion.json and manifest.json record exact fresh-process CSV replay and
all new artifact hashes/mtimes. Earlier studies and all 143 protected R8 files
are verified unchanged. This is replay in the recorded environment, not a
clean-environment reinstall.

| Source | SHA-256 |
|---|---|
'''
    for name in ('histories.csv','dependence.csv','bands.csv','bootstrap.csv','unavailable.csv'):
        text+=f'| `primary/{name}` | `{sha(OUT/"primary"/name)}` |\n'
    (OUT/'RESULTS.md').write_text(text)

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    lock=validate()
    if a.report:write_report();print('Research decision report written.');return
    env=os.environ.copy();env['MPLCONFIGDIR']='/private/tmp/irfa-theory-v2-mpl'
    old=subprocess.run([sys.executable,str(ROOT/'research/r8_theory_loop_v3/finalize.py'),'--verify'],capture_output=True,text=True,check=True,env=env)
    if a.verify:
        data=json.loads(MANIFEST.read_text())
        for r in data['files']:
            now=bind(ROOT/r['relative_path'])
            assert all(now[k]==r[k] for k in ('sha256','mtime_ns','size')),r['relative_path']
        print('Verified',len(data['files']),'AR artifacts; earlier studies/R8 preserved; financial panel NOT_RUN.');return
    assert not MANIFEST.exists()
    review=json.loads((OUT/'independent_verification.json').read_text());assert review['status']=='PASS'
    assert (DOC/'STATISTICAL_REVIEW.md').is_file()
    replay=[]
    for f in sorted((OUT/'primary').glob('*.csv')):
        b=(OUT/'replay'/f.name).read_bytes();good=f.read_bytes()
        assert good!=b+b'\nDELIBERATE_CORRUPTION\n';assert good==b
        replay.append(dict(file=f.name,sha256=sha(f),negative_control_rejected=True,exact_match=True))
    dump(OUT/'completion.json',dict(status='COMPLETED_ARCHIVED_PATH_CONTROL',protocol_commit=lock['protocol_commit'],
        cells=6,derived_rows=3000,independent_latent_histories=500,bootstrap_replicates=999,replay=replay,
        independent_verification='PASS',new_random_histories=0,new_forecaster_fits=0,
        manuscript_changed=False,financial_panel='NOT_RUN',original_admission='FAIL_UNCHANGED',
        conservation=old.stdout.strip(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__))
    files=[]
    for folder in (BASE,OUT,DOC):
        for f in sorted(folder.rglob('*')):
            if not f.is_file() or f==MANIFEST or '.git' in f.parts or '__pycache__' in f.parts:continue
            rec=bind(f);rec['relative_path']=str(f.relative_to(ROOT));files.append(rec)
    dump(MANIFEST,dict(files=files,protocol_commit=lock['protocol_commit'],
        exclusions=['manifest itself','Git internals','interpreter caches']))
    print('Bound',len(files),'artifacts;',len(replay),'exact CSV replays.')

if __name__=='__main__':main()
