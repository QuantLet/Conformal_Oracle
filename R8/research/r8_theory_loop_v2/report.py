"""Standalone diagnostic evidence; does not edit or build the R8 manuscript."""
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from common import ROOT,OUT,validate_lock,sha

PURPLE='#6633FF';ORANGE='#E64B35';BLUE='#008FE8';TEAL='#00A878';INK='#263248'

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#B6BECB',alpha=.25,lw=.6)
    ax.set_axisbelow(True);ax.tick_params(labelsize=9)

def figures(path):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':INK,
        'text.color':INK,'axes.titleweight':'bold','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.transparent':True})
    risk=pd.read_csv(OUT/'risk/risk_map.csv');view=risk[risk.a.between(0,3)]
    crossing=pd.read_csv(OUT/'risk/crossings.csv')
    dec=pd.read_csv(OUT/'diagnostic/density_decomposition.csv')
    sub=dec[(dec.transformation=='identity') & (dec.setting=='default_1000')].copy()
    tight=dec[(dec.transformation=='identity') & (dec.setting=='tight_16384')].copy()
    with PdfPages(path,metadata={'Title':'Theory-loop diagnostic: risk and density estimation',
        'Author':'Research diagnostic','CreationDate':None,'ModDate':None}) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(11.7,8.3));fig.subplots_adjust(left=.09,right=.97,bottom=.16,top=.80,hspace=.52,wspace=.27)
        fig.suptitle('Estimating the correction fraction has its own cost',x=.08,ha='left',y=.95,fontsize=19,weight='bold')
        fig.text(.08,.895,'Gaussian local experiment with the nuisance parameters known. Positive values mean higher loss.',fontsize=10.5)
        for ax,(name,label) in zip(axes.flat,[('raw','A  Proposed rule minus Raw'),('full','B  Proposed rule minus Full'),
                ('half','C  Proposed rule minus Half'),('fixed_oracle','D  Proposed rule minus fixed oracle')]):
            y=view['minus_'+name]
            ax.plot(view.a,y,color=PURPLE,lw=2.5,label='Expected loss difference')
            ax.axhline(0,color=INK,lw=.9,ls='--',label='Equal expected loss')
            root=crossing[crossing.comparator==name]
            if len(root):
                x=float(root.iloc[0].root);ax.scatter([x],[0],color=ORANGE,zorder=5,s=28)
                ax.annotate(f'{x:.3f}',(x,0),xytext=(8,8),textcoords='offset points',fontsize=9,color=ORANGE)
            ax.set(title=label,xlabel='Absolute local signal  |a|',ylabel='Difference in units of A₀',xlim=(0,3))
            style(ax)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.072),ncol=2,frameon=False)
        fig.text(.08,.035,'At zero signal: proposed 0.3333, Half 0.2500, Raw 0.0000. Crossings are numerical findings on the declared grid.',fontsize=9)
        fig.patch.set_alpha(0);pdf.savefig(fig,transparent=True);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11.7,8.3),gridspec_kw={'width_ratios':[1,1.55]})
        fig.subplots_adjust(left=.08,right=.98,bottom=.32,top=.78,wspace=.29)
        fig.suptitle('Numerical precision does not resolve the tail-density error',x=.08,ha='left',y=.95,fontsize=18,weight='bold')
        fig.text(.08,.893,'12 fixed windows from existing synthetic histories. These checks do not certify statistical accuracy.',fontsize=10.5)
        ax=axes[0]
        for law,col,label in [('normal',PURPLE,'Normal'),('t5',ORANGE,'Student-t(5)')]:
            s=sub[sub.law==law];t=tight[tight.law==law]
            ax.scatter(s.fhat_C,t.fhat_C,color=col,label=label,s=50,edgecolors='white',linewidths=.7,zorder=3)
        top=max(sub.fhat_C.max(),tight.fhat_C.max())*1.1
        ax.plot([0,top],[0,top],color=INK,ls='--',lw=1,label='Identical density')
        ax.set(title='A  Default versus refined computation',xlabel='Default density at C',ylabel='Refined density at C',xlim=(0,top),ylim=(0,top))
        style(ax);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),frameon=False,fontsize=9)
        ax=axes[1];x=np.arange(len(sub));w=.36
        ax.bar(x-w/2,100*sub.relative_error_fhat_C,width=w,color=PURPLE,label='Evaluated at estimated quantile C')
        ax.bar(x+w/2,100*sub.relative_error_fhat_cstar,width=w,color=BLUE,label='Evaluated at true quantile 0')
        labels=[f'{"N" if r.law=="normal" else "t5"}/{r.n}/{r.rep}' for r in sub.itertuples()]
        ax.set_xticks(x,labels,rotation=60,ha='right',fontsize=8)
        ax.set(title='B  Error relative to density at the true quantile',ylabel='Absolute relative error (%)')
        style(ax);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.35),frameon=False,fontsize=9)
        fig.text(.08,.09,'Labels: innovation law / window length / history index. Default-to-refined density changes are at most 0.139%.',fontsize=9)
        fig.text(.08,.05,'One of 60 bin-resolution comparisons fails its 0.1% bandwidth criterion; all 60 solver-refinement comparisons pass.',fontsize=9)
        fig.patch.set_alpha(0);pdf.savefig(fig,transparent=True);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pdf',type=Path,default=OUT/'diagnostic_evidence.pdf');a=p.parse_args()
    validate_lock();figures(a.pdf)
    risk=pd.read_csv(OUT/'risk/risk_map.csv');zero=risk[risk.a==0].iloc[0]
    cmp=pd.read_csv(OUT/'diagnostic/numerical_comparisons.csv')
    dec=pd.read_csv(OUT/'diagnostic/density_decomposition.csv')
    old=pd.read_csv(ROOT/'results/theory_loop/synthetic/validation_summary.csv')
    crossing=pd.read_csv(OUT/'risk/crossings.csv');uniform=pd.read_csv(OUT/'risk/uniform.csv')
    verification=json.loads((OUT/'risk/independent_verification.json').read_text())
    text=f'''# Theory-loop v2: completed bounded diagnostic

The proposed plug-in fraction should not be advanced to the financial panel
under the current protocol. The Gaussian calculation exposes a decision cost
even when nuisance parameters are known. The twelve-window numerical check
does not identify a computational repair that explains the earlier density
errors. The original 15% accuracy gate remains **FAIL**; panel deliverables
remain **NOT_RUN**. No manuscript, market data, model or forecast was changed.

## Gaussian risk: what was established

All 329 declared points were evaluated. At zero local signal the proposed rule
has risk {zero.risk:.12f} in units of A0, compared with 0 for Raw, .25 for
Half and 1 for Full. Its fraction is positive with probability
{zero.active_probability:.12f} even at zero signal. It therefore has 33.30%
more risk than Half at the null. This concerns the feasible positive-part
plug-in construction; it does not invalidate the population optimal fixed
fraction in the paper.

The positive-axis crossings found on [0,8] are:

| Compared with | Crossing | Proposed rule below crossing | Above crossing on checked grid |
|---|---:|---|---|
| Raw | {crossing.iloc[0].root:.9f} | Higher loss | Lower loss |
| Full | {crossing.iloc[1].root:.9f} | Lower loss | Higher loss |
| Half | {crossing.iloc[2].root:.9f} | Higher loss | Lower loss |

The large-signal scaled excess a²[r(a)-1] is 3.0088323, 3.0022000 and
3.0005495 at a=32,64,128, consistent with the derived limit 3. Neither the
crossing scan nor the oracle comparator constitutes a universally deployable
dominance rule. The experiment is local Gaussian expected loss, not a
financial coverage statement or a finite-sample GARCH theorem.

Independent full-real-line Stein integration agrees with primary squared-error
quadrature to {verification['max_risk_discrepancy']:.3g}; all 29 mathematical
checks rejected their defective counterpart before accepting the valid one.
Three checks exercise actual implementation mutants. Details are in
`docs/theory_loop_v2_20260911/MATH_VERIFICATION.md`.

## Training optimism: an exact example

For Uniform scores the independent future loss is computed exactly, separately
from the calibration ECDF integral. At n=1000, the true future loss change is
{uniform[uniform.n==1000].iloc[0].test_delta:.12g}; I+A predicts
{uniform[uniform.n==1000].iloc[0].one_A_prediction:.12g}; I+2A predicts
{uniform[uniform.n==1000].iloc[0].two_A_prediction:.12g}.
Thus I+A omits a leading estimation-cost term in this example. The six exact
rational rows, including n=250 where finite-rank effects are larger, are in
`risk/uniform_exact.csv`. This is not a justification for applying 2 Ahat to
the panel: estimated nuisances, dependence and finite-rank corrections remain.

## SJ numerical diagnostic

The design was fixed before execution: Normal/t5, n=700/1000, histories
0/17/499, five transformations and five settings. All 300 calculations are
finite and reproduce their independently reconstructed pair counts and root
equations, with zero warnings. RDS retains arguments, counts and residuals.

Across all 60 window/transform combinations:

| Refinement | Maximum relative change in bandwidth | Density at mapped C | Density at mapped true 0 |
|---|---:|---:|---:|
'''
    for name,label in [('default_tolerance','Default to tighter root tolerance'),('bins','4096 to 16384 bins'),
                       ('root','Root tolerance 1e-10 to 1e-12'),('total_numerical_change','Default to tight 16384')]:
        r=cmp[cmp.comparison==name]
        text+=f'| {label} | {100*r.bw_original_units.max():.9g}% | {100*r.fhat_C.max():.9g}% | {100*r.fhat_cstar.max():.9g}% |\n'
    text+='''
**One numerical criterion failed:** t5, n=1000, history 499, x+b, bandwidth
4096-to-16384 relative difference 0.12446937%, against the fixed 0.1% limit.
No tolerance was widened. The bin criterion passes jointly in 59/60 cases;
its two density components pass in all 60. The stricter root-tolerance
criterion passes in all 60.

The maximum default-to-refined density change is 0.13865%. This is very small
compared with the earlier 500-history median density errors of 25.12% (Normal)
and 28.67% (t5) at n=1000. This diagnostic provides no numerical explanation
for those errors. Twelve chosen histories do not certify aggregate accuracy
or justify changing the primary estimator after its failed admission.

`diagnostic/density_decomposition.csv` separates the exact ratio into
fhat(C)/f_ref(C) times f_ref(C)/f_ref(0). It also records fhat(0). The reference
is the original million saved conditional scales, not a new simulation.
Selected three-history medians are descriptive and are not substituted for
the original 500-history results. The medians of two factors need not multiply
to the median total. Four nuisance combinations in the variance diagnostic
keep variance, squared bias and MSE distinct; ddof=1 variance uses all 500
original histories at each law and n.

Translation produces small changes in binned SJ; scale and reflection match
to numerical precision on these cases. Fixed-bandwidth kernel equivariance
passes separately. Reflection maps the original C to -C; no incorrect upper
tail is substituted. Default results reproduce the selected original outputs.

## Decision and effect on R8

- Do not run the three financial-panel deliverables with this estimator.
- Retain the current population shrinkage proposition. Do not claim that its
  positive-part plug-in implementation inherits its expected-risk formula.
- Do not replace I+A by an unvalidated I+2 Ahat panel predictor.
- Treat the original tail-density failure as unresolved statistical accuracy,
  with one explicitly recorded numerical convergence failure in this diagnostic.
- Keep this diagnostic outside the manuscript until its role in the paper is
  assessed. A longer appendix and another rule are not automatically progress.

The current R8 manuscript and its publication assessment do not change merely
because these checks completed. The next decision should be based on the
independent interpretation report, not a new benchmark search.

## Reproduction and source bindings

The protocol was committed before numerical execution:
`de834692c7fddc61e6d28051ccf2c81a47965e00`. Entry points rebind original inputs.
Separate fresh processes replay the R calculations, the primary risk map and
the density decomposition. R summary CSV and numerical RDS content replay
identically; provenance paths are distinguished from numerical contents.
No clean-environment dependency reinstall is claimed.

Final verification and every artifact SHA-256/mtime are in
`completion.json` and `manifest.json`. The old study's own read-only verifier
checks that its recorded artifacts and 143 protected R8 files remain unchanged.
'''
    files=['risk/risk_map.csv','risk/crossings.csv','risk/uniform.csv','risk/uniform_exact.csv',
        'sj/summary.csv','diagnostic/density_decomposition.csv','diagnostic/numerical_comparisons.csv',
        'diagnostic/variance_reference.csv','diagnostic/selected_window_summary.csv']
    text+='\n| Numerical source | SHA-256 |\n|---|---|\n'
    for f in files:text+=f'| `{f}` | `{sha(OUT/f)}` |\n'
    (OUT/'RESULTS.md').write_text(text)
    validate_lock();print('Created standalone two-page evidence PDF and source-bound results report.')

if __name__=='__main__': main()
