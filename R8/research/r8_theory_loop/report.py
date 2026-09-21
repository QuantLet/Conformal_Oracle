"""Generate research tables, negative-result report and two diagnostic figures."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/theory_loop';SYN=OUT/'synthetic'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evidence(path):
    return f'`{path.relative_to(ROOT)}`; SHA-256 `{sha(path)}`'


def figures(summ,exp):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'axes.labelcolor':'#23262B','text.color':'#23262B',
        'axes.edgecolor':'#69717B','savefig.transparent':True})
    colours={'normal':'#6633FF','t5':'#E64B35'}
    names={'normal':'Normal innovations','t5':'Student-t(5) innovations'}
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.9))
    specs=[('median_density_relative_error','density_error_lo','density_error_hi',100,
            'Density error','Median absolute relative error (%)'),
           ('median_omega_relative_error','omega_error_lo','omega_error_hi',100,
            'Long-run variance error','Median absolute relative error (%)'),
           ('variance_ratio','variance_ratio_lo','variance_ratio_hi',1,
            'Quantile variance','Estimated / empirical variance')]
    for ax,(col,lo,hi,mult,title,label) in zip(axes,specs):
        for law,g in summ.groupby('law',sort=False):
            ax.plot(g.n,g[col]*mult,color=colours[law],marker='o',markersize=4,
                    linewidth=1.7,linestyle='-' if law=='normal' else '--')
            ax.fill_between(g.n,g[lo]*mult,g[hi]*mult,color=colours[law],alpha=.1)
        if col=='variance_ratio':
            ax.axhspan(.85,.98,color='#89919D',alpha=.18)
            ax.axhline(.85,color='#69717B',lw=.9,ls=':');ax.axhline(.98,color='#69717B',lw=.9,ls=':')
        else:
            ax.axhline(15,color='#69717B',ls=':',lw=1)
        ax.set(title=title,xlabel='Calibration observations',ylabel=label)
        ax.set_xscale('log');ax.set_xticks([250,500,700,1000,2000],[250,500,700,1000,2000])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.18)
    handles=[Line2D([0],[0],color=colours[l],marker='o',ls='-' if l=='normal' else '--',label=names[l])
             for l in colours]
    handles.append(Line2D([0],[0],color='#69717B',ls=':',label='Prespecified admission limits'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False)
    fig.subplots_adjust(left=.075,right=.99,top=.89,bottom=.26,wspace=.42)
    fig.suptitle('The proposed plug-in estimators fail the synthetic admission gate',fontsize=12,y=.99)
    for suffix in ('pdf','png'):
        fig.savefig(SYN/f'estimator_validation.{suffix}',dpi=180,transparent=True,
                    metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else None)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.8))
    for row,law in enumerate(('normal','t5')):
        for col,bias in enumerate(sorted(exp.bias.unique())):
            ax=axes[row,col];g=exp[(exp.law==law)&np.isclose(exp.bias,bias)].sort_values('n')
            colour=colours[law]
            ax.errorbar(g.n,g.observed_delta*1e4,yerr=1.959963984540054*g.mcse*1e4,
                        color=colour,marker='o',markersize=4,capsize=2,linewidth=1.6)
            ax.plot(g.n,g.prediction*1e4,color=colour,ls='--',linewidth=1.6)
            ax.axhline(0,color='#69717B',lw=.8)
            title=names[law]+('\nCorrect raw quantile' if col==0 else '\nConstant raw bias: 0.25 V0')
            ax.set(title=title,xlabel='Calibration observations',ylabel='Static minus raw QS '+r'$\times 10^4$')
            ax.set_xscale('log');ax.set_xticks([250,500,700,1000,2000],[250,500,700,1000,2000])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.18)
    handles=[Line2D([0],[0],color='#343941',marker='o',label='Simulated mean (pointwise 95% MC interval)'),
             Line2D([0],[0],color='#343941',ls='--',label='First-order population prediction')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=1,frameon=False)
    fig.suptitle('Loss expansion under a common reference distribution',fontsize=12,y=.98)
    fig.subplots_adjust(left=.105,right=.985,top=.86,bottom=.17,hspace=.70,wspace=.32)
    for suffix in ('pdf','png'):
        fig.savefig(SYN/f'loss_expansion.{suffix}',dpi=180,transparent=True,
                    metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else None)
    plt.close(fig)


def main():
    status=json.loads((SYN/'admission.json').read_text())
    checked=json.loads((SYN/'independent_validation.json').read_text())
    if checked['computational_checks']!='PASS':
        raise RuntimeError('Unverified results')
    if status['status']!='FAIL':
        raise RuntimeError('This report is specifically the failed-admission branch')
    summ=pd.read_csv(SYN/'validation_summary.csv');exp=pd.read_csv(SYN/'expansion.csv')
    truth=pd.read_csv(SYN/'truth.csv');est=pd.read_csv(SYN/'estimators.csv')
    figures(summ,exp)
    statuses=[]
    for deliverable in (1,2,3):
        for panel,n in (('main',240),('external',48)):
            row=dict(deliverable=deliverable,panel=panel,planned_pairs=n,
                     computed_pairs=0,status='NOT_RUN_SYNTHETIC_ADMISSION_FAILED',
                     reason='Locked density/LRV/variance criteria failed before panel access')
            if deliverable==3 and panel=='external':
                row['status']='NOT_REQUESTED';row['reason']='Boundary request covers main240 and existing161 subset'
            statuses.append(row)
        subset=[r for r in statuses if r['deliverable']==deliverable]
        pd.DataFrame(subset).to_csv(OUT/f'deliverable{deliverable}/status.csv',index=False)
        (OUT/f'deliverable{deliverable}/README.md').write_text(
            '# Not run\n\nThe committed synthetic gate failed before financial-panel access.\n'
            'There are no pair-level estimates, new forecasts, confidence bands or financial plots here.\n'
            'The status table contains design counts, not computed observations.\n')
    pd.DataFrame(statuses).to_csv(OUT/'deliverable_status.csv',index=False)
    first=[]
    for law,g in summ.groupby('law',sort=False):
        for criterion,col in [('Omega median error <15%','omega_pass'),('Density median error <15%','density_pass'),
                              ('Variance ratio in [.85,.98]','variance_ratio_pass'),('All criteria','joint_pass')]:
            matched=g[g[col]].n
            first.append(dict(law=law,criterion=criterion,first_tested_n=int(matched.min()) if len(matched) else None,
                              status='OBSERVED' if len(matched) else 'NOT_OBSERVED_BY_2000'))
    pd.DataFrame(first).to_csv(SYN/'first_passing_size.csv',index=False)
    floors=est.groupby(['law','n','bias']).agg(h2_zero_fraction=('h2_floor','mean'),
              omega_floor_fraction=('omega_floor','mean'),lambda_mean=('lambda_hat','mean'),
              lambda_median=('lambda_hat','median')).reset_index()
    floors.to_csv(SYN/'floor_diagnostics.csv',index=False)
    spacing=est[['law','n','sbg_m','sbg_lo','sbg_hi','sbg_status']].drop_duplicates()
    spacing.to_csv(SYN/'spacing_status.csv',index=False)
    # Every displayed data-derived number points to its authoritative CSV hash.
    number_rows=[]
    for filename,frame in [('validation_summary.csv',summ),('expansion.csv',exp),('truth.csv',truth),
                           ('floor_diagnostics.csv',floors),('first_passing_size.csv',pd.DataFrame(first))]:
        path=SYN/filename
        for index,row in frame.iterrows():
            for column,value in row.items():
                if isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)):
                    number_rows.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),
                         data_row=index+1,law=row.get('law',''),n=row.get('n',''),bias=row.get('bias',''),
                         field=column,value=value))
    pd.DataFrame(number_rows).to_csv(OUT/'manuscript_numbers.csv',index=False)
    tex=['% Synthetic estimator validation; NOT a financial panel result.',
         r'\begin{tabular}{llrrrr}',r'\toprule',
         r'Law & $n$ & $\widehat\Omega$ error (\%) & $\hat f_*$ error (\%) & Variance ratio & Coverage (\%) \\',r'\midrule']
    for r in summ.itertuples():
        tex.append(f'{r.law} & {r.n} & {100*r.median_omega_relative_error:.4f} & '
                   f'{100*r.median_density_relative_error:.4f} & {r.variance_ratio:.4f} & {100*r.coverage:.4f} '+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (SYN/'tab_validation.tex').write_text('\n'.join(tex)+'\n')
    tex=[r'\begin{tabular}{llrrrr}',r'\toprule',
         r'Law & $n$ & Bias & Simulated QS change & Predicted change & MC interval \\',r'\midrule']
    for r in exp.itertuples():
        lo=(r.observed_delta-1.959963984540054*r.mcse)*1e4
        hi=(r.observed_delta+1.959963984540054*r.mcse)*1e4
        tex.append(f'{r.law} & {r.n} & {r.bias:.4f} & {r.observed_delta_x1e4:.4f} & '
                   f'{r.prediction_x1e4:.4f} & [{lo:.4f}, {hi:.4f}] '+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (SYN/'tab_expansion.tex').write_text('\n'.join(tex)+'\n')
    primary=summ[summ.n.isin([700,1000])]
    rows=['| Law | n | Omega median error | Density median error | Estimated/empirical variance | 95% interval coverage |',
          '|---|---:|---:|---:|---:|---:|']
    for r in summ.itertuples():
        rows.append(f'| {r.law} | {r.n} | {100*r.median_omega_relative_error:.2f}% | '
                    f'{100*r.median_density_relative_error:.2f}% | {r.variance_ratio:.4f} | {100*r.coverage:.1f}% |')
    expansion_rows=['| Law | n | Raw bias | Mean change | Prediction | Relative discrepancy |',
                    '|---|---:|---:|---:|---:|---:|']
    for r in exp.itertuples():
        expansion_rows.append(f'| {r.law} | {r.n} | {r.bias:.6f} | {r.observed_delta_x1e4:.4f} | '
                              f'{r.prediction_x1e4:.4f} | {100*r.relative_discrepancy:.2f}% |')
    q1000=summ[summ.n==1000].set_index('law')
    totalchecks=sum(json.loads(p.read_text()).get('negative_controls',0)
                    for p in [OUT/'preflight.json',SYN/'runtime_checks.json',SYN/'independent_validation.json'])
    lock=json.loads((OUT/'lock.json').read_text())
    report=f'''# R8 theory–empirics loop: results and stopping decision

**Synthetic admission: FAILED. Financial deliverables 1–3: NOT RUN.**
The estimators were implemented and tested, but the calibration-density and
short-sample long-run-variance estimates did not meet the author's fixed
precision requirements. No financial forecasts were refitted or read by the
new experiment. No panel shrinkage, prediction correlation, boundary concordance
or financial confidence band is reported as computed.

## Locked protocol and interpretation

The complete plan was committed before numerical execution as
`{lock['protocol_commit']}` in the local protocol-only repository.
Plan SHA-256: `{lock['plan_sha256']}`. This is not historical Git provenance.
The original plan remains intact. `DECISIONS.md` records both runtime and
mathematical corrections; the earlier full run is retained under
`prior_expansion_definition/`. No tolerance or primary estimator was changed.

The ECDF integral requested as B is the signed empirical corrected-minus-raw
loss. We preserve it and define removable error as its negative. The prescribed
SBG spacing size exceeds the available upper order statistics at every tested
n, so the secondary sensitivity is unavailable rather than silently modified.

## Synthetic validation

There are 500 independent GARCH histories per innovation law, with nested
prefixes at five sizes; zero/constant translations share those histories.
Thus the derived rows are not independent new simulations. Two independent
million-observation references establish density truth by conditional-law
integration; target hits in this oracle-shape GARCH control are iid and give
Omega=.0099 exactly. The exercise does not establish accuracy for arbitrary
financial hit dependence. Warm-up approximates stationarity.

Admission required BOTH laws at BOTH n=700 and n=1000 to have median absolute
relative errors below15% and a mean estimated/empirical quantile-variance ratio
inside [.85,.98]. Nominal95% interval coverage is reported separately and never
substitutes for the variance ratio. The failure persists under the alternative
coverage reading because the density criterion fails in both laws.

{chr(10).join(rows)}

Source for every cell: {evidence(SYN/'validation_summary.csv')}.
This CSV also gives bootstrap Monte Carlo intervals for the medians/variance
ratios and Wilson intervals for empirical coverage. At n=1000 density errors
are {100*q1000.loc['normal','median_density_relative_error']:.2f}% and
{100*q1000.loc['t5','median_density_relative_error']:.2f}%, exceeding15%.
No tested n through2000 passes the density gate or the joint gate.
The first passing size for each individual criterion is preserved in
{evidence(SYN/'first_passing_size.csv')}.

At a conformal empirical rank the number of scores strictly above C_n is fixed.
The primary HAC estimator uses exactly these sample-threshold hits and sample
centering as requested. No population threshold, population centering, or
finite-rank bias correction was substituted to improve the result.

The long-reference precision, including simulated indicator LRV alongside its
known truth, is in {evidence(SYN/'truth.csv')}.
All bias-corrected h-squared zero fractions, LRV floor fractions and synthetic
lambda summaries are in {evidence(SYN/'floor_diagnostics.csv')}.
They are synthetic diagnostics, not panel evidence for deployable shrinkage.
Spacing failure ranks are in {evidence(SYN/'spacing_status.csv')}.
All primary and Andrews estimates, floor flags, invalid SBG statuses and literal
wrong-sign predictions remain in {evidence(SYN/'estimators.csv')}.

## Expansion: population result versus an estimated plug-in

The following comparison uses the same long-reference volatility distribution
for risk integration and density. The earlier1,024-state calculation mixed a
finite evaluation law with a different calibration curvature; Deviation02
corrects this and preserves both its literal and two-density predictions.
The primary comparison is independent-marginal expected loss, not a numerical
test of contiguous transfer. GARCH population moments are approximated by the
long reference, with the saved reference precision; this does not prove a theorem.

Changes below are QS x10^4. Relative discrepancy is (observed-predicted)/
abs(predicted), not an equivalence test. A sign match alone is not statistical
confirmation; the complete MCSE/interval evidence accompanies all cells.

{chr(10).join(expansion_rows)}

Source for every cell: {evidence(SYN/'expansion.csv')}.
Paired history losses and MCSE inputs: {evidence(SYN/'loss_histories.csv')}.
The fixed-state diagnostic, including its corrected two-density leading term:
{evidence(SYN/'finite_reference_diagnostic.csv')}.
The 63/127-degree integration comparison and direct checks use the original
4e-11 loss tolerances; saved nodes and coefficients permit independent replay.

## Deliverable status and prohibited conclusions

- Deliverable1: no predicted-versus-realised panel loss, correlations, regression,
  sign concordance or order-of-magnitude success verdict.
- Deliverable2: no panel plug-in shrinkage forecasts, comparison family or
  superiority claim. The fixed-half, five-fraction and Vol-ERM results are intact.
- Deliverable3: no newly computed ERM/CP boundary statistic or state concordance.
  The existing161-pair exploratory results remain unchanged.
- The main240 and external48 panels were not entered. Their original results
  remain valid on their original evidence, not validated by this unfinished loop.

Machine-readable status: {evidence(OUT/'deliverable_status.csv')}.
No empty placeholder is labelled a produced per-pair result. This stopping rule
is the author's requested response to failed synthetic admission, not an
unreported omission of unfavourable financial outcomes.

## What this changes in R8

No manuscript source was edited. Proposed exact statements, if this negative
validation is integrated in a separately authorised editing step:

| Section | Sentence supported, or claim withheld |
|---|---|
| Introduction | "Population estimation-cost formulas motivate a feasible correction, but their plug-in implementation requires separate validation at the target tail level." |
| Section4, Proposition4.1/Corollary4.2 | "The expansion concerns population density and long-run variance; substituting estimated quantities does not inherit its accuracy automatically." |
| Section4, shrinkage | "The oracle shrinkage fraction remains a population benchmark; the specified kernel-based plug-in did not pass the prespecified synthetic precision gate." |
| Section5 | "With 1,000 calibration observations, median absolute relative density errors were {100*q1000.loc['normal','median_density_relative_error']:.1f}% under Normal innovations and {100*q1000.loc['t5','median_density_relative_error']:.1f}% under Student-t(5) innovations, above the 15% threshold." |
| Section6, main results | WITHHOLD: "The theoretical estimation-cost prediction explains realised panel loss changes." The requested panel test was not run. |
| Section7, indication | WITHHOLD: "Calibration-only plug-in shrinkage improves on full, half or selected shifts." No new forecast comparison exists. |
| External evaluation | WITHHOLD: any new transfer claim for this plug-in; the existing external comparisons are unchanged. |
| Proposition4.4 companion | WITHHOLD: any panel confirmation of the estimated volatility boundary; a failed density gate blocks its use here. |
| Discussion | "At the 1% tail, estimating the inputs to a population correction rule can itself be a substantial source of error." Scope: these prescribed estimators in the stated GARCH controls. |

No claim or formula in the existing manuscript is invalidated merely by this
failed plug-in. The theory–empirics loop remains open for the proposed estimator.
A different estimator would require a new declared protocol and fresh synthetic
validation; this run does not select one after seeing financial outcomes.

## Verification and reproducibility

At report generation, {totalchecks} paired rejection-first checks validate code,
source preservation, exact R bandwidth replay, array arithmetic, summaries,
integration and the financial execution barrier. Computational checks pass;
statistical admission fails. Reports: `preflight.json`,
`synthetic/runtime_checks.json`, `synthetic/independent_validation.json`.
All143 previously validated R8 source/release files retain their hashes.
The final provenance receipt separately binds current files, their mtimes,
the generated figures/tables and this report; it never lists nonexistent results.
No manuscript edit occurred, so no new R8 PDF build is claimed or needed.

See `research/r8_theory_loop/README.md` for verification and full isolated replay.
`manuscript_numbers.csv` maps each data-derived numeric field to an input row and
SHA-256. PDF figures are `synthetic/estimator_validation.pdf` and
`synthetic/loss_expansion.pdf`, with transparent backgrounds and legends below.
Figure1 ribbons are descriptive Monte Carlo intervals; Figure2 intervals are
pointwise over independent simulated histories, not financial simultaneous bands.
'''
    (OUT/'RESULTS.md').write_text(report)
    print('Negative-validation report, CSV statuses, LaTeX tables and two figures generated.')


if __name__=='__main__':
    main()
