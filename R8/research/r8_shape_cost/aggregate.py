"""Whole-history simultaneous Monte Carlo bands, predeclared families."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import json
import numpy as np
import pandas as pd
import engine as e


def bands(values, indices):
    means = values.mean(axis=0)
    se = values.std(axis=0, ddof=1)/np.sqrt(len(values))
    assert np.all(se > 0)
    boot = np.empty((len(indices), values.shape[1]))
    for start in range(0, len(indices), 25):
        boot[start:start+25] = values[indices[start:start+25]].mean(axis=1)
    maximum = np.max(np.abs((boot-means)/se),axis=1)
    critical = float(np.quantile(maximum,.95,method='higher'))
    return means, se, critical, boot


def main():
    lock=json.loads((e.OUT/'lock.json').read_text())
    for name,digest in lock['files'].items():
        assert e.sha(e.ROOT/name)==digest, name
    execution=json.loads((e.OUT/'execution.json').read_text())
    for name,digest in execution['outputs'].items():
        assert e.sha(e.OUT/name)==digest, name
    frame = pd.read_parquet(e.OUT/'replications.parquet')
    info = e.cells()
    cells = [c['cell'] for c in info]
    matrices = {m: frame[frame.method==m].pivot(index='replication',columns='cell',values='loss')
                .reindex(index=np.arange(e.REPS),columns=cells).to_numpy() for m in e.METHODS}
    rng = np.random.default_rng(e.BOOT_SEED)
    indices = rng.integers(0,e.REPS,size=(e.BOOT_DRAWS,e.REPS),dtype=np.int32)
    np.savez_compressed(e.OUT/'bootstrap_indices.npz', indices=indices)
    primary = matrices['Vol-ERM']-matrices['Shift-ERM']
    sens_methods = ('Vol-UERM','Vol-CP','Shift-CP')
    sensitivity = np.concatenate([matrices[m]-matrices['Shift-ERM'] for m in sens_methods],axis=1)
    rows=[]; receipts={}
    for family,values,methods in [('primary',primary,['Vol-ERM']),
                                   ('sensitivity',sensitivity,list(sens_methods))]:
        mean,se,critical,bootstrap = bands(values,indices)
        receipts[family] = dict(contrasts=values.shape[1],critical=critical)
        np.savez_compressed(e.OUT/(family+'_bootstrap.npz'),means=bootstrap)
        for mpos,method in enumerate(methods):
            for cpos,cell in enumerate(info):
                j=mpos*len(cells)+cpos
                target=cell['predicted_delta'] if family=='primary' else (
                    0. if method=='Shift-CP' else (e.A-e.C)*(
                    e.ALPHA*(1-e.ALPHA)/cell['g']-cell['g']*cell['h']**2)/(2*cell['n']))
                rows.append(dict(family=family,method=method,reference='Shift-ERM',**cell,
                    mean_delta=mean[j],MCSE=se[j],simultaneous_lower=mean[j]-critical*se[j],
                    simultaneous_upper=mean[j]+critical*se[j],critical=critical,
                    leading_prediction=target,discrepancy=mean[j]-target,
                    n_scaled_delta=cell['n']*mean[j],n_scaled_prediction=cell['n']*target))
    summary=pd.DataFrame(rows)
    summary.to_csv(e.OUT/'contrasts.csv',index=False)
    metrics=['loss','marginal_loss','excess_vs_conditional_oracle','violation','coefficient',
             'low_state_loss','high_state_loss','low_state_violation','high_state_violation']
    marginal=frame.groupby(['cell','kind','n','h_factor','method'],sort=False)[metrics].agg(['mean','std'])
    marginal.columns=['_'.join(c) for c in marginal.columns]
    marginal.reset_index().to_csv(e.OUT/'method_summary.csv',index=False)
    primary_rows=summary[(summary.family=='primary') & summary.primary]
    low=primary_rows[primary_rows.h_factor==.5].iloc[0]
    high=primary_rows[primary_rows.h_factor==2.].iloc[0]
    passed=bool(low.simultaneous_lower>0 and high.simultaneous_upper<0)
    adverse=bool(low.simultaneous_upper<0 or high.simultaneous_lower>0)
    verdict='supported' if passed else ('contradicted_at_primary_finite_n' if adverse else 'unresolved')
    result=dict(status='complete',primary_sign_crossing=verdict,
        primary_cells=primary_rows.cell.tolist(),families=receipts,
        bootstrap_draws=e.BOOT_DRAWS,independent_histories=e.REPS,
        bootstrap_seed=e.BOOT_SEED,conditioning='Exact contiguous expectation conditional on calibration history',
        quantitative_equivalence_claimed=False,financial_generalisation_claimed=False,
        all_primary_cells_observed_sign_agrees=bool(np.all(np.sign(summary[summary.family=='primary'].mean_delta)
            ==np.sign(summary[summary.family=='primary'].leading_prediction))))
    files=['bootstrap_indices.npz','primary_bootstrap.npz','sensitivity_bootstrap.npz','contrasts.csv','method_summary.csv']
    result['outputs']={p:e.sha(e.OUT/p) for p in files}
    (e.OUT/'findings.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
