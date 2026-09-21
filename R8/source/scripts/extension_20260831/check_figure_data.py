#!/usr/bin/env python3
"""Check restored R8 figures against daily paths and simulation replications.

Read-only statistical checks: no forecast fitting, inference or simulation.
Run after build_paper_outputs.py. Display hashing/replay is handled separately
by validate_r8.py.
"""
import json
import numpy as np
import pandas as pd
from panel_statistics import ROOT, load_pair


def main():
    table=pd.read_csv(ROOT/'results/posthoc.csv')
    cells=0
    for (model,asset),group in table.groupby(['model','asset']):
        daily=pd.read_parquet(ROOT/'posthoc'/f'{model}__{asset}.parquet')
        for method in ['Raw','Conformal']:
            row=group[group.method==method].iloc[0]
            n=len(daily);viol=int((daily.r<daily[method]).sum())
            # Integer comparisons avoid ambiguity at the zone boundaries.
            zone='Green' if 250*viol<=4*n else ('Yellow' if 250*viol<=9*n else 'Red')
            assert row.n_test==n and row.viol==viol and row.TL==zone,(model,asset,method)
            cells+=1
    assert cells==432

    grid=pd.read_csv(ROOT/'results/monte_carlo/grid.csv')
    reps=pd.read_csv(ROOT/'results/monte_carlo/replications.csv')
    for row in grid.itertuples():
        d=reps[(reps.dgp==row.dgp)&(reps['T']==row.T)]
        assert len(d)==500
        expected=[100*d.corr_pi.mean(),(d.raw_QS-d.corr_QS).mean()*1e4,
                  100*(d.raw_pi<=4/250).mean(),d.qV.std(ddof=1)*100]
        actual=[row.Corr_pi*100,row.DQS_mean*1e4,row.RawGreen,row.Std_qV*100]
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)

    count=0;maximum=0.
    for model in ['TimesFM-2.5','Moirai-2.0','Moirai-1.1','Lag-Llama','GJR-GARCH']:
        y,p=load_pair(model,'SP500');score=p['VaR_0.01'].to_numpy()-y
        daily=pd.read_parquet(ROOT/'posthoc'/f'{model}__SP500.parquet')
        locations=p.index.get_indexer(daily.index);assert (locations>=250).all()
        # Sort each preceding window independently, using the finite-sample
        # rank. The plotted path is instead recovered by subtracting forecasts.
        rank=int(np.ceil(251*.99))
        expected=np.array([np.sort(score[t-250:t])[rank-1] for t in locations])
        actual=(daily.Raw-daily['rolling']).to_numpy()
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
        maximum=max(maximum,float(np.max(np.abs(actual-expected))))
        count+=len(actual)
    result=dict(traffic_light_cells=cells,simulation_cells=len(grid),
                simulation_replications=len(reps),rolling_daily_shifts=count,
                maximum_shift_difference=maximum,passed=True,
                inference_fitting_or_simulation_run=False)
    (ROOT/'quality/figure_data_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
