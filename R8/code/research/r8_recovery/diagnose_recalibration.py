"""Exploratory diagnostics from stored R8 forecasts; no new forecast models.

The score-optimal test-window shift is a hindsight diagnostic, never a
deployable comparator. Quantile minimisers use the inverse empirical CDF.
Nothing is written into the manuscript's numerical dependency closure.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO=Path(__file__).resolve().parents[2]
CODE=REPO/'source/scripts/extension_20260831'
sys.path.insert(0,str(CODE))
from panel_statistics import ROOT,MODELS,ALPHAS,load_pair,qshift
OUT=Path(__file__).resolve().parent

def loss(y,q,alpha):
    error=y-q
    return float(np.mean((alpha-(error<0))*error))

def optimum(s,alpha):
    p=1-alpha
    value=float(np.quantile(s,p,method='inverted_cdf'))
    assert np.mean(s<value)<=p+1e-12 and np.mean(s<=value)>=p-1e-12
    return value

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows=[];bindings={};max_optimum_error=0.
    for asset in sorted(p.stem for p in (ROOT/'data/returns').glob('*.csv')):
        for model in MODELS:
            y,forecast=load_pair(model,asset);ncal=int(.7*len(y));test_y=y[ncal:]
            daily_path=ROOT/'posthoc'/f'{model}__{asset}.parquet'
            fit_path=daily_path.with_suffix('.json')
            daily=pd.read_parquet(daily_path);fit=json.loads(fit_path.read_text())
            assert sha(daily_path)==fit['daily_sha256']
            directory,suffix=MODELS[model]
            forecast_path=ROOT/'data'/directory/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet')
            return_path=ROOT/'data/returns'/f'{asset}.csv'
            for path,key in [(forecast_path,'forecast_sha256'),(return_path,'return_sha256')]:
                assert sha(path)==fit['binding'][key]
                bindings[str(path.relative_to(REPO))]=sha(path)
            assert daily.index.equals(forecast.index[ncal:]) and np.array_equal(daily.r,test_y)
            bindings[str(daily_path.relative_to(REPO))]=sha(daily_path)
            bindings[str(fit_path.relative_to(REPO))]=sha(fit_path)
            for alpha in ALPHAS:
                q=forecast[f'VaR_{alpha:g}'].to_numpy();qcal=q[:ncal];qtest=q[ncal:]
                shift=qshift(qcal-y[:ncal],alpha)
                cal_opt=optimum(qcal-y[:ncal],alpha)
                test_opt=optimum(qtest-test_y,alpha)
                raw=loss(test_y,qtest,alpha)
                corrected=loss(test_y,daily[f'static_{alpha:g}'].to_numpy(),alpha)
                hindsight=loss(test_y,qtest-test_opt,alpha)
                # Independent global check at every breakpoint of the convex
                # empirical loss, using sorted prefix sums rather than quantiles.
                ordered=np.sort(qtest-test_y);count=len(ordered);k=np.arange(1,count+1)
                prefix=np.cumsum(ordered)
                all_losses=(alpha*(k*ordered-prefix)+(alpha-1)*((count-k)*ordered-(prefix[-1]-prefix)))/count
                error=abs(hindsight-float(all_losses.min()))
                max_optimum_error=max(max_optimum_error,error)
                assert error<1e-12
                assert np.allclose(qtest-shift,daily[f'static_{alpha:g}'],atol=1e-15,rtol=0)
                potential=raw-hindsight;transfer=corrected-hindsight;realised=raw-corrected
                assert potential>=-1e-14 and transfer>=-1e-14
                assert abs(realised-(potential-transfer))<1e-15
                row=dict(model=model,asset=asset,alpha=alpha,n_cal=ncal,n_test=len(test_y),
                    calibration_shift=shift,calibration_optimum=cal_opt,test_hindsight_optimum=test_opt,
                    raw_QS=raw,static_QS=corrected,hindsight_shift_QS=hindsight,
                    possible_shift_gain=potential,transfer_shortfall=transfer,realised_gain=realised,
                    calibration_rank_cost=loss(y[:ncal],qcal-shift,alpha)-loss(y[:ncal],qcal-cal_opt,alpha))
                if alpha==.01:
                    historical=loss(test_y,daily['Hist-Quantile'].to_numpy(),alpha)
                    row.update(historical_QS=historical,value_over_historical=historical-corrected)
                    for method,expected in [('Raw',raw),('Conformal',corrected),('Hist-Quantile',historical)]:
                        stored=next(m['QS'] for m in fit['metrics'] if m['method']==method)
                        assert abs(stored-expected)<1e-14
                rows.append(row)
    data=pd.DataFrame(rows);data.to_csv(OUT/'diagnostic_pairs.csv',index=False)
    byalpha=[]
    for alpha,d in data.groupby('alpha'):
        p=d.possible_shift_gain.mean();gap=d.transfer_shortfall.mean();gain=d.realised_gain.mean()
        byalpha.append(dict(alpha=alpha,pairs=len(d),raw_QS_x1e4=d.raw_QS.mean()*1e4,
            static_QS_x1e4=d.static_QS.mean()*1e4,possible_gain_x1e4=p*1e4,
            transfer_shortfall_x1e4=gap*1e4,realised_gain_x1e4=gain*1e4,
            fraction_of_aggregate_hindsight_gain_realised=gain/p,
            deteriorations=int((d.realised_gain<0).sum()),
            shift_rank_correlation_calibration_test=float(spearmanr(d.calibration_shift,d.test_hindsight_optimum).statistic)))
    pd.DataFrame(byalpha).to_csv(OUT/'diagnostic_by_alpha.csv',index=False)
    d=data[data.alpha==.01]
    bymodel=d.groupby('model').agg(raw_QS=('raw_QS','mean'),static_QS=('static_QS','mean'),
        historical_QS=('historical_QS','mean'),possible_shift_gain=('possible_shift_gain','mean'),
        transfer_shortfall=('transfer_shortfall','mean'),realised_gain=('realised_gain','mean'),
        static_beats_historical=('value_over_historical',lambda x:int((x>0).sum())))
    bymodel.to_csv(OUT/'diagnostic_by_model.csv')
    ci=pd.read_csv(ROOT/'results/paired_loss_intervals.csv')
    relevant=ci[(ci.comparator=='Hist-Quantile')&(ci.reference=='Conformal')].drop_duplicates('block_calendar_days')
    relevant.to_csv(OUT/'existing_historical_comparison_intervals.csv',index=False)
    result=dict(exploratory=True,inference_or_simulations_run=False,diagnostic_only=True,
        alpha_comparison=byalpha,historical_baseline=dict(
            mean_QS_x1e4=d.historical_QS.mean()*1e4,static_QS_x1e4=d.static_QS.mean()*1e4,
            relative_mean_loss_reduction=1-d.static_QS.mean()/d.historical_QS.mean(),
            pairs_improved=int((d.value_over_historical>0).sum()),pairs=len(d),
            models_improved=int((bymodel.static_QS<bymodel.historical_QS).sum()),
            existing_paired_intervals=relevant.to_dict('records')),
        verification={'stored_metrics_match':True,'exact_loss_identity':True,'quantile_subgradient_check':True,
            'independent_all_breakpoint_checks':len(rows),'maximum_loss_difference':max_optimum_error},
        interpretation='The hindsight gap combines calibration estimation, rank convention and temporal distribution change; these components are not identified separately.',
        inputs=bindings,producer_sha256=sha(Path(__file__)))
    (OUT/'diagnostic_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','producer_sha256']},indent=2))

if __name__=='__main__':main()
