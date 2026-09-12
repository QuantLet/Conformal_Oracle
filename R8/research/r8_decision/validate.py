"""Fresh-process replay and future-outcome perturbation for representative pairs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import methods as m
from run import OUT,PROJECT,work,load,compute,sha

REPLAY=[('GJR-GARCH-t','SP500'),('Moirai-2.0','NATGAS'),('Lag-Llama','BTC'),('TimesFM-2.5','DJCI')]


def main():
    records=[]
    for model,asset in REPLAY:
        work(model,asset,replay=True);key=f'{model}__{asset}'
        expected=OUT/'pairs'/key;actual=OUT/'replay'/key
        for file in ['daily.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(expected/file),pd.read_parquet(actual/file),check_exact=True)
        for file in ['metrics.csv','dtaci_seed_metrics.csv','fits.json']:
            assert (expected/file).read_bytes()==(actual/file).read_bytes(),(key,file)
        with np.load(expected/'dtaci_experts.npz') as a,np.load(actual/'dtaci_experts.npz') as b:
            assert a.files==b.files
            for name in a.files:np.testing.assert_array_equal(a[name],b[name])
        records.append({'pair':key,'all_numeric_outputs_exact':True})
        print('Exact fresh replay',key,flush=True)
    # The outcome perturbation checks the complete actual fitting/selection
    # pipeline, not only isolated helper functions.
    model,asset=REPLAY[0];key=f'{model}__{asset}'
    y,q,sigma,index,ref,binding=load(model,asset);nc=int(.7*len(y))
    changed=y.copy();changed[nc:]+=np.linspace(.1,1,len(y)-nc)
    preds,params,proj,orig,mix,seeds=compute(changed,q,sigma,nc,key)
    expected=json.loads((OUT/'pairs'/key/'fits.json').read_text())
    for name in ['state','POT-Shift','POT-Vol','gate']:
        assert params[name]==expected[name],name
    previous=pd.read_parquet(OUT/'pairs'/key/'daily.parquet')
    for name in ['State-L1','State-L1-clipped','POT-Shift','POT-Vol']:
        np.testing.assert_array_equal(preds[name],previous[name].to_numpy())
    with np.load(OUT/'pairs'/key/'dtaci_experts.npz') as z:
        np.testing.assert_array_equal(proj['predictions'][:nc+1],z['projected_q'][:nc+1])
        np.testing.assert_array_equal(proj['probabilities'][:nc+1],z['projected_p'][:nc+1])
        np.testing.assert_array_equal(orig['predictions'][:nc+1],z['unprojected_q'][:nc+1])
    assert preds['Loss-gate'][0]==previous['Loss-gate'].iloc[0]
    before=json.loads((OUT/'before.json').read_text())
    assert all(sha(PROJECT/p)==h for p,h in before['canonical'].items())
    report={'fresh_replays':records,'future_outcome_perturbation':{'pair':key,'all_selections_and_static_fits_unchanged':True,
             'adaptive_prefix_through_first_test_forecast_unchanged':True},
            'canonical_files_unchanged':len(before['canonical']),
            'code':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md'))}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Future-outcome perturbation passed; canonical files unchanged.',flush=True)


if __name__=='__main__':main()
