"""Validate unchanged inputs, replacement provenance and all daily panel losses."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from panel_scope import PROJECT, OLD, NEW, ART, ROOT, DECISION, MODELS, ASSETS, REPLACEMENTS, source


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stage(record):
    assert record['producer_sha256']==sha(Path(__file__).with_name('stage_panel.py'))
    assert record['scope_sha256']==sha(Path(__file__).with_name('panel_scope.py'))
    for path,digest in record['inputs'].items(): assert sha(PROJECT/path)==digest,('source changed',path)
    for name,item in record['copies'].items():
        assert sha(ROOT/name)==item['sha256']==sha(PROJECT/item['source']),('copy changed',name)
    assert record['replacements']==REPLACEMENTS and record['assets']==ASSETS
    assert set(p.stem for p in (ROOT/'data/returns').glob('*.csv'))==set(ASSETS)


def native():
    rows=[]
    for model in ['moirai','lagllama']:
        old=json.loads((OLD/'quality'/f'native_replay_{model}_primary.json').read_text())
        for asset in ASSETS:
            expected=len(pd.read_csv(ROOT/'data/returns'/f'{asset}.csv'))-512
            fp=ROOT/'data'/model/f'{asset}.parquet'
            assert sha(fp)==sha(source(asset)/'data'/model/f'{asset}.parquet')
            if asset not in REPLACEMENTS.values():
                row=next(r for r in old['rows'] if r['asset']==asset)
                assert row['exact'] and all(b['exact'] for b in row['batches']) and row['hashed_rows']==expected
                folder=PROJECT/row['native_directory'] if 'native_directory' in row else OLD/'native'/model/asset
                assert sha(folder/'binding.json')==row['binding_sha256']
                binding=json.loads((folder/'binding.json').read_text())
                assert binding['input_sha256']==sha(ROOT/'data/returns'/f'{asset}.csv')
                total=0
                for chunk in sorted(folder.glob('*.npz')):
                    item=json.loads(chunk.with_suffix('.json').read_text())
                    assert sha(chunk)==item['sha256']
                    total+=item['rows']
                assert total==expected
            else:
                folder=NEW/'native'/model
                report=json.loads((folder/'replay.json').read_text())
                assert report['exact_fresh_replay'] and report['binding_sha256']==sha(folder/'binding.json')
                item=next(r for r in report['assets'] if r['asset']==asset)
                assert item['rows']==expected==sum(c['rows'] for c in item['chunks'])
                for chunk in item['chunks']:
                    assert sha(folder/asset/chunk['file'])==chunk['sha256']
                    assert chunk['input_sha256']==sha(ROOT/'data/returns'/f'{asset}.csv')
                reduction_name='reduction_lagllama.json' if model=='lagllama' else 'reduction_moirai_chronos2_patchtst.json'
                reductions=json.loads((NEW/'quality'/reduction_name).read_text())['records']
                red=next(r for r in reductions if r['model']==model and r['asset']==asset)
                assert red['output_sha256']==sha(fp) and red['exact_native_replay']
            rows.append(dict(model=model,asset=asset,rows=expected))
    assert len(rows)==48
    return rows


def numerical():
    metrics=pd.read_csv(ROOT/'results/posthoc.csv'); decisions=pd.read_csv(DECISION/'pairs.csv')
    assert len(metrics)==1680 and len(decisions)==3360
    assert not metrics.duplicated(['model','asset','method']).any()
    assert not decisions.duplicated(['model','asset','method']).any()
    count=0
    for model,(folder,suffix) in MODELS.items():
        for asset in ASSETS:
            daily=pd.read_parquet(ROOT/'posthoc'/f'{model}__{asset}.parquet')
            pred=pd.read_parquet(ROOT/'data'/folder/(f'{asset}_{suffix}.parquet' if suffix else f'{asset}.parquet'))
            ret=pd.read_csv(ROOT/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True).log_return
            nc=int(.7*len(pred)); assert daily.index.equals(pred.index[nc:])
            np.testing.assert_array_equal(pred.loc[daily.index,'VaR_0.01'],daily.Raw)
            np.testing.assert_array_equal(ret.loc[daily.index],daily.r)
            for row in metrics[(metrics.model==model)&(metrics.asset==asset)].itertuples():
                q=daily[row.method].to_numpy();y=daily.r.to_numpy();hits=y<q
                loss=np.where(hits,.99*(q-y),.01*(y-q))
                assert int(hits.sum())==row.viol and len(y)==row.n_test
                np.testing.assert_allclose(loss.mean(),row.QS,atol=1e-15,rtol=1e-11)
                count+=len(y)
    for kind in ['review','decision']:
        record=json.loads((ART/f'aggregation_{kind}.json').read_text())
        assert record['pairs']==168
        for path,digest in record['inputs'].items():assert sha(PROJECT/path)==digest
        for path,digest in record['outputs'].items():assert sha(ART/path)==digest
    common=pd.read_csv(ROOT/'results/common_support.csv')
    assert len(common)==24*22 and common.groupby('asset')[['first','last','n_test']].nunique().eq(1).all().all()
    assert set(common.asset)==set(ASSETS)
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');args=p.parse_args()
    record=json.loads((ART/'stage.json').read_text());stage(record)
    damaged=json.loads(json.dumps(record));damaged['copies'][next(iter(damaged['copies']))]['sha256']='0'*64
    try:stage(damaged)
    except AssertionError:pass
    else:raise AssertionError('Input-corruption negative control failed')
    if args.replay:
        files=['windows.csv','gaps.csv','master.csv','policy.csv','strata.csv','order_convention.csv','common_support.csv',
               'grid_fit_quality.csv','class_sensitivity.csv','paired_loss_intervals.csv','bootstrap_L20.npz','bootstrap_L60.npz']
        before={name:sha(ROOT/'results'/name) for name in files}
        subprocess.run([sys.executable,str(Path(__file__).with_name('analyse_panel.py'))],check=True)
        assert before=={name:sha(ROOT/'results'/name) for name in files},'Base aggregation replay differs'
        for kind in ['review','decision']:
            subprocess.run([sys.executable,str(Path(__file__).with_name('aggregate_panel.py')),'--kind',kind,'--check'],check=True)
    natives=native();losses=numerical()
    result=dict(status='passed',stage_sha256=sha(ART/'stage.json'),assets=24,pairs=168,
        historical_assets_byte_identical=21,reestimated_assets=3,native_rows=sum(r['rows'] for r in natives),
        native_series=len(natives),daily_posthoc_losses=losses,exact_aggregation_replay=args.replay,
        input_corruption_negative_control=True,producer_sha256=sha(__file__))
    (ART/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
