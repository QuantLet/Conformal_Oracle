"""Adopt independently replayed actual-calendar outputs; retain old controls."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import reduce_calendar as reduce

PROJECT=reduce.PROJECT;ROOT=reduce.OLD;CAL=reduce.ROOT


def forecasts(root=ROOT):
    reduction=json.loads((CAL/'reduction.json').read_text());assert reduction['complete'] and reduction['n_assets']==24
    replay=json.loads((CAL/'actual_full_replay.json').read_text());assert replay['complete'] and len(replay['rows'])==24
    for row in replay['rows']:
        asset=row['asset'];native=CAL/'native'/asset
        assert row['exact'] and row['parameters_exact'] and all(b['exact'] for b in row['batches'])
        assert row['binding_sha256']==reduce.sha(native/'binding.json')
        b=reduction['assets'][asset];assert b['return_sha256']==reduce.sha(root/'data/returns'/f'{asset}.csv')
        # Re-reduce the native samples when used in a fresh reconstruction.
        import numpy as np
        import pandas as pd
        blocks=[]
        for p in sorted(native.glob('*.npz')):
            assert reduce.sha(p)==b['chunks'][p.name]
            with np.load(p) as z:
                samples=z['actual_samples'];index=pd.DatetimeIndex(z['date'].astype('datetime64[ns]'),name='date')
                block=pd.DataFrame({'mean':samples.mean(axis=1),'std':samples.std(axis=1)},index=index)
                quantiles=np.percentile(samples,np.array(reduce.ALPHAS)*100,axis=1).T
                for j,alpha in enumerate(reduce.ALPHAS):block[f'VaR_{alpha:g}']=quantiles[:,j]
                blocks.append(block)
        target=root/'data/lagllama'/f'{asset}.parquet';target.parent.mkdir(parents=True,exist_ok=True)
        pd.concat(blocks).to_parquet(target)
        # Percentile vectorisation can affect no values; require exact equality.
        expected=pd.read_parquet(CAL/'data/lagllama'/target.name)
        pd.testing.assert_frame_equal(pd.read_parquet(target),expected,check_exact=True)
        params=root/'parameters/lagllama'/target.name;params.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(CAL/'parameters'/target.name,params)
        meta=dict(forecast_sha256=reduce.sha(target),parameters_sha256=reduce.sha(params),
                  native_chunks=b['chunks'],native_directory=str(native.relative_to(PROJECT)),native_array='actual_samples',
                  input_sha256=b['return_sha256'],producer_sha256=reduce.sha(__file__),
                  reduction_sha256=reduce.sha(CAL/'reduction.json'),calendar='actual observed timestamps')
        prov=root/'provenance/lagllama'/f'{asset}.json';prov.parent.mkdir(parents=True,exist_ok=True)
        prov.write_text(json.dumps(meta,indent=2)+'\n')
        row['native_directory']=str(native.relative_to(PROJECT))
    quality=root/'quality';quality.mkdir(exist_ok=True)
    (quality/'native_replay_lagllama_primary.json').write_text(json.dumps(replay,indent=2)+'\n')
    (root/'calendar_primary.json').write_text(json.dumps(dict(complete=True,assets=24,
        reduction_sha256=reduce.sha(CAL/'reduction.json'),replay_sha256=reduce.sha(CAL/'actual_full_replay.json')),indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',type=Path);a=ap.parse_args()
    if a.destination:
        forecasts(a.destination.resolve());return
    archive=CAL/'before_adoption'
    if not archive.exists():
        archive.mkdir()
        for name in ['data/lagllama','parameters/lagllama','provenance/lagllama','draws/lagllama','results','quality']:
            target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ROOT/name,target)
        (archive/'posthoc').mkdir()
        for p in (ROOT/'posthoc').glob('Lag-Llama__*'):shutil.copy2(p,archive/'posthoc'/p.name)
    assert json.loads((CAL/'pools_replay.json').read_text())['exact']
    forecasts()
    for p in (CAL/'draws/lagllama').iterdir():shutil.copy2(p,ROOT/'draws/lagllama'/p.name)
    shutil.copy2(CAL/'pools_replay.json',ROOT/'quality/pools_replay_lagllama.json')
    # Refit affected methods only. Existing unaffected bindings remain valid.
    for p in (ROOT/'posthoc').glob('Lag-Llama__*'):p.unlink()
    (ROOT/'quality/full_reconstruction.json').write_text(json.dumps(dict(complete=False,
        reason='Actual-calendar outputs adopted; affected downstream refits and fresh comparison pending.'),indent=2)+'\n')
    print('Adopted 24 actual-calendar forecasts and two independently replayed pools',flush=True)


if __name__=='__main__':main()
