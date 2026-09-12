"""Independent full replay of corrected calendar forecasts in a fresh process."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import lag_calendar as calendar


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2)
    assert torch.backends.mps.is_available()
    interface=calendar.Interface('mps')
    records=[]
    for retpath in sorted((calendar.OLD/'data/returns').glob('*.csv')):
        asset=retpath.stem;folder=calendar.OUT/'native'/asset
        report=folder/'actual_replay.json'
        if report.exists():
            saved=json.loads(report.read_text())
            assert saved['checker_sha256']==calendar.sha(__file__)
            assert saved['binding_sha256']==calendar.sha(folder/'binding.json')
            records.append(saved);continue
        while not (folder/'complete.json').exists():
            if not a.watch:raise RuntimeError(f'Incomplete corrected asset: {asset}')
            time.sleep(5)
        start=time.monotonic()
        ret=pd.read_csv(retpath,index_col='date',parse_dates=True).log_return
        values=ret.to_numpy(np.float32);batches=[]
        for path in sorted(folder.glob('*.npz')):
            assert calendar.sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
            with np.load(path) as z:
                positions=z['positions']
                assert np.array_equal(z['date'],ret.index[positions].values.astype('datetime64[D]'))
                for i in range(0,len(positions),calendar.BATCH):
                    pos=positions[i:i+calendar.BATCH]
                    contexts=np.stack([values[t-calendar.CONTEXT:t] for t in pos])
                    dates=[ret.index[t-calendar.CONTEXT:t+1] for t in pos]
                    sample,params=interface.predict(contexts,dates,'actual',calendar.seed_for(asset,int(pos[0])))
                    assert np.array_equal(sample,z['actual_samples'][i:i+len(pos)]),(asset,path.name,i,'samples')
                    assert np.array_equal(params,z['actual_parameters'][i:i+len(pos)]),(asset,path.name,i,'parameters')
                    batches.append({'offset':int(pos[0])-calendar.CONTEXT,'rows':len(pos),'exact':True,'max_abs':0.})
        record={'asset':asset,'model':'lagllama','checker_sha256':calendar.sha(__file__),
                'binding_sha256':calendar.sha(folder/'binding.json'),'hashed_rows':len(ret)-calendar.CONTEXT,
                'batches':batches,'exact':True,'parameters_exact':True,'elapsed_seconds':time.monotonic()-start}
        assert sum(b['rows'] for b in batches)==record['hashed_rows']
        report.write_text(json.dumps(record,indent=2)+'\n');records.append(record)
        print(asset,'every corrected sample and parameter replayed',round(time.monotonic()-start,1),'s',flush=True)
    (calendar.OUT/'actual_full_replay.json').write_text(json.dumps({'checker_sha256':calendar.sha(__file__),
        'scope':'All corrected native forecasts and parameters replayed in a fresh process',
        'complete':len(records)==24,'rows':records},indent=2)+'\n')


if __name__=='__main__':main()
