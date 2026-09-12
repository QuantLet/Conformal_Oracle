"""Admit complete TS-ICL native outputs only after every exact fresh replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scope import PROJECT, ROOT, EXT, ASSETS, sha, dump, bind


def main():
    target = ROOT/'preparation_ten.json'
    assert not target.exists(), 'Completed ten-model admission is immutable'
    base = json.loads((ROOT/'preparation.json').read_text())
    binding = {**base['binding'], **bind([ROOT/'preparation.json', Path(__file__)])}
    records = list(base['native'])
    out = ROOT/'tsicl_assembled'; out.mkdir(exist_ok=True)
    for asset in ASSETS:
        folder = ROOT/'tsicl_full'/asset
        first = json.loads((folder/'complete.json').read_text())
        second = json.loads((folder/'replay.json').read_text())
        assert first['status'] == second['status'] == 'complete' and second['fresh_replay_exact']
        assert first['chunks'] == second['chunks'] and first['binding'] == second['binding']
        binding.update(first['binding']); binding.update(bind([folder/'complete.json',folder/'replay.json']))
        rp = EXT/'data/returns'/f'{asset}.csv'
        assert sha(rp) == first['input_sha256'] == second['input_sha256']
        series = pd.read_csv(rp,index_col='date',parse_dates=True).log_return
        values = series.to_numpy(np.float32); blocks = []; dates = []; positions = []
        for item in first['chunks']:
            fp = folder/item['file']; assert sha(fp) == item['sha256']
            binding.update(bind([fp])); saved = np.load(fp,allow_pickle=False)
            contexts = np.stack([values[t-512:t] for t in saved['positions']])
            assert hashlib.sha256(contexts.tobytes()).hexdigest() == item['contexts_sha256']
            np.testing.assert_array_equal(saved['levels'],np.arange(1,100)/100)
            np.testing.assert_array_equal(saved['dates'],series.index[saved['positions']].to_numpy())
            blocks.append(saved['native']); dates.append(saved['dates']); positions.append(saved['positions'])
        native = np.concatenate(blocks); dates = np.concatenate(dates); positions = np.concatenate(positions)
        np.testing.assert_array_equal(positions,np.arange(512,len(series)))
        assert np.isfinite(native).all() and native.shape == (len(series)-512,99)
        destination = out/f'{asset}.npz'
        np.savez_compressed(destination,native=native,dates=dates,positions=positions,levels=np.arange(1,100)/100)
        records.append(dict(model='TS-ICL',asset=asset,rows=len(native),native_values=native.size,
            output=str(destination.relative_to(ROOT)),sha256=sha(destination),
            crossings=int((np.diff(native,axis=1)<0).any(1).sum()),
            q01_crossings=int((native[:,:1]>native[:,1:]).any(1).sum()),fresh_native_replay_exact=True))
    for name,expected in binding.items(): assert sha(PROJECT/name) == expected,name
    dump(target,dict(status='complete',binding=binding,native=records,assets=24,
        forecasts_per_model=base['forecasts_per_model'],asset_test_dates=base['asset_test_dates']))
    print('Three native candidates fully assembled and replay-verified.',flush=True)


if __name__ == '__main__': main()
