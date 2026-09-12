"""Independently check reuse provenance, native values and past-context digests."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scope import PROJECT, ROOT, GRID, CHRONOS, FUNDS, EXT, sha, dump, bind


def main():
    preparation = json.loads((ROOT/'preparation.json').read_text())
    binding = bind([Path(__file__), ROOT/'preparation.json'])
    for name, expected in preparation['binding'].items(): assert sha(PROJECT/name) == expected,name
    patch = json.loads((GRID/'patchtst_full/complete.json').read_text())
    for name, expected in patch['binding'].items():
        if name != 'python_source_files_verified':
            assert sha(PROJECT/name) == expected,name
            binding[name] = expected
    chrono = json.loads((CHRONOS/'complete.json').read_text())
    for path,key in [(PROJECT/'research/r8_native_candidates/full_chronos.py','producer_sha256'),
                     (PROJECT/'research/r8_native_candidates/PROTOCOL.md','protocol_sha256'),
                     (PROJECT/'artifacts/r8_native_candidates/models/manifest.json','model_manifest_sha256')]:
        assert sha(path) == chrono[key]; binding.update(bind([path]))
    for short in ['chronos2','patchtst']:
        bpath = FUNDS/'native'/short/'binding.json'; b = json.loads(bpath.read_text())
        assert sha(PROJECT/'research/r8_commodity_etp/native.py') == b['producer_sha256']
        assert sha(PROJECT/'research/r8_commodity_etp/PROTOCOL.md') == b['protocol_sha256']
        assert sha(FUNDS/'data_admission.json') == b['admission_sha256']
        for name,expected in b['sources'].items(): assert sha(PROJECT/name) == expected,name
        binding.update(b['sources']); binding.update(bind([bpath]))
    rebuilt = digest_verified = values_verified = 0
    for item in preparation['native']:
        asset = item['asset']; model = item['model']; short = 'patchtst' if model == 'PatchTST-FM' else 'chronos2'
        rp = EXT/'data/returns'/f'{asset}.csv'
        series = pd.read_csv(rp,index_col='date',parse_dates=True).log_return
        values = series.to_numpy(np.float32)
        contexts = np.array([values[t-512:t] for t in range(512,len(series))])
        rebuilt += len(contexts)
        staged = np.load(ROOT/item['output'],allow_pickle=False)
        if asset in ['USO','GLD','UNG']:
            folder = FUNDS/'native'/short/asset
            first = json.loads((folder/'complete.json').read_text()); replay = json.loads((folder/'replay.json').read_text())
            assert replay['exact_fresh_replay'] and first['chunks'] == replay['chunks']
            offset = 0
            for chunk in first['chunks']:
                saved = np.load(folder/chunk['file'],allow_pickle=False); n = len(saved['positions'])
                assert hashlib.sha256(contexts[offset:offset+n].astype('<f4').tobytes()).hexdigest() == chunk['contexts_sha256']
                np.testing.assert_array_equal(staged['native'][offset:offset+n],saved['native'])
                offset += n; digest_verified += n; values_verified += saved['native'].size
            assert offset == len(contexts)
        else:
            old = GRID/'patchtst_full' if short == 'patchtst' else CHRONOS
            original = np.load(old/f'{asset}.npz',allow_pickle=False)
            for key in ['native','positions','dates','levels']: np.testing.assert_array_equal(staged[key],original[key])
            values_verified += staged['native'].size
            if short == 'patchtst':
                note = next(x for x in patch['assets'] if x['asset']==asset)
                assert hashlib.sha256(contexts.tobytes()).hexdigest() == note['contexts_sha256']
                digest_verified += len(contexts)
        assert staged['levels'][0] == .01 and (np.diff(staged['levels'])>0).all()
    dump(ROOT/'source_validation.json',dict(status='passed',binding=binding,
        independently_rebuilt_contexts=rebuilt,contexts_matched_to_saved_digest=digest_verified,
        native_values_compared_to_original=values_verified,
        note='Old Chronos has exact input/date/position and full replay bindings; its archive has no separate context digest.'))
    print('Native source, value and context validation passed.',flush=True)


if __name__ == '__main__': main()
