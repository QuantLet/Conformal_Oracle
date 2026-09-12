"""Independent checks of saved candidate artefacts; does not load models."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'artifacts/r8_grid_candidates'
RESEARCH = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(8*1024*1024): h.update(part)
    return h.hexdigest()


def main():
    result = {'status': 'passed', 'scope': 'technical pilots only; no empirical model admission',
              'source_files_verified': 0, 'models': {}, 'manuscript_modified': False}
    for kind in ('source', 'weights'):
        manifest = json.loads((ROOT/f'{kind}_manifest.json').read_text())
        for name, entry in manifest['files'].items():
            path = ROOT/name
            assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
    # Verify the actual imported Python sources against their immutable archives.
    for archive in (ROOT/'sources').glob('*.tar.gz'):
        extracted = archive.parent/archive.name.removesuffix('.tar.gz')
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                if member.isfile() and member.name.endswith('.py'):
                    assert (extracted/member.name).read_bytes() == stream.extractfile(member).read(), member.name
                    result['source_files_verified'] += 1
    cpu_receipt = json.loads((ROOT/'patchtst/cpu_check.json').read_text())
    assert cpu_receipt['status'] == 'complete'
    assert cpu_receipt['producer_sha256'] == sha(RESEARCH/'pilot.py')
    for name, digest in cpu_receipt['outputs'].items(): assert sha(ROOT/'patchtst'/name) == digest
    for model in ('patchtst', 'tsicl', 'patchtst_float32'):
        out = ROOT/model
        pilot = json.loads((out/'pilot.json').read_text())
        replay = json.loads((out/'replay.json').read_text())
        for receipt in (pilot, replay):
            for name, digest in receipt['outputs'].items(): assert sha(out/name) == digest
            assert receipt['source_manifest_sha256'] == sha(ROOT/'source_manifest.json')
            assert receipt['weights_manifest_sha256'] == sha(ROOT/'weights_manifest.json')
            for asset, binding in receipt['bindings'].items():
                assert binding['input_sha256'] == sha(PROJECT/f'artifacts/extension_20260831/data/returns/{asset}.csv')
        expected_script = 'precision_check.py' if model == 'patchtst_float32' else 'pilot.py'
        expected_protocol = 'NUMERICS.md' if model == 'patchtst_float32' else 'PROTOCOL.md'
        assert pilot['producer_sha256'] == replay['producer_sha256'] == sha(RESEARCH/expected_script)
        assert pilot['protocol_sha256'] == replay['protocol_sha256'] == sha(RESEARCH/expected_protocol)
        assert all(x['replay']['exact'] for x in replay['checks'])
        values = []; count_cross = 0; q01_cross = 0; dates_count = 0
        for asset in ('SP500', 'BTC'):
            data = np.load(out/f'{asset}_native.npz', allow_pickle=False)
            q = data['native']; levels = data['levels']; context = data['contexts']
            assert q.shape == (32,99) and np.isfinite(q).all()
            np.testing.assert_array_equal(levels, np.arange(1,100)/100)
            source = pd.read_csv(PROJECT/f'artifacts/extension_20260831/data/returns/{asset}.csv',
                                 index_col='date', parse_dates=True).log_return
            np.testing.assert_array_equal(data['dates'], source.index[-32:].to_numpy())
            for i, date in enumerate(data['dates']):
                position = source.index.get_loc(date)
                np.testing.assert_array_equal(context[i], source.iloc[position-512:position].to_numpy(dtype=np.float32))
            assert hashlib.sha256(context.tobytes()).hexdigest() == pilot['bindings'][asset]['context_sha256']
            count_cross += int((np.diff(q, axis=1)<0).any(1).sum())
            q01_cross += int((q[:,:1]>q[:,1:]).any(1).sum())
            dates_count += len(q); values.extend(q[:,0].tolist())
        if model != 'patchtst_float32':
            with (out/'quantiles.csv').open() as stream: saved = [float(row['q01']) for row in csv.DictReader(stream)]
            np.testing.assert_array_equal(saved, values)
            assert pilot['status'] == replay['status'] == 'complete'
            assert all(x['public_selected_q01']['exact'] for x in pilot['checks'])
            assert all(x['past_only'] for x in pilot['model_details']['normalization_checks'])
        else:
            assert pilot['all_consistency_checks_pass'] and replay['all_consistency_checks_pass']
            for asset in ('SP500', 'BTC'):
                current = np.load(out/f'{asset}_native.npz')['native'][[0,31]]
                reference = np.load(ROOT/'patchtst'/f'{asset}_cpu.npz')['native']
                np.testing.assert_allclose(current, reference, rtol=1e-5, atol=1e-7)
        timings = pilot['checks'] if model == 'patchtst_float32' else pilot['timings']
        seconds = sum(x['seconds'] for x in timings)
        replay_times = replay['checks'] if model == 'patchtst_float32' else replay['timings']
        result['models'][model] = {'forecast_dates': dates_count, 'native_values': dates_count*99,
            'fresh_replay_exact': True, 'dates_with_crossing': count_cross, 'q01_crossing_dates': q01_cross,
            'pilot_seconds': seconds, 'replay_seconds': sum(x['seconds'] for x in replay_times),
            'pilot_receipt_sha256': sha(out/'pilot.json'), 'replay_receipt_sha256': sha(out/'replay.json')}
    result['producer_sha256'] = sha(__file__)
    result['recreated_environment_replays'] = {}
    for name in ('patchtst_float32', 'tsicl'):
        path = ROOT/name/'environment_replay.json'
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'passed' and all(x['exact'] for x in receipt['checks'])
        assert receipt['bootstrap_sha256'] == sha(RESEARCH/'bootstrap.py')
        assert receipt['producer_sha256'] == sha(RESEARCH/'environment_replay.py')
        assert receipt['adapter_sha256'] == sha(RESEARCH/'pilot.py')
        result['recreated_environment_replays'][name] = {'native_values': 6336, 'exact': True, 'receipt_sha256': sha(path)}
    (ROOT/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
