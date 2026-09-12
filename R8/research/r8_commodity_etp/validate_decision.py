"""Compare every replacement decision replay and perturb future outcomes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from decision import ROOT, OUT, MODELS, load, compute, sha


def main():
    records = []
    for model in MODELS:
        for asset in ['USO', 'GLD', 'UNG']:
            key = f'{model}__{asset}'
            expected, actual = OUT/'pairs'/key, OUT/'replay'/key
            for folder in [expected, actual]:
                record = json.loads((folder/'complete.json').read_text())
                assert record['binding'] == load(model, asset)[-1]
                assert all(sha(folder/k) == h for k, h in record['outputs'].items())
            pd.testing.assert_frame_equal(pd.read_parquet(expected/'daily.parquet'),
                                          pd.read_parquet(actual/'daily.parquet'), check_exact=True)
            for name in ['metrics.csv', 'dtaci_seed_metrics.csv', 'fits.json']:
                assert (expected/name).read_bytes() == (actual/name).read_bytes(), (key, name)
            with np.load(expected/'dtaci_experts.npz') as a, np.load(actual/'dtaci_experts.npz') as b:
                assert a.files == b.files
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])
            records.append({'pair':key, 'all_numeric_outputs_exact':True})
    # Exercise the unchanged actual selection pipeline, including adaptive
    # prefixes. Future returns may change later adaptive predictions only.
    perturbations = []
    for model, asset in [('Moirai-1.1', 'USO'), ('GJR-GARCH-t', 'GLD'), ('Lag-Llama', 'UNG')]:
        key = f'{model}__{asset}'
        y, q, sigma, index, ref, binding = load(model, asset)
        nc = int(.7*len(y)); changed = y.copy()
        changed[nc:] += np.linspace(.1, 1, len(y)-nc)
        pred, params, proj, orig, mix, seeds = compute(changed, q, sigma, nc, key)
        expected = json.loads((OUT/'pairs'/key/'fits.json').read_text())
        for name in ['state', 'POT-Shift', 'POT-Vol', 'gate']:
            assert params[name] == expected[name], (key, name)
        previous = pd.read_parquet(OUT/'pairs'/key/'daily.parquet')
        for name in ['State-L1', 'State-L1-clipped', 'POT-Shift', 'POT-Vol']:
            np.testing.assert_array_equal(pred[name], previous[name])
        with np.load(OUT/'pairs'/key/'dtaci_experts.npz') as data:
            np.testing.assert_array_equal(proj['predictions'][:nc+1], data['projected_q'][:nc+1])
            np.testing.assert_array_equal(proj['probabilities'][:nc+1], data['projected_p'][:nc+1])
            np.testing.assert_array_equal(orig['predictions'][:nc+1], data['unprojected_q'][:nc+1])
        assert pred['Loss-gate'][0] == previous['Loss-gate'].iloc[0]
        perturbations.append({'pair':key, 'past_only_selection_and_adaptive_prefix':True})
    report = dict(status='passed', fresh_replays=records, future_outcome_perturbations=perturbations,
                  producer_sha256=sha(__file__),
                  protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')))
    (ROOT/'quality/decision_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('21 complete fresh decision replays and 3 future-outcome perturbations passed')


if __name__ == '__main__':
    main()
