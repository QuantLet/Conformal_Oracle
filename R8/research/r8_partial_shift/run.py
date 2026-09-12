"""Run the fixed partial-shift diagnostic on saved histories only."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import engine as e

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_partial_shift'
KEYS = ['module', 'innovation', 'phi', 'n_cal', 'alpha', 'truth']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    names = ['research/r8_partial_shift/PROTOCOL.md', 'research/r8_partial_shift/engine.py',
             'research/r8_partial_shift/run.py', 'artifacts/r8_partial_shift/before.json',
             'artifacts/r8_mechanism/paths.npz', 'artifacts/r8_mechanism/paths.json',
             'artifacts/r8_mechanism/results/replications.parquet']
    return {name: sha(ROOT / name) for name in names}


def configurations():
    for module in ('ar', 'garch'):
        for kind in ('normal', 't5'):
            for phi in ((0., .5, .8) if module == 'ar' else (0.,)):
                for n in (125, 250, 500, 1000):
                    for alpha in (.01, .05):
                        for truth in (('none', 'constant') if module == 'ar' else ('none', 'constant', 'state')):
                            yield dict(zip(KEYS, (module, kind, phi, n, alpha, truth)))


def forecasts(paths, cfg):
    module, kind, phi, n, alpha, truth = [cfg[k] for k in KEYS]
    if module == 'ar':
        y = paths[f'ar_{kind}_{phi:g}'][:, -n:]
        sigma = np.full_like(y, e.V0)
        test_sigma = np.array([e.V0])
    else:
        y = paths[f'garch_y_{kind}'][:, -n:]
        sigma = paths[f'garch_sigma_{kind}'][:, -n:]
        test_sigma = paths[f'test_sigma_{kind}']
    raw = e.quantile(kind, sigma, alpha) + e.distortion(sigma, truth)
    test_raw = e.quantile(kind, test_sigma, alpha) + e.distortion(test_sigma, truth)
    return y, raw, test_raw, test_sigma


def rows(cfg, risks, violations, extra=None):
    n = risks.shape[0]
    frame = pd.DataFrame({**{k: v for k, v in cfg.items()},
                          'replication': np.repeat(np.arange(n), len(e.METHODS)),
                          'method': np.tile(e.METHODS, n),
                          'expected_loss': risks.ravel(), 'expected_violation': violations.ravel()})
    if extra:
        for k, v in extra.items():
            frame[k] = v
    return frame


def summarise(frame):
    records = []
    group_keys = KEYS + (['horizon'] if 'horizon' in frame else [])
    for values, group in frame.groupby(group_keys, sort=True):
        wide = group.pivot(index='replication', columns='method', values='expected_loss')
        pi = group.pivot(index='replication', columns='method', values='expected_violation')
        for method in e.METHODS:
            row = dict(zip(group_keys, values), method=method, histories=len(wide),
                       expected_loss=float(wide[method].mean()),
                       expected_violation=float(pi[method].mean()))
            for reference in ['Raw', 'Full-CP', 'Half-Inner', 'Oracle-Grid']:
                d = wide[method] - wide[reference]
                row['difference_vs_' + reference] = float(d.mean())
                row['se_vs_' + reference] = float(d.std(ddof=1) / np.sqrt(len(d)))
            records.append(row)
    return pd.DataFrame(records)


def main(replay=False):
    folder = OUT / ('replay' if replay else 'run')
    folder.mkdir(parents=True, exist_ok=False)
    binding = inputs()
    before = json.loads((OUT / 'before.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in before['canonical'].items())
    assert sha(ROOT / 'research/r8_partial_shift/PROTOCOL.md') == before['protocol_sha256']
    assert sha(ROOT / 'artifacts/r8_mechanism/paths.npz') == json.loads((ROOT / 'artifacts/r8_mechanism/paths.json').read_text())['sha256']
    paths = np.load(ROOT / 'artifacts/r8_mechanism/paths.npz')
    all_rows, decisions, contiguous = [], [], []
    for i, cfg in enumerate(configurations(), 1):
        y, raw, test_raw, sigma = forecasts(paths, cfg)
        a, kind = cfg['alpha'], cfg['innovation']
        choice = e.select(y, raw, a)
        optimum = e.constant_optimum(kind, test_raw, sigma, a)
        scalar_risk = float(e.expected_loss(kind, test_raw - optimum, sigma, a).mean())
        conditional_risk = float(e.expected_loss(kind, e.quantile(kind, sigma, a), sigma, a).mean())
        candidates, fraction = e.candidate_corrections(choice, optimum)
        grid_risk, grid_pi = e.independent_metrics(kind, test_raw, sigma, candidates, a)
        risk, pi, oracle = e.method_columns(grid_risk, grid_pi, choice)
        all_rows.append(rows(cfg, risk, pi))
        record = {**cfg, 'replication': np.arange(500), 'inner_shift': choice['inner'],
                  'full_shift': choice['full'], 'fit_size': choice['fit_size'],
                  'rank_inner': choice['rank_inner'], 'rank_full': choice['rank_full'],
                  'selected_fraction': choice['selected_fraction'],
                  'flat_validation_minimum': choice['flat_minimum'],
                  'exact_comparison_fallback': choice['exact_fallback'],
                  'oracle_grid_fraction': e.FRACTIONS[oracle], 'oracle_continuous_fraction': fraction,
                  'best_constant_shift': optimum, 'best_constant_loss': scalar_risk,
                  'conditional_oracle_loss': conditional_risk,
                  'constant_family_gap': scalar_risk - conditional_risk,
                  **e.accounting(risk, scalar_risk)}
        for j in range(len(e.FRACTIONS)):
            record[f'validation_loss_{j}'] = choice['validation'][:, j]
            record[f'population_grid_loss_{j}'] = grid_risk[:, j]
        decisions.append(pd.DataFrame(record))
        if cfg['module'] == 'ar' and kind == 'normal':
            h = 3 * cfg['n_cal'] // 7
            future = e.contiguous_metrics(float(test_raw[0]), candidates, a,
                       paths[f"ar_z_{cfg['phi']:g}"][:, -1], cfg['phi'], h)
            future_risk, future_pi, _ = e.method_columns(*future, choice)
            contiguous.append(rows(cfg, future_risk, future_pi, {'horizon': h}))
        if i % 12 == 0:
            print('Completed', i, '/ 144 configurations', flush=True)
    frame, dec, cont = pd.concat(all_rows, ignore_index=True), pd.concat(decisions, ignore_index=True), pd.concat(contiguous, ignore_index=True)
    assert len(frame) == 576000 and len(dec) == 72000 and len(cont) == 192000
    old = pd.read_parquet(ROOT / 'artifacts/r8_mechanism/results/replications.parquet')
    old = old[old.method.isin(['Raw', 'Shift-CP'])].copy()
    old['method'] = old.method.replace({'Shift-CP': 'Full-CP'})
    matched = frame[frame.method.isin(['Raw', 'Full-CP'])].merge(old, on=KEYS + ['replication', 'method'], validate='one_to_one', suffixes=('_new', '_old'))
    assert len(matched) == 144000
    old_error = float(np.max(np.abs(matched.expected_loss - matched.expected_QS)))
    pi_error = float(np.max(np.abs(matched.expected_violation_new - matched.expected_violation_old)))
    assert old_error < 2e-15 and pi_error < 2e-13, (old_error, pi_error)
    for name, data in [('replications', frame), ('decisions', dec), ('contiguous', cont)]:
        data.to_parquet(folder / (name + '.parquet'), index=False)
    summarise(frame).to_csv(folder / 'summary.csv', index=False)
    summarise(cont).to_csv(folder / 'contiguous_summary.csv', index=False)
    components = ['removable_loss', 'inner_estimation_cost', 'oracle_shrinkage_gain', 'selection_regret',
                  'grid_discretisation_gap', 'validation_reservation_effect', 'full_estimation_cost',
                  'selected_change', 'constant_family_gap']
    dec.groupby(KEYS)[components].mean().reset_index().to_csv(folder / 'decomposition.csv', index=False)
    dec.groupby(KEYS + ['selected_fraction']).size().rename('count').reset_index().to_csv(folder / 'selection.csv', index=False)
    outputs = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    assert all(sha(ROOT / p) == h for p, h in before['canonical'].items())
    result = dict(status='passed',configurations=144,histories_per_configuration=500,
                  independent_latent_histories=1500,method_history_rows=len(frame),
                  contiguous_configurations=48,old_loss_max_error=old_error,old_probability_max_error=pi_error,
                  no_new_paths=True,no_new_base_forecasts=True,canonical_unchanged=True,
                  inputs=binding,outputs=outputs,
                  environment={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','pyarrow']})
    (folder / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    if replay:
        assert result == json.loads((OUT / 'run/validation.json').read_text()), 'Fresh replay differs'
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','outputs')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    main(parser.parse_args().replay)
