"""Independent CSV-to-LaTeX check for the authorised optimism integration.

No producer imports, estimation, simulation, source edits, or old-lock rebinding.
Default execution reviews current source and writes a separately bound receipt.
--check-receipt also rejects any change since that completed source review.
Every numerical/family guard is exercised on a defective copy before acceptance.
"""
import argparse
from copy import deepcopy
import csv
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import io
import json
from pathlib import Path
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_optimism_integration/independent_numbers.json'
BINDINGS = ROOT / 'artifacts/r8_optimism_integration/study_bindings.json'
SOURCES = ['source/sections_r8/' + n + '.tex' for n in
           ('numbers_optimism', 'tab_optimism', 'supp_optimism', 'montecarlo',
            'risk', 'optimism_proof')]
AR = 'results/optimism_ar/primary/'
V3 = 'results/theory_loop_v3/diagnostic/'
SYN = 'results/theory_loop/synthetic/'
INPUTS = [AR + n + '.csv' for n in ('bands', 'dependence', 'histories', 'unavailable')]
INPUTS += [V3 + 'simultaneous_bands.csv', V3 + 'unavailable.csv',
           SYN + 'validation_summary.csv', SYN + 'history_bootstrap_indices.npy',
           'results/theory_loop/deliverable_status.csv',
           'results/optimism_ar/independent_verification.json',
           'results/theory_loop_v3/independent_verification.json']
EXPECTED = {(p, n, 'independent') for p in (0., .8) for n in (500, 1000)}
EXPECTED |= {(p, 500, 'contiguous') for p in (0., .8)}
CHECKS = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject_first(name, function, bad, good):
    try:
        function(bad)
    except (AssertionError, ValueError, KeyError):
        pass
    else:
        raise AssertionError('Defective case accepted: ' + name)
    function(good)
    CHECKS.append({'name': name, 'defect_evaluated_first': True,
                   'defect_rejected': True, 'valid_accepted': True})


def records(content):
    return list(csv.DictReader(io.StringIO(content.decode())))


def key(row):
    return (float(row['phi']), int(row['n']), row['evaluation'])


def family(rows):
    keys = [key(r) for r in rows]
    require(len(keys) == len(EXPECTED) and set(keys) == EXPECTED,
            'Family must have exactly the six distinct prespecified cells')


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def rounded(value, places):
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-places),
                                       rounding=ROUND_HALF_EVEN)


def table_rows(source):
    out = []
    for line in source.splitlines():
        if not re.match(r'^\s*[0-9]', line):
            continue
        cols = [c.strip() for c in line.split('&')]
        require(len(cols) == 5, 'Table column count')
        band = re.fullmatch(r'\$\[([-0-9.]+),\s*([-0-9.]+)\]\$\s*\\\\', cols[4])
        require(band is not None, 'Malformed table band')
        names = {'Contiguous': 'contiguous', 'Independent marginal': 'independent'}
        out.append({'phi': float(cols[0]), 'n': int(cols[1].replace('{,}', '')),
                    'evaluation': names[cols[2]], 'mean_ratio': Decimal(cols[3]),
                    'lower': Decimal(band[1]), 'upper': Decimal(band[2])})
    return out


def check_table(rows, expected):
    family(rows)
    for row in rows:
        for column in ('mean_ratio', 'lower', 'upper'):
            require(row[column] == rounded(expected[key(row)][column], 4),
                    'Incorrect displayed ' + str(key(row)) + ' ' + column)


def check_macros(source, expected):
    found = re.findall(r'\\newcommand\{\\(nOpt\w+)\}\{([^{}]+)\}', source)
    require(len(found) == len(expected) and len(dict(found)) == len(expected),
            'Missing/duplicate optimism macro')
    require(set(dict(found)) == set(expected), 'Unexpected macro names')
    for name, value in found:
        require(Decimal(value) == expected[name], 'Incorrect macro ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-receipt', action='store_true')
    args = parser.parse_args()
    started = time.time_ns()
    bindings = json.loads(BINDINGS.read_text())
    raw = {name: (ROOT / name).read_bytes() for name in INPUTS}
    source = {name: (ROOT / name).read_bytes() for name in SOURCES}
    if args.check_receipt:
        prior = json.loads(OUT.read_text())
        require(prior['status'] == 'PASS', 'Prior review did not pass')
        for name, digest in prior['source_sha256'].items():
            require(sha((ROOT / name).read_bytes()) == digest, 'Stale source: ' + name)
        require(prior['verifier_sha256'] == sha(Path(__file__).read_bytes()), 'Stale verifier')
    for name in INPUTS:
        bound = bindings[name]
        predicate = lambda data, b=bound: require(sha(data) == b['sha256'], 'Changed study input')
        reject_first('stale_study_input:' + name, predicate, raw[name] + b'\n', raw[name])
        require((ROOT / name).stat().st_mtime_ns == bound['mtime_ns'], 'Input mtime changed: ' + name)
    for name, data in source.items():
        digest = sha(data)
        predicate = lambda b, expected=digest: require(sha(b) == expected, 'Stale reviewed source')
        reject_first('stale_source:' + name, predicate, data + b'% altered after review\n', data)

    bands = records(raw[AR + 'bands.csv'])
    reject_first('missing_family_cell', family, bands[:-1], bands)
    reject_first('duplicate_family_cell', family, bands[:-1] + [bands[0]], bands)
    expected = {key(r): r for r in bands}
    dep = {(float(r['phi']), int(r['n'])): r for r in records(raw[AR + 'dependence.csv'])}
    hist = records(raw[AR + 'histories.csv'])
    require(len(hist) == 3000, '3000 history rows required')
    histkeys = [(key(r), int(r['rep'])) for r in hist]
    require(len(set(histkeys)) == 3000, 'Duplicate history keys')
    matrix = []
    reviewed_rows = []
    for cell in sorted(EXPECTED):
        part = sorted((r for r in hist if key(r) == cell), key=lambda r: int(r['rep']))
        require([int(r['rep']) for r in part] == list(range(500)), 'Incomplete cell histories')
        n = cell[1]
        H = 214 if cell[2] == 'contiguous' else 0
        rank = (99 * (n + 1) + 99) // 100
        require({(int(r['k']), int(r['H'])) for r in part} == {(rank, H)}, 'Rank/future chronology')
        d = dep[cell[:2]]
        omega, density = float(d['omega']), float(d['f_true'])
        require(omega > 0 and density > 0, 'Population constants must be positive')
        A0 = omega / (2 * n * density)
        Aiid = .0099 / (2 * n * density)
        require(all(np.isfinite(float(r[c])) for r in part for c in
                    ('C', 'J', 'V', 'optimism', 'A0', 'A_iid', 'ratio', 'ratio_iid')),
                'Nonfinite history')
        numerical = np.array([[float(r[c]) for c in
                               ('J', 'V', 'optimism', 'A0', 'A_iid', 'ratio', 'ratio_iid')]
                              for r in part])
        target = np.column_stack([numerical[:, 1] - numerical[:, 0],
                                  np.full(500, A0), np.full(500, Aiid),
                                  (numerical[:, 1] - numerical[:, 0]) / (2 * A0),
                                  (numerical[:, 1] - numerical[:, 0]) / (2 * Aiid)])
        bad = numerical[:, 2:].copy(); bad[:, 0] *= -1
        reject_first('optimism_sign_and_scaling:' + str(cell),
                     lambda arr, target=target: close(arr, target), bad, numerical[:, 2:])
        matrix.append(target[:, 3])
        r = expected[cell]
        close([float(r['mean_ratio']), float(r['se'])],
              [target[:, 3].mean(), target[:, 3].std(ddof=1) / np.sqrt(500)])
        close(float(r['iid_reference']), .0099 / omega)
        require(int(r['family_size']) == 6 and int(r['bootstrap']) == 999, 'Incorrect family metadata')
        L, U = float(r['lower']), float(r['upper'])
        includes_one = L <= 1 <= U
        includes_iid = L <= .0099 / omega <= U
        require(includes_one and includes_iid == (cell[0] == 0), 'Main comparison claim fails')
        require(r['includes_LRV_reference'] == str(includes_one) and
                r['includes_iid_reference'] == str(includes_iid), 'Stored band flag inconsistent')
        reviewed_rows.append({'phi': cell[0], 'n': n, 'evaluation': cell[2], 'k': rank, 'H': H,
                              'mean': float(r['mean_ratio']), 'lower': L, 'upper': U,
                              'iid_reference': .0099 / omega,
                              'includes_LRV_reference': includes_one, 'includes_iid_reference': includes_iid})

    # Reproduce the existing six-cell band from CSV losses and saved indices,
    # independently through a multiplicity matrix, not producer code.
    values = np.column_stack(matrix)
    indices = np.load(io.BytesIO(raw[SYN + 'history_bootstrap_indices.npy']))
    require(indices.shape == (999, 500) and np.issubdtype(indices.dtype, np.integer), 'Bootstrap shape/type')
    require(indices.min() >= 0 and indices.max() < 500, 'Bootstrap index support')
    weights = np.stack([np.bincount(row, minlength=500) for row in indices]) / 500
    means = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(500)
    max_t = np.max(np.abs((weights @ values - means) / se), axis=1)
    critical = np.sort(max_t)[int(np.ceil(.95 * (len(max_t) - 1)))]
    actual_bands = np.array([[float(expected[c][v]) for v in ('lower', 'upper', 'critical_value')]
                            for c in sorted(EXPECTED)])
    wanted_bands = np.column_stack([means - critical * se, means + critical * se, np.full(6, critical)])
    wrong = actual_bands.copy(); wrong[0, 0] *= -1
    reject_first('six_cell_simultaneous_bands', lambda x: close(x, wanted_bands), wrong, actual_bands)

    unavailable = records(raw[AR + 'unavailable.csv'])
    require({key(r) for r in unavailable} == {(p, 1000, 'contiguous') for p in (0., .8)} and
            len(unavailable) == 2 and all(r['status'] == 'NOT_AVAILABLE' for r in unavailable),
            'Unavailable AR futures incorrectly represented')

    validation = records(raw[SYN + 'validation_summary.csv'])
    require(len(validation) == 10 and {(r['law'], int(r['n'])) for r in validation} ==
            {(law, n) for law in ('normal', 't5') for n in (250, 500, 700, 1000, 2000)}, 'Density validation grid')
    require(all(float(r['median_density_relative_error']) > .15 and r['density_pass'] == 'False'
                and int(r['finite_histories']) == 500 for r in validation), 'Density failure statement')
    gate = records(raw['results/theory_loop/deliverable_status.csv'])
    require(len(gate) == 6 and {(int(r['deliverable']), r['panel']) for r in gate} ==
            {(d, p) for d in (1, 2, 3) for p in ('main', 'external')}, 'Deliverable status grid')
    require(all(int(r['computed_pairs']) == 0 and r['status'] ==
                ('NOT_REQUESTED' if (int(r['deliverable']), r['panel']) == (3, 'external')
                 else 'NOT_RUN_SYNTHETIC_ADMISSION_FAILED') for r in gate),
            'Financial-panel admission statement')

    nuisance = {r['law']: r for r in validation if int(r['n']) == 1000}
    d = dep[(.8, 1000)]
    c = expected[(.8, 500, 'contiguous')]
    wanted_macros = {
        'nOptBootstrap': Decimal(len(indices)), 'nOptFamily': Decimal(len(EXPECTED)),
        'nOptHistories': Decimal(len(values)),
        'nOptContHi': rounded(c['upper'], 4), 'nOptContLo': rounded(c['lower'], 4),
        'nOptContMean': rounded(c['mean_ratio'], 4),
        'nOptDensityNormal': rounded(Decimal(nuisance['normal']['median_density_relative_error']) * 100, 1),
        'nOptDensityT': rounded(Decimal(nuisance['t5']['median_density_relative_error']) * 100, 1),
        'nOptIidReference': rounded(.0099 / float(d['omega']), 4),
        'nOptInflation': rounded(float(d['omega']) / .0099, 2),
        'nOptOmegaDep': rounded(d['omega'], 8)}
    macrotext = source[SOURCES[0]].decode()
    for name in wanted_macros:
        match = re.search(r'(\\newcommand\{\\' + name + r'\}\{)([^{}]+)(\})', macrotext)
        require(match is not None, 'Missing macro ' + name)
        bad = macrotext[:match.start(2)] + str(-Decimal(match[2])) + macrotext[match.end(2):]
        reject_first('macro_sign:' + name, lambda x: check_macros(x, wanted_macros), bad, macrotext)
    parsed_table = table_rows(source[SOURCES[1]].decode())
    for i, row in enumerate(parsed_table):
        for column in ('mean_ratio', 'lower', 'upper'):
            wrong = deepcopy(parsed_table); wrong[i][column] += Decimal('.0001')
            reject_first('table_value:' + str(key(row)) + ':' + column,
                         lambda x: check_table(x, expected), wrong, parsed_table)
    reject_first('missing_displayed_row', lambda x: check_table(x, expected), parsed_table[:-1], parsed_table)
    reject_first('duplicate_displayed_row', lambda x: check_table(x, expected),
                 parsed_table[:-1] + [parsed_table[0]], parsed_table)

    # Do not pool the earlier 18-cell iid-hit diagnostic with the six-cell AR family.
    v3 = records(raw[V3 + 'simultaneous_bands.csv'])
    v3keys = {(r['law'], int(r['n']), r['evaluation']) for r in v3}
    v3wanted = {(law, n, 'independent') for law in ('normal', 't5') for n in (250, 500, 700, 1000, 2000)}
    v3wanted |= {(law, n, 'contiguous') for law in ('normal', 't5') for n in (250, 500, 700, 1000)}
    require(len(v3) == 18 and v3keys == v3wanted, 'Earlier diagnostic family')
    require(all(int(r['family_size']) == 18 and int(r['bootstrap_replicates']) == 999 and
                float(r['lower']) <= 1 <= float(r['upper']) and
                int(r['H']) == (3 * int(r['n']) // 7 if r['evaluation'] == 'contiguous' else 0)
                for r in v3), 'Earlier diagnostic scope/bands/horizons')
    for prefix, receipt_name, names in [
        (AR, 'results/optimism_ar/independent_verification.json', ('bands.csv', 'histories.csv', 'dependence.csv', 'unavailable.csv')),
        (V3, 'results/theory_loop_v3/independent_verification.json', ('simultaneous_bands.csv', 'unavailable.csv'))]:
        receipt = json.loads(raw[receipt_name])
        require(receipt['status'] == 'PASS' and receipt['financial_panel'] == 'NOT_RUN', 'Original validation status')
        for name in names:
            require(sha(raw[prefix + name]) == receipt['output_sha256'][name], 'Original numerical receipt mismatch')

    supp = ' '.join(source[SOURCES[2]].decode().split())
    mc = ' '.join(source[SOURCES[3]].decode().split())
    risk = ' '.join(source[SOURCES[4]].decode().split())
    clauses = {
        'retrospective': (supp, 'retrospective control reuses'),
        'level': (supp, r'$\alpha=0.01$'),
        'prefix': (supp, 'Calibration uses the first'),
        'fixed_future': (supp, 'next 214 stored scores at $n=500$, holding the shift fixed'),
        'no_1000_future': (supp, r'No future is available at $n=1{,}000$'),
        'shared_family': (supp, 'they are one comparison family'),
        'population_inputs': (supp, 'Population density and hit covariances are known'),
        'marginal_target': (supp, 'Independent evaluation integrates marginal loss'),
        'failed_density': (supp, r"exceeding the protocol's 15\% criterion"),
        'financial_not_run': (supp, 'were not evaluated on the financial panels'),
        'main_six_family': (mc, 'All three dependent cells in a six-cell simultaneous family exclude'),
        'no_equality': (mc, 'without establishing finite-sample equality'),
        'expectation': (risk, r'\E I_n(C_n)=-\mathcal B-A_{0,n}+o(n^{-1})'),
        'twice_population_cost': (risk, r'\E\{\bar D_{n,H_n}-I_n(C_n)\}=2A_{0,n}+o(n^{-1})'),
        'contiguous_horizon': (risk, r'second relation requires $H_n/\sqrt n\to\infty$'),
    }
    for name, (content, fragment) in clauses.items():
        reject_first('scope:' + name, lambda s, f=fragment: require(f in s, 'Reviewed scope missing'),
                     content.replace(fragment, '[ALTERED CLAIM]'), content)

    # Reject concurrent source/input mutation during this review.
    for name, data in {**raw, **source}.items():
        require((ROOT / name).read_bytes() == data, 'Changed during review: ' + name)
    result = {'status': 'PASS', 'started_ns': started, 'finished_ns': time.time_ns(),
              'verifier_sha256': sha(Path(__file__).read_bytes()),
              'study_bindings_sha256': sha(BINDINGS.read_bytes()),
              'source_sha256': {name: sha(data) for name, data in source.items()},
              'input_sha256': {name: sha(data) for name, data in raw.items()},
              'negative_checks': CHECKS, 'check_count': len(CHECKS),
              'macro_values': {k: str(v) for k, v in wanted_macros.items()},
              'table_rows': reviewed_rows, 'critical_value_recomputed': float(critical),
              'main_claim': {'dependent_rows': 3, 'exclude_iid_reference': 3,
                             'include_population_LRV_reference': 3,
                             'family_size': 6, 'interpretation': 'compatibility, not equivalence'},
              'v3_family': {'distinct_cells': 18, 'independent': 10, 'contiguous': 8,
                            'all_include_first_order_reference': True, 'pooled_with_AR': False},
              'nuisance_density': {'all_ten_cells_fail_15_percent': True,
                                   'normal_n1000': float(nuisance['normal']['median_density_relative_error']),
                                   't5_n1000': float(nuisance['t5']['median_density_relative_error'])},
              'new_simulations': False, 'financial_panel': 'NOT_RUN',
              'scope': 'Independent source/CSV numerical and statistical-scope review; no manuscript edits or theorem re-proof.'}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(CHECKS), 'table_rows': 6,
                      'macros': len(wanted_macros), 'receipt': str(OUT.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
