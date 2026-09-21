"""Validate the integration independently of its CSV display builder.

Historical guards run against their original immutable archive, preserving
their canonical-file checks. Current documents have a separate scope guard.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile
import numpy as np
import pandas as pd
from pypdf import PdfReader
from build import PROJECT, SOURCE, OUT, sha, check

KEYS = ['module', 'innovation', 'phi', 'n_cal', 'alpha', 'truth']
STUDY = PROJECT / 'artifacts/r8_partial_shift'


def study_replay():
    receipt = json.loads((STUDY / 'package.json').read_text())
    archive_path = PROJECT / receipt['archive']
    assert receipt['status'] == 'complete' and sha(archive_path) == receipt['archive_sha256']
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix='irfa-partial-historical-') as tmp:
        manifest_name, = [p for p in archive.namelist() if p.endswith('/STUDY_MANIFEST.json')]
        prefix = manifest_name.split('/')[0]
        manifest = json.loads(archive.read(manifest_name))
        for name, expected in manifest['files'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            content = archive.read(prefix + '/' + name)
            assert hashlib.sha256(content).hexdigest() == expected
            path = Path(tmp) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        process = subprocess.run([sys.executable, 'research/r8_partial_shift/validate.py'],
                                 cwd=tmp, capture_output=True, text=True)
        assert process.returncode == 0, (process.stdout, process.stderr)
        audit = json.loads((Path(tmp) / 'artifacts/r8_partial_shift/independent_validation.json').read_text())
        assert audit == json.loads((STUDY / 'independent_validation.json').read_text())
    return dict(archive=receipt['archive'], archive_sha256=receipt['archive_sha256'],
                verified_members=len(manifest['files']), numerical_replay_exact=True,
                original_canonical_checks_preserved=True, choices=audit['independent_rank_and_selection_checks'])


def numerical_displays():
    producer = json.loads((STUDY / 'run/validation.json').read_text())
    assert producer['status'] == 'passed'
    for name, expected in producer['inputs'].items():
        assert sha(PROJECT / name) == expected, name
    for name, expected in producer['outputs'].items():
        assert sha(STUDY / 'run' / name) == expected, name
    data = pd.read_parquet(STUDY / 'run/replications.parquet')
    decisions = pd.read_parquet(STUDY / 'run/decisions.parquet')
    assert len(data) == 576000 and len(decisions) == 72000
    index = KEYS + ['replication']
    wide = data.pivot(index=index, columns='method', values='expected_loss').sort_index()
    d = decisions.set_index(index).reindex(wide.index)
    oracle = wide['Oracle-Grid']
    components = {
        'removable_loss': wide.Raw - d.best_constant_loss,
        'inner_estimation_cost': wide['Inner-CP'] - d.best_constant_loss,
        'oracle_shrinkage_gain': wide['Inner-CP'] - oracle,
        'selection_regret': wide['Selected-Inner'] - oracle,
    }
    error = 0.
    for name, value in components.items():
        error = max(error, float(np.max(np.abs(value - d[name]))))
        assert (value >= -1e-16).all()
    identity = -components['removable_loss'] + components['inner_estimation_cost'] \
        - components['oracle_shrinkage_gain'] + components['selection_regret']
    error = max(error, float(np.max(np.abs(identity - (wide['Selected-Inner'] - wide.Raw)))))
    assert error < 1e-16
    summary = pd.read_csv(STUDY / 'run/summary.csv').set_index(KEYS + ['method']).sort_index()
    independent = data.groupby(KEYS + ['method'], dropna=False)[['expected_loss', 'expected_violation']].mean()
    assert independent.index.equals(summary.index)
    summary_error = float(np.max(np.abs(independent.to_numpy() - summary[independent.columns].to_numpy())))
    assert summary_error < 2e-15
    macros = dict(re.findall(r'\\newcommand\{\\([^}]+)\}\{([^}]+)\}',
                             (SOURCE / 'sections_r8/numbers_partial.tex').read_text()))
    means = wide.groupby(level=KEYS, dropna=False).mean()
    counts = {}
    for alpha, tag in [(.01, 'One'), (.05, 'Five')]:
        part = means[means.index.get_level_values('alpha') == alpha]
        assert len(part) == 72
        counts[str(alpha)] = {}
        for reference, name in [('Raw', 'Raw'), ('Full-CP', 'Full'), ('Half-Inner', 'Half')]:
            count = int((part['Selected-Inner'] < part[reference]).sum())
            assert macros['nPartial' + tag + 'Better' + name] == str(count)
            counts[str(alpha)][reference] = count
        rows = d[d.index.get_level_values('alpha') == alpha]
        ratio = 100 * rows.selection_regret.mean() / rows.oracle_shrinkage_gain.mean()
        assert macros['nPartial' + tag + 'RegretPct'] == f'{ratio:.1f}'
    table = (SOURCE / 'sections_r8/tab_partial.tex').read_text()
    for method in data.method.unique():
        observed = re.search(r'^' + re.escape(method) + r' & (.+) \\\\$', table, re.M)
        assert observed, method
        cells = []
        for alpha in [.01, .05]:
            rows = data[(data.alpha == alpha) & (data.method == method)]
            cells.extend([f'{rows.expected_loss.mean() * 1e4:.4f}', f'{rows.expected_violation.mean() * 100:.3f}'])
        assert observed.group(1).split(' & ') == cells, method
    return dict(replication_rows=len(data), decisions=len(decisions), table_cells=32,
                summary_max_error=summary_error, exact_accounting_error=error, counts=counts)


def current_scope():
    snapshot = json.loads((OUT / 'before.json').read_text())
    assert sha(OUT / 'before_sources.zip') == snapshot['snapshot_sha256']
    allowed = {'source/main_R8.tex', 'source/supplement_R8.tex', 'source/sections_r8/deployment.tex',
               'source/analysis/provenance_r8/PRODUCERS.tsv', 'source/scripts/extension_20260831/validate_r8.py',
               'research/r8_ten_integration/package_sources.py', 'docs/IRFA_REVIEW_STATE.md',
               'source/main_R8.pdf', 'source/supplement_R8.pdf', 'Manuscript_R8.pdf',
               'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json',
               'artifacts/r8_ten_integration/source_package.json', 'release/R8_LaTeX_sources.zip',
               'release/R8_20260910_ten_external_sources.zip'}
    changed = []; preserved = []; equations = 0
    with zipfile.ZipFile(OUT / 'before_sources.zip') as old, tempfile.TemporaryDirectory(prefix='irfa-partial-scope-') as folder:
        for name, expected in snapshot['files'].items():
            before = old.read(name)
            assert hashlib.sha256(before).hexdigest() == expected
            now = (PROJECT / name).read_bytes()
            if name.endswith('.tex'):
                blocks = re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', before.decode(), re.S)
                assert all(block in now.decode() for block in blocks), name
                equations += len(blocks)
            if now == before:
                preserved.append(name)
            else:
                assert name in allowed, ('unexpected change', name)
                changed.append(name)
            if name.endswith(('.tex', '.py', '.md', '.tsv')) and now != before:
                a = Path(folder) / 'before'; a.write_bytes(before)
                process = subprocess.run(['git', 'diff', '--no-index', '--check', str(a), str(PROJECT / name)],
                                         capture_output=True, text=True)
                assert not process.stdout and not process.stderr and process.returncode in (0, 1)
        a = Path(folder) / 'empty'; a.write_text('')
        b = Path(folder) / 'bad'; b.write_text('bad whitespace \n')
        negative = subprocess.run(['git', 'diff', '--no-index', '--check', str(a), str(b)], capture_output=True, text=True)
        assert 'trailing whitespace' in negative.stdout and negative.returncode not in (0, 1)
    old_result = json.loads((PROJECT / 'artifacts/r8_count_law/final_validation.json').read_text())
    for name, expected in old_result['current_manuscript'].items():
        if name not in allowed:
            assert sha(PROJECT / name) == expected, ('previous theory/bibliography changed', name)
    protected = json.loads((STUDY / 'before.json').read_text())['canonical']
    for name, expected in protected.items():
        if name not in allowed:
            assert sha(PROJECT / name) == expected, name
    fresh = ['source/sections_r8/supp_partial.tex', 'source/sections_r8/numbers_partial.tex',
             'source/sections_r8/tab_partial.tex']
    fresh += [str(p.relative_to(PROJECT)) for p in (PROJECT / 'research/r8_partial_integration').glob('*') if p.is_file()]
    for name in fresh:
        assert all(line == line.rstrip() for line in (PROJECT / name).read_text().splitlines()), name
    return dict(changed_snapshot_files=sorted(changed), unchanged_snapshot_files=len(preserved),
                old_numbered_equations_preserved=equations, financial_displays_unchanged=True,
                previous_theory_and_bibliography_unchanged=True, whitespace_negative_control=True,
                git_metadata_present=(PROJECT / '.git').exists(),
                current_files={name: sha(PROJECT / name) for name in sorted(set(snapshot['files']) | set(fresh))})


def documents():
    guards = json.loads((PROJECT / 'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    package = json.loads((PROJECT / 'artifacts/r8_ten_integration/source_package.json').read_text())
    assert guards['documents']['negative_controls'] == 4
    for key in ['undefined_references', 'undefined_citations', 'overfull_boxes']:
        assert guards['documents'][key] == 0
    assert guards['partial_displays']['exact_display_replay'] == 2
    assert sha(PROJECT / package['source_zip']) == package['source_zip_sha256']
    assert sha(PROJECT / 'release/R8_LaTeX_sources.zip') == package['source_zip_sha256']
    for name, result in package['documents'].items():
        assert sha(SOURCE / (name + '.pdf')) == result['pdf_sha256']
        assert result['normalised_text_matches'] and result['clean_diagnostics']
        pages = PdfReader(SOURCE / (name + '.pdf')).pages
        assert len(pages) == result['pages']
        assert all(len(page.extract_text().strip()) > 10 for page in pages)
        if name == 'main_R8':
            assert 'JEL:' in pages[0].extract_text()
            assert '1. Introduction' in pages[1].extract_text()
        log = (SOURCE / (name + '.log')).read_text(errors='backslashreplace')
        assert not re.search(r'Citation .*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!', log, re.M)
    assert sha(PROJECT / 'Manuscript_R8.pdf') == package['documents']['main_R8']['pdf_sha256']
    return dict(guards=guards, source_package=package, no_blank_pages=True, title_page_complete=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-study', action='store_true')
    args = parser.parse_args()
    if args.replay_study:
        replay = study_replay()
        (OUT / 'study_replay.json').write_text(json.dumps(replay, indent=2) + '\n')
    replay = json.loads((OUT / 'study_replay.json').read_text())
    assert replay['numerical_replay_exact'] and replay['choices'] == 72000
    assert sha(PROJECT / replay['archive']) == replay['archive_sha256']
    result = dict(status='passed', producer_sha256=sha(__file__), numerical=numerical_displays(),
                  displays=check(), scope=current_scope(), documents=documents(), study_replay=replay)
    (OUT / 'final_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['scope', 'documents']}, indent=2))


if __name__ == '__main__':
    main()
