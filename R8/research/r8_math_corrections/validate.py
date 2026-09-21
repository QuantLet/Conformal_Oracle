"""Validate the authorised local correction and current document receipts."""
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

from pypdf import PdfReader
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_math_corrections'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before = json.loads((OUT / 'before.json').read_text())['files']
    allowed = {
        'source/sections_r8/risk_proofs.tex', 'docs/IRFA_REVIEW_STATE.md',
        'source/main_R8.pdf', 'source/supplement_R8.pdf', 'Manuscript_R8.pdf',
    }
    changed = [n for n, h in before.items() if sha(ROOT / n) != h]
    assert set(changed) <= allowed, changed
    proof_name = 'source/sections_r8/risk_proofs.tex'
    assert proof_name in changed
    with zipfile.ZipFile(OUT / 'before_sources.zip') as z, tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for n, h in before.items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h, n
        target = work / proof_name
        target.parent.mkdir(parents=True)
        target.write_bytes(z.read(proof_name))
        run = subprocess.run(
            ['patch', '--fuzz=0', '-p1', '-d', str(work)],
            input=(OUT / 'authorised.patch').read_text(),
            text=True, capture_output=True,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert target.read_bytes() == (ROOT / proof_name).read_bytes()
        for n in changed:
            if n.endswith(('.tex', '.md')):
                old = work / 'before.txt'
                old.write_bytes(z.read(n))
                check = subprocess.run(
                    ['git', 'diff', '--no-index', '--check', str(old), str(ROOT / n)],
                    capture_output=True, text=True,
                )
                assert check.returncode in (0, 1) and not check.stdout and not check.stderr
        empty, bad = work / 'empty', work / 'bad'
        empty.write_text('')
        bad.write_text('trailing \n')
        check = subprocess.run(
            ['git', 'diff', '--no-index', '--check', str(empty), str(bad)],
            capture_output=True, text=True,
        )
        assert 'trailing whitespace' in check.stdout
        page_comparison = {}
        for doc in ['main_R8', 'supplement_R8']:
            old = PdfReader(io.BytesIO(z.read(f'source/{doc}.pdf')))
            new = PdfReader(ROOT / 'source' / f'{doc}.pdf')
            norm = lambda p: re.sub(r'\s+', ' ', p.extract_text() or '').strip()
            assert len(old.pages) == len(new.pages)
            page_comparison[doc] = {
                'pages': len(new.pages),
                'changed_text_pages': [i + 1 for i, (a, b) in enumerate(zip(old.pages, new.pages)) if norm(a) != norm(b)],
            }
    assert page_comparison == json.loads((OUT / 'page_comparison.json').read_text())
    assert page_comparison['main_R8'] == {'pages': 44, 'changed_text_pages': []}
    assert page_comparison['supplement_R8'] == {'pages': 36, 'changed_text_pages': [18, 20, 21]}

    guard_path = ROOT / 'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'
    guard = json.loads(guard_path.read_text())
    assert guard['documents']['negative_controls'] == 4
    assert all(guard['documents'][x] == 0 for x in ['undefined_references', 'undefined_citations', 'overfull_boxes'])
    displays = guard['displays']['displays'] + sum(
        guard[k]['exact_display_replay'] for k in [
            'risk_displays', 'regime_displays', 'ten_external_displays',
            'partial_displays', 'information_displays',
        ]
    )
    assert displays == 57
    shape_path = ROOT / 'artifacts/r8_shape_integration/display_check.json'
    shape = json.loads(shape_path.read_text())
    assert shape['status'] == 'passed' and shape['macros'] == 33 and shape['exact_vector_pdf_replay']
    assert all(sha(ROOT / n) == h for n, h in shape['inputs'].items())

    package = json.loads((OUT / 'source_package.json').read_text())
    assert package['source_members'] == 73
    assert sha(ROOT / package['source_zip']) == package['source_zip_sha256']
    assert sha(ROOT / 'release/R8_LaTeX_sources.zip') == package['source_zip_sha256']
    with zipfile.ZipFile(ROOT / package['source_zip']) as z:
        assert len(z.namelist()) == 73
        for name in z.namelist():
            if name != 'BUILD.txt':
                assert z.read(name) == (ROOT / 'source' / name).read_bytes(), name
    visual = json.loads((OUT / 'visual_validation.json').read_text())
    assert visual['inspected_pages'] == {'supplement_R8': [18, 20, 21]}
    assert visual['clipping_overlap_or_legibility_defects'] == 0
    assert all(sha(ROOT / n) == h for n, h in visual['rendered_files'].items())
    for doc, info in package['documents'].items():
        pdf = ROOT / 'source' / f'{doc}.pdf'
        assert sha(pdf) == info['pdf_sha256'] == visual['documents'][doc]['pdf_sha256']
        assert info['normalised_text_matches'] and info['clean_diagnostics']
        log = (ROOT / 'source' / f'{doc}.log').read_text(errors='replace')
        assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!', log, re.M)
    assert sha(ROOT / 'Manuscript_R8.pdf') == package['documents']['main_R8']['pdf_sha256']

    # The general CDF identity also holds in the two innovation laws used by
    # the existing controls. Deliberately reusing the Normal cutoff fails t5.
    quantile_checks = []
    scale = (3 / 5) ** .5
    for alpha in [.01, .05, .5]:
        for law, q, probability in [
            ('normal', stats.norm.ppf(alpha), lambda x: stats.norm.cdf(x)),
            ('standardised_t5', scale * stats.t.ppf(alpha, 5), lambda x: stats.t.cdf(x / scale, 5)),
        ]:
            achieved = float(probability(q))
            assert abs(achieved - alpha) < 1e-10
            quantile_checks.append({'law': law, 'alpha': alpha, 'cdf_at_quantile': achieved})
    wrong = float(stats.t.cdf(stats.norm.ppf(.01) / scale, 5))
    assert abs(wrong - .01) > .001

    files = set(before) | {
        str(p.relative_to(ROOT))
        for base in [ROOT / 'research/r8_math_corrections', OUT]
        for p in base.iterdir()
        if p.is_file() and p.name != 'final_validation.json'
    }
    files.update([
        'docs/IRFA_MATH_CORRECTIONS_20260911.md',
        package['source_zip'], 'release/R8_LaTeX_sources.zip',
        str(guard_path.relative_to(ROOT)), str(shape_path.relative_to(ROOT)),
    ])
    result = {
        'status': 'passed', 'revision': 'R8', 'authorised_corrections_applied': 3,
        'patch_matches_exactly': True, 'snapshot_files': len(before),
        'changed_snapshot_files': changed,
        'unchanged_snapshot_files': len(before) - len(changed),
        'git_metadata_present': (ROOT / '.git').exists(),
        'no_index_whitespace_check': True, 'whitespace_negative_control': True,
        'page_comparison': page_comparison, 'global_display_count': displays,
        'global_document_guards': guard['documents'],
        'shape_macros': 33, 'shape_vector_replay': True,
        'source_package': package, 'visual_inspection': visual,
        'innovation_quantile_checks': quantile_checks,
        'wrong_normal_cutoff_t5_probability': wrong,
        'new_model_inference_or_simulation': False,
        'new_full_empirical_archive': False,
        'current_file_sha256': {n: sha(ROOT / n) for n in sorted(files)},
    }
    (OUT / 'final_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['current_file_sha256', 'visual_inspection', 'source_package']}, indent=2))


if __name__ == '__main__':
    main()
