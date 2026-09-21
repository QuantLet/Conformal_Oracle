"""Validate an editorial change against unchanged mathematics and evidence."""
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_financial_argument'
DOC = ROOT / 'docs/financial_argument_20260911'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    before = json.loads((OUT / 'before.json').read_text())['files']
    prose = {'source/sections_r8/' + n + '.tex' for n in ['introduction', 'results', 'discussion']}
    allowed = prose | {'source/main_R8.pdf', 'source/supplement_R8.pdf', 'Manuscript_R8.pdf',
                       'docs/IRFA_REVIEW_STATE.md', 'docs/IRFA_CLAIM_EVIDENCE_MAP.md'}
    changed = [n for n, h in before.items() if sha(ROOT / n) != h]
    assert set(changed) <= allowed
    patterns = [r'\\begin\{(theorem|proposition|corollary|lemma|assumption)\}.*?\\end\{\1\}',
                r'\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}|\\\[.*?\\\]|(?<!\\)\$.*?(?<!\\)\$',
                r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}', r'\\label\{[^}]+\}']
    extract = lambda pat, s: [m.group() for m in re.finditer(pat, s, re.S)]
    cite_keys = lambda s: Counter(k for keys in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', s) for k in keys.split(','))
    words = lambda s: len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", re.sub(r'\\[A-Za-z]+', '', s)))
    counts = {}; pages = {}
    with zipfile.ZipFile(OUT / 'before_sources.zip') as z, tempfile.TemporaryDirectory() as tmp:
        for n, h in before.items():
            a = z.read(n); b = (ROOT / n).read_bytes()
            assert hashlib.sha256(a).hexdigest() == h
            if n.endswith('.tex'):
                a, b = a.decode(), b.decode()
                assert cite_keys(a) == cite_keys(b), (n, 'citation keys')
                for pat in patterns:
                    assert extract(pat, a) == extract(pat, b), (n, 'protected structure')
                assert Counter(re.findall(r'\\n[A-Z]\w*', a)) == Counter(re.findall(r'\\n[A-Z]\w*', b)), (n, 'numeric macro use')
                if n in prose:
                    counts[n] = {'before': words(a), 'after': words(b)}
            if n in changed and n.endswith(('.md', '.tex')):
                p = Path(tmp) / 'before'; p.write_bytes(z.read(n))
                check = subprocess.run(['git', 'diff', '--no-index', '--check', str(p), str(ROOT / n)], capture_output=True, text=True)
                assert check.returncode in (0, 1) and not check.stdout and not check.stderr
        blank, bad = Path(tmp) / 'blank', Path(tmp) / 'bad'
        blank.write_text(''); bad.write_text('trailing \n')
        check = subprocess.run(['git', 'diff', '--no-index', '--check', str(blank), str(bad)], capture_output=True, text=True)
        assert 'trailing whitespace' in check.stdout
        for d in ['main_R8', 'supplement_R8']:
            a = PdfReader(io.BytesIO(z.read(f'source/{d}.pdf'))); b = PdfReader(ROOT / 'source' / f'{d}.pdf')
            norm = lambda p: re.sub(r'\s+', ' ', p.extract_text() or '').strip()
            assert len(a.pages) == len(b.pages)
            pages[d] = {'before_pages': len(a.pages), 'pages': len(b.pages),
                        'changed_text_pages': [i + 1 for i, (x, y) in enumerate(zip(a.pages, b.pages)) if norm(x) != norm(y)]}
    assert pages == json.loads((OUT / 'page_comparison.json').read_text())
    assert pages['main_R8']['pages'] == 44 and pages['supplement_R8']['pages'] == 36
    assert not pages['supplement_R8']['changed_text_pages']

    evidence = ROOT / 'artifacts/r8_shape_cost/financial/summary.csv'
    with evidence.open() as f:
        rows = {r['state']: r for r in csv.DictReader(f) if r['population'] == 'matched161'}
    num = lambda state, col: Decimal(rows[state][col])
    assert set(rows) == {'all', 'high', 'other'} and all(r['pairs'] == '161' for r in rows.values())
    assert abs(num('all', 'shift_pi') - Decimal('.01')) < Decimal('.00001')
    assert num('high', 'shift_pi') > Decimal('.01')
    assert num('high', 'vol_pi') < num('high', 'shift_pi') and num('high', 'delta_normalized') < 0
    assert num('other', 'vol_pi') > num('other', 'shift_pi') and num('other', 'delta_normalized') > 0
    assert num('high', 'threshold_normalized') > 0
    assert num('high', 'overshoot_normalized') < -num('high', 'threshold_normalized')
    for state in rows:
        assert abs(num(state, 'threshold_normalized') + num(state, 'overshoot_normalized') - num(state, 'delta_normalized')) < Decimal('1e-15')
    review = (DOC / 'POST_EDIT_REVIEW.md').read_text()
    for n in prose:
        assert sha(ROOT / n) in review, ('stale independent review', n)
    bindings = json.loads(re.search(r'```json\n(.*?)\n```', review, re.S).group(1))
    assert all(sha(ROOT / n) == h for n, h in bindings.items())
    inference = json.loads((ROOT / 'artifacts/r8_shape_cost/financial/inference_status.json').read_text())
    assert len(inference) == 2 and all(r['inference'] == 'aborted_empty_state' and r['primary_family_size'] == 4 for r in inference)

    guard_path = ROOT / 'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'
    guard = json.loads(guard_path.read_text())
    assert guard['documents']['negative_controls'] == 4
    assert all(guard['documents'][k] == 0 for k in ['undefined_references', 'undefined_citations', 'overfull_boxes'])
    total = guard['displays']['displays'] + sum(guard[k]['exact_display_replay'] for k in ['risk_displays', 'regime_displays', 'ten_external_displays', 'partial_displays', 'information_displays'])
    assert total == 57
    shape_path = ROOT / 'artifacts/r8_shape_integration/display_check.json'
    shape = json.loads(shape_path.read_text())
    assert shape['status'] == 'passed' and shape['macros'] == 33 and shape['exact_vector_pdf_replay']
    assert all(sha(ROOT / n) == h for n, h in shape['inputs'].items())
    package = json.loads((OUT / 'source_package.json').read_text())
    assert package['source_members'] == 73
    assert sha(ROOT / package['source_zip']) == package['source_zip_sha256'] == sha(ROOT / 'release/R8_LaTeX_sources.zip')
    with zipfile.ZipFile(ROOT / package['source_zip']) as z:
        assert len(z.namelist()) == 73
        for n in z.namelist():
            if n != 'BUILD.txt': assert z.read(n) == (ROOT / 'source' / n).read_bytes()
    visual = json.loads((OUT / 'visual_validation.json').read_text())
    assert visual['inspected_pages'] == {d: p['changed_text_pages'] for d, p in pages.items()}
    assert visual['layout_defects'] == 0
    for n, h in visual['rendered_files'].items(): assert sha(ROOT / n) == h
    for d, item in package['documents'].items():
        assert sha(ROOT / 'source' / f'{d}.pdf') == item['pdf_sha256'] == visual['documents'][d]['pdf_sha256']
        assert item['normalised_text_matches'] and item['clean_diagnostics']
        log = (ROOT / 'source' / f'{d}.log').read_text(errors='replace')
        assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!', log, re.M)
    assert sha(ROOT / 'Manuscript_R8.pdf') == package['documents']['main_R8']['pdf_sha256']
    files = set(before) | {str(p.relative_to(ROOT)) for base in [ROOT / 'research/r8_financial_argument', DOC, OUT] for p in base.iterdir() if p.is_file() and p.name != 'final_validation.json'}
    files.update([str(evidence.relative_to(ROOT)), 'artifacts/r8_shape_cost/financial/inference_status.json', str(guard_path.relative_to(ROOT)), str(shape_path.relative_to(ROOT)), package['source_zip'], 'release/R8_LaTeX_sources.zip'])
    receipt = {'status': 'passed', 'changed_snapshot_files': changed, 'snapshot_files': len(before),
               'protected_math_labels_citation_keys_floats_and_numeric_macros': 'unchanged',
               'word_counts': counts, 'page_comparison': pages, 'state_loss_signs_and_identity': 'passed',
               'source_bound_independent_challenge': 'passed', 'global_displays': total, 'shape_macros': 33,
               'guards': guard['documents'], 'source_package': package, 'visual_inspection': visual,
               'git_metadata_present': (ROOT / '.git').exists(), 'no_index_diff_check': True,
               'whitespace_negative_control': True, 'new_fits_or_inference_or_simulations': False,
               'current_file_sha256': {n: sha(ROOT / n) for n in sorted(files)}}
    (OUT / 'final_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['current_file_sha256', 'source_package', 'visual_inspection']}, indent=2))


if __name__ == '__main__':
    main()
