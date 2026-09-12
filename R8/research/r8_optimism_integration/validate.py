"""Validate the authorised integration without rebinding historical studies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_optimism_integration'
DOC = ROOT / 'docs/optimism_integration_20260911'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def negative_first(check, bad, good):
    try:
        check(bad)
    except (AssertionError, KeyError, FileNotFoundError):
        pass
    else:
        raise AssertionError('Deliberately invalid case was not rejected')
    check(good)


def main():
    before = json.loads((OUT / 'before.json').read_text())['files']
    allowed = {'source/' + n for n in ['main_R8.tex', 'supplement_R8.tex',
               'main_R8.pdf', 'supplement_R8.pdf', 'calibrating_the_oracle.bib']}
    allowed |= {'source/sections_r8/' + n + '.tex' for n in
                ['risk', 'introduction', 'montecarlo', 'discussion']}
    allowed |= {'Manuscript_R8.pdf', 'release/R8_LaTeX_sources.zip',
                'docs/IRFA_REVIEW_STATE.md', 'docs/IRFA_CLAIM_EVIDENCE_MAP.md'}
    changed = [n for n, h in before.items() if sha(ROOT / n) != h]
    def preservation(names):
        assert set(names) <= allowed
    negative_first(preservation, changed + ['source/sections_r8/tab_methods.tex'], changed)

    bindings = json.loads((OUT / 'study_bindings.json').read_text())
    def immutable(items):
        for n, v in items.items():
            assert sha(ROOT / n) == v['sha256']
            assert (ROOT / n).stat().st_mtime_ns == v['mtime_ns']
    bad = json.loads(json.dumps(bindings)); bad[next(iter(bad))]['sha256'] = '0'*64
    negative_first(immutable, bad, bindings)

    old_statement_count = old_display_count = 0
    statements = r'\\begin\{(theorem|proposition|corollary|lemma|assumption)\}.*?\\end\{\1\}'
    displays = r'\\begin\{(equation|align)\}.*?\\end\{\1\}'
    with zipfile.ZipFile(OUT / 'before_sources.zip') as z, tempfile.TemporaryDirectory() as tmp:
        for n, h in before.items():
            a = z.read(n); b = (ROOT / n).read_bytes()
            assert hashlib.sha256(a).hexdigest() == h
            if n.endswith('.tex'):
                old, new = a.decode(), b.decode()
                for m in re.finditer(statements, old, re.S):
                    assert m.group() in new, (n, 'changed old formal statement')
                    old_statement_count += 1
                for m in re.finditer(displays, old, re.S):
                    assert m.group() in new, (n, 'changed old numbered display')
                    old_display_count += 1
                labels = lambda s: Counter(re.findall(r'\\label\{([^}]+)\}', s))
                assert not (labels(old) - labels(new)), (n, 'lost label')
            if n in changed and n.endswith(('.tex', '.md', '.bib')):
                p = Path(tmp) / 'before'; p.write_bytes(a)
                r = subprocess.run(['git', 'diff', '--no-index', '--check', str(p), str(ROOT / n)], capture_output=True, text=True)
                assert not r.stdout and not r.stderr and r.returncode in (0, 1), (n, r.stdout, r.stderr)
        oldbib = z.read('source/calibrating_the_oracle.bib')
        assert (ROOT / 'source/calibrating_the_oracle.bib').read_bytes().startswith(oldbib.rstrip())
        blank, bad = Path(tmp)/'blank', Path(tmp)/'bad'
        blank.write_text(''); bad.write_text('trailing \n')
        r = subprocess.run(['git', 'diff', '--no-index', '--check', str(blank), str(bad)], capture_output=True, text=True)
        assert 'trailing whitespace' in r.stdout
    for folder in [ROOT/'research/r8_optimism_integration', DOC]:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ('.py', '.md', '.tex'):
                assert all(line == line.rstrip() for line in p.read_text().splitlines()), p

    math = (DOC/'MATH_REVIEW.md').read_text().split('### Final source bindings')[-1]
    pairs = re.findall(r'([0-9a-f]{64})\s+(\S+)', math)
    assert len(pairs) == 5
    def current_sources(items):
        for h, n in items:
            assert sha(ROOT/n) == h
    negative_first(current_sources, [('0'*64, pairs[0][1])], pairs)
    numeric = json.loads((OUT/'independent_numbers.json').read_text())
    assert numeric['status'].lower() == 'pass'
    for field in ['source_sha256', 'input_sha256']:
        current_sources([(h,n) for n,h in numeric[field].items()])
    assert numeric['verifier_sha256'] == sha(ROOT/'research/r8_optimism_integration/check_numbers.py')
    statistical = (DOC/'STATISTICAL_REVIEW.md').read_text()
    for n in ['numbers_optimism', 'tab_optimism', 'supp_optimism', 'montecarlo']:
        assert sha(ROOT/'source/sections_r8'/f'{n}.tex') in statistical, n

    newdisplays = json.loads((OUT/'display_check.json').read_text())
    for field in ['inputs', 'outputs']:
        for n,h in newdisplays[field].items(): assert sha(ROOT/n) == h
    assert newdisplays['producer_sha256'] == sha(ROOT/'research/r8_optimism_integration/displays.py')
    guard_path = ROOT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'
    guard = json.loads(guard_path.read_text())
    assert guard['documents']['negative_controls'] == 4
    assert all(guard['documents'][k] == 0 for k in ['undefined_references', 'undefined_citations', 'overfull_boxes'])
    total = guard['displays']['displays'] + sum(guard[k]['exact_display_replay'] for k in
              ['risk_displays','regime_displays','ten_external_displays','partial_displays','information_displays'])
    assert total == 57
    shape = json.loads((ROOT/'artifacts/r8_shape_integration/display_check.json').read_text())
    assert shape['status'] == 'passed' and shape['macros'] == 33 and shape['exact_vector_pdf_replay']
    for n,h in shape['inputs'].items(): assert sha(ROOT/n) == h

    package = json.loads((OUT/'source_package.json').read_text())
    assert package['source_members'] == 77
    assert sha(ROOT/package['source_zip']) == package['source_zip_sha256'] == sha(ROOT/'release/R8_LaTeX_sources.zip')
    with zipfile.ZipFile(ROOT/package['source_zip']) as z:
        assert len(z.namelist()) == 77
        for n in z.namelist():
            if n != 'BUILD.txt': assert z.read(n) == (ROOT/'source'/n).read_bytes()
    visual = json.loads((OUT/'visual_validation.json').read_text())
    assert visual['layout_defects'] == 0
    for n,h in visual['rendered_files'].items(): assert sha(ROOT/n) == h
    for d, item in package['documents'].items():
        assert sha(ROOT/'source'/f'{d}.pdf') == item['pdf_sha256'] == visual['documents'][d]['pdf_sha256']
        assert item['normalised_text_matches'] and item['clean_diagnostics']
        assert len(PdfReader(ROOT/'source'/f'{d}.pdf').pages) == item['pages']
        assert visual['documents'][d]['all_pages_layout_inspected'] == item['pages']
        log = (ROOT/'source'/f'{d}.log').read_text(errors='replace')
        assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!', log, re.M)
    assert sha(ROOT/'Manuscript_R8.pdf') == package['documents']['main_R8']['pdf_sha256']
    files = set(before) | {'source/analysis/provenance_r8/PRODUCERS.tsv',
                          'source/analysis/provenance_r8/DECLARED_CONSTANTS.md'}
    for folder in [ROOT/'research/r8_optimism_integration', DOC, OUT]:
        files.update(str(p.relative_to(ROOT)) for p in folder.rglob('*') if p.is_file()
                     and p.name != 'final_validation.json' and '__pycache__' not in p.parts)
    files.update(newdisplays['outputs']); files.add(package['source_zip'])
    receipt = {'status':'passed','changed_snapshot_files':changed,'snapshot_files':len(before),
        'old_formal_statements_preserved':old_statement_count,'old_numbered_displays_preserved':old_display_count,
        'research_artifacts_unchanged_by_hash_and_mtime':len(bindings),
        'old_global_displays':total,'shape_macros':33,'new_macros':11,'new_table_rows':6,
        'source_bound_mathematics':'passed','independent_numeric_verification':'passed',
        'guards':guard['documents'],'package':package,'visual':visual,
        'no_index_diff_check':True,'git_metadata_present':(ROOT/'.git').exists(),
        'synthetic_admission':'FAIL_UNCHANGED','financial_panel_estimated_penalty':'NOT_RUN',
        'new_fits_or_paths':False,'complete_empirical_archive_rebuilt':False,
        'files':{n:{'sha256':sha(ROOT/n),'mtime_ns':(ROOT/n).stat().st_mtime_ns} for n in sorted(files)}}
    (OUT/'final_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','package','visual']},indent=2))


if __name__ == '__main__':
    main()
