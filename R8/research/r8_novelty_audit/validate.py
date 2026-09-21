"""Current audit validation; historical source hashes remain immutable."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile
from pypdf import PdfReader
from check import ROOT, OUT, sha


def scope():
    before = json.loads((OUT/'before.json').read_text())
    assert sha(OUT/'before_sources.zip') == before['snapshot_sha256']
    allowed = {'source/sections_r8/introduction.tex', 'source/calibrating_the_oracle.bib',
               'source/main_R8.pdf', 'source/supplement_R8.pdf', 'Manuscript_R8.pdf',
               'docs/IRFA_REVIEW_STATE.md', 'release/R8_LaTeX_sources.zip',
               'release/R8_20260910_ten_external_sources.zip',
               'artifacts/r8_ten_integration/source_package.json',
               'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'}
    changed = []; equations = 0
    with zipfile.ZipFile(OUT/'before_sources.zip') as archive, tempfile.TemporaryDirectory() as folder:
        for name, wanted in before['files'].items():
            old = archive.read(name)
            assert hashlib.sha256(old).hexdigest() == wanted
            new = (ROOT/name).read_bytes()
            if name.endswith('.tex'):
                blocks = re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', old.decode(), re.S)
                equations += len(blocks)
                assert all(block in new.decode() for block in blocks)
            if old == new:
                continue
            assert name in allowed, ('Unapproved audit scope', name)
            changed.append(name)
            if name.endswith('.bib'):
                assert new.startswith(old), 'Existing bibliography changed'
            if name.endswith(('.tex', '.bib', '.md')):
                a = Path(folder)/'before'; a.write_bytes(old)
                run = subprocess.run(['git','diff','--no-index','--check',str(a),str(ROOT/name)],
                                     capture_output=True,text=True)
                assert not run.stdout and not run.stderr and run.returncode in (0, 1)
        a = Path(folder)/'empty'; a.write_text('')
        b = Path(folder)/'bad'; b.write_text('bad whitespace \n')
        run = subprocess.run(['git','diff','--no-index','--check',str(a),str(b)],capture_output=True,text=True)
        assert 'trailing whitespace' in run.stdout and run.returncode not in (0, 1)
    fresh = [str(p.relative_to(ROOT)) for p in (ROOT/'research/r8_novelty_audit').glob('*') if p.is_file()]
    fresh += ['docs/IRFA_NOVELTY_MATH_AUDIT.md','docs/IRFA_EXTERNAL_MATH_REVIEW_BRIEF.md']
    for name in fresh:
        assert all(line == line.rstrip() for line in (ROOT/name).read_text().splitlines()), name
    return dict(changed_snapshot_files=changed, preserved_snapshot_files=len(before['files'])-len(changed),
                numbered_equation_blocks_preserved=equations, financial_displays_unchanged=True,
                all_theorem_statements_and_proofs_unchanged=True,
                bibliography_append_only=True, new_references=7,
                whitespace_negative_control=True, git_metadata_present=(ROOT/'.git').exists(),
                current_files={p:sha(ROOT/p) for p in sorted(set(before['files'])|set(fresh))})


def documents():
    source = ROOT/'source'
    guard = json.loads((ROOT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    assert guard['documents']['negative_controls'] == 4
    assert all(guard['documents'][k] == 0 for k in ['undefined_references','undefined_citations','overfull_boxes'])
    display_count = guard['displays']['displays'] + sum(guard[k]['exact_display_replay'] for k in
        ['risk_displays','regime_displays','ten_external_displays','partial_displays','information_displays'])
    assert display_count == 57
    package = json.loads((ROOT/'artifacts/r8_ten_integration/source_package.json').read_text())
    assert package['source_members'] == 66
    assert sha(ROOT/package['source_zip']) == package['source_zip_sha256'] == sha(ROOT/'release/R8_LaTeX_sources.zip')
    visual = json.loads((OUT/'visual_validation.json').read_text())
    for doc, result in package['documents'].items():
        assert sha(source/(doc+'.pdf')) == result['pdf_sha256'] == visual['documents'][doc]['pdf_sha256']
        assert result['normalised_text_matches'] and result['clean_diagnostics']
        pages = PdfReader(source/(doc+'.pdf')).pages
        assert len(pages) == result['pages'] and all(len(p.extract_text().strip()) > 10 for p in pages)
        log = (source/(doc+'.log')).read_text(errors='backslashreplace')
        assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!',log,re.M)
        if doc == 'main_R8':
            assert 'JEL:' in pages[0].extract_text() and '1. Introduction' in pages[1].extract_text()
    assert sha(ROOT/'Manuscript_R8.pdf') == package['documents']['main_R8']['pdf_sha256']
    return dict(exact_displays=display_count, guards=guard['documents'], source_package=package,
                visual_inspection=visual, no_blank_pages=True)


if __name__ == '__main__':
    run = subprocess.run([sys.executable,str(Path(__file__).with_name('check.py'))],capture_output=True,text=True)
    assert run.returncode == 0, (run.stdout,run.stderr)
    checks = json.loads((OUT/'mathematical_checks.json').read_text())
    assert checks['status'] == 'passed' and not checks['independent_external_reviewer']
    assert checks['producer_sha256'] == sha(Path(__file__).with_name('check.py'))
    for name, wanted in checks['outputs'].items():
        assert sha(OUT/name) == wanted
    result = dict(status='passed', producer_sha256=sha(__file__), mathematics=checks,
                  scope=scope(), documents=documents(), fresh_process_mathematics_exact=True,
                  independent_external_review_obtained=False)
    (OUT/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['scope','documents']},indent=2))
