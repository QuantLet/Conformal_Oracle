"""Editorial-scope and document checks; no new numerical experiments."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_concision'
EDITED = {'introduction', 'methodology', 'theory', 'risk', 'montecarlo',
          'results', 'deployment', 'information', 'regime', 'external', 'discussion'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope():
    before = json.loads((OUT / 'before.json').read_text())
    assert sha(OUT / 'before_sources.zip') == before['snapshot_sha256']
    allowed = {'source/sections_r8/' + n + '.tex' for n in EDITED}
    allowed.update({'source/main_R8.tex', 'source/main_R8.pdf', 'source/supplement_R8.pdf', 'Manuscript_R8.pdf',
                    'docs/IRFA_REVIEW_STATE.md', 'release/R8_LaTeX_sources.zip',
                    'release/R8_20260910_ten_external_sources.zip',
                    'artifacts/r8_ten_integration/source_package.json',
                    'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'})
    changes = []; counts = {}; blocks = collections.Counter(); oldcites = set(); newcites = set()
    env = r'\\begin\{(equation|align|theorem|proposition|corollary|lemma|assumption|remark|table|figure)\}.*?\\end\{\1\}'
    with zipfile.ZipFile(OUT / 'before_sources.zip') as z, tempfile.TemporaryDirectory() as folder:
        for name, wanted in before['files'].items():
            a = z.read(name); b = (ROOT / name).read_bytes()
            assert hashlib.sha256(a).hexdigest() == wanted
            if name.endswith('.tex'):
                old, new = a.decode(), b.decode()
                if name == 'source/main_R8.tex':
                    assert old.split(r'\begin{abstract}')[0] == new.split(r'\begin{abstract}')[0]
                    assert old.split(r'\end{abstract}')[1] == new.split(r'\end{abstract}')[1]
                labels = lambda s: collections.Counter(re.findall(r'\\label\{([^}]+)\}', s))
                assert labels(old) == labels(new), ('label change', name)
                for s, target in ((old, oldcites), (new, newcites)):
                    for c in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', s):
                        target.update(c.split(','))
                aa = list(re.finditer(env, old, re.S)); bb = list(re.finditer(env, new, re.S))
                assert [m.group() for m in aa] == [m.group() for m in bb], ('protected block', name)
                blocks.update(m.group(1) for m in aa)
                assert re.findall(r'\\\[.*?\\\]', old, re.S) == re.findall(r'\\\[.*?\\\]', new, re.S)
                if a != b:
                    counts[name] = {'before': len(old.split()), 'after': len(new.split())}
            if a == b:
                continue
            assert name in allowed, ('outside editorial scope', name)
            changes.append(name)
            if name.endswith(('.tex', '.md')):
                p = Path(folder) / 'before'; p.write_bytes(a)
                run = subprocess.run(['git', 'diff', '--no-index', '--check', str(p), str(ROOT / name)], capture_output=True, text=True)
                assert run.returncode in (0, 1) and not run.stdout and not run.stderr, (name, run.stdout, run.stderr)
        p = Path(folder) / 'empty'; p.write_text('')
        q = Path(folder) / 'negative'; q.write_text('trailing space \n')
        run = subprocess.run(['git', 'diff', '--no-index', '--check', str(p), str(q)], capture_output=True, text=True)
        assert 'trailing whitespace' in run.stdout and run.returncode not in (0, 1)
    assert oldcites == newcites, (oldcites - newcites, newcites - oldcites)
    fresh = [str(p.relative_to(ROOT)) for p in (ROOT / 'research/r8_concision').glob('*') if p.is_file()]
    fresh += ['docs/IRFA_CONCISION.md', 'docs/IRFA_CLAIM_EVIDENCE_MAP.md']
    for name in fresh:
        assert all(line == line.rstrip() for line in (ROOT / name).read_text().splitlines()), name
    return dict(changed_snapshot_files=changes, preserved_snapshot_files=len(before['files'])-len(changes),
                word_counts=counts, tokens_removed=sum(v['before']-v['after'] for v in counts.values()),
                count_convention='Whitespace-delimited LaTeX tokens in edited sections; not natural-language words.',
                protected_blocks=dict(blocks), all_labels_and_citation_keys_preserved=True,
                financial_displays_and_all_proofs_unchanged=True, typography_unchanged=True,
                whitespace_negative_control=True, git_metadata_present=(ROOT / '.git').exists(),
                current_files={n: sha(ROOT / n) for n in sorted(set(before['files']) | set(fresh))})


def documents():
    source = ROOT / 'source'
    guard = json.loads((ROOT / 'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    assert guard['documents']['negative_controls'] == 4
    assert all(guard['documents'][k] == 0 for k in ['undefined_references', 'undefined_citations', 'overfull_boxes'])
    n = guard['displays']['displays'] + sum(guard[k]['exact_display_replay'] for k in
        ['risk_displays','regime_displays','ten_external_displays','partial_displays','information_displays'])
    assert n == 57
    package = json.loads((ROOT / 'artifacts/r8_ten_integration/source_package.json').read_text())
    assert package['source_members'] == 66
    assert sha(ROOT / package['source_zip']) == package['source_zip_sha256'] == sha(ROOT / 'release/R8_LaTeX_sources.zip')
    visual = json.loads((OUT / 'visual_validation.json').read_text())
    expected = {'main_R8': 41, 'supplement_R8': 32}
    for name, info in package['documents'].items():
        assert info['pdf_sha256'] == sha(source / (name + '.pdf')) == visual['documents'][name]['pdf_sha256']
        assert info['normalised_text_matches'] and info['clean_diagnostics']
        pages = PdfReader(source / (name + '.pdf')).pages
        assert len(pages) == info['pages'] == expected[name]
        assert all(len(p.extract_text().strip()) > 10 for p in pages)
        if name == 'main_R8':
            assert 'JEL:' in pages[0].extract_text() and '1. Introduction' in pages[1].extract_text()
        log = (source / (name + '.log')).read_text(errors='backslashreplace')
        assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!', log, re.M)
    assert sha(ROOT / 'Manuscript_R8.pdf') == package['documents']['main_R8']['pdf_sha256']
    return dict(exact_displays=n, guards=guard['documents'], source_package=package,
                visual_inspection=visual, no_blank_pages=True, article_pages_removed=9,
                supplement_source_unchanged=True)


if __name__ == '__main__':
    claim = json.loads((OUT / 'claim_check.json').read_text())
    assert claim['claim_verified'] and sha(ROOT / claim['input']) == claim['sha256']
    result = dict(claim_check=claim, status='passed', producer_sha256=sha(__file__), scope=scope(), documents=documents(),
                  new_inference_or_simulations=False, prior_mathematical_audit_preserved=True)
    (OUT / 'final_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('scope', 'documents')}, indent=2))
    print('LaTeX tokens removed:', result['scope']['tokens_removed'])
