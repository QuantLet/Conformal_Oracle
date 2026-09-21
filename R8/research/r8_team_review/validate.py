"""Conservation and document checks for the focused reviewer revision.

Run the existing R8 display/document guard and portable source build first.
This script adds snapshot conservation and honest review-count accounting;
it neither generates observations nor changes empirical specifications.
"""
import collections
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader
from review_index import collect
from ewma_check import check as check_ewma

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_team_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_displays(text):
    # Only the listed, reviewed caption clarifications are permitted inside floats.
    text = text.replace('QS in return units is multiplied by $10^4$.',
                        'QS is multiplied by $10^4$.')
    approved_caption_edits = [('Top: paired loss differences\n'
      'from Shift-CP, with simultaneous 95\\% bands across eight comparisons,\n'
      'using common 60-calendar-day blocks. Bottom: mean loss and violation rate\n'
      'of feasible policies.',
      'Top: method-minus-Shift-CP QS differences, with simultaneous 95\\%\n'
      'bands across eight comparisons separately for all assets and after excluding\n'
      'Bitcoin and Ethereum, using common 60-calendar-day blocks. Negative differences\n'
      'favour the named method. The asset-sample colour legend applies only to the\n'
      'top panel. Bottom: mean loss and violation rate of feasible policies on all assets.'),
     ('Bands use 60-calendar-day blocks',
      'Simultaneous 95\\% bands use 60-calendar-day blocks'),
     ('Bands cover loss differences', '95\\% bands cover loss differences'),
     ('Shading gives pointwise paired Monte Carlo\nuncertainty.',
      'Shading gives pointwise approximate 95\\% paired Monte Carlo\nintervals.')]
    for old_caption, new_caption in approved_caption_edits:
        text = text.replace(new_caption, old_caption)
    patterns = [r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}',
                r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{[^}]+\}']
    return [[m.group() for m in re.finditer(p, text, re.S)] for p in patterns]


def scope():
    before = json.loads((OUT / 'before.json').read_text())
    assert sha(OUT / 'before_sources.zip') == before['snapshot_sha256']
    allowed = {'source/main_R8.tex', 'source/main_R8.pdf',
               'source/supplement_R8.pdf', 'Manuscript_R8.pdf',
               'docs/IRFA_REVIEW_STATE.md', 'docs/IRFA_CLAIM_EVIDENCE_MAP.md',
               'release/R8_LaTeX_sources.zip',
               'release/R8_20260910_ten_external_sources.zip',
               'artifacts/r8_ten_integration/source_package.json',
               'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'}
    allowed.update('source/sections_r8/' + n + '.tex' for n in
                   ['introduction', 'methodology', 'risk', 'information', 'external',
                    'theory', 'montecarlo', 'results', 'deployment', 'discussion', 'regime', 'supp_specs'])
    changes, counts = [], {}
    oldcites, newcites = set(), set()
    protected = collections.Counter()
    formal = r'\\begin\{(theorem|proposition|corollary|lemma|assumption)\}.*?\\end\{\1\}'
    maths = r'\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}|\\\[.*?\\\]'
    with zipfile.ZipFile(OUT / 'before_sources.zip') as archive, tempfile.TemporaryDirectory() as folder:
        for name, digest in before['files'].items():
            old = archive.read(name)
            new = (ROOT / name).read_bytes()
            assert hashlib.sha256(old).hexdigest() == digest
            if name.endswith('.tex'):
                a, b = old.decode(), new.decode()
                if name == 'source/main_R8.tex':
                    assert a.split(r'\begin{abstract}')[0] == b.split(r'\begin{abstract}')[0]
                    assert a.split(r'\end{abstract}')[1] == b.split(r'\end{abstract}')[1]
                labels = lambda s: collections.Counter(re.findall(r'\\label\{([^}]+)\}', s))
                assert labels(a) == labels(b), ('Changed labels', name)
                assert protected_displays(a) == protected_displays(b), ('Changed float or input selection', name)
                for pattern in [formal, maths]:
                    extract = lambda s: [m.group() for m in re.finditer(pattern, s, re.S)]
                    assert extract(a) == extract(b), ('Changed formal statement/math', name)
                protected.update(m.group(1) for m in re.finditer(formal, a, re.S))
                for s, target in [(a, oldcites), (b, newcites)]:
                    for keys in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', s):
                        target.update(keys.split(','))
                if old != new:
                    counts[name] = dict(before=len(a.split()), after=len(b.split()))
            if old == new:
                continue
            assert name in allowed, ('Out-of-scope change', name)
            changes.append(name)
            if name.endswith(('.tex', '.md')):
                original = Path(folder) / 'original'
                original.write_bytes(old)
                r = subprocess.run(['git', 'diff', '--no-index', '--check', str(original), str(ROOT / name)],
                                   capture_output=True, text=True)
                assert r.returncode in [0, 1] and not r.stdout and not r.stderr, (name, r.stdout, r.stderr)
        original = Path(folder) / 'empty'
        original.write_text('')
        bad = Path(folder) / 'bad'
        bad.write_text('trailing space \n')
        r = subprocess.run(['git', 'diff', '--no-index', '--check', str(original), str(bad)],
                           capture_output=True, text=True)
        assert r.returncode not in [0, 1] and 'trailing whitespace' in r.stdout
    assert oldcites == newcites
    control = r'\begin{figure}\includegraphics{fig_ten_traffic}\end{figure}'
    assert protected_displays(control) != protected_displays(control.replace('fig_ten_traffic', 'fig_ten_strong'))
    fresh = [str(p.relative_to(ROOT)) for directory in ['research/r8_team_review', 'docs/team_review_20260910']
             for p in (ROOT / directory).glob('*') if p.is_file()]
    fresh.extend(['source/scripts/extension_20260831/build_paper_outputs.py',
                  'artifacts/r8_commodity_etp/panel/base/results/paper_outputs_manifest.json'])
    return dict(changed_snapshot_files=changes,
                unchanged_snapshot_files=len(before['files']) - len(changes),
                word_counts=counts, count_convention='Whitespace-delimited LaTeX tokens.',
                formal_statements_preserved=dict(protected), display_maths_preserved=True,
                labels_and_citation_keys_preserved=True, generated_tables_and_figures_preserved=True,
                original_proof_sources_preserved=True, typography_preserved=True,
                git_metadata_present=(ROOT / '.git').exists(), no_index_whitespace_negative_control=True,
                float_selection_negative_control=True,
                current_files={n: sha(ROOT / n) for n in sorted(set(before['files']) | set(fresh))})


def display_ownership():
    producer = ROOT / 'source/scripts/extension_20260831/build_paper_outputs.py'
    manifest = ROOT / 'artifacts/r8_commodity_etp/panel/base/results/paper_outputs_manifest.json'
    old = json.loads((OUT / 'before_display_manifest.json').read_text())
    new = json.loads(manifest.read_text())
    assert old['producer_sha256'] == sha(OUT / 'before_display_producer.py')
    assert new['producer_sha256'] == sha(producer)
    assert {k: v for k, v in old.items() if k != 'producer_sha256'} == {
        k: v for k, v in new.items() if k != 'producer_sha256'}
    names, = [ast.literal_eval(node.value) for node in ast.walk(ast.parse(producer.read_text()))
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
              t.id == 'own_table_names' for t in node.targets)]
    own = {Path(p).name for p in new['outputs'] if '/tab_' in p}
    assert names == own
    foreign = {'tab_external.tex', 'tab_native_forecasters.tex', 'tab_ten_strong.tex', 'tab_partial.tex'}
    assert not names & foreign
    for name, expected in new['outputs'].items():
        assert sha(ROOT / 'source' / name) == expected
    return dict(outputs_unchanged=len(new['outputs']), explicit_table_ownership=len(names),
                foreign_tables_excluded=sorted(foreign), only_producer_binding_changed=True)


def environment():
    recorded = json.loads((OUT / 'environment.json').read_text())
    assert platform.python_version() == recorded['python']
    for name, version in recorded['packages'].items():
        assert metadata.version(name) == version, ('Different validation dependency', name)
    return dict(python=recorded['python'], checked_package_versions=len(recorded['packages']),
                captured_environment_sha256=sha(OUT / 'environment.json'),
                tex_toolchain='Recorded from the actual build; not rerun by this receipt check.')


def documents():
    guard = json.loads((ROOT / 'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    assert guard['documents']['negative_controls'] == 4
    assert all(guard['documents'][k] == 0 for k in ['undefined_references', 'undefined_citations', 'overfull_boxes'])
    displays = guard['displays']['displays'] + sum(guard[k]['exact_display_replay'] for k in
        ['risk_displays', 'regime_displays', 'ten_external_displays', 'partial_displays', 'information_displays'])
    assert displays == 57
    package = json.loads((ROOT / 'artifacts/r8_ten_integration/source_package.json').read_text())
    assert package['source_members'] == 66
    assert sha(ROOT / package['source_zip']) == package['source_zip_sha256'] == sha(ROOT / 'release/R8_LaTeX_sources.zip')
    with zipfile.ZipFile(ROOT / package['source_zip']) as archive:
        for name in archive.namelist():
            if name != 'BUILD.txt':
                assert archive.read(name) == (ROOT / 'source' / name).read_bytes(), ('Stale packaged source', name)
    visual = json.loads((OUT / 'visual_validation.json').read_text())
    for name, info in package['documents'].items():
        pdf = ROOT / 'source' / (name + '.pdf')
        assert info['pdf_sha256'] == sha(pdf) == visual['documents'][name]['pdf_sha256']
        assert info['normalised_text_matches'] and info['clean_diagnostics']
        pages = PdfReader(pdf).pages
        assert len(pages) == info['pages']
        assert all(len(p.extract_text().strip()) > 10 for p in pages)
        log = (ROOT / 'source' / (name + '.log')).read_text(errors='backslashreplace')
        assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!', log, re.M)
    assert sha(ROOT / 'Manuscript_R8.pdf') == package['documents']['main_R8']['pdf_sha256']
    return dict(exact_displays=displays, guards=guard['documents'], source_package=package,
                packaged_sources_match_current_bytes=True,
                visual_inspection=visual, no_blank_pages=True)


if __name__ == '__main__':
    reviews = collect(complete=True)
    result = dict(status='passed', producer_sha256=sha(__file__), reviews=reviews,
                  scope=scope(), documents=documents(), ewma_current_replay=check_ewma(),
                  display_producer_repair=display_ownership(),
                  validation_environment=environment(),
                  new_inference_or_simulations=False)
    (OUT / 'final_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], reviews=reviews['completed'],
                         displays=result['documents']['exact_displays']), indent=2))
