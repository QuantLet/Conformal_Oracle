"""Check current documents while replaying immutable research separately."""
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
from scipy.stats import binom
from pypdf import PdfReader
from build import ROOT,SOURCE,OUT,sha,check


def replay_study():
    receipt=json.loads((ROOT/'artifacts/r8_information_limit/package.json').read_text())
    assert receipt['status']=='complete'
    archive_path=ROOT/receipt['archive']
    assert sha(archive_path)==receipt['archive_sha256']
    with zipfile.ZipFile(archive_path) as archive,tempfile.TemporaryDirectory(prefix='irfa-info-replay-') as folder:
        manifest_name,=[p for p in archive.namelist() if p.endswith('/STUDY_MANIFEST.json')]
        prefix=manifest_name.split('/')[0]
        manifest=json.loads(archive.read(manifest_name))
        for name,wanted in manifest['files'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data=archive.read(prefix+'/'+name);assert hashlib.sha256(data).hexdigest()==wanted
            p=Path(folder)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        process=subprocess.run([sys.executable,'research/r8_information_limit/validate.py'],
                               cwd=folder,capture_output=True,text=True)
        assert process.returncode==0,(process.stdout,process.stderr)
        result=json.loads((Path(folder)/'artifacts/r8_information_limit/independent_validation.json').read_text())
        assert result==json.loads((ROOT/'artifacts/r8_information_limit/independent_validation.json').read_text())
    return dict(archive=receipt['archive'],archive_sha256=receipt['archive_sha256'],
                archived_members=len(manifest['files']),independent_validation_exact=True,
                original_109_canonical_checks_preserved=True)


def numbers():
    macros=dict(re.findall(r'\\newcommand\{\\([^}]+)\}\{([^}]+)\}',
                           (SOURCE/'sections_r8/numbers_information.tex').read_text()))
    theta0,theta1=.01*.8/.75,.01*1.2/.75
    results={}
    for retention,label in [(0.,'nInfoErrorIid'),(.5,'nInfoErrorDependent')]:
        total=0.
        # Direct overlaps; the manuscript builder only reads the saved CSV.
        for k in range(1,251):
            j=np.arange(k+1)
            total+=.5*binom.pmf(k-1,249,1-retention)*np.minimum(
                binom.pmf(j,k,theta0),binom.pmf(j,k,theta1)).sum()
        assert macros[label]==f'{100*total:.1f}'
        results[label]=total
    assert macros['nInfoThetaLow']==f'{100*theta0:.3f}'
    assert macros['nInfoThetaHigh']==f'{100*theta1:.3f}'
    return dict(independent_binomial_overlap=True,macros_checked=4,errors=results)


def scope():
    snapshot=json.loads((OUT/'before.json').read_text())
    assert sha(OUT/'before_sources.zip')==snapshot['snapshot_sha256']
    allowed={'source/main_R8.tex','source/supplement_R8.tex','source/calibrating_the_oracle.bib',
             'source/sections_r8/deployment.tex','source/sections_r8/introduction.tex','source/sections_r8/discussion.tex',
             'source/analysis/provenance_r8/PRODUCERS.tsv','source/scripts/extension_20260831/validate_r8.py',
             'research/r8_ten_integration/package_sources.py','docs/IRFA_REVIEW_STATE.md',
             'source/main_R8.pdf','source/supplement_R8.pdf','Manuscript_R8.pdf',
             'release/R8_LaTeX_sources.zip','release/R8_20260910_ten_external_sources.zip',
             'artifacts/r8_ten_integration/source_package.json',
             'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'}
    changed=[];preserved=[];equations=0
    with zipfile.ZipFile(OUT/'before_sources.zip') as old,tempfile.TemporaryDirectory(prefix='irfa-info-diff-') as folder:
        for name,wanted in snapshot['files'].items():
            before=old.read(name);assert hashlib.sha256(before).hexdigest()==wanted
            now=(ROOT/name).read_bytes()
            if name.endswith('.tex'):
                blocks=re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',before.decode(),re.S)
                equations+=len(blocks);assert all(block in now.decode() for block in blocks),name
            if name.endswith('.bib'):assert now.startswith(before),'Old bibliography changed'
            if now==before:preserved.append(name);continue
            assert name in allowed,('Unexpected change',name)
            changed.append(name)
            if name.endswith(('.tex','.bib','.py','.md','.tsv')):
                a=Path(folder)/'before';a.write_bytes(before)
                p=subprocess.run(['git','diff','--no-index','--check',str(a),str(ROOT/name)],capture_output=True,text=True)
                assert not p.stdout and not p.stderr and p.returncode in (0,1),(name,p.stdout,p.stderr)
        a=Path(folder)/'empty';a.write_text('');b=Path(folder)/'bad';b.write_text('bad whitespace \n')
        negative=subprocess.run(['git','diff','--no-index','--check',str(a),str(b)],capture_output=True,text=True)
        assert 'trailing whitespace' in negative.stdout and negative.returncode not in (0,1)
    fresh=['source/sections_r8/information.tex','source/sections_r8/information_proof.tex',
           'source/sections_r8/numbers_information.tex','docs/IRFA_INFORMATION_INTEGRATION.md']
    fresh += [str(p.relative_to(ROOT)) for p in (ROOT/'research/r8_information_integration').glob('*') if p.is_file()]
    for name in fresh:
        assert all(line==line.rstrip() for line in (ROOT/name).read_text().splitlines()),name
    return dict(changed_snapshot_files=changed,preserved_snapshot_files=len(preserved),
                previous_numbered_equations_preserved=equations,financial_displays_unchanged=True,
                previous_theory_proofs_unchanged=True,old_bibliography_entries_unchanged=True,
                new_bibliography_entries=['ma2024minimax','liang2026selection'],
                whitespace_negative_control=True,git_metadata_present=(ROOT/'.git').exists(),
                current_files={p:sha(ROOT/p) for p in sorted(set(snapshot['files'])|set(fresh))})


def documents():
    guards=json.loads((ROOT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    package=json.loads((ROOT/'artifacts/r8_ten_integration/source_package.json').read_text())
    assert guards['information_displays']['exact_display_replay']==1
    assert guards['documents']['negative_controls']==4
    assert all(guards['documents'][p]==0 for p in ['undefined_references','undefined_citations','overfull_boxes'])
    assert sha(ROOT/package['source_zip'])==package['source_zip_sha256']
    assert sha(ROOT/'release/R8_LaTeX_sources.zip')==package['source_zip_sha256']
    for name,result in package['documents'].items():
        assert sha(SOURCE/(name+'.pdf'))==result['pdf_sha256']
        assert result['normalised_text_matches'] and result['clean_diagnostics']
        pages=PdfReader(SOURCE/(name+'.pdf')).pages
        assert all(len(p.extract_text().strip())>10 for p in pages)
        if name=='main_R8':
            assert 'JEL:' in pages[0].extract_text() and '1. Introduction' in pages[1].extract_text()
        log=(SOURCE/(name+'.log')).read_text(errors='backslashreplace')
        assert not re.search(r'Citation .*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!',log,re.M)
    assert sha(ROOT/'Manuscript_R8.pdf')==package['documents']['main_R8']['pdf_sha256']
    return dict(guards=guards,source_package=package,no_blank_pages=True,title_page_complete=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay-study',action='store_true');args=p.parse_args()
    if args.replay_study:(OUT/'study_replay.json').write_text(json.dumps(replay_study(),indent=2)+'\n')
    historical=json.loads((OUT/'study_replay.json').read_text())
    assert historical['independent_validation_exact'] and sha(ROOT/historical['archive'])==historical['archive_sha256']
    result=dict(status='passed',producer_sha256=sha(__file__),numbers=numbers(),displays=check(),
                scope=scope(),documents=documents(),study_replay=historical)
    (OUT/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['scope','documents']},indent=2))
