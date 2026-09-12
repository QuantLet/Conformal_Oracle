"""Conserve prior R8 results while checking the authorised coverage addition."""
import collections
import hashlib
from importlib import metadata
import platform
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
from pypdf import PdfReader
from check import check as mathematical_check

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_contiguous_integration'
DOC=ROOT/'docs/contiguous_integration_20260911'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope():
    before=json.loads((OUT/'before.json').read_text())
    assert sha(OUT/'before_sources.zip')==before['snapshot_sha256']
    allowed={'source/main_R8.tex','source/supplement_R8.tex','source/calibrating_the_oracle.bib',
             'source/main_R8.pdf','source/supplement_R8.pdf','Manuscript_R8.pdf',
             'release/R8_LaTeX_sources.zip','source/analysis/provenance_r8/PRODUCERS.tsv','docs/IRFA_REVIEW_STATE.md','docs/IRFA_CLAIM_EVIDENCE_MAP.md',
             'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json'}
    allowed.update('source/sections_r8/'+n+'.tex' for n in ['introduction','methodology','theory','discussion'])
    removed_inputs={r'\input{sections_r8/contiguous_coverage}',r'\input{sections_r8/contiguous_proof}'}
    formal=r'\\begin\{(theorem|proposition|corollary|lemma|assumption)\}.*?\\end\{\1\}'
    math=r'\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}|\\\[.*?\\\]'
    floats=r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}'
    inputs=r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{[^}]+\}'
    extract=lambda pattern,s:[m.group() for m in re.finditer(pattern,s,re.S)]
    oldcites,newcites=set(),set();changes=[];protected=collections.Counter()
    with zipfile.ZipFile(OUT/'before_sources.zip') as archive,tempfile.TemporaryDirectory() as tmp:
        for name,digest in before['files'].items():
            old=archive.read(name);new=(ROOT/name).read_bytes()
            assert hashlib.sha256(old).hexdigest()==digest
            if name.endswith('.tex'):
                a,b=old.decode(),new.decode()
                assert extract(r'\\label\{[^}]+\}',a)==extract(r'\\label\{[^}]+\}',b),name
                for pattern in [formal,math,floats]:assert extract(pattern,a)==extract(pattern,b),(name,'altered existing formal/display block')
                assert extract(inputs,a)==[i for i in extract(inputs,b) if i not in removed_inputs],(name,'input/figure changed')
                protected.update(m.group(1) for m in re.finditer(formal,a,re.S))
                if name=='source/main_R8.tex':
                    assert a.split(r'\begin{abstract}')[0]==b.split(r'\begin{abstract}')[0]
                    assert a.split(r'\end{abstract}')[1]==b.split(r'\end{abstract}')[1]
                if name=='source/supplement_R8.tex':
                    assert b.replace(r'\input{sections_r8/contiguous_proof}'+'\n','')==a
                for s,target in [(a,oldcites),(b,newcites)]:
                    for keys in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',s):target.update(keys.split(','))
            if old==new:continue
            assert name in allowed,('Unapproved change',name)
            changes.append(name)
            if name.endswith(('.tex','.md','.bib')):
                p=Path(tmp)/'before';p.write_bytes(old)
                run=subprocess.run(['git','diff','--no-index','--check',str(p),str(ROOT/name)],capture_output=True,text=True)
                assert run.returncode in [0,1] and not run.stdout and not run.stderr,(name,run.stdout,run.stderr)
        registration='source/analysis/provenance_r8/PRODUCERS.tsv'
        old_register=archive.read(registration).decode()
        new_register=(ROOT/registration).read_text()
        assert new_register==old_register+'sections_r8/contiguous_coverage\tauthored\tsections_r8/contiguous_coverage.tex\nsections_r8/contiguous_proof\tauthored\tsections_r8/contiguous_proof.tex\n'
        oldbib=archive.read('source/calibrating_the_oracle.bib').decode()
        newbib=(ROOT/'source/calibrating_the_oracle.bib').read_text()
        prefix,entry=newbib.split('@article{besbes2023data,')
        assert prefix.strip()==oldbib.strip() and '10.1287/mnsc.2023.4725' in entry
        p=Path(tmp)/'empty';p.write_text('');bad=Path(tmp)/'bad';bad.write_text('x \n')
        run=subprocess.run(['git','diff','--no-index','--check',str(p),str(bad)],capture_output=True,text=True)
        assert run.returncode not in [0,1] and 'trailing whitespace' in run.stdout
    assert newcites-oldcites=={'besbes2023data'} and not oldcites-newcites
    reviewed=(DOC/'PROOF_RECHECK.md').read_text()
    for name in ['contiguous_coverage','contiguous_proof','theory']:
        assert sha(ROOT/'source/sections_r8'/f'{name}.tex') in reviewed,('Unreviewed mathematical source',name)
    cor=(ROOT/'source/sections_r8/contiguous_coverage.tex').read_text()
    proof=(ROOT/'source/sections_r8/contiguous_proof.tex').read_text()
    assert len(extract(formal,cor))==1 and '\\label{cor:contiguous_coverage}' in cor
    assert proof.count(r'\begin{proof}')==proof.count(r'\end{proof}')==1
    fresh={str(p.relative_to(ROOT)) for folder in [ROOT/'research/r8_contiguous_integration',DOC]
           for p in folder.glob('*') if p.is_file()}
    fresh.update(['source/sections_r8/contiguous_coverage.tex','source/sections_r8/contiguous_proof.tex',
                  'release/R8_20260911_contiguous_sources.zip','artifacts/r8_team_review/environment.json'])
    files=set(before['files'])|fresh
    return dict(changed_snapshot_files=changes,unchanged_snapshot_files=len(before['files'])-len(changes),
                original_formal_statements_preserved=dict(protected),original_proof_files_preserved=True,
                original_display_math_floats_and_assets_preserved=True,
                added_citation='besbes2023data',new_result='cor:contiguous_coverage',
                git_metadata_present=(ROOT/'.git').exists(),no_index_whitespace_negative_control=True,
                current_files={n:sha(ROOT/n) for n in sorted(files)})


def environment():
    path=ROOT/'artifacts/r8_team_review/environment.json'
    record=json.loads(path.read_text())
    assert platform.python_version()==record['python']
    for name,version in record['packages'].items():assert metadata.version(name)==version,name
    return dict(record_sha256=sha(path),python=record['python'],checked_versions=len(record['packages']))


def documents():
    guard=json.loads((ROOT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    assert guard['documents']['negative_controls']==4
    assert all(guard['documents'][k]==0 for k in ['undefined_references','undefined_citations','overfull_boxes'])
    displays=guard['displays']['displays']+sum(guard[k]['exact_display_replay'] for k in
        ['risk_displays','regime_displays','ten_external_displays','partial_displays','information_displays'])
    assert displays==57
    package=json.loads((OUT/'source_package.json').read_text());assert package['source_members']==68
    assert sha(ROOT/package['source_zip'])==package['source_zip_sha256']==sha(ROOT/'release/R8_LaTeX_sources.zip')
    with zipfile.ZipFile(ROOT/package['source_zip']) as archive:
        for name in archive.namelist():
            if name!='BUILD.txt':assert archive.read(name)==(ROOT/'source'/name).read_bytes(),name
    visual=json.loads((OUT/'visual_validation.json').read_text())
    for name,info in package['documents'].items():
        pdf=ROOT/'source'/(name+'.pdf');pages=PdfReader(pdf).pages
        assert sha(pdf)==info['pdf_sha256']==visual['documents'][name]['pdf_sha256']
        assert len(pages)==info['pages'] and info['normalised_text_matches'] and info['clean_diagnostics']
        assert all(len(p.extract_text().strip())>10 for p in pages)
        log=(ROOT/'source'/(name+'.log')).read_text(errors='backslashreplace')
        assert not re.search(r'(?:Reference|Citation).*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!',log,re.M)
    assert sha(ROOT/'Manuscript_R8.pdf')==package['documents']['main_R8']['pdf_sha256']
    return dict(exact_displays=displays,guards=guard['documents'],source_package=package,
                visual_inspection=visual,packaged_sources_match_current_bytes=True,no_blank_pages=True)


if __name__=='__main__':
    result=dict(status='passed',producer_sha256=sha(__file__),scope=scope(),documents=documents(),
                mathematical_checks=mathematical_check(),environment=environment(),
                new_financial_inference_or_simulations=False)
    (OUT/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],displays=result['documents']['exact_displays']),indent=2))
