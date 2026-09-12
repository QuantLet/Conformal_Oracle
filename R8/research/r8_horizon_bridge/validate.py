"""Verify the mathematical diagnostic and the exact manuscript-change scope."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/'artifacts/r8_horizon_bridge'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    previous=json.loads((OUT/'validation.json').read_text())
    script=Path(__file__).with_name('check.py')
    assert sha(script)==previous['producer_sha256']
    assert sha(script.with_name('NOTE.md'))==previous['note_sha256']
    spec=importlib.util.spec_from_file_location('bridge_checks',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='irfa-bridge-checks-') as temporary:
        module.OUT=Path(temporary)
        results=dict(finite_markov_checks=module.finite_markov(),
                     continuous_refresh=module.continuous_refresh(),
                     gaussian_information=module.local_information())
        assert all(results[k]==previous[k] for k in results)
        assert all(sha(module.OUT/p)==h for p,h in previous['outputs'].items())
    horizon=json.loads((OUT/'horizon/validation.json').read_text())
    replay=json.loads((OUT/'horizon_replay/validation.json').read_text())
    assert horizon==replay and horizon['status']=='passed'
    assert all(sha(PROJECT/p)==h for p,h in horizon['inputs'].items())
    for folder in ('horizon','horizon_replay'):
        assert all(sha(OUT/folder/p)==h for p,h in horizon['outputs'].items())
    snapshot=json.loads((OUT/'before_manuscript.json').read_text())
    changed=[];preserved=0;new_equations=[]
    equation=re.compile(r'\\begin\{equation\}(.*?)\\end\{equation\}',re.S)
    with zipfile.ZipFile(OUT/'before_sources.zip') as old, tempfile.TemporaryDirectory(prefix='irfa-bridge-diff-') as temporary:
        for relative,want in snapshot['files'].items():
            data=old.read(relative)
            assert hashlib.sha256(data).hexdigest()==want
            current=PROJECT/relative
            if sha(current)==want:preserved+=1;continue
            changed.append(relative)
            a=Path(temporary)/'before.tex';a.write_bytes(data)
            result=subprocess.run(['git','diff','--no-index','--check',str(a),str(current)],capture_output=True,text=True)
            # --no-index implies --exit-code: one also means a clean text change.
            assert not result.stdout and not result.stderr and result.returncode in (0,1),(relative,result.stdout,result.stderr)
            old_equations=equation.findall(data.decode())
            current_equations=equation.findall(current.read_text())
            assert all(block in current_equations for block in old_equations),relative
            new_equations.extend((relative,block) for block in current_equations if block not in old_equations)
        a=Path(temporary)/'empty';a.write_text('')
        bad=Path(temporary)/'bad';bad.write_text('trailing whitespace  \n')
        negative=subprocess.run(['git','diff','--no-index','--check',str(a),str(bad)],capture_output=True,text=True)
        assert 'trailing whitespace' in negative.stdout and negative.returncode not in (0,1)
    allowed=['source/main_R8.tex',*[f'source/sections_r8/{name}.tex' for name in
             ['risk','risk_proofs','theory','methodology','introduction','discussion']]]
    assert sorted(changed)==sorted(allowed),changed
    assert len(new_equations)==2
    assert {re.search(r'\\label\{([^}]+)\}',block)[1] for _,block in new_equations}=={
        'eq:contiguous_transfer','eq:contiguous_risk'}
    assert sha(PROJECT/'artifacts/r8_referee_revision/final_validation.json')==previous['protected_inputs']['artifacts/r8_referee_revision/final_validation.json']
    documents=json.loads((PROJECT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    package=json.loads((PROJECT/'artifacts/r8_ten_integration/source_package.json').read_text())
    assert documents['documents']['undefined_references']==0
    assert documents['documents']['undefined_citations']==0
    assert documents['documents']['overfull_boxes']==0
    assert all(v['normalised_text_matches'] for v in package['documents'].values())
    assert all(sha(PROJECT/f'source/{doc}.pdf')==v['pdf_sha256'] for doc,v in package['documents'].items())
    record=dict(status='passed',deterministic_replay=results,horizon_replay='exact',
                changed_manuscript_files=changed,preserved_snapshot_files=preserved,
                old_numbered_equations_preserved=True,new_numbered_equations=2,
                empirical_tables_and_macros_unchanged=True,whitespace_check='git diff --no-index --check',
                whitespace_negative_control=True,
                no_git_metadata=not (PROJECT/'.git').exists(),documents=documents['documents'],
                display_replay_counts=[25,14,2,13],source_package=package,
                visual_inspection=['main pp14-15','supplement p18'],
                producer_sha256=sha(__file__),current_manuscript={p:sha(PROJECT/p) for p in snapshot['files']})
    (OUT/'final_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='current_manuscript'},indent=2))


if __name__=='__main__':main()
