"""Final conservation, replay and provenance for v3 only."""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy
import pandas
import scipy
from engine import ROOT,BASE,OUT,LOCK,validate_lock,record,sha,dump

MANIFEST=OUT/'manifest.json'

def preserved():
    env=os.environ.copy();env['MPLCONFIGDIR']='/private/tmp/irfa-theory-v2-mpl'
    logs=[]
    for stage in ('r8_theory_loop','r8_theory_loop_v2'):
        r=subprocess.run([sys.executable,str(ROOT/'research'/stage/'finalize.py'),'--verify'],
            check=True,text=True,capture_output=True,env=env)
        logs.append(dict(stage=stage,stdout=r.stdout.strip()))
    return logs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');a=parser.parse_args()
    lock=validate_lock();old=preserved()
    committed=subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'show',
        lock['protocol_commit']+':PROTOCOL.md'])
    assert committed==(BASE/'PROTOCOL.md').read_bytes()
    if a.verify:
        manifest=json.loads(MANIFEST.read_text())
        for item in manifest['files']:
            now=record(ROOT/item['relative_path'])
            assert all(now[k]==item[k] for k in ('sha256','size','mtime_ns')),item['relative_path']
        print('Verified',len(manifest['files']),'v3 bindings; original and v2 artifacts preserved; panel NOT_RUN.');return
    assert not MANIFEST.exists(),'Refuse to overwrite completed provenance'
    independent=json.loads((OUT/'independent_verification.json').read_text())
    assert independent['status']=='PASS'
    assert json.loads((OUT/'diagnostic/checks.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'exact_checks.json').read_text())['status']=='PASS'
    replays=[]
    for original in sorted((OUT/'diagnostic').glob('*.csv')):
        replay=OUT/'replay'/original.name
        source=original.read_bytes();other=replay.read_bytes()
        assert source!=other+b'\nDELIBERATE_CORRUPTION\n'
        assert source==other,original.name
        replays.append(dict(file=original.name,sha256=sha(original),negative_control_rejected=True,exact_match=True))
    reports=ROOT/'docs/theory_loop_v3_20260911'
    for name in ('MATHEMATICAL_REVIEW.md','STATISTICAL_REVIEW.md','INTERPRETATION.md'):
        assert (reports/name).is_file()
    for name in ('MATHEMATICAL_REVIEW.md','STATISTICAL_REVIEW.md'):
        assert (reports/name).stat().st_size>1000
    dump(OUT/'completion.json',dict(status='MATHEMATICAL_AND_STORED_DATA_DIAGNOSTIC_COMPLETED',
        protocol_commit=lock['protocol_commit'],independent_verification='PASS',replay=replays,
        simultaneous_family_size=18,financial_panel='NOT_RUN',original_admission='FAIL_UNCHANGED',
        new_random_histories=0,forecaster_refits=0,manuscript_changed=False,
        unavailable_contiguous_cells=['normal/n2000','t5/n2000'],old_artifact_conservation=old,
        python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,
        scope='Reused previously inspected histories. Oracle expectation checks, not financial deployability or clean-environment reinstall.'))
    files=[]
    for folder in (BASE,OUT,reports):
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path==MANIFEST or '.git' in path.parts or '__pycache__' in path.parts:continue
            item=record(path);item['relative_path']=str(path.relative_to(ROOT));files.append(item)
    dump(MANIFEST,dict(protocol_commit=lock['protocol_commit'],files=files,
        note='Excludes manifest itself, Git internals and interpreter caches. Original input binding is in lock.json.'))
    print('Completed:',len(files),'artifact bindings;',len(replays),'exact CSV replays; financial admission remains failed.')

if __name__=='__main__':main()
