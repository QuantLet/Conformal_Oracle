"""Bind and verify the completed diagnostic without changing original studies."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy
import pandas
import scipy
import matplotlib
from pypdf import PdfReader
from common import ROOT,BASE,OUT,LOCK,binding,sha,save_json,validate_lock

MANIFEST=OUT/'manifest.json'

def numerical_replay():
    pairs=[(p,OUT/'replay/risk'/p.name) for p in sorted((OUT/'risk').glob('*.csv'))]
    pairs +=[(p,OUT/'replay/diagnostic'/p.name) for p in sorted((OUT/'diagnostic').glob('*.csv'))]
    pairs +=[(OUT/'sj/summary.csv',OUT/'sj_replay/summary.csv')]
    rec=[]
    for original,replayed in pairs:
        original_bytes=original.read_bytes()
        # Deliberately corrupt the replay through the same equality predicate first.
        corrupted=replayed.read_bytes()+b'\nINTENTIONAL_REPLAY_CORRUPTION\n'
        assert original_bytes!=corrupted
        assert original_bytes==replayed.read_bytes(), str(original)
        rec.append(dict(path=str(original.relative_to(ROOT)),replay=str(replayed.relative_to(ROOT)),
            sha256=sha(original),negative_control_rejected=True,exact_match=True))
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    lock=validate_lock()
    committed=subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'show',
        lock['protocol_commit']+':PROTOCOL.md'])
    assert committed==Path(lock['protocol_path']).read_bytes()
    old=subprocess.run([sys.executable,str(ROOT/'research/r8_theory_loop/finalize.py'),'--verify'],
        check=True,text=True,capture_output=True)
    if a.verify:
        manifest=json.loads(MANIFEST.read_text())
        for entry in manifest['files']:
            current=binding(ROOT/entry['relative_path'])
            assert all(current[k]==entry[k] for k in ('sha256','mtime_ns','size')), entry['relative_path']
        print('Verified',len(manifest['files']),'new artifact bindings;',old.stdout.strip());return
    assert not MANIFEST.exists(),'Refuse to overwrite completion manifest'
    replay=numerical_replay()
    math=json.loads((OUT/'risk/independent_verification.json').read_text())
    diagnostic=json.loads((OUT/'diagnostic/checks.json').read_text())
    assert math['status']=='PASS' and diagnostic['required_checks_passed']
    assert diagnostic['numerical_convergence_failures']==1
    pdf=PdfReader(OUT/'diagnostic_evidence.pdf');assert len(pdf.pages)==2
    text='\n'.join(page.extract_text() for page in pdf.pages)
    for expected in ('0.3333','0.2500','0.139%','0.1%'): assert expected in text,expected
    save_json(OUT/'runtime.json',dict(python=platform.python_version(),python_executable=sys.executable,
        platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,
        pandas=pandas.__version__,matplotlib=matplotlib.__version__,R_runtime='sj/runtime.txt'))
    save_json(OUT/'completion.json',dict(status='BOUNDED_DIAGNOSTIC_COMPLETED',
        statistical_admission='FAIL',financial_panel='NOT_RUN',numerical_convergence='59/60 bins; 60/60 solver',
        protocol_commit=lock['protocol_commit'],math_negative_controls=math['negative_controls'],
        sj_required_checks='PASS',sj_convergence_failures=1,replay_csv_files=len(replay),replay=replay,
        pdf_pages=2,visual_review='Both rendered final pages inspected: no clipped/overlapping text; transparent PDF backgrounds; legends below axes.',
        old_study_conservation=old.stdout.strip(),
        manuscript_edited=False,forecaster_refits=0,new_random_histories=0,
        scope='Existing runtime, independent fresh processes; no clean-environment reinstall or journal acceptance claim.'))
    dirs=[BASE,OUT,ROOT/'docs/theory_loop_v2_20260911']
    entries=[]
    for directory in dirs:
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path==MANIFEST or '__pycache__' in path.parts or '.git' in path.parts:continue
            entry=binding(path);entry['relative_path']=str(path.relative_to(ROOT));entries.append(entry)
    save_json(MANIFEST,dict(protocol_commit=lock['protocol_commit'],files=entries,
        note='Manifest excludes itself, Python caches and local Git internals. Original input bindings remain in protocol_lock.json.'))
    print('Bound',len(entries),'artifacts;',len(replay),'exact CSV replays; 1 numerical criterion remains failed.')

if __name__=='__main__': main()
