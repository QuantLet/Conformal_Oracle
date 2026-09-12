"""Full same-seed reconstruction from an extracted research archive."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from checks import record_check,sha

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/theory_loop'


def main():
    archive=OUT/'replay_package.zip';checks=[]
    package=json.loads((OUT/'package_manifest.json').read_text())
    record_check(checks,'archive_digest','0'*64,sha(archive),lambda x:x==package['archive_sha256'])
    work=Path(tempfile.mkdtemp(prefix='irfa-theory-loop-replay-',dir='/private/tmp'))
    with zipfile.ZipFile(archive) as z:
        # Only our own, hash-bound archive is extracted; reject unsafe members.
        safe=lambda names:all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        record_check(checks,'archive_paths',['../escape'],z.namelist(),safe)
        z.extractall(work)
    original=package['members'];bad={**original}
    key=next(iter(bad));bad[key]={**bad[key],'sha256':'0'*64}
    valid=lambda m:all((work/n).is_file() and sha(work/n)==v['sha256'] for n,v in m.items())
    record_check(checks,'all_archive_member_hashes',bad,original,valid)
    expected=work/'expected_synthetic';(work/'results/theory_loop/synthetic').rename(expected)
    (work/'results/theory_loop/synthetic').mkdir()
    environment=dict(os.environ,MPLCONFIGDIR=str(work/'mpl_cache'))
    commands=['checks.py','run_synthetic.py','verify_synthetic.py','report.py']
    for script in commands:
        print('Archive-only fresh process:',script,flush=True)
        log=work/(script+'.log')
        with log.open('w') as f:
            subprocess.run([sys.executable,str(work/'research/r8_theory_loop'/script)],cwd=work,
                           env=environment,stdout=f,stderr=subprocess.STDOUT,check=True)
    current=work/'results/theory_loop/synthetic'
    exact=['estimators.csv','truth.csv','validation_summary.csv','expansion.csv','loss_histories.csv',
           'finite_reference_diagnostic.csv','finite_loss_histories.csv','sj_normal.csv','sj_t5.csv',
           'floor_diagnostics.csv','first_passing_size.csv','spacing_status.csv',
           'tab_validation.tex','tab_expansion.tex','estimator_validation.pdf','loss_expansion.pdf']
    for name in exact:
        actual=(current/name).read_bytes();reference=(expected/name).read_bytes()
        record_check(checks,'exact_archive_replay:'+name,actual+b'x',actual,lambda x:x==reference)
    status=json.loads((current/'admission.json').read_text())
    record_check(checks,'repeated_failed_admission',{**status,'status':'PASS'},status,
                 lambda r:r['status']=='FAIL' and r['financial_panel_reads']==0)
    report=dict(status='PASS',runtime_scope='Fresh extraction/processes; same recorded Python/R environment',
        archive_sha256=sha(archive),members=len(original),replayed_scripts=commands,
        exact_replayed_outputs=exact,financial_panel_reads=0,synthetic_admission='FAIL',
        extraction_directory=str(work),checks=checks,negative_controls=len(checks))
    (OUT/'archive_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','exact_replayed_outputs')}))


if __name__=='__main__':
    main()
