"""Fail-closed provenance and numeric display checks for this research stage."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from checks import record_check,sha

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/theory_loop'
EXCLUDE={'provenance_manifest.json','final_receipt.json'}


def verify_manifest(manifest):
    for name,entry in manifest['files'].items():
        p=ROOT/name
        if not p.is_file() or sha(p)!=entry['sha256'] or p.stat().st_mtime_ns!=entry['mtime_ns']:
            return False
    return True


def finalize():
    records=[]
    protected=json.loads((OUT/'preserved_R8.json').read_text());bad=dict(protected)
    bad[next(iter(bad))]='0'*64
    record_check(records,'protected_final',bad,protected,lambda d:all(sha(ROOT/p)==h for p,h in d.items()))
    numbers=pd.read_csv(OUT/'manuscript_numbers.csv')
    def number_check(frame):
        for row in frame.itertuples():
            if sha(ROOT/row.file)!=row.sha256:
                return False
            value=pd.read_csv(ROOT/row.file).iloc[row.data_row-1][row.field]
            if pd.isna(value) and pd.isna(row.value):
                continue
            if not np.isclose(float(value),float(row.value),rtol=1e-12,atol=1e-14):
                return False
        return True
    bad=numbers.copy();bad.loc[0,'value']+=1
    record_check(records,'number_index',bad,numbers,number_check)
    # The transparent saved source image and explicit opaque mutant are checked.
    for name in ('estimator_validation','loss_expansion'):
        a=np.asarray(Image.open(OUT/f'synthetic/{name}.png').convert('RGBA'))
        bad=a.copy();bad[:,:,3]=255
        record_check(records,'transparent_'+name,bad,a,lambda x:x[0,0,3]==0 and np.any(x[:,:,3]>0))
    allpaths=[ROOT/'analysis_plan_theory_loop.md',ROOT/'DECISIONS.md']
    for folder in (ROOT/'research/r8_theory_loop',OUT):
        allpaths.extend(p for p in folder.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and '.git' not in p.parts and p.name not in EXCLUDE)
    allpaths=sorted(set(allpaths))
    def whitespace(p):
        return all(line==line.rstrip() for line in p.read_text().splitlines())
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);bad=td/'trailing.md';bad.write_text('bad \n')
        for p in allpaths:
            if p.suffix in ('.py','.md','.R','.tex','.json','.lock'):
                record_check(records,'whitespace:'+str(p.relative_to(ROOT)),bad,p,whitespace)
    files={}
    for p in allpaths:
        st=p.stat()
        files[str(p.relative_to(ROOT))]=dict(sha256=sha(p),size=st.st_size,mtime_ns=st.st_mtime_ns,
             mtime_utc=datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat())
    manifest={'status':'HASHED_EXISTING_FILES_ONLY','files':files,
              'excluded':'This manifest and its detached receipt; ephemeral caches and Git internals',
              'historical_attempts':'Retained and labelled; only current synthetic outputs are authoritative'}
    bad=json.loads(json.dumps(manifest));bad['files'][next(iter(files))]['sha256']='0'*64
    record_check(records,'manifest_hash_verification',bad,manifest,verify_manifest)
    bad=json.loads(json.dumps(manifest));bad['files']['does_not_exist.csv']=next(iter(files.values()))
    record_check(records,'manifest_missing_artifact',bad,manifest,verify_manifest)
    bad=json.loads(json.dumps(manifest));bad['files'][next(iter(files))]['mtime_ns']+=1
    record_check(records,'manifest_mtime_verification',bad,manifest,verify_manifest)
    text=(OUT/'RESULTS.md').read_text()
    record_check(records,'report_stopping_status',text.replace('Financial deliverables 1–3: NOT RUN.','Financial deliverables complete.'),
                 text,lambda s:'Financial deliverables 1–3: NOT RUN.' in s)
    archive=json.loads((OUT/'archive_replay.json').read_text())
    record_check(records,'archive_replay_verdict',{**archive,'status':'FAIL'},archive,
                 lambda r:r['status']=='PASS' and r['financial_panel_reads']==0)
    path=OUT/'provenance_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    receipt={'status':'RESEARCH_STAGE_RECORDED','synthetic_admission':'FAIL',
        'financial_deliverables':'NOT_RUN','financial_panel_reads':0,
        'manifest_sha256':sha(path),'artifact_count':len(files),'protected_R8_files':len(protected),
        'checks':records,'negative_controls':len(records),
        'protocol_commit':json.loads((OUT/'lock.json').read_text())['protocol_commit'],
        'archive_replay':archive,'pdf_visual_review':json.loads((OUT/'visual_review.json').read_text()),
        'manuscript_edited':False,'forecasters_refitted':False,
        'full_repository_reproduction_claimed':False}
    (OUT/'final_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('checks','archive_replay','pdf_visual_review')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:
        m=OUT/'provenance_manifest.json';receipt=json.loads((OUT/'final_receipt.json').read_text())
        if sha(m)!=receipt['manifest_sha256'] or not verify_manifest(json.loads(m.read_text())):
            raise SystemExit('FAIL: changed, absent or stale manifest member')
        print('All recorded artifact hashes and mtimes match; synthetic admission remains FAIL.')
    else:
        finalize()
