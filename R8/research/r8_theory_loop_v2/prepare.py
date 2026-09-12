"""Lock and extract the twelve already-generated windows; no random draws."""
import copy
import csv
import subprocess
import numpy as np
from common import ROOT,BASE,OUT,LOCK,binding,sha,save_json,validate_lock

def main():
    assert not LOCK.exists(), 'Refuse to replace the protocol lock'
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=BASE/'PROTOCOL.md'
    commit=subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'rev-parse','HEAD'],text=True).strip()
    committed=subprocess.check_output(['git','-C',str(BASE/'protocol_repository'),'show','HEAD:PROTOCOL.md'])
    assert committed==protocol.read_bytes()
    paths=[ROOT/'results/theory_loop'/p for p in ('provenance_manifest.json','final_receipt.json','preserved_R8.json')]
    paths += [ROOT/'results/theory_loop/synthetic'/p for p in (
        'calibration_normal.npz','calibration_t5.npz','truth_normal.npz','truth_t5.npz',
        'estimators.csv','truth.csv','validation_summary.csv','admission.json','sj_normal.csv','sj_t5.csv')]
    lock=dict(repository_root=str(ROOT),protocol_path=str(protocol),protocol_sha256=sha(protocol),
              protocol_commit=commit,input_files=[binding(p) for p in paths])
    save_json(LOCK,lock)
    # Failing fixtures precede checks of real inputs.
    bad=copy.deepcopy(lock);bad['protocol_sha256']='0'*64
    fixture=OUT/'fixtures/bad_protocol_lock.json';save_json(fixture,bad)
    rejected=[]
    try: validate_lock(fixture)
    except AssertionError: rejected.append('protocol_digest')
    else: raise AssertionError('Bad protocol accepted')
    bad=copy.deepcopy(lock);bad['input_files'][0]['sha256']='0'*64
    fixture=OUT/'fixtures/stale_input_lock.json';save_json(fixture,bad)
    try: validate_lock(fixture)
    except AssertionError: rejected.append('input_binding')
    else: raise AssertionError('Stale input accepted')
    validate_lock()
    rows=[];files=[]
    for law in ('normal','t5'):
        with np.load(ROOT/f'results/theory_loop/synthetic/calibration_{law}.npz') as z:
            x=z['scores']
            assert x.shape==(500,2000) and np.isfinite(x).all()
            for n in (700,1000):
                k=(99*(n+1)+99)//100
                for rep in (0,17,499):
                    scores=x[rep,:n].astype('<f8')
                    path=OUT/f'inputs/{law}_{n}_{rep}.bin';path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(scores.tobytes());files.append(binding(path))
                    rows.append(dict(law=law,n=n,rep=rep,path=str(path),C=float(np.sort(scores)[k-1]),
                                     bias=.25*np.sqrt(1e-5/(1-.10-.85))))
    metadata=OUT/'inputs/metadata.csv'
    with metadata.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lock.update(metadata_path=str(metadata),metadata_sha256=sha(metadata),window_files=files)
    save_json(LOCK,lock);validate_lock()
    save_json(OUT/'prepare_checks.json',dict(negative_controls_rejected=rejected,windows=len(rows),status='PASS'))
    print('Protocol committed:',commit,'; twelve windows bound; original inputs preserved.')

if __name__=='__main__': main()
