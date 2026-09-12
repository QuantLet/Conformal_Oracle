"""Initialise, execute and checksum every predeclared simulation block."""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.metadata as md
import json
from pathlib import Path
import time
import zlib
import numpy as np
import pandas as pd
import engine as e


def initialise():
    e.OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((e.PROJECT/'artifacts/r8_decision/before.json').read_text())
    before={'canonical':{p:e.sha(e.PROJECT/p) for p in old['canonical']},
            'protocol_sha256':e.sha(Path(__file__).with_name('PROTOCOL.md')),
            'packages':{p:md.version(p) for p in ('numpy','scipy','pandas','pyarrow')},
            'independent_histories':1500,'configuration_replications':72000}
    target=e.OUT/'before.json'
    if target.exists():assert json.loads(target.read_text())==before
    else:target.write_text(json.dumps(before,indent=2)+'\n')
    target=e.OUT/'paths.npz'
    if not target.exists():
        paths={};seeds={};clipping={}
        eps=np.empty((e.REPS,1000))
        for rep in range(e.REPS):
            seed=e.m.seed_for('mechanism/ar',rep);seeds[f'ar/{rep}']=seed
            eps[rep]=np.random.default_rng(seed).standard_normal(1000)
        paths['ar_innovations']=eps
        for phi in (0.,.5,.8):
            z=np.array([e.ar_path(x,phi) for x in eps]);paths[f'ar_z_{phi:g}']=z
            for kind in ('normal','t5'):
                y,c=e.marginal(z,kind);paths[f'ar_{kind}_{phi:g}']=y;clipping[f'{kind}/{phi:g}']=c
        for kind in ('normal','t5'):
            ys=[];ss=[]
            for rep in range(e.REPS):
                seed=zlib.crc32(f'20260909|cal|{kind}|{rep}'.encode())&0xffffffff
                seeds[f'garch/{kind}/{rep}']=seed
                y,s=e.old.garch(kind,1000,seed);ys.append(y);ss.append(s)
            paths[f'garch_y_{kind}']=np.array(ys);paths[f'garch_sigma_{kind}']=np.array(ss)
            seed=zlib.crc32(f'20260909|test|{kind}'.encode())&0xffffffff
            seeds[f'test/{kind}']=seed
            _,paths[f'test_sigma_{kind}']=e.old.garch(kind,1024,seed)
        np.savez_compressed(target,**paths)
        (e.OUT/'paths.json').write_text(json.dumps({'seeds':seeds,'cdf_endpoint_clipping':clipping,
            'sha256':e.sha(target),'source':e.sha(__file__)},indent=2)+'\n')
    assert e.sha(target)==json.loads((e.OUT/'paths.json').read_text())['sha256']


def binding():
    files=[Path(__file__),Path(e.__file__),Path(e.m.__file__),Path(e.old.__file__),
           e.PROJECT/'research/r8_review/controlled_comparisons.py',
           Path(__file__).with_name('PROTOCOL.md'),e.OUT/'paths.npz',e.OUT/'before.json']
    return {str(p.relative_to(e.PROJECT)):e.sha(p) for p in files}


def work(module,kind,phi,n,start,stop,replay=False):
    clock=time.monotonic();key=f'{module}_{kind}_{phi:g}_{n}_{start:03d}_{stop:03d}'
    folder=e.OUT/('replay' if replay else 'blocks')/key;folder.mkdir(parents=True,exist_ok=True)
    bound=binding();done=folder/'complete.json'
    if done.exists():
        stored=json.loads(done.read_text());assert stored['binding']==bound
        assert all(e.sha(folder/p)==h for p,h in stored['outputs'].items())
        return key,'verified'
    paths=np.load(e.OUT/'paths.npz');rows=[];fits=[];moments={}
    for rep in range(start,stop):
        if module=='ar':
            y=paths[f'ar_{kind}_{phi:g}'][rep,-n:];sigma=e.V0
            predictions,params=e.ar_calculation(y,kind)
        else:
            y=paths[f'garch_y_{kind}'][rep,-n:];s=paths[f'garch_sigma_{kind}'][rep,-n:]
            sigma=paths[f'test_sigma_{kind}']
            predictions,params=e.garch_calculation(y,s,sigma,kind)
        fits.append({'replication':rep,'parameters':params})
        for (alpha,truth),pred in predictions.items():
            oracle=e.old.conditional_quantile(kind,sigma,alpha)
            for method,p in pred.items():
                row={'module':module,'innovation':kind,'phi':phi,'n_cal':n,'alpha':alpha,
                     'truth':truth,'replication':rep,'method':method,**e.metric(kind,p,sigma,alpha,oracle)}
                if module=='ar':
                    row['oracle_tail_count']=int(np.sum(y<oracle))
                else:
                    qc=e.old.conditional_quantile(kind,s,alpha)
                    row['oracle_tail_count']=int(np.sum(y<qc))
                rows.append(row)
                mk=f'{alpha:g}/{truth}/{method}'
                if mk not in moments:moments[mk]=np.zeros((3,)+np.shape(p))
                moments[mk][0]+=p;moments[mk][1]+=p*p;moments[mk][2]+=(p-oracle)**2
    pd.DataFrame(rows).to_csv(folder/'replications.csv',index=False)
    (folder/'fits.jsonl').write_text(''.join(json.dumps(f,allow_nan=False)+'\n' for f in fits))
    np.savez_compressed(folder/'moments.npz',**moments)
    files=('replications.csv','fits.jsonl','moments.npz')
    done.write_text(json.dumps({'binding':bound,'start':start,'stop':stop,
        'outputs':{p:e.sha(folder/p) for p in files},'elapsed_seconds':time.monotonic()-clock},indent=2)+'\n')
    return key,round(time.monotonic()-clock,2)


def tasks(module='all'):
    if module in ('ar','all'):
        for kind in ('normal','t5'):
            for phi in (0.,.5,.8):
                for n in e.SIZES:
                    for start in range(0,e.REPS,25):yield ('ar',kind,phi,n,start,start+25)
    if module in ('garch','all'):
        for kind in ('normal','t5'):
            for n in e.SIZES:
                for start in range(0,e.REPS,25):yield ('garch',kind,0.,n,start,start+25)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--init',action='store_true')
    p.add_argument('--module',choices=['ar','garch','all'],default='all')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--single',nargs=6)
    p.add_argument('--replay',action='store_true');args=p.parse_args()
    initialise()
    if args.init:print('Inputs and protocol recorded',flush=True)
    elif args.single:
        a=args.single;print(work(a[0],a[1],float(a[2]),int(a[3]),int(a[4]),int(a[5]),args.replay),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending=[pool.submit(work,*t) for t in tasks(args.module)]
            for future in as_completed(pending):print(*future.result(),flush=True)
