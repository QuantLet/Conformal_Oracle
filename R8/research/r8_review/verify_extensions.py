"""Fresh refitting/recalculation of every completed scientific extension."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import controlled_comparisons as controlled
import full_candidates as full
import complexity_simulation as simulation

SOURCE=controlled.PROJECT/'artifacts/review_20260909'


def compare_file(left,right):
    if left.suffix=='.parquet':
        pd.testing.assert_frame_equal(pd.read_parquet(left),pd.read_parquet(right),check_exact=True)
    elif left.suffix=='.csv':
        pd.testing.assert_frame_equal(pd.read_csv(left),pd.read_csv(right),check_exact=True)
    elif left.suffix in ['.json','.jsonl']:
        assert left.read_bytes()==right.read_bytes(),left
    else:
        assert left.read_bytes()==right.read_bytes(),left


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--scope',choices=['empirical','simulation'],required=True)
    ap.add_argument('--available-only',action='store_true')
    a=ap.parse_args();a.destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    if a.scope=='empirical':
        controlled.OUT=a.destination/'controlled';full.OUT=controlled.OUT
        assets=sorted(p.stem for p in (controlled.ROOT/'data/returns').glob('*.csv'))
        for model in controlled.MODELS:
            for asset in assets:
                original=SOURCE/'controlled'/f'{model}__{asset}'
                if not (original/'full/complete.json').exists():
                    if a.available_only:continue
                    raise RuntimeError(f'Incomplete pair: {model}/{asset}')
                controlled.work(model,asset);full.work(model,asset)
                copy=controlled.OUT/original.name
                for part in [Path('.'),Path('full')]:
                    meta=json.loads((original/part/'complete.json').read_text())
                    fresh=json.loads((copy/part/'complete.json').read_text())
                    assert meta['binding']==fresh['binding']
                    for filename in meta['outputs']:compare_file(original/part/filename,copy/part/filename)
                rows.append({'model':model,'asset':asset,'fresh_fits_and_daily_paths_exact':True})
                print('Exact fresh empirical fit:',model,asset,flush=True)
        complete=len(rows)==216
    else:
        simulation.OUT=a.destination/'complexity_mc';simulation.OUT.mkdir(exist_ok=True)
        for kind in ['normal','t5']:
            for truth in ['constant','state']:
                for n in [125,250,500,1000]:
                    for alpha in [.01,.05]:
                        name,_=simulation.run(kind,truth,n,alpha)
                        original=SOURCE/'complexity_mc'/name;copy=simulation.OUT/name
                        meta=json.loads((original/'complete.json').read_text())
                        fresh=json.loads((copy/'complete.json').read_text())
                        assert meta['binding']==fresh['binding']
                        for filename in meta['outputs']:compare_file(original/filename,copy/filename)
                        rows.append({'configuration':name,'all_replications_parameters_and_moments_exact':True})
                        print('Exact fresh simulation:',name,flush=True)
        complete=len(rows)==32
    quality=SOURCE/'quality';quality.mkdir(exist_ok=True)
    result={'checker_sha256':controlled.sha(__file__),'scope':a.scope,'complete':complete,'rows':rows,
            'environment':'Recorded binary-locked R8 analysis environment',
            'destination':str(a.destination.resolve())}
    (quality/f'fresh_{a.scope}_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Replay complete:',complete,len(rows),flush=True)


if __name__=='__main__':main()
