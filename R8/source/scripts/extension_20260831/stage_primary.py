#!/usr/bin/env python3
"""Assemble a read-only input overlay while old-vintage replay is running."""
import json
import hashlib
from pathlib import Path
import pandas as pd
from panel_statistics import ROOT,MODELS


def main():
    repair=ROOT/'unfiltered_repair';out=ROOT/'primary_candidate';out.mkdir(exist_ok=True)
    for folder in ['data','parameters','provenance','posthoc','evt_fhs']:
        sources={str(p.relative_to(ROOT/folder)):p for p in (ROOT/folder).rglob('*') if p.is_file()}
        sources.update({str(p.relative_to(repair/folder)):p for p in (repair/folder).rglob('*') if p.is_file()})
        for rel,source in sources.items():
            dest=out/folder/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(source.resolve())
    (out/'results').mkdir(exist_ok=True);(out/'quality').mkdir(exist_ok=True)
    for sub in ['monte_carlo','predictive_sampling']:
        p=out/'results'/sub
        if not p.exists():p.symlink_to(ROOT/'results'/sub,target_is_directory=True)
    inv=pd.read_csv(ROOT/'quality/asset_inventory.csv')
    for row in json.loads((repair/'restored_returns.json').read_text()):
        mask=inv.asset==row['asset'];inv.loc[mask,'n_returns']=row['n_returns'];inv.loc[mask,'sha256']=row['sha256'];inv.loc[mask,'n_exclusions_after_initial']=0
    inv.to_csv(out/'quality/asset_inventory.csv',index=False)
    metrics=[];ledger=[]
    for model in MODELS:
        for asset in inv.asset:
            f=out/'posthoc'/f'{model}__{asset}.json';j=json.loads(f.read_text())
            ret=out/'data/returns'/f'{asset}.csv'
            assert j['binding']['return_sha256']==hashlib.sha256(ret.read_bytes()).hexdigest()
            metrics.extend(j['metrics']);ledger.extend(j['indication'])
    pd.DataFrame(metrics).sort_values(['model','asset','method']).to_csv(out/'results/posthoc.csv',index=False)
    pd.DataFrame(ledger).sort_values(['model','asset','alpha']).to_csv(out/'results/indication.csv',index=False)
    (out/'primary_ready.json').write_text(json.dumps(dict(stage='Complete forecast inputs assembled; validation still running',
        retained_large_returns=True,restored=json.loads((repair/'restored_returns.json').read_text())),indent=2)+'\n')
    print('24-asset primary overlay ready for downstream analysis; original paths unchanged',flush=True)


if __name__=='__main__':main()
