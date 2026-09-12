#!/usr/bin/env python3
"""Promote verified tail-restored inputs while retaining the complete prior vintage."""
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
from panel_statistics import ROOT


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    repair=ROOT/'unfiltered_repair';candidate=ROOT/'primary_candidate';before=ROOT/'before_tail_restoration'
    if (ROOT/'primary_ready.json').exists():raise RuntimeError('Primary promotion already performed')
    if before.exists():raise RuntimeError('Prior promotion archive exists; inspect before retrying')
    for model in ['moirai','timesfm25','moirai2','lagllama']:
        for parent,n in [(ROOT,24),(repair,2)]:
            report=json.loads((parent/'quality'/f'native_replay_{model}_full.json').read_text())
            assert len(report['rows'])==n and all(r['exact'] for r in report['rows'])
    assert len(pd.read_csv(candidate/'results/indication.csv'))==864
    before.mkdir();shutil.copytree(ROOT/'results',before/'results');shutil.copytree(ROOT/'quality',before/'quality')
    (before/'code').mkdir();shutil.copy2(Path(__file__).with_name('prepare_returns.py'),before/'code/prepare_returns_filtered.py')
    changes=[]
    for folder in ['data','parameters','provenance','posthoc','evt_fhs','native']:
        for new in sorted((repair/folder).rglob('*')):
            if not new.is_file():continue
            rel=new.relative_to(repair);old=ROOT/rel;saved=before/rel
            if old.exists():
                saved.parent.mkdir(parents=True,exist_ok=True);old.rename(saved)
            old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(new,old)
            changes.append(dict(path=str(rel),before_sha256=sha(saved) if saved.exists() else None,after_sha256=sha(old)))
    shutil.copy2(candidate/'quality/asset_inventory.csv',ROOT/'quality/asset_inventory.csv')
    inv=pd.read_csv(ROOT/'quality/asset_inventory.csv')
    (ROOT/'quality/returns_manifest.json').write_text(inv.to_json(orient='records',indent=2)+'\n')
    for asset in ['ETH','NATGAS']:
        f=ROOT/'quality/exclusions'/f'{asset}.csv';d=pd.read_csv(f);d[d.reason=='initial_observation'].to_csv(f,index=False)
    restored=json.loads((repair/'restored_returns.json').read_text())
    pd.DataFrame(restored).to_csv(ROOT/'quality/restored_tail_observations.csv',index=False)
    (ROOT/'quality/primary_returns_sources.json').write_text(json.dumps(dict(
        normalized_price_roundtrip=['ETH','NATGAS'],
        reason='The two restored series were regenerated from the frozen 17-significant-digit normalised price CSVs; their CSV roundtrip is part of the exact numerical input specification.'),indent=2)+'\n')
    for source in (candidate/'results').iterdir():
        if source.is_symlink():continue
        target=ROOT/'results'/source.name
        if source.is_dir():shutil.copytree(source,target,dirs_exist_ok=True)
        else:shutil.copy2(source,target)
    for model in ['moirai','timesfm25','moirai2','lagllama']:
        old=json.loads((before/'quality'/f'native_replay_{model}_full.json').read_text())
        new=json.loads((repair/'quality'/f'native_replay_{model}_full.json').read_text())
        rows={r['asset']:r for r in old['rows']};rows.update({r['asset']:r for r in new['rows']})
        for asset,row in rows.items():
            assert row['hashed_rows']==int(inv.loc[inv.asset==asset,'n_returns'].iloc[0])-512
            row['binding_sha256']=sha(ROOT/'native'/model/asset/'binding.json')
        (ROOT/'quality'/f'native_replay_{model}_primary.json').write_text(json.dumps(dict(scope='Every original batch on the final 24-asset inputs',exact=True,
            source_reports=[f'before_tail_restoration/quality/native_replay_{model}_full.json',f'unfiltered_repair/quality/native_replay_{model}_full.json'],
            rows=list(rows.values())),indent=2)+'\n')
    (ROOT/'primary_ready.json').write_text(json.dumps(dict(retained_large_returns=True,restored=restored,changes=changes,
        native_replay='All final native forecasts replayed bit-for-bit; combined 22 unchanged and two restored histories'),indent=2)+'\n')
    print('Promoted',len(changes),'files; former results and inputs preserved',flush=True)


if __name__=='__main__':main()
