"""Prepare disposable research replay while preserving historical safeguards.

The original research validators deliberately require the pre-integration
manuscript hashes. Restore those files only in this new copy; never weaken
their assertions or replace the current article in the working project.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('destination',type=Path);a=ap.parse_args()
    dest=a.destination.resolve()
    if dest.exists():raise FileExistsError('Use a new disposable directory')
    dest.mkdir(parents=True)
    directories=['research/r8_review','research/r8_decision','research/r8_mechanism',
        'source/scripts/extension_20260831','source/Quantlets/CO_gamlss',
        'artifacts/review_20260909','artifacts/r8_decision','artifacts/r8_mechanism',
        'artifacts/extension_20260831/data','artifacts/extension_20260831/posthoc',
        'artifacts/extension_20260831/results','artifacts/extension_20260831/quality']
    for name in directories:
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(ROOT/name,p,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.DS_Store'))
    shutil.copy2(ROOT/'source/Quantlets/cfp_config.py',dest/'source/Quantlets/cfp_config.py')
    manifests=[json.loads((ROOT/f'artifacts/{phase}/before.json').read_text())['canonical']
               for phase in ['r8_decision','r8_mechanism']]
    assert manifests[0]==manifests[1],'Research phases have different manuscript baselines'
    restored={}
    for name,want in manifests[0].items():
        candidates=[ROOT/'artifacts/r8_risk_integration/before'/name,ROOT/name]
        valid=next((p for p in candidates if p.is_file() and sha(p)==want),None)
        if valid is None:raise ValueError(('Historical protected file unavailable',name))
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(valid,target)
        assert sha(target)==want;restored[name]=want
    (dest/'REPLAY_STAGE.json').write_text(json.dumps({'canonical_files_restored':restored,
        'purpose':'Original numerical research replay, not the current manuscript build'},indent=2)+'\n')
    print(json.dumps({'destination':str(dest),'restored':len(restored)}),flush=True)


if __name__=='__main__':main()
