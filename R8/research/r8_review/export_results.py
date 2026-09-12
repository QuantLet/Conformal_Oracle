"""Copy verified review result inputs into the canonical display dependency tree."""
import json
import shutil
from pathlib import Path
from controlled_comparisons import PROJECT,ROOT,sha


def main():
    review=PROJECT/'artifacts/review_20260909';source=review/'results'
    meta=json.loads((source/'complete.json').read_text());assert meta['pairs']==216
    for name,digest in meta['outputs'].items():assert sha(source/name)==digest
    for scope in ['empirical','simulation']:
        assert json.loads((review/'quality'/f'fresh_{scope}_replay.json').read_text())['complete']
    out=ROOT/'results/review';out.mkdir(parents=True,exist_ok=True)
    for p in source.glob('*.csv'):shutil.copy2(p,out/p.name)
    shutil.copy2(review/'complexity_mc/summary.csv',out/'complexity_mc.csv')
    for name in ['calendar_effects.csv','calendar_scores.csv']:shutil.copy2(review/'calendar'/name,out/name)
    print('Exported complete and independently replayed review results',flush=True)


if __name__=='__main__':main()
