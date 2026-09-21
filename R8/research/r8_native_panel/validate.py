"""Bind current displays to the selected panel and reject stale/corrupt results."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'source/scripts/extension_20260831'))
from paper_scope import ROOT,ARCHIVE,MODELS,N_PAIRS,DECISION


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(receipt):
    assert receipt['models']==list(MODELS) and receipt['pairs']==N_PAIRS
    for path,want in receipt['inputs'].items():assert sha(PROJECT/path)==want,('input',path)
    for path,want in receipt['outputs'].items():assert sha(ROOT.parent/path)==want,('output',path)
    for name in ['posthoc.csv','indication.csv','common_support.csv','dq_diagnostics.csv']:
        old=pd.read_csv(ARCHIVE/'results'/name)
        expected=old[~old.model.isin(receipt['excluded'])].reset_index(drop=True)
        actual=pd.read_csv(ROOT/'results'/name)
        pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-12,atol=1e-15)
    pairs=pd.read_csv(DECISION/'pairs.csv')
    assert set(pairs.model)==set(MODELS) and pairs.groupby('method').size().eq(N_PAIRS).all()
    summary=pd.read_csv(DECISION/'summary.csv').set_index('method')
    for name,group in pairs.groupby('method'):
        np.testing.assert_allclose(summary.loc[name,'QS'],group.QS.mean(),rtol=1e-12,atol=1e-15)
        assert summary.loc[name,'kupiec_rejections']==int((group.p_kup<.05).sum())


if __name__=='__main__':
    receipt=json.loads((ROOT.parent/'aggregation.json').read_text());verify(receipt)
    bad=json.loads(json.dumps(receipt));bad['outputs'][next(iter(bad['outputs']))]='0'*64
    try:verify(bad)
    except AssertionError:pass
    else:raise AssertionError('Corruption control was accepted')
    out={'pairs':N_PAIRS,'input_bindings':len(receipt['inputs']),'output_bindings':len(receipt['outputs']),
         'retained_pair_statistics_unchanged':True,'independent_summary_checks':True,'corruption_control':True}
    (ROOT.parent/'selection_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
