"""Deterministic checks of the new prefix-rank argument; no simulations."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/r8_contiguous_integration'


def check():
    rank_cases=0
    wrong_rank_witness=None
    for n in range(2,8):
        for scores in product(range(3),repeat=n):
            full=sorted(scores)
            for k in range(2,n+1):
                for ell in range(1,k):
                    m=n-ell;r=k-ell;prefix=sorted(scores[:m])
                    assert 1<=r<=m and prefix[r-1]<=full[k-1]
                    rank_cases+=1
                    if k<=m and prefix[k-1]>full[k-1] and wrong_rank_witness is None:
                        wrong_rank_witness=dict(scores=scores,k=k,ell=ell)
    assert wrong_rank_witness is not None
    cap_cases=0
    for n in range(2,401):
        for alpha in [F(1,100),F(1,40),F(1,20),F(1,2),F(99,100)]:
            p=1-alpha;x=(n+1)*p;k=min(n,-(-x.numerator//x.denominator))
            assert F(k,n)>=p
            for ell in range(1,k):
                assert F(k-ell,n-ell)>=p-alpha*F(ell,n-ell)
                cap_cases+=1
    return dict(status='passed',prefix_rank_cases_with_ties=rank_cases,
                exact_rational_rank_penalties=cap_cases,
                undecremented_prefix_rank_negative_control=wrong_rank_witness,
                purpose='Checks a finite grid of algebraic consequences, not a proof of the theorem.',
                new_inference_or_simulations=False)


if __name__=='__main__':
    result=check();OUT.mkdir(exist_ok=True)
    (OUT/'mathematical_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
