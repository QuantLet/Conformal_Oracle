#!/usr/bin/env python3
"""Replay all 12,500 simulation paths from their seeds and compare output bytes."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import monte_carlo as mc
ORIGINAL=mc.ROOT;TARGET=ORIGINAL/'quality/mc_replay'


def replay(kind,T):
    mc.ROOT=TARGET
    mc.run(kind,T)
    rel=f'results/monte_carlo/{kind}_{T}.csv'
    want=(ORIGINAL/rel).read_bytes();got=(TARGET/rel).read_bytes();assert want==got,(kind,T)
    return dict(dgp=kind,T=T,replications=500,sha256=hashlib.sha256(got).hexdigest())


if __name__=='__main__':
    (TARGET/'results/monte_carlo').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[f.result() for f in as_completed([pool.submit(replay,k,T) for k in mc.DGPS for T in mc.GRID])]
    (ORIGINAL/'quality/monte_carlo_replay.json').write_text(json.dumps(dict(exact=True,replications=12500,cells=rows),indent=2)+'\n')
    print('PASS all 12500 seeded replications, byte-identical outputs',flush=True)
