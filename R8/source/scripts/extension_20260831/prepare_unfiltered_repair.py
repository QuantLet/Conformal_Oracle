#!/usr/bin/env python3
"""Restore the two calculable tail returns removed by the inherited size filter.

Stage separate inputs; promotion waits until original replay and replacement
forecasts are complete. Never splice old forecasts across the added observation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from panel_statistics import ROOT


def main():
    target=ROOT/'unfiltered_repair';(target/'data/returns').mkdir(parents=True,exist_ok=True)
    if not (target/'models').exists():(target/'models').symlink_to('../models',target_is_directory=True)
    rows=[]
    for asset in ['ETH','NATGAS']:
        prices=pd.read_csv(ROOT/'prices'/f'{asset}.csv',index_col='date',parse_dates=True).adjusted_close.dropna()
        prior=prices.shift(1);valid=(prices>0)&(prior>0)
        ret=np.log(prices.where(valid)/prior.where(valid)).dropna().to_frame('log_return')
        old=pd.read_csv(ROOT/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True)
        added=ret.index.difference(old.index);assert len(added)==1
        assert np.max(np.abs(ret.loc[old.index,'log_return']-old.log_return))<1e-15
        file=target/'data/returns'/f'{asset}.csv';ret.to_csv(file,float_format='%.17g')
        rows.append(dict(asset=asset,date=str(added[0].date()),log_return=float(ret.loc[added[0],'log_return']),
                         reason='Remove outcome-based absolute-return filter; retain every calculable log return',
                         n_returns=len(ret),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    (target/'restored_returns.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
