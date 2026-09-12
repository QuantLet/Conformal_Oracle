#!/usr/bin/env python3
"""Compare all daily forecasts after a fresh fit, retaining every discrepancy."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from panel_statistics import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path)
    ap.add_argument('--report',type=Path,required=True);ap.add_argument('--require-exact',action='store_true')
    a=ap.parse_args();rows=[]
    files=sorted((ROOT/'data').glob('*/*.parquet'))+sorted((ROOT/'evt_fhs').glob('*.parquet'))+sorted((ROOT/'posthoc').glob('*.parquet'))
    for old in files:
        if old.stem.endswith('_parameters'):continue
        rel=old.relative_to(ROOT);new=a.candidate/rel
        assert new.exists(),('missing reconstruction',str(rel))
        x=pd.read_parquet(old);y=pd.read_parquet(new)
        assert x.index.equals(y.index) and list(x)==list(y),(str(rel),'calendar/schema')
        # Parameters have their own audit; every numerical column in every daily
        # forecast file is compared, not just averages or reported tail scores.
        cols=list(x.select_dtypes(include='number'))
        xx=x[cols].to_numpy();yy=y[cols].to_numpy()
        assert np.array_equal(np.isnan(xx),np.isnan(yy)),(str(rel),'missingness')
        delta=np.abs(xx-yy);finite=np.isfinite(delta)
        same=np.array_equal(xx,yy,equal_nan=True)
        rows.append(dict(file=str(rel),rows=len(x),cells=xx.size,exact=same,
                         changed_cells=int(((xx!=yy)&~(np.isnan(xx)&np.isnan(yy))).sum()),
                         max_abs=float(delta[finite].max()) if finite.any() else 0.))
    frame=pd.DataFrame(rows);a.report.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(a.report.with_suffix('.csv'),index=False)
    summary=dict(files=len(frame),cells=int(frame.cells.sum()),all_exact=bool(frame.exact.all()),
                 nonexact_files=int((~frame.exact).sum()),changed_cells=int(frame.changed_cells.sum()),
                 maximum_absolute_difference=float(frame.max_abs.max()))
    a.report.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if a.require_exact:assert summary['all_exact'],'Fresh reconstruction differs; see complete daily comparison'


if __name__=='__main__':main()
