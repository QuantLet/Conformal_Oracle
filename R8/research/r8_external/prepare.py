"""Admit and reconstruct the value-weighted daily industry returns only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile
import numpy as np
import pandas as pd
from fetch import PROJECT, OUT, sha

ASSETS = ['NoDur','Durbl','Manuf','Enrgy','Chems','BusEq','Telcm','Utils','Shops','Hlth','Money','Other']
END = '2026-07-31'


def parse(raw):
    with ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist();assert len(names)==1
        lines=z.read(names[0]).decode('utf-8-sig').splitlines()
    starts=[i for i,s in enumerate(lines) if s.strip()=='Average Value Weighted Returns -- Daily']
    assert len(starts)==1
    start=starts[0]+1; stop=start+1
    while stop<len(lines) and re.match(r'^\d{8},',lines[stop]):stop+=1
    f=pd.read_csv(io.StringIO('\n'.join(lines[start:stop])),index_col=0)
    assert f.columns.tolist()==ASSETS
    f.index=pd.to_datetime(f.index.astype(str),format='%Y%m%d');f.index.name='date'
    assert f.index.is_unique and f.index.is_monotonic_increasing
    f=f.replace([-99.99,-999.],np.nan)
    return f,dict(zip_member=names[0],source_header=lines[0],value_table_start_line=start+1,
                  value_table_end_line=stop,rows=len(f),first=str(f.index[0].date()),last=str(f.index[-1].date()))


def prepare(check=False):
    raw=OUT/'raw/12_Industry_Portfolios_daily_CSV.zip'
    meta=json.loads(raw.with_suffix(raw.suffix+'.json').read_text());assert sha(raw)==meta['sha256']
    f,details=parse(raw.read_bytes());assert f.index[-1]>=pd.Timestamp(END)
    n0=int(f.index.searchsorted('2000-01-01'));assert n0>=1250
    used=f.iloc[n0-1250:].loc[:END].copy()
    assert used.index[-1]==pd.Timestamp(END)
    assert np.isfinite(used).all().all() and (used>-100).all().all()
    result=np.log1p(used/100)
    # The contemporaneously archived S&P calendar is an independent source
    # for US market dates; only its timestamps, never outcomes, enter here.
    calendar=PROJECT/'artifacts/extension_20260831/data/returns/SP500.csv'
    idx=pd.read_csv(calendar,index_col='date',parse_dates=True).index
    start=max(pd.Timestamp('2000-01-01'),idx[0]);expected=idx[(idx>=start)&(idx<=END)]
    actual=result.index[(result.index>=start)&(result.index<=END)]
    assert actual.equals(expected),dict(missing=expected.difference(actual).astype(str).tolist(),extra=actual.difference(expected).astype(str).tolist())
    outputs={}
    for asset in ASSETS:
        target=OUT/'data/returns'/f'{asset}.csv'
        encoded=result[asset].rename('log_return').to_csv(float_format='%.17g').encode()
        if check:assert target.read_bytes()==encoded,asset
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(encoded)
        outputs[str(target.relative_to(OUT))]=hashlib.sha256(encoded).hexdigest()
    record=dict(status='passed',producer_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
                raw_sha256=sha(raw),calendar_reference=str(calendar.relative_to(PROJECT)),calendar_sha256=sha(calendar),
                source=details,portfolios=ASSETS,weighting='value',dividends='included',transformation='log1p(percent_simple_return/100)',
                missing_in_admitted_window=0,pre_2000_context_rows=1250,first_context=str(result.index[0].date()),
                first_forecast=str(result.index[result.index.year>=2000][0].date()),last_forecast=END,
                calibration_dates=int(((result.index.year>=2000)&(result.index.year<=2014)).sum()),
                test_dates=int((result.index.year>=2015).sum()),outputs=outputs)
    path=OUT/'admission.json'
    if check:assert json.loads(path.read_text())==record
    else:path.write_text(json.dumps(record,indent=2)+'\n')
    print({k:v for k,v in record.items() if k not in ['outputs','source']},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');prepare(ap.parse_args().check)
