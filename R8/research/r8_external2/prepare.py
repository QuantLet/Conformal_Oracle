"""Admit and reconstruct the 25 value-weighted Developed ex-US ME x BE-ME daily returns only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile
import numpy as np
import pandas as pd
from fetch import PROJECT, OUT, sha, DETAILS, PROTOCOL_DETAILS

COLUMNS=['SMALL LoBM','ME1 BM2','ME1 BM3','ME1 BM4','SMALL HiBM',
         'ME2 BM1','ME2 BM2','ME2 BM3','ME2 BM4','ME2 BM5',
         'ME3 BM1','ME3 BM2','ME3 BM3','ME3 BM4','ME3 BM5',
         'ME4 BM1','ME4 BM2','ME4 BM3','ME4 BM4','ME4 BM5',
         'BIG LoBM','ME5 BM2','ME5 BM3','ME5 BM4','BIG HiBM']
ASSETS=[c.replace(' ','_') for c in COLUMNS]
RAW_NAME='Developed_ex_US_25_Portfolios_ME_BE-ME_daily_CSV.zip'
TABLE='Average Value Weighted Returns -- Daily'
SENTINELS=[-99.99]          # the only missing-data code documented in the archived file header
UNDOCUMENTED=[-999.]        # documented in the CRSP industry file, absent from this header; counted, never mapped
MIN_END='2025-12-31'        # protocol: stop if the file ends before this date
END='2026-07-31'            # last date in the archived file; verified against the file at every run
END_YEAR=2026
END_LABEL='July2026'
YEARS=range(2000,END_YEAR+1)


def parse(raw):
    with ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist();assert len(names)==1
        lines=z.read(names[0]).decode('utf-8-sig').splitlines()
    starts=[i for i,s in enumerate(lines) if s.strip()==TABLE]
    assert len(starts)==1
    start=starts[0]+1;stop=start+1
    while stop<len(lines) and re.match(r'^\s*\d{8}\s*,',lines[stop]):stop+=1
    f=pd.read_csv(io.StringIO('\n'.join(lines[start:stop])),index_col=0,skipinitialspace=True)
    f.columns=[c.strip() for c in f.columns]
    assert f.columns.tolist()==COLUMNS
    f.index=pd.to_datetime(f.index.astype(str).str.strip(),format='%Y%m%d');f.index.name='date'
    assert f.index.is_unique and f.index.is_monotonic_increasing
    assert all(pd.api.types.is_numeric_dtype(t) for t in f.dtypes);f=f.astype(float)
    counts={str(v):int((f==v).sum().sum()) for v in SENTINELS+UNDOCUMENTED}
    f=f.replace(SENTINELS,np.nan)
    f.columns=ASSETS
    header=[s for s in lines[:starts[0]] if s.strip()]
    return f,dict(zip_member=names[0],source_header=header,missing_statement=[s for s in header if 'Missing' in s],
                  sentinel_codes_mapped=SENTINELS,undocumented_codes_counted=UNDOCUMENTED,sentinel_counts_value_table=counts,
                  value_table_title=TABLE,value_table_start_line=start+1,value_table_end_line=stop,
                  other_tables=[s.strip() for s in lines if s.strip().startswith('Average ') and s.strip()!=TABLE],
                  rows=len(f),first=str(f.index[0].date()),last=str(f.index[-1].date()))


def prepare(check=False):
    raw=OUT/'raw'/RAW_NAME
    meta=json.loads(raw.with_suffix(raw.suffix+'.json').read_text());assert sha(raw)==meta['sha256']
    f,details=parse(raw.read_bytes())
    assert f.index[-1]>=pd.Timestamp(MIN_END),('file ends before the protocol minimum endpoint',details['last'])
    assert f.index[-1]==pd.Timestamp(END),('recorded endpoint differs from the file',details['last'])
    n0=int(f.index.searchsorted('2000-01-01'));assert n0>=1250
    used=f.iloc[n0-1250:].loc[:END].copy()
    assert used.index[-1]==pd.Timestamp(END)
    # Common calendar: every admitted date carries all 25 portfolios, and the
    # file's own calendar is the complete Monday-Friday sequence between its
    # first and last admitted dates (the international files list every weekday).
    assert np.isfinite(used).all().all(),dict(missing_cells=int(used.isna().sum().sum()))
    assert (used>-100).all().all()
    weekday=pd.bdate_range(used.index[0],used.index[-1])
    assert used.index.equals(weekday),dict(missing=weekday.difference(used.index).astype(str).tolist(),extra=used.index.difference(weekday).astype(str).tolist())
    result=np.log1p(used/100)
    # The archived S&P calendar is recorded for comparison only; the Developed
    # ex-US file uses a weekday calendar that includes US holidays.
    calendar=PROJECT/'artifacts/extension_20260831/data/returns/SP500.csv'
    idx=pd.read_csv(calendar,index_col='date',parse_dates=True).index
    start=pd.Timestamp('2000-01-01');expected=idx[(idx>=start)&(idx<=END)]
    actual=result.index[(result.index>=start)&(result.index<=END)]
    comparison=dict(reference='artifacts/extension_20260831/data/returns/SP500.csv',reference_sha256=sha(calendar),
                    dates_here=int(len(actual)),dates_sp500=int(len(expected)),
                    here_not_in_sp500=int(len(actual.difference(expected))),sp500_not_here=int(len(expected.difference(actual))))
    outputs={}
    for asset in ASSETS:
        target=OUT/'data/returns'/f'{asset}.csv'
        encoded=result[asset].rename('log_return').to_csv(float_format='%.17g').encode()
        if check:assert target.read_bytes()==encoded,asset
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(encoded)
        outputs[str(target.relative_to(OUT))]=hashlib.sha256(encoded).hexdigest()
    probe=json.loads((OUT/'raw/protocol_details_page_probe.json').read_text())
    record=dict(status='passed',producer_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
                raw_sha256=sha(raw),raw_retrieved_utc=meta['retrieved_utc'],raw_source=meta['source'],
                details_page=DETAILS,details_page_sha256=sha(OUT/'raw/availability.html'),
                protocol_details_page=dict(url=PROTOCOL_DETAILS,http_status=probe['status']),
                calendar='file weekday calendar (Monday-Friday, complete between first and last admitted date)',
                calendar_sp500_comparison=comparison,
                endpoint=dict(rule='last date in the file',value=END,minimum_required=MIN_END),
                source=details,portfolios=ASSETS,source_columns=COLUMNS,weighting='value',dividends='included (details page: returns in US dollars, include dividends and capital gains)',
                currency='USD',transformation='log1p(percent_simple_return/100)',
                sentinel_handling=dict(mapped_to_missing=SENTINELS,counted_only=UNDOCUMENTED,counts=details['sentinel_counts_value_table']),
                missing_in_admitted_window=0,pre_2000_context_rows=1250,first_context=str(result.index[0].date()),
                first_forecast=str(result.index[result.index.year>=2000][0].date()),last_forecast=END,
                first_test=str(result.index[result.index.year>=2015][0].date()),
                calibration_dates=int(((result.index.year>=2000)&(result.index.year<=2014)).sum()),
                test_dates=int((result.index.year>=2015).sum()),outputs=outputs)
    path=OUT/'admission.json'
    if check:assert json.loads(path.read_text())==record
    else:path.write_text(json.dumps(record,indent=2)+'\n')
    print({k:v for k,v in record.items() if k not in ['outputs','source','source_columns','portfolios']},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');prepare(ap.parse_args().check)
