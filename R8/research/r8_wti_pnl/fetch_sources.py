#!/usr/bin/env python3
"""Archive public source evidence without changing the existing market panel."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_wti_pnl/raw_sources'
SOURCES = {
    'eodhd_commodities.html': 'https://eodhd.com/financial-apis/commodities-api-historical-prices-for-oil-gas-metals-agriculture-beta',
    'eodhd_wti_demo.json': 'https://eodhd.com/api/commodities/historical/WTI?api_token=demo&interval=daily&fmt=json',
    'eodhd_wti_demo_offset1000.json': 'https://eodhd.com/api/commodities/historical/WTI?api_token=demo&interval=daily&fmt=json&offset=1000',
    'eia_nearby1.html': 'https://www.eia.gov/dnav/pet/hist/RCLC1D.htm',
    'eia_nearby2.html': 'https://www.eia.gov/dnav/pet/hist/RCLC2D.htm',
    'eia_definitions.html': 'https://www.eia.gov/dnav/pet/TblDefs/pet_pri_fut_tbldef2.asp',
    'eia_coverage.html': 'https://www.eia.gov/dnav/pet/pet_pri_fut_s1_d.htm',
    'cftc_april2020.pdf': 'https://www.cftc.gov/media/5296/InterimStaffReportNYMEX_WTICrudeOil/download',
    'cme_cl_rules.pdf': 'https://www.cmegroup.com/rulebook/NYMEX/2/200.pdf',
    'cme_continuous.html': 'https://www.cmegroup.com/market-data/cme-group-continuous-price-series.html',
}


def download(item):
    name, url = item
    target = OUT / name
    receipt = OUT / (name + '.json')
    if target.exists() or receipt.exists():
        previous = json.loads(receipt.read_text())
        assert previous['url'] == url
        assert hashlib.sha256(target.read_bytes()).hexdigest() == previous['sha256']
        return previous
    try:
        with urlopen(Request(url, headers={'User-Agent': 'Academic research data audit'}), timeout=40) as response:
            payload = response.read()
            record = dict(url=url, resolved_url=response.url, status=response.status,
                          content_type=response.headers.get('Content-Type'), bytes=len(payload))
        if name.endswith('.pdf') and not payload.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF')
        if name.startswith('eia_nearby') and b'Week Of' not in payload:
            raise ValueError('Expected EIA historical table missing')
        record.update(file=name, sha256=hashlib.sha256(payload).hexdigest(),
                      retrieved_utc=datetime.now(timezone.utc).isoformat())
        target.write_bytes(payload)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        return record
    except Exception as error:
        return dict(file=name, url=url, error=type(error).__name__ + ': ' + str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=SOURCES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = [(k, v) for k, v in SOURCES.items() if args.only is None or k in args.only]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(download, selected):
            print(json.dumps(result), flush=True)
