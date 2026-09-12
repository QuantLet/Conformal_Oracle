#!/usr/bin/env python3
"""Archive named commodity ETP prices and issuer pages before model scoring."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT/'artifacts/r8_commodity_etp/raw'
ASSETS = ['USO', 'GLD', 'UNG']


def fetch(item):
    name, url = item
    target = OUT/name
    receipt = OUT/(name+'.receipt.json')
    if target.exists():
        old = json.loads(receipt.read_text())
        assert old['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert old['url'] == url
        return name+' verified existing'
    try:
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=40) as response:
            payload = response.read()
            status = response.status
        if name.endswith('.json'):
            result = json.loads(payload)['chart']
            assert result.get('error') is None and len(result['result']) == 1
            assert result['result'][0]['meta']['symbol'] == name.split('.')[0]
        target.write_bytes(payload)
        record = dict(url=url, http_status=status, bytes=len(payload),
                      retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      sha256=hashlib.sha256(payload).hexdigest())
        receipt.write_text(json.dumps(record, indent=2)+'\n')
        return name+' downloaded '+str(len(payload))+' bytes'
    except Exception as error:
        # Public URLs only; retain concise failure evidence, never fabricate data.
        record = dict(url=url, error_type=type(error).__name__,
                      status=getattr(error, 'code', None),
                      attempted_utc=datetime.now(timezone.utc).isoformat())
        (OUT/(name+'.failure.json')).write_text(json.dumps(record, indent=2)+'\n')
        return name+' failed '+type(error).__name__


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = urlencode(dict(period1=946684800, period2=1788220800, interval='1d',
                            events='div,splits', includeAdjustedClose='true'))
    queries = [(asset+'.json', f'https://query2.finance.yahoo.com/v8/finance/chart/{asset}?{params}')
               for asset in ASSETS]
    queries += [(asset+'.stooq.csv', f'https://stooq.com/q/d/l/?s={asset.lower()}.us&i=d&d1=20000101&d2=20260831')
                for asset in ASSETS]
    queries += [('uso_issuer.html', 'https://www.uscfinvestments.com/uso'),
                ('ung_issuer.html', 'https://www.uscfinvestments.com/ung'),
                ('gld_issuer.html', 'https://www.spdrgoldshares.com/usa/')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, queries):
            print(result, flush=True)


if __name__ == '__main__':
    main()
