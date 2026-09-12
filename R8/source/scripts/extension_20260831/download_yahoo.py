#!/usr/bin/env python3
"""Freeze Yahoo daily chart responses for the authorised August 2026 extension.

DJCI is deliberately rejected: Yahoo DJCI is the retired ETN, not the index.
Raw responses are immutable within this vintage; rerunning resumes downloads.
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/extension_20260831')
    ap.add_argument('--assets', nargs='+')
    args = ap.parse_args()
    mapping = json.loads((ROOT / 'source/Quantlets/CO_data_returns/ticker_mapping.json').read_text())
    mapping.pop('DJCI')
    if args.assets:
        mapping = {k: mapping[k] for k in args.assets}
    raw = args.output / 'raw_responses'
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for asset, ticker in mapping.items():
        params = urlencode(dict(period1=946684800, period2=1788220800,
                                interval='1d', events='div,splits', includeAdjustedClose='true'))
        url = f'https://query2.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe="")}?{params}'
        target = raw / f'{asset}.json'
        metadata = raw / f'{asset}.provenance.json'
        if not target.exists():
            temporary = target.with_suffix('.download')
            result = subprocess.run(['curl', '-L', '--fail', '--silent', '--show-error',
                                     '--max-time', '45', '--retry', '2', '--retry-delay', '10',
                                     '-A', 'Mozilla/5.0', url, '-o', str(temporary)])
            if result.returncode:
                records.append(dict(asset=asset, status='download_failed', exit_code=result.returncode))
                print(asset, 'FAILED', flush=True)
                continue
            payload = json.loads(temporary.read_text())
            if payload['chart'].get('error') or not payload['chart'].get('result'):
                raise ValueError(f'{asset}: invalid chart response')
            temporary.rename(target)
        payload = json.loads(target.read_text())['chart']['result'][0]
        if payload['meta']['symbol'] != ticker:
            raise ValueError(f'{asset}: unexpected instrument {payload["meta"]["symbol"]}')
        record = dict(asset=asset, ticker=ticker, url=url,
                      retrieved_at_utc=datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat(),
                      sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      instrument=payload['meta'], observations=len(payload.get('timestamp', [])),
                      requested_end_exclusive='2026-09-01', status='downloaded')
        metadata.write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(asset, ticker, record['observations'], 'OK', flush=True)
        time.sleep(1)
    (args.output / 'download_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    if any(x['status'] != 'downloaded' for x in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
