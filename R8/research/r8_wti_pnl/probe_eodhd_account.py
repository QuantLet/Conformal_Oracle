#!/usr/bin/env python3
"""Read-only EODHD catalog check; credentials stay out of argv, files and logs."""
import argparse
import getpass
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        # Never forward an authentication URL to another location.
        return None


def scrub(value, token):
    if isinstance(value, str):
        return value.replace(token, '[REDACTED]')
    if isinstance(value, list):
        return [scrub(item, token) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item, token) for key, item in value.items()
                if not any(term in key.lower() for term in ['token', 'password', 'secret', 'email'])}
    return value


def run(token, out):
    if out.exists():
        raise ValueError('Use a new output directory to preserve prior evidence')
    opener = build_opener(NoRedirects())
    queries = [('account', '/user', {}), ('exchanges', '/exchanges-list', {})]
    queries += [(name, '/search/'+quote(term, safe=''), {'limit': 500})
                for name, term in [('wti', 'WTI'), ('crude_oil', 'Crude Oil'),
                                   ('may2020', 'CLK20'), ('june2020', 'CLM20')]]
    receipts = []
    out.mkdir(parents=True)
    for name, path, params in queries:
        record = dict(name=name, endpoint='https://eodhd.com/api'+path,
                      parameters=params, requested_utc=datetime.now(timezone.utc).isoformat())
        try:
            url = record['endpoint']+'?'+urlencode(dict(params, fmt='json', api_token=token))
            request = Request(url, headers={'User-Agent': 'Academic WTI contract-data audit'})
            with opener.open(request, timeout=25) as response:
                data = json.loads(response.read())
                record['http_status'] = response.status
            if name == 'account':
                # Exclude identity, payment and authentication fields entirely.
                allowed = {'subscriptionType', 'subscriptionMode', 'apiRequests',
                           'apiRequestsDate', 'dailyRateLimit', 'extraLimit', 'requests'}
                data = {key: value for key, value in data.items() if key in allowed}
            data = scrub(data, token)
            payload = json.dumps(data, indent=2)+'\n'
            if token in payload:
                raise ValueError('Credential redaction failed')
            file = out/(name+'.json')
            file.write_text(payload)
            record.update(file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                          count=len(data) if isinstance(data, list) else None,
                          list_complete_within_limit=(len(data)<500 if isinstance(data, list) and path.startswith('/search/') else None))
            if isinstance(data, list) and path.startswith('/search/'):
                counts = {}
                for item in data:
                    kind = item.get('Type', 'Unspecified')
                    counts[kind] = counts.get(kind, 0)+1
                record['returned_types'] = counts
            print(json.dumps(record), flush=True)
        except HTTPError as error:
            record['http_status'] = error.code
            print(json.dumps({'name': name, 'http_status': error.code}), flush=True)
        except Exception as error:
            # Exception strings can contain the authenticated URL: do not log them.
            record['error_type'] = type(error).__name__
            print(json.dumps({'name': name, 'error_type': type(error).__name__}), flush=True)
        receipts.append(record)
    manifest = dict(queries=receipts, credential_persisted=False,
                    requested_hosts=['eodhd.com'], browser_used=False,
                    producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scope='Documented standard API catalog. Search covers active instruments only; an empty expired-symbol query does not prove absence of historical data.',
                    model_inference_run=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/r8_wti_pnl/eodhd_account')
    args = parser.parse_args()
    credential = os.environ.get('EODHD_API_TOKEN') or getpass.getpass('EODHD token (hidden, memory only): ')
    if not credential.strip():
        raise ValueError('No credential supplied')
    try:
        run(credential.strip(), args.out)
    finally:
        credential = None
