#!/usr/bin/env python3
"""Replay the saved, sanitised EODHD catalog audit without credentials/network."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'artifacts/r8_wti_pnl/eodhd_account'


def main():
    manifest = json.loads((DATA/'manifest.json').read_text())
    producer = ROOT/'research/r8_wti_pnl/probe_eodhd_account.py'
    assert hashlib.sha256(producer.read_bytes()).hexdigest() == manifest['producer_sha256']
    expected = {'account', 'exchanges', 'wti', 'crude_oil', 'may2020', 'june2020'}
    assert len(manifest['queries']) == len(expected)
    assert {q['name'] for q in manifest['queries']} == expected
    payloads, bindings, counts = {}, {}, {}
    for query in manifest['queries']:
        assert query['http_status'] == 200
        endpoint = urlsplit(query['endpoint'])
        assert endpoint.scheme == 'https' and endpoint.netloc == 'eodhd.com'
        assert not endpoint.query and not endpoint.fragment
        assert set(query['parameters']) <= {'limit'}
        assert query['file'] == query['name']+'.json'
        path = DATA/query['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == query['sha256']
        bindings[path.name] = digest
        data = json.loads(path.read_text())
        payloads[query['name']] = data
        if isinstance(data, list):
            assert len(data) == query['count']
        if query['name'] in {'wti', 'crude_oil', 'may2020', 'june2020'}:
            counts[query['name']] = dict(Counter(item['Type'] for item in data))
            assert counts[query['name']] == query['returned_types']
            assert query['list_complete_within_limit'] == (len(data) < query['parameters']['limit'])
    assert set(payloads['account']) <= {
        'subscriptionType', 'subscriptionMode', 'apiRequests', 'apiRequestsDate',
        'dailyRateLimit', 'extraLimit', 'requests'}
    matches = [item for item in payloads['exchanges']
               if re.search(r'\b(CME|NYMEX|COMEX|futures|commodity|commodities)\b',
                            ' '.join(str(item.get(key, '')) for key in ['Name', 'Code']), re.I)]
    named = [{key: item.get(key) for key in ['Code', 'Exchange', 'Name', 'Type']}
             for item in payloads['wti']
             if (item['Code'], item['Exchange']) in
             {('WTI', 'US'), ('WTI', 'LSE'), ('CRUD', 'LSE'), ('42GG', 'XETRA')}]
    result = {
        'status': 'saved_authenticated_catalog_audit_passed',
        'successful_authenticated_requests': len(payloads),
        'input_sha256': bindings,
        'exchanges_returned': len(payloads['exchanges']),
        'futures_exchange_name_matches': matches,
        'search_counts': {key: len(payloads[key]) for key in counts},
        'search_types': counts,
        'instrument_identity_examples': named,
        'scope': 'Observed standard exchange/search catalog; search covers active instruments only.',
        'inference': 'These responses do not establish access to individual CL contract histories.',
        'empty_expired_search_proves_historical_absence': False,
        'custom_product_availability': 'Not established',
        'full_wti_contract_history_admitted': False,
        'credential_required_for_offline_replay': False,
        'model_inference_run': False,
        'canonical_manuscript_modified': False,
    }
    (DATA.parent/'eodhd_account_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
