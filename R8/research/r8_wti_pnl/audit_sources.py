#!/usr/bin/env python3
"""Reconcile observed source data; never admit spot/nearby data as contracts."""
import hashlib
import html
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/r8_wti_pnl'
RAW = OUT/'raw_sources'


def parse_eia(path):
    rows = []
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', path.read_text(), re.S | re.I):
        label = re.search(r"<td class='B6'>(.*?)</td>", row, re.S)
        if label is None:
            continue
        label = html.unescape(label.group(1)).strip()
        match = re.fullmatch(r'(\d{4}) ([A-Za-z]{3})-\s*(\d+) to ([A-Za-z]{3})-\s*(\d+)', label)
        if match is None:
            raise ValueError('Unrecognised EIA weekly date: '+label)
        year, month, day = match.group(1, 2, 3)
        monday = datetime.strptime(f'{year} {month} {day}', '%Y %b %d')
        if monday.weekday() != 0:
            raise ValueError('EIA row does not start on Monday')
        cells = re.findall(r"<td class='B3'>(.*?)</td>", row, re.S)
        if len(cells) != 5:
            raise ValueError('Expected five daily cells')
        for offset, value in enumerate(cells):
            value = html.unescape(value).strip()
            if value:
                rows.append(dict(date=monday+timedelta(days=offset), value=float(value)))
    frame = pd.DataFrame(rows).set_index('date').sort_index()
    if frame.index.has_duplicates or not np.isfinite(frame.value).all():
        raise ValueError('Invalid parsed EIA data')
    return frame


def main():
    bindings = {}
    for receipt in sorted([*RAW.glob('*.json.json'), *RAW.glob('*.html.json'),
                           *RAW.glob('*.pdf.json')]):
        record = json.loads(receipt.read_text())
        file = RAW/record['file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256']
        bindings[file.name] = record['sha256']
    first, second = [parse_eia(RAW/f'eia_nearby{i}.html') for i in (1, 2)]
    first.to_csv(OUT/'eia_nearby1.csv', float_format='%.17g')
    second.to_csv(OUT/'eia_nearby2.csv', float_format='%.17g')
    assert str(first.index.max().date()) == str(second.index.max().date()) == '2024-04-05'
    dates = ['2020-04-17', '2020-04-20', '2020-04-21', '2020-04-22']
    np.testing.assert_allclose(first.loc[dates, 'value'], [18.27, -37.63, 10.01, 13.78], atol=1e-12)
    np.testing.assert_allclose(second.loc[dates, 'value'], [25.03, 20.43, 11.57, 20.69], atol=1e-12)
    yahoo = pd.read_csv(ROOT/'artifacts/extension_20260831/prices/WTI.csv',
                        index_col='date', parse_dates=True)
    event = pd.DataFrame({'eia_first_nearby': first.loc[dates, 'value'],
                          'eia_second_nearby': second.loc[dates, 'value'],
                          'yahoo_close': yahoo.loc[dates, 'adjusted_close']})
    event['yahoo_minus_eia_first'] = event.yahoo_close-event.eia_first_nearby
    assert event.yahoo_minus_eia_first.abs().max() < 2e-6
    event['first_contract'] = ['CLK20', 'CLK20', 'CLK20', 'CLM20']
    event['second_contract'] = ['CLM20', 'CLM20', 'CLM20', 'CLN20']
    event.to_csv(OUT/'april2020_reconciliation.csv', float_format='%.17g')
    pnl = pd.DataFrame([
        ['2020-04-20', 'CLK20', 18.27, -37.63, -55900., 'Hold May through penultimate settlement, then roll to June'],
        ['2020-04-21', 'CLM20', 20.43, 11.57, -8860., 'June held after the 20 April roll'],
        ['2020-04-22', 'CLM20', 11.57, 13.78, 2210., 'Same June contract, across the nearby-series label change'],
    ], columns=['date', 'held_contract', 'prior_settlement', 'current_settlement', 'pnl_usd', 'explanation'])
    np.testing.assert_allclose(1000*(pnl.current_settlement-pnl.prior_settlement), pnl.pnl_usd, atol=1e-10)
    pnl.to_csv(OUT/'april2020_position_pnl.csv', index=False, float_format='%.17g')
    demo = json.loads((RAW/'eodhd_wti_demo.json').read_text())
    demo_offset = json.loads((RAW/'eodhd_wti_demo_offset1000.json').read_text())
    spot = pd.DataFrame(demo['data']).set_index('date').sort_index()
    assert not spot.index.has_duplicates and np.isfinite(spot.value).all()
    spot.to_csv(OUT/'eodhd_spot_probe.csv', float_format='%.17g')
    result = dict(
        status='source_audit_passed_full_contract_history_not_admitted',
        primary_sources_sha256=bindings,
        eia=dict(nearby1_observations=len(first), nearby2_observations=len(second),
                 endpoint=str(first.index.max().date()), full_august2026_admission=False),
        april2020=dict(source_reconciliation_passed=True, event_pnl_passed=True,
                       largest_yahoo_rounding_difference=float(event.yahoo_minus_eia_first.abs().max()),
                       interpretation='Retrospective penultimate-close roll exposure; no execution guarantee',
                       observed_may_expiry_recovery_usd=1000*(10.01-(-37.63)),
                       may_recovery_not_owned_after_roll=True),
        eodhd=dict(metadata=demo['meta'], returned_count=len(spot),
                   returned_start=str(spot.index.min()), returned_end=str(spot.index.max()),
                   august31_price=float(spot.loc['2026-08-31', 'value']),
                   schema_fields=list(demo['data'][0]),
                   instrument='WTI spot from FRED/EIA, not individual NYMEX futures',
                   offset1000_returns_same_response=demo == demo_offset,
                   full_history_downloaded=False, private_account_accessed=False,
                   individual_futures_available='not established by the documented product or response'),
        full_reconstruction=dict(ready=False,
            missing=['Unadjusted individual CL settlements with roll overlap through August 2026',
                     'Verified full historical contract expirations and exchange sessions']),
        model_inference_run=False, canonical_manuscript_modified=False)
    (OUT/'source_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
