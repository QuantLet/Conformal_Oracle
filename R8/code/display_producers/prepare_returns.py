#!/usr/bin/env python3
"""Build audited prices/returns from frozen Yahoo responses, without redownloading.

Retain every calculable log return. Nonpositive price transitions cannot
define real log returns and are recorded. Never forward-fill prices.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def prepare(payload):
    result = payload['chart']['result'][0]
    meta = result['meta']
    dates = pd.to_datetime(result['timestamp'], unit='s', utc=True)
    dates = dates.tz_convert(meta['exchangeTimezoneName']).tz_localize(None).normalize()
    frame = pd.DataFrame(result['indicators']['quote'][0], index=dates)
    frame['adjusted_close'] = result['indicators']['adjclose'][0]['adjclose']
    frame.index.name = 'date'
    frame = frame.loc['2000-01-01':'2026-08-31'].sort_index()
    if frame.index.has_duplicates:
        raise ValueError('Duplicate local trading dates')
    return prepare_price_frame(frame)


def prepare_price_frame(frame):
    observed = frame['adjusted_close'].dropna()
    prior = observed.shift(1)
    valid_prices = (observed > 0) & (prior > 0)
    log_return = np.log(observed.where(valid_prices) / prior.where(valid_prices))
    kept = log_return.notna()
    returns = log_return[kept].to_frame('log_return')
    audit = pd.DataFrame({'price': observed, 'prior_price': prior, 'log_return': log_return})
    audit['reason'] = np.select([prior.isna(), ~valid_prices],
                               ['initial_observation', 'nonpositive_price_transition'], default='retained')
    exclusions = audit[audit.reason != 'retained']
    return frame, returns, exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'artifacts/extension_20260831')
    args = ap.parse_args()
    out = args.root
    source_spec = out / 'quality/primary_returns_sources.json'
    normalized = set(json.loads(source_spec.read_text())['normalized_price_roundtrip']) if source_spec.exists() else set()
    for sub in ['prices', 'data/returns', 'quality/exclusions']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    records = []
    for manifest in sorted((out / 'raw_responses').glob('*.provenance.json')):
        provenance = json.loads(manifest.read_text())
        asset = provenance['asset']
        raw = out / 'raw_responses' / f'{asset}.json'
        if hashlib.sha256(raw.read_bytes()).hexdigest() != provenance['sha256']:
            raise ValueError(f'{asset}: raw response changed')
        frame, returns, exclusions = prepare(json.loads(raw.read_text()))
        if len(returns) < 1000 or returns.index[-1] < pd.Timestamp('2026-08-28'):
            raise ValueError(f'{asset}: insufficient history or stale endpoint')
        frame.to_csv(out / 'prices' / f'{asset}.csv', float_format='%.17g')
        if asset in normalized:
            # Preserve the exact numerical specification of the staged repair:
            # these inputs were rebuilt from the frozen normalised price CSVs.
            _, returns, exclusions = prepare_price_frame(pd.read_csv(
                out / 'prices' / f'{asset}.csv', index_col='date', parse_dates=True))
        target = out / 'data/returns' / f'{asset}.csv'
        returns.to_csv(target, float_format='%.17g')
        exclusions.to_csv(out / 'quality/exclusions' / f'{asset}.csv', float_format='%.17g')
        m = provenance['instrument']
        record = dict(asset=asset, ticker=provenance['ticker'], instrument=m.get('longName', m.get('shortName')),
                      currency=m['currency'], exchange=m.get('fullExchangeName'), n_returns=len(returns),
                      first_date=str(returns.index[0].date()), last_date=str(returns.index[-1].date()),
                      n_exclusions_after_initial=int((exclusions.reason != 'initial_observation').sum()),
                      n_missing_prices=int(frame.adjusted_close.isna().sum()),
                      sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        # Historical comparisons are optional audit metadata, never a dependency
        # of the analytical returns or of a portable current-vintage replay.
        legacy = ROOT / 'cfp_ijf_data/returns' / f'{asset}.csv'
        if legacy.exists():
            old = pd.read_csv(legacy, index_col='date', parse_dates=True)
            common = old.index.intersection(returns.index)
            delta = (old.loc[common, 'log_return'] - returns.loc[common, 'log_return']).abs()
            record.update(old_n=len(old),old_last=str(old.index[-1].date()),common_dates=len(common),
                          overlap_max_abs_change=float(delta.max()),overlap_changed_above_1e_12=int((delta>1e-12).sum()),
                          old_dates_absent=int(len(old.index.difference(returns.index))))
        records.append(record)
        print(asset, len(returns), record['last_date'], 'excluded', record['n_exclusions_after_initial'],flush=True)
    raw = out / 'raw_responses/DJCI_investing_table.csv'
    provenance = json.loads((out / 'raw_responses/DJCI_source.json').read_text())
    if hashlib.sha256(raw.read_bytes()).hexdigest() != provenance['sha256']:
        raise ValueError('DJCI: raw table changed')
    frame = pd.read_csv(raw, index_col='date', parse_dates=True).sort_index()
    if frame.index.has_duplicates or len(frame) != 3062 or (frame.close <= 0).any():
        raise ValueError('DJCI: invalid raw table')
    frame['adjusted_close'] = frame.close  # spot index, not a dividend-paying security
    frame.to_csv(out / 'prices/DJCI.csv', float_format='%.17g')
    # CME has no trade date on these holidays. Keep the vendor rows in prices
    # and a separate calendar-exclusion file; calculate consecutive valid closes.
    closed = frame.index.isin(pd.to_datetime(provenance['excluded_closed_dates']))
    frame[closed].to_csv(out / 'quality/DJCI_closed_market_rows.csv', float_format='%.17g')
    _, returns, exclusions = prepare_price_frame(frame[~closed])
    target = out / 'data/returns/DJCI.csv'
    returns.to_csv(target, float_format='%.17g')
    exclusions.to_csv(out / 'quality/exclusions/DJCI.csv', float_format='%.17g')
    records.append(dict(asset='DJCI', ticker='.DJCI', instrument='Dow Jones Commodity Index (spot)',
                        currency='USD', exchange='Index follows CME holiday calendar',
                        n_returns=len(returns), first_date=str(returns.index[0].date()),
                        last_date=str(returns.index[-1].date()), n_exclusions_after_initial=0,
                        n_closed_market_rows=int(closed.sum()), n_missing_prices=0,
                        overlap_comparison='Not comparable: old DJCI was a different instrument (retired ETN)',
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    if len(records) != 24 or set(x['asset'] for x in records) != set(
            p.stem for p in (out / 'data/returns').glob('*.csv')):
        raise ValueError('Expected exactly 24 assets')
    pd.DataFrame(records).to_csv(out / 'quality/asset_inventory.csv', index=False)
    (out / 'quality/returns_manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
