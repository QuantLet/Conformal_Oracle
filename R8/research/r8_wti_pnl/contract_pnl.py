#!/usr/bin/env python3
"""Same-contract settlement P&L; refuse incomplete input rather than splice."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(settlements, contracts, sessions, start, end, reference_capital=100000.):
    """Start is an initial settlement date; outputs begin the next session.

    Calendar and expirations must be externally sourced. Validation here
    proves internal coverage, not the authenticity of a supplied calendar.
    """
    if not np.isfinite(reference_capital) or reference_capital <= 0:
        raise ValueError('Reference capital must be positive and finite')
    calendar = pd.DatetimeIndex(pd.to_datetime(sessions['date']))
    if calendar.has_duplicates or calendar.hasnans or not calendar.is_monotonic_increasing:
        raise ValueError('Calendar must contain unique ordered sessions')
    if not calendar.equals(calendar.normalize()):
        raise ValueError('Calendar entries must be exchange dates, without intraday times')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start not in calendar or end not in calendar or start >= end:
        raise ValueError('Exact requested endpoints are not covered by the calendar')
    days = calendar[(calendar >= start) & (calendar <= end)]
    metadata = contracts.copy()
    if metadata['contract'].isna().any() or metadata['contract'].duplicated().any():
        raise ValueError('Missing or duplicate contract identity')
    metadata['last_trade_date'] = pd.to_datetime(metadata['last_trade_date'])
    metadata['delivery_month'] = pd.to_datetime(metadata['delivery_month'])
    if metadata[['last_trade_date', 'delivery_month']].isna().any().any():
        raise ValueError('Missing contract dates')
    if metadata['last_trade_date'].duplicated().any() or metadata['delivery_month'].duplicated().any():
        raise ValueError('Ambiguous CL monthly contract schedule')
    if (metadata['last_trade_date'] >= metadata['delivery_month']).any():
        raise ValueError('CL last trade must precede the delivery month')
    locations = calendar.get_indexer(metadata['last_trade_date'])
    if np.any(locations <= 0):
        raise ValueError('Calendar must include expiry and its preceding exchange session')
    metadata['roll_date'] = calendar[locations - 1]
    metadata = metadata.sort_values('delivery_month').reset_index(drop=True)
    months = metadata['delivery_month'].dt.to_period('M').astype('int64').to_numpy()
    if np.any(np.diff(months) != 1):
        raise ValueError('Contract schedule has a missing delivery month')
    if not metadata['roll_date'].is_monotonic_increasing:
        raise ValueError('Contract roll dates are not ordered')
    prices = settlements.copy()
    prices['date'] = pd.to_datetime(prices['date'])
    if prices[['date', 'contract']].isna().any().any() or prices.duplicated(['date', 'contract']).any():
        raise ValueError('Missing or duplicate settlement key')
    if not prices['date'].isin(calendar).all():
        raise ValueError('Settlement dated outside the supplied exchange calendar')
    if not prices['contract'].isin(metadata['contract']).all():
        raise ValueError('Settlement has no contract metadata')
    values = prices['settlement_usd_per_barrel'].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite settlement')
    lookup = prices.set_index(['date', 'contract'])['settlement_usd_per_barrel']

    def price(day, contract):
        try:
            return float(lookup.loc[(day, contract)])
        except KeyError as error:
            raise ValueError(f'Missing required same-contract settlement: {day.date()} {contract}') from error

    rows = []
    for previous, day in zip(days[:-1], days[1:]):
        # The schedule is known in advance; selection uses the previous close.
        eligible = metadata[metadata['roll_date'] > previous]
        if eligible.empty:
            raise ValueError('Contract schedule ends before the requested exposure')
        contract = eligible.iloc[0]
        held = contract['contract']
        if day > contract['roll_date']:
            raise ValueError('A roll session is missing between evaluated settlements')
        old, new = price(previous, held), price(day, held)
        pnl = 1000. * (new - old)
        row = dict(date=day, previous_date=previous, held_contract=held,
                   prior_settlement=old, current_settlement=new,
                   pnl_usd=pnl, reference_capital_usd=reference_capital,
                   capital_scaled_pnl=pnl/reference_capital,
                   roll_at_close=bool(day == contract['roll_date']),
                   next_contract=held, incoming_settlement=np.nan,
                   roll_price_gap_excluded=np.nan)
        if row['roll_at_close']:
            following = metadata[metadata['roll_date'] > day]
            if following.empty:
                raise ValueError('Incoming contract missing at roll')
            incoming = following.iloc[0]['contract']
            incoming_price = price(day, incoming)
            row.update(next_contract=incoming, incoming_settlement=incoming_price,
                       roll_price_gap_excluded=incoming_price-new)
        rows.append(row)
    return pd.DataFrame(rows), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settlements', type=Path, required=True)
    parser.add_argument('--contracts', type=Path, required=True)
    parser.add_argument('--sessions', type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output exists: use a new directory to preserve provenance')
    ledger, metadata = build(pd.read_csv(args.settlements), pd.read_csv(args.contracts),
                             pd.read_csv(args.sessions), args.start, args.end)
    args.out.mkdir(parents=True)
    ledger.to_csv(args.out/'daily_pnl.csv', index=False, float_format='%.17g')
    metadata.to_csv(args.out/'roll_schedule.csv', index=False)
    files = [args.settlements, args.contracts, args.sessions, Path(__file__),
             Path(__file__).with_name('PROTOCOL.md'), args.out/'daily_pnl.csv', args.out/'roll_schedule.csv']
    record = dict(status='internal_input_checks_passed',
                  external_source_admission='requires provenance and calendar/expiry reconciliation',
                  start=args.start, end=args.end, n_pnl=len(ledger),
                  contract_size=1000, reference_capital=100000,
                  hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.out/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
