#!/usr/bin/env python3
"""Admit observable ETP share returns with an explicit corporate-action audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'artifacts/r8_commodity_etp'
OLD = PROJECT/'artifacts/extension_20260831'
SPEC = {
    'USO': ('WTI', '2006-04-10', {'2020-04-29': '1:8'}),
    'GLD': ('GOLD', '2004-11-18', {}),
    'UNG': ('NATGAS', '2007-04-18', {'2011-03-09': '1:2', '2012-02-22': '1:4',
                                   '2018-01-05': '1:4', '2024-01-24': '1:4'}),
}
ACTION_SOURCES = {
    'USO_2020': 'https://www.sec.gov/Archives/edgar/data/1327068/000117120020000271/i20263_ex99-1.htm',
    'UNG_2011_2012': 'https://secure.alpsinc.com/MarketingAPI/api/v1/Content/uscfinvestments/ung-8-k-20140331.pdf',
    'UNG_2018': 'https://www.sec.gov/Archives/edgar/data/1376227/000117120018000005/i18003_ung-8k.htm',
    'UNG_2024': 'https://www.sec.gov/Archives/edgar/data/1376227/000117120024000010/i24014_ung-8k.htm',
    'GLD_inception': 'https://www.ssga.com/us/en/individual/etfs/spdr-gold-shares-gld',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_dates(timestamps):
    return pd.to_datetime(timestamps, unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()


def main():
    bindings = {}
    for receipt in sorted((ROOT/'raw').glob('*.receipt.json')):
        raw = receipt.with_name(receipt.name.removesuffix('.receipt.json'))
        assert sha(raw) == json.loads(receipt.read_text())['sha256']
        bindings[str(raw.relative_to(PROJECT))] = sha(raw)
    calendar_file = OLD/'prices/SP500.csv'
    calendar = pd.read_csv(calendar_file, index_col='date', parse_dates=True).index
    bindings[str(calendar_file.relative_to(PROJECT))] = sha(calendar_file)
    assert calendar.is_unique and calendar.is_monotonic_increasing
    assert calendar[-1] == pd.Timestamp('2026-08-31')
    records, actions, audits = [], [], []
    for asset, (previous, inception, expected_actions) in SPEC.items():
        raw = ROOT/'raw'/f'{asset}.json'
        obj = json.loads(raw.read_text())['chart']['result'][0]
        assert obj['meta']['symbol'] == asset and obj['meta']['instrumentType'] == 'ETF'
        assert obj['meta']['currency'] == 'USD'
        assert obj['meta']['exchangeTimezoneName'] == 'America/New_York'
        dates = pd.DatetimeIndex(local_dates(obj['timestamp']), name='date')
        assert dates.is_unique and dates.is_monotonic_increasing
        assert dates[0] == pd.Timestamp(inception) and dates[-1] == pd.Timestamp('2026-08-31')
        assert dates.equals(calendar[(calendar >= dates[0]) & (calendar <= dates[-1])])
        quote = obj['indicators']['quote'][0]
        close = np.asarray(quote['close'], dtype=float)
        adjusted = np.asarray(obj['indicators']['adjclose'][0]['adjclose'], dtype=float)
        assert np.isfinite(adjusted).all() and (adjusted > 0).all()
        assert np.isfinite(close).all() and (close > 0).all()
        # These three responses contain no distributions; Yahoo's historical
        # Close already uses split-adjusted share units. Do not split it again.
        assert not obj.get('events', {}).get('dividends')
        np.testing.assert_array_equal(close, adjusted)
        observed_actions = {}
        for split in obj.get('events', {}).get('splits', {}).values():
            day = local_dates([split['date']])[0]
            observed_actions[str(day.date())] = split['splitRatio']
            pos = dates.get_loc(day)
            actions.append(dict(asset=asset, first_post_split_session=str(day.date()),
                new_shares=split['numerator'], old_shares=split['denominator'],
                previous_adjusted_close=adjusted[pos-1], adjusted_close=adjusted[pos],
                log_return=float(np.log(adjusted[pos]/adjusted[pos-1])),
                second_adjustment_applied=False))
        assert observed_actions == expected_actions, (asset, observed_actions)
        price = pd.DataFrame({'close': close, 'adjusted_close': adjusted,
                              'adjustment_factor': adjusted/close}, index=dates)
        ret = pd.DataFrame({'log_return': np.diff(np.log(adjusted))}, index=dates[1:])
        np.testing.assert_allclose(ret.log_return, np.log(adjusted[1:]/adjusted[:-1]), atol=2e-15, rtol=0)
        for folder in ['prices', 'data/returns', 'quality']:
            (ROOT/folder).mkdir(parents=True, exist_ok=True)
        price.to_csv(ROOT/'prices'/f'{asset}.csv', float_format='%.17g')
        ret.to_csv(ROOT/'data/returns'/f'{asset}.csv', float_format='%.17g')
        ret.loc[ret.log_return.abs().nlargest(20).index].sort_index().to_csv(
            ROOT/'quality'/f'{asset}_largest_moves.csv', float_format='%.17g')
        ret.loc['2020-03-01':'2020-05-29'].to_csv(ROOT/'quality'/f'{asset}_2020_stress.csv', float_format='%.17g')
        before = OLD/'raw_responses'/f'{previous}.json'
        old = json.loads(before.read_text())['chart']['result'][0]
        assert not any(k in old for k in ['contract', 'contracts', 'roll_dates', 'settlements'])
        assert 'contractSymbol' not in old['meta']
        bindings[str(before.relative_to(PROJECT))] = sha(before)
        audits.append(dict(old_asset=previous, symbol=old['meta']['symbol'],
            instrument_type=old['meta']['instrumentType'], fields=list(old),
            daily_contract_identifiers=False, roll_schedule=False,
            finding='Saved continuous quotes do not identify same-contract holding returns.',
            replacement=asset, claim_that_all_price_jumps_are_roll_artifacts=False))
        eligible = len(ret)-512; nc = int(.7*eligible)
        records.append(dict(asset=asset, replaced_asset=previous, price_start=inception,
            first_return=str(ret.index[0].date()), last_date=str(ret.index[-1].date()),
            prices=len(price), returns=len(ret), eligible_forecasts=eligible,
            n_cal=nc, n_test=eligible-nc, first_test=str(ret.index[512+nc].date()),
            corporate_actions=len(observed_actions), missing_sessions=0, excluded_returns=0,
            input_sha256=sha(ROOT/'data/returns'/f'{asset}.csv')))
    pd.DataFrame(records).to_csv(ROOT/'quality/support.csv', index=False)
    pd.DataFrame(actions).to_csv(ROOT/'quality/corporate_actions.csv', index=False)
    report = dict(status='admitted_for_forecast_recomputation', assets=records,
        old_futures_audit=audits, source_bindings=bindings,
        corporate_action_sources_reviewed=ACTION_SOURCES,
        corporate_action_note='Issuer close-of-day effective dates reconciled with the following first post-split trading session.',
        vendor_adjustment='Yahoo Close and Adjusted Close coincide and are already adjusted for splits; no second adjustment applied.',
        independent_daily_vendor_check='Stooq returned HTML browser verification, not price CSV; not used as data.',
        calendar_check='Every date equals the corresponding archived SP500 exchange session; not an independent official exchange-calendar audit.',
        protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
        producer_sha256=sha(__file__),
        canonical_manuscript_modified=False, model_scores_used_in_selection=False)
    (ROOT/'data_admission.json').write_text(json.dumps(report, indent=2)+'\n')
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
