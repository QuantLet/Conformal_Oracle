"""Hand-calculated exposure cases, including negative prices and a roll."""
import unittest

import numpy as np
import pandas as pd

from contract_pnl import build


class ContractPnlTests(unittest.TestCase):
    def setUp(self):
        # Deliberately a tiny fixture, not a purported complete market calendar.
        self.sessions = pd.DataFrame({'date': ['2020-04-17', '2020-04-20', '2020-04-21',
                                               '2020-04-22', '2020-05-18', '2020-05-19']})
        self.contracts = pd.DataFrame([
            ['CLK20', '2020-05-01', '2020-04-21'],
            ['CLM20', '2020-06-01', '2020-05-19']],
            columns=['contract', 'delivery_month', 'last_trade_date'])
        self.prices = pd.DataFrame([
            ['2020-04-17', 'CLK20', 18.27], ['2020-04-20', 'CLK20', -37.63],
            ['2020-04-21', 'CLK20', 10.01], ['2020-04-20', 'CLM20', 20.43],
            ['2020-04-21', 'CLM20', 11.57], ['2020-04-22', 'CLM20', 13.78]],
            columns=['date', 'contract', 'settlement_usd_per_barrel'])

    def run_builder(self, prices=None, **kwargs):
        return build(self.prices if prices is None else prices, self.contracts,
                     self.sessions, '2020-04-17', kwargs.pop('end', '2020-04-22'), **kwargs)[0]

    def test_negative_price_and_same_contract_roll(self):
        result = self.run_builder()
        np.testing.assert_allclose(result.pnl_usd, [-55900., -8860., 2210.], atol=1e-10)
        self.assertEqual(result.held_contract.tolist(), ['CLK20', 'CLM20', 'CLM20'])
        self.assertAlmostEqual(result.loc[0, 'roll_price_gap_excluded'], 58.06)
        self.assertFalse(np.any(np.isclose(result.pnl_usd, 47640.)))

    def test_zero_settlement_allowed(self):
        prices = self.prices.copy()
        prices.loc[1, 'settlement_usd_per_barrel'] = 0.
        self.assertAlmostEqual(self.run_builder(prices).loc[0, 'pnl_usd'], -18270.)

    def test_future_prices_do_not_change_previous_pnl_or_position(self):
        original = self.run_builder()
        prices = self.prices.copy()
        prices.loc[prices.date >= '2020-04-21', 'settlement_usd_per_barrel'] *= -100
        changed = self.run_builder(prices)
        pd.testing.assert_series_equal(original.iloc[0], changed.iloc[0])
        self.assertEqual(original.held_contract.tolist(), changed.held_contract.tolist())

    def test_missing_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing required same-contract'):
            self.run_builder(self.prices.drop(index=3))

    def test_missing_held_settlement_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing required same-contract'):
            self.run_builder(self.prices.drop(index=4))

    def test_endpoint_not_silently_shortened(self):
        with self.assertRaisesRegex(ValueError, 'Exact requested endpoints'):
            self.run_builder(end='2026-08-31')

    def test_duplicate_settlement_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate settlement'):
            self.run_builder(pd.concat([self.prices, self.prices.iloc[[0]]]))

    def test_capital_is_only_fixed_reporting_scale(self):
        first, second = self.run_builder(), self.run_builder(reference_capital=200000.)
        np.testing.assert_array_equal(first.pnl_usd, second.pnl_usd)
        np.testing.assert_allclose(first.capital_scaled_pnl, 2*second.capital_scaled_pnl)


if __name__ == '__main__':
    unittest.main()
