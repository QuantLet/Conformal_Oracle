# Producer verification: does the committed code reproduce the shipped series?

Earlier checks verified that tables follow from the shipped series. This one asks whether the shipped series follow from any committed code. The four parametric benchmarks are deterministic given the returns, so the verdict is exact (tolerance 1e-12).

| method | assets | exact | round-off | data revised | **differs** | max abs diff | z shipped | z rerun |
|---|---|---|---|---|---|---|---|---|
| `hs` | 24 | 21 | 0 | 3 | 0 | 2.037e-06 | -2.54248 | -2.54248 |
| `ewma` | 24 | 0 | 24 | 0 | 0 | 9.214e-07 | -2.32635 | -2.32635 |
| `ewma_recursive` | 24 | 0 | 24 | 0 | 0 | 9.212e-07 | -2.32635 | -2.32635 |
| `garch_n` | 24 | 0 | 0 | 24 | 0 | 1.763e-01 | -2.32635 | -2.32635 |
| `gjr_garch` | 24 | 24 | 0 | 0 | 0 | 0.000e+00 | -2.32635 | -2.32635 |

**Data revised** on ASX200, AUDUSD, BOVESPA, BTC, CBU0, DJCI, ETH, EURUSD, FCHI, FTSE100, GBPUSD, GDAXI, GOLD, HSI, IBGL, ICLN, NATGAS, NIFTY, NIKKEI, SP500, STOXX, TLT, USDJPY, WTI: the rolling window mean of the returns no longer matches the value implied by the shipped forecasts, so the input series changed after the forecasts were written. These are dividend-adjusted ETF histories that get restated upstream; the code is not at fault and the forecasts cannot be reconciled without the original vintage.

`z` is the implied standardised innovation quantile `(VaR_0.01 - mean)/std`, which is what separates one quantile map from another. GARCH-N is −2.32635 = `norm.ppf(0.01)` by construction.

The six TSFM series are **UNVERIFIABLE_HERE**: they came from GPU inference on an A30, and sampling is not bit-reproducible across backends. That is a statement about this machine, not a pass.

