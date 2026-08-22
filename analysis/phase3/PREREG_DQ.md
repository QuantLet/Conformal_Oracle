# Pre-registration: dynamic quantile test on the sequence panel

Rule 1 (PROTOCOL.md). Declared before running.

**Unit of analysis:** the *cell* — one forecaster on one asset. Expected row
count 312 (13 series x 24 assets), matching `all_results.csv` at alpha = 0.01.

**What varies between rows:** the forecaster and the asset, jointly. Nothing else
varies — same alpha, same 70/30 split, same conformal order statistic, same
instrument set, same lag count.

**Statistic.** Engle-Manganelli dynamic quantile regression on the test window:
Hit_t - alpha on a constant, four lagged Hits, and the reported lower quantile.
Wald statistic, chi-squared with 6 degrees of freedom.

**Expected.** Corrected series reject less often than raw. Truncated Chronos
rejected on close to 24/24 raw. DQ rejection on raw series between the Kupiec
rate (69.9%) and the Christoffersen rate, because DQ nests both.

**Falsification.** If DQ rejects essentially everything including well-specified
series, it does not discriminate on this panel and I report that rather than
presenting it as a discriminating test.

## Outcome

312 cells, 2 seconds. Raw 255/312 = 81.7% rejected; corrected 168/312 = 53.8%.
The truncated Chronos series are rejected 24/24 raw, as expected — but so are
19/24 for GARCH-N, EWMA and GJR-GARCH, and 14/24 for GJR-GARCH-t.

**The falsification condition fired.** DQ rejects *more* of the panel than
Kupiec's 69.9%, well-specified series included. It does not discriminate here,
and that is the finding: the most elaborate of the standard backtests, nesting
both coverage and independence and reading the reported quantile as an
instrument, is the least discriminating of the three on this panel. Reported as
such in Section 6.2 rather than presented as a test that separates.
