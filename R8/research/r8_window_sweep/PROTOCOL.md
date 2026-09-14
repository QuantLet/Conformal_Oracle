# Estimation-window sweep of the classical forecasters — display protocol

Fixed 14 September 2026, before the display producer was written. This is a
display of an existing computation, not a new one.

Source: `source/analysis/phase3_windows/window_sensitivity.csv` (288 rows),
written by `source/analysis/phase3_windows/run_window_sensitivity.py` on
29 August 2026 and verified for SP500 by `verify_window_sp500.py`
(`verify_window_sp500.log`: violation rates and Kupiec p-values reproduced to
1e-16, QS to 6e-9). The script re-estimates GARCH-N and GJR-GARCH (zero mean,
Normal innovations), EWMA and Historical Simulation on trailing windows of
w in {250, 500, 1000} returns, applies the 70/30 split and the static shift
of equation (3), and evaluates raw and corrected forecasts on the test block.

Vintage: the inputs are the 24 return histories of `cfp_ijf_data/returns/`
ending 18 March 2026, in which GOLD, NATGAS and WTI futures series stand where
the current panel uses the GLD, USO and UNG fund shares. Per
`artifacts/r8_commodity_etp/panel/base/quality/asset_inventory.csv`, the
current histories extend the 18 March 2026 vintage with unchanged overlap for
18 series; IBGL, ICLN and TLT differ on the overlap (adjusted-price revisions).
The display states this provenance; no value is recomputed on the current panel.

Displayed: per model and window, the mean over the 24 series of the QS gain
(raw minus corrected, times 1e4), the mean raw and corrected violation rates and
the number of series whose raw forecast is Green under the scaled rule.
Outputs: `artifacts/r8_window_sweep/{sweep_summary.csv, run.json}`; displays via
`displays.py` (`numbers_windows.tex`, `tab_windows_sweep.tex`) with `--check`.
