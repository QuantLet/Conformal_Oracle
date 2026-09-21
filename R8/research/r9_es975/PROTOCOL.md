# 97.5% Expected Shortfall and the multiplied capital component — protocol

Fixed 17 September 2026, before computation. Descriptive: the panel's VaR and
1% ES results have been inspected. The study replaces the capital
illustration's supposition (that the 97.5% ES widens like the 1% VaR) by a
measured quantity on the seven forecasters whose stored output defines a
predictive law below the 2.5% quantile; Chronos-2, PatchTST-FM and TS-ICL have
no such output and are excluded.

Inputs (read-only; bound by SHA-256 in `run.json`): the returns, VaR forecast
files, native draws, GJR-GARCH-t degrees of freedom and ten-model calibration
file of `research/r8_es_fz0/PROTOCOL.md`; the binomial traffic-light zones per
pair and method in `artifacts/r8_basel_binomial/zones.csv`.

ES forecasts at alpha = 0.025, one per date, in log-return units (negative):
Moirai-1.1 and Lag-Llama, the mean of the floor(1000 x 0.025) = 25 smallest
draws (Acerbi and Tasche 2002, Proposition 4.1); GARCH-N, GJR-N, EWMA, the
Normal tail mean from the stored mean and standard deviation; GJR-GARCH-t, the
unit-variance Student-t tail mean with the stored degrees of freedom (Normal
where undefined); HS, the mean of the floor(250 x 0.025) = 6 smallest returns
of its 250-day window. Guard: each law reproduces the stored `VaR_0.025`.

Support and correction as in the ten-model panel: eligible dates after 512
observations, the first 70% for calibration, the rest for test. The static
shift is the 1% shift qV of `calibration.csv` (the shift the panel fits);
static ES is ES - qV, the same location shift applied to the whole law.

Capital component: the Basel Framework's internal-models charge multiplies a
60-day average of the 97.5% ES by m_c, 1.5 in Green, 1.70 to 1.92 in Amber and
2.00 in Red (MAR32.9, MAR33.42). Per pair, the component is m(zone) times the
mean 97.5% ES on the test dates, with the pair's binomial zone under raw
forecasts for the raw component and under Shift-CP for the static component
(`zones.csv`), m = 1.5 (Green), 1.70 (Amber, its lower value; 1.92 as
sensitivity) and 2.00 (Red). The 60-day average and the maximum with the
latest measure are not applied; the component is the mean level.

Outputs: `artifacts/r9_es975/{metrics.csv, summary.csv, run.json, RESULTS.md}`:
per pair and per method, mean VaR at 1%, mean ES at 97.5%, zone, multiplier,
component; per method, pooled and pair-equal means, the ES ratio static/raw,
the component ratio under both Amber multipliers, and the number of pairs
whose component falls. Displays via `displays.py` (`numbers_es975.tex`,
`tab_es975.tex`) with `--check`.
