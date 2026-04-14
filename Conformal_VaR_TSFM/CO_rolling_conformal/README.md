# CO_rolling_conformal

Rolling 250-day conformal recalibration vs static 70/30 split (Table 10).

## Method

- Static: single calibration set (first 70% of dates), fixed q_V
- Rolling: 250-day trailing window, q_V updated daily

## Output

- `results/tab_rolling_vs_static.tex`
- `results/fig_rolling_coverage.pdf`
