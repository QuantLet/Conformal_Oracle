# Panel-level violation rate with dependence-aware uncertainty

Fixed 14 September 2026 before computation, in response to the statistics
referee's point that 240 Kupiec and 240 Christoffersen counts have no stated
null and that pairs share dates. Declared descriptive: the 240-pair panel and
its per-method violation rates have been inspected. No new forecasts, fits or
corrections. Inputs are read-only: `artifacts/r8_ten_comparators/pairs/*/daily.parquet`
(daily returns `r` and the quantile forecast of every method per pair),
`metrics.csv` (per-pair `pihat`, `p_kup`, `p_ind`, `p_cc`) and
`artifacts/r8_ten_comparators/results/pairs.csv`.

## Statistic

For method $m$ and pair $i$ the violation indicator is $V_{it}=\mathbf 1\{r_t<q^{(m)}_{it}\}$,
the convention of the manuscript's Quantile Score definition. The pair violation rate is
the mean of $V_{it}$ over the pair's test dates; the panel statistic is the unweighted mean
of the 240 pair rates (pair-equal weighting, as for QS). It must equal the mean of `pihat`
in `metrics.csv` to 1e-12 for every method.

## Uncertainty

The 95% interval is the percentile interval of the 999 common-calendar circular block
bootstrap draws of the panel statistic, blocks of 20 and 60 calendar days, with the
resampling convention of `research/r8_power_analysis/analysis.py` (seed
`sha256('20260909/panel-calendar/{block}')[:4]`, identical draw order), applied to the
violation indicators instead of the loss differences. Within a pair the resampled rate is
the count-weighted sum of indicators divided by the count-weighted number of trading days.
No binomial reference is used because pairs share dates.

## Methods

Raw, Shift-CP, Rolling250, Rolling500, Selected-rolling, Gate-selected-rolling, Loss-gate,
Past-minimum, in that order. All are columns of `daily.parquet`.

## Backtest counts

For each method, the number of pairs (of 240) with Kupiec $p<0.05$, Christoffersen
independence $p<0.05$ and joint conditional-coverage $p<0.05$, from `pairs.csv`, with the
number of pairs whose statistic is undefined reported separately. Boundary conventions
are those of the archived producer (`0\log0=0`; undefined independence statistics).

## Outputs

`artifacts/r8_coverage_panel/coverage.csv` (one row per method and block length),
`run.json`, `RESULTS.md`. `displays.py` writes `source/sections_r8/numbers_coverage.tex`
(prefix `\nCov`) and `tab_coverage.tex`; `--check` regenerates both into memory and
compares with disk. No existing R8 file is modified.
