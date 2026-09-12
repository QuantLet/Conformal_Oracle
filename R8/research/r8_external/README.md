# External industry-portfolio test

The author approved the 31 July 2026 endpoint on 10 September, before
external return ingestion or outcome calculation. The original August
protocol and dated amendment are preserved under
`artifacts/r8_external/july2026`. This changes only the external endpoint;
the main market panel remains through August. The test is an external
asset-universe retrospective exercise, with shared US market shocks.

Use the existing locked analysis environment; its Python/library versions
are recorded in `runtime.json`. The archived public source response is
the input to reproduction. A later CRSP download may revise history and
must not replace the saved response when claiming an exact replay.

## Data and forecasts

From the project root:

```sh
python research/r8_external/fetch.py
python research/r8_external/prepare.py
python research/r8_external/prepare.py --check
python -m pytest -q research/r8_external/test_external.py
python source/scripts/extension_20260831/classical.py --root artifacts/r8_external/july2026 --models hs gjr_t --workers 3
python research/r8_external/dynamic.py --workers 3
```

The fetcher verifies and reuses an existing archived response. The parser
selects only the value-weighted table, maps the documented missing-value
sentinels, retains every valid large return and verifies the calendar
against the independently stored S&P timestamps. It retains 1,250 pre-2000
observations for initial contexts. HS/GJR have daily 250-observation fits;
CAViaR-AS/GAS-t refit on 1,250 preceding returns each January. GAS uses the
Student-t scale implied by its likelihood. All attempts and yearly
parameters are retained, including nonconvergence flags.

## Full fresh replay

Keep the original outputs. In a disposable copy, create
`artifacts/r8_external/july2026/classical_replay/data/returns` and copy the
twelve admitted return CSVs there unchanged. Then run:

```sh
python source/scripts/extension_20260831/classical.py --root artifacts/r8_external/july2026/classical_replay --models hs gjr_t --workers 3
python research/r8_external/dynamic.py --replay --workers 3
python research/r8_external/validate_base.py
```

Completed outputs are checked before reuse. To force a new calculation,
remove only the replay outputs in the disposable copy; never delete the
sole archived production result. The base validator compares every fitted
parameter and forecast, reconstructs all classical forecasts, and checks
each dynamic recursion independently.

## Corrections and evaluation

```sh
python research/r8_external/corrections.py --workers 3
python research/r8_external/corrections.py --replay --workers 3
python research/r8_external/validate_corrections.py
python research/r8_external/aggregate.py
python research/r8_external/validate_aggregate.py
```

The correction validator reconstructs ranks, weighted quantiles, every
rolling path and selection, inner bootstrap gates, LP objectives, POT
inversions, seeded paths, mixture losses and backtests. It also perturbs
future outcomes for each base model. The aggregate validator rebuilds all
calendar resamples from individual pair losses, the two simultaneous
families, normalised primary decisions and all three descriptive periods.

Primary loss is divided by each pair's calibration-return standard
deviation. All 48 pairs remain. A claimed transfer requires the normalised
simultaneous upper endpoint to be below zero under both block lengths.
Coverage is assessed separately; failed transfer criteria are reported.
These commands do not run foundation-model inference or alter the main
market-panel calculations.
