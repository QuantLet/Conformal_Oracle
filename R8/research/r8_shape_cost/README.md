# Reproduce the shape-cost research extension

This is a standalone research-study package. It includes the new simulation,
its saved histories, the exact existing financial inputs needed for the
secondary analysis, candidate mathematics and independent validation. It is
not a new canonical manuscript release or a full TeX source package.

## Environment

Use Python 3.13.9 and the package versions in `environment.json` and
`requirements.txt` beside this file. These describe the tested environment,
not a claim of exact binary portability on every operating system. Set
OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and
VECLIB_MAXIMUM_THREADS=1 when reproducing numerical reductions.

## Validate saved computations

From the package root:

```sh
python research/r8_shape_cost/validate_simulation.py
python research/r8_shape_cost/financial.py --validate
python research/r8_shape_cost/validate.py
```

The first reconstructs histories and checks all coefficients, analytic loss,
conditional Markov occupancy and both bootstrap families independently. The
second replays the financial paths, support and score identities. The third
checks the locked files, all execution receipts, the pre-extension canonical
snapshot, figures and review bindings. The figure's visual receipt applies
to its recorded PNG/SVG bytes; changed plots require another visual inspection.

## Rerun, without overwriting the archived results

Make a separate working copy. The simulation runner refuses to overwrite
`replications.parquet`. Preserve the archived outputs for comparison, then
move only the following generated simulation files out of that working copy:
`histories.npz`, `replications.parquet`, `cells.json`, `execution.json`,
`bootstrap_indices.npz`, `primary_bootstrap.npz`, `sensitivity_bootstrap.npz`,
`contrasts.csv`, `method_summary.csv`, `findings.json` and the run logs.
Keep `design.json`, `before.json`, `preexecution_checks.json` and `lock.json`.

```sh
python research/r8_shape_cost/run.py
python research/r8_shape_cost/aggregate.py
python research/r8_shape_cost/validate_simulation.py
```

The producer checks the locked protocol/code before executing the same5,000
histories. Compare arrays and numerical tables with the archived values;
wall-clock metadata and ZIP timestamps need not reproduce byte for byte.
No additional seeds or configurations should be substituted under this study.

The financial paths are already fitted. `financial.py --run` verifies their
original completion receipts and regenerates only this new analysis; follow
with `--validate`. It does not estimate new financial models. Both bootstrap
families are intentionally suppressed because the predefined ratio statistic
has empty pair-state denominators in some draws. Their absence is a recorded
scientific limitation, not a missing execution step to repair by dropping data.

`plot.py` creates the transparent PNG/SVG and `report.py` renders the research
summary. Existing R8 sources/PDFs included for snapshot verification are
unchanged reference inputs; use the preceding full manuscript release for
its complete TeX build closure.

## Interpretation

The simulation supports the registered sign-crossing prediction, while the
leading approximation has material finite-sample magnitude error. The
financial decomposition is retrospective and descriptive, with no claimed
confidence band. The new results do not establish a feasible financial gate,
universal method dominance, capital savings or publication acceptance.
