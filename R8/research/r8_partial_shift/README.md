# Partial-shift development study

The author approved this study on 10 September 2026. It reuses the complete
stored mechanism grid, chooses among five correction fractions using only
past observations, and separates estimation cost from selection regret.
The protocol predates the new rule's results. The histories and earlier
results were already inspected, so this is development, not confirmation.

The canonical article and supplement remain the validated R8 documents.
This study does not fit financial models or generate new simulation paths.
Its results and proposed manuscript wording are in
`docs/IRFA_PARTIAL_SHIFT_RESULTS.md`.

## Reproduction

Use Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3 and
pyarrow 21.0.0, with Matplotlib 3.10.6 and Pillow for the figures.
The input path archive, old results and 53 protected document/receipt
files are bound by hashes. All commands run from the project root.

```sh
python research/r8_partial_shift/validate.py
```

This verifies the complete production/replay agreement, reconstructs all
72,000 choices independently, checks exact rational tie comparisons,
numerically integrates both loss functions, verifies oracle optima and
the loss decomposition, and compares all 24,000 contiguous Full-CP
differences with the earlier independent implementation. It tests future
invariance and a deliberately leaking negative control.

To repeat the full calculation, make a disposable copy or extract the
standalone study archive into a new directory. Within that copy only,
rename `artifacts/r8_partial_shift/run` and `replay` to `run_saved` and
`replay_saved`. Then run:

```sh
python research/r8_partial_shift/run.py
python research/r8_partial_shift/run.py --replay
python research/r8_partial_shift/validate.py
python research/r8_partial_shift/plot.py
python research/r8_partial_shift/report.py
```

The calculation refuses to overwrite an existing run directory. It reads
the stored histories and evaluates future losses analytically. No random
number generator is called. The study archive includes every bound input;
the previous large release is not needed for these commands.

## Decisions and numerical ties

The first max(100,floor(.7*n)) scores fit the inner shift; the rest select
lambda from 0, .25, .5, .75 and 1. Neither parameter is refitted afterwards.
At n=125 there are only 25 validation scores. Both full and inner conformal
ranks must be finite; there is no silent rank cap.

If validation size times (1-alpha) is an integer, the empirical pinball
objective can have an exactly flat interval. The implementation recognises
this interval and chooses the smallest allowed fraction attaining it.
Any unresolved floating comparison falls back to rational arithmetic.
Independent rational comparisons validate the decisions. Initial outputs
that used floating argmin without this tie correction are preserved under
`initial_numeric_ties`, labelled superseded. This repairs the prescribed
rule; it does not select a new policy based on test performance.

## Interpretation

`run/summary.csv` and `contiguous_summary.csv` contain every method and
configuration, paired Monte Carlo errors and expected violation rates.
Pointwise errors are descriptive. The 144 configurations share 1,500
latent histories; their signs are not independent replications.
`decisions.parquet` stores every fitted shift, fraction, validation loss,
oracle reference and exact decomposition component. `decomposition.csv`
and `selection.csv` aggregate these without dropping cases.

The oracle fractions are infeasible references. They use the population
loss only after feasible selection has finished. Partial correction has
no asserted conformal coverage guarantee. The independent-marginal primary
evaluation and the Normal-AR contiguous sensitivity are kept separate.

The current pilot does not justify presenting the selected fraction as a
new validated financial method. Its useful result is an explicit accounting
of the price of choosing correction strength, even with a scalar family.
The previously inspected French results cannot validate a new choice as
untouched evidence.

`package.py` writes the self-contained study ZIP, verifies every member and
runs the validator from an isolated extraction before writing its receipt.
The ZIP is a research supplement to the existing complete R8 release, not a
replacement for the financial data/model replication package.
