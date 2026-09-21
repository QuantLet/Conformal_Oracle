# Current R8: expected loss, dependence and stronger comparators

The canonical documents are `source/main_R8.tex` and
`source/supplement_R8.tex`. The September 9 risk integration adds Proposition
4.7 and its proof, the completed controlled-mechanism experiments, and the
stronger-comparator results. Coverage Theorem 4.5 and its proof are unchanged.
The manuscript has 42 pages and six figures; the supplement has 23 pages.
The shorter abstract has 99 whitespace-delimited words.

The September 10 update adds the fully replayed predetermined-regime study
to the deployment argument. The new control separates immediate shock
response from longer-horizon loss and tests an initial Kupiec-only decision
that remains fixed through a break. It is not the empirical gate's complete
specification or a new break-detection rule. The theorem and the 216-pair
market panel remain unchanged; all main figures are retained.

This file supersedes the older five-figure/base-archive instructions below
for the current document build. It does not replace the locked environments
or native/statistical reconstruction records. External industry-portfolio
validation is pending August data; it is not included as a completed result.

## Rebuild current documents without fitting

Use the locked analysis Python described in the base-data README. From the
project root, with that Python active:

```sh
python source/scripts/extension_20260831/build_paper_outputs.py
python research/r8_integration/build.py
python research/r8_regime/build_paper.py
cd source
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main_R8.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error supplement_R8.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main_R8.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error supplement_R8.tex
cd ..
python source/scripts/extension_20260831/validate_r8.py
```

The final command checks the current 216-pair input panel, exactly replays
26 original, 14 risk-study and two regime-study display files, runs corruption negative controls,
and checks numerical literals, input producers, references and fresh PDF
compilation. New empirical numbers are generated from the recorded result
matrices. The figure builder performs no estimation or simulation. Every
new plot has transparent PDF/PNG/SVG exports and a bottom legend.

## Reproduce the completed research phases

The original research programs deliberately protect the manuscript hashes
recorded before integration. Preserve that safeguard. Prepare a **new**
disposable project copy; the staging script restores those 42 historical
files only in the copy, using the archived pre-integration snapshot:

```sh
python research/r8_integration/stage_research_replay.py /tmp/r8-research-replay
cd /tmp/r8-research-replay
python research/r8_decision/validate.py
python research/r8_mechanism/validate.py
```

The decision validator checks the archived fresh-process pair replays and
repeats the full future-outcome perturbation. The mechanism validator
regenerates all 1,500 independent calibration paths and both test-state
paths, checks the recorded block replays and evaluates independent iid
quadratures. Completed outputs are verified and skipped by the fitting
programs; a cache hit is not a newly estimated model.

For fresh re-estimation, remove only the `replay/` directories **in the
disposable copy**, then follow `research/r8_decision/README.md` and
`research/r8_mechanism/README.md`. `run.py --replay` writes separate outputs;
the original `pairs/` and `blocks/` remain the comparison references. The
mechanism README gives the four individually selected replay commands.
The complete run can be reproduced by replaying all pairs/blocks. To
reaggregate saved results, run each phase's `aggregate.py` in the copy and
compare with the original `results/` CSV/Parquet matrices. Study-specific
protocols, fit attempts, parameters, seeds, daily predictions and runtime
bindings are retained with each phase. Same-version numerical libraries
from a different binary distribution need not reproduce optimiser paths.

Tests for these phases are separate processes to avoid module-name collisions:

```sh
python -m pytest -q research/r8_decision/test_methods.py
python -m pytest -q research/r8_mechanism/test_engine.py
```

Ancillary broad-baseline specifications were moved, without numerical
changes, to `research/r8_integration/broad_baseline_specs.tex`. The original
generated `tab_methods`, `tab_classes`, `tab_quotations` and
`tab_controlled_sweep` remain available even where omitted from the PDFs.
Their producers and machine-readable sources are unchanged.

## Reproduce the regime study

`research/r8_regime/README.md` gives production and full fresh-process replay
instructions. Its 1,000 independent innovation histories are shared across
32 scenario/law/tail configurations and eleven policies. All ten production
blocks have been regenerated and every numerical array matched exactly.
After manuscript integration, prepare a disposable copy with the historical
protected files; the staging helper never changes the current article:

```sh
python research/r8_regime/stage_replay.py /tmp/r8-regime-replay
cd /tmp/r8-regime-replay
python research/r8_regime/run.py --replay --workers 3
python research/r8_regime/validate.py
python research/r8_regime/aggregate.py
```

The result and validation report is `docs/IRFA_REGIME_CHANGE_RESULTS.md`.
All eight scenario figures at both levels and both laws are in the regime
artifact folder; the article adds only the argument and a compact supplement
table. The external evaluation remains unexecuted.

## Pending external study

`research/r8_external/PROTOCOL.md` fixes the proposed 12-industry universe,
four econometric forecasters, chronological fitting/selection rules and
evaluation criteria before external outcome inspection. `readiness.py`
reads only the official source's availability metadata and currently
records 31 July 2026. It does not silently shorten the required 31 August
endpoint or claim the external forecasting adapter has been completed.

The current integration report is `docs/IRFA_R8_RISK_INTEGRATION.md`.
The base-data reconstruction instructions follow in the packaged root README.
