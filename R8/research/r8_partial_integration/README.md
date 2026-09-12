# Current R8: partial-shift study integrated

The canonical article is `source/main_R8.tex`; its companion is
`source/supplement_R8.tex`. Section 7.1 adds one paragraph on choosing
correction strength. S.4.8 gives the fixed development design, exact loss
accounting and Table S.11. The financial panels, inference, base forecasts,
theory and bibliography are unchanged. This adds an explanation of selection
cost, not a new recommended financial method or untouched external test.

The article and supplement have 48 and 30 pages. Main inference still does
not establish a simultaneous static-shift advantage over raw. The previous
French evaluation cannot confirm a rule developed after its inspection.

## Document and display reproduction

Use Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, arch 8.0.0,
pyarrow 21.0.0, Matplotlib 3.10.6, pypdf 6.14.2 and TeX Live. Run from the
project root:

```sh
python source/scripts/extension_20260831/build_paper_outputs.py
python research/r8_integration/build.py
python research/r8_regime/build_paper.py
python research/r8_ten_integration/build.py
python research/r8_partial_integration/build.py
```

From `source`, compile `main_R8.tex`, then `supplement_R8.tex`, then both
again, each with `latexmk -g -pdf -interaction=nonstopmode -halt-on-error`.
Back at the project root:

```sh
python source/scripts/extension_20260831/validate_r8.py
python research/r8_ten_integration/package_sources.py
python research/r8_partial_integration/validate.py --replay-study
```

The first command checks all 56 generated displays, their corruption
controls, the data/forecast structure, literal numbers, producers,
references and compilation. The second builds a 63-member portable
LaTeX archive in isolation and compares both PDF texts. The third verifies
the new table and counts directly from all 576,000 method-history losses,
checks the exact accounting, preserves the old financial displays and
replays the independent 72,000-choice validation from the original study
archive. The workspace has no Git metadata; source checks use
`git diff --no-index --check` and a deliberate whitespace negative control.

## Numerical reproduction and historical snapshots

`release/R8_20260910_partial_shift_study.zip` is an immutable, self-contained
study archive, included in the complete release. Its original validators
protect the pre-integration canonical files. Extract that archive to a new
directory before using the commands in its `research/r8_partial_shift/README.md`.
Do not run the historical study validator directly against the edited
manuscript or disable its canonical-file checks. The integration validator
performs this extraction automatically with `--replay-study`.

The separate original study includes saved histories, all producers,
fresh-process replay, exact tie checks and superseded initial floating-tie
outputs. To reproduce its calculation, rename `run` and `replay` within
the extracted copy, then execute its `run.py`, `run.py --replay` and
`validate.py`. No new random paths or base forecasts are needed.

Financial reproduction remains documented in these packages:

- `research/r8_model_extension`: model inference and common-support inputs.
- `research/r8_ten_comparators/README.md`: stronger-comparator fits and replays.
- `research/r8_external/README.md`: July industry test, inputs and validations.
- `research/r8_referee_revision/README.md`: simultaneous inference and the
  separate supervised GBM experiment.
- `research/r8_horizon_bridge/README.md` and `research/r8_count_law/README.md`:
  contiguous expected loss and the finite-count counterexample.

Those research-stage validators may also bind their original manuscript
snapshots. Their documented disposable-copy restoration remains necessary;
the current integration does not weaken historical checks. The count-law
release manifest is preserved inside the current full archive.

## Packaging and scope

`package_release.py` overlays the integrated documents and study on the
verified preceding full archive. It verifies every member and runs the
integration validator using only extracted archive files before finalising
`release/R8_20260910_partial_integration.zip`. The receipt is
`artifacts/r8_partial_integration/full_release.json` in the working project.
It preserves both preceding full releases and the separate study archive.
The numerical financial calculations are retained from their validated
release; they are not newly reestimated by this integration step.

`release/R8_LaTeX_sources.zip` is only the portable document closure.
Neither packaging operation publishes or submits files. The change and
validation report is `docs/IRFA_PARTIAL_INTEGRATION_VALIDATION.md`.
