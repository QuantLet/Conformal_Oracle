# R8 with observable commodity-fund exposures

The active manuscript uses 24 assets and seven forecasters, with USO, GLD
and UNG replacing WTI, GOLD and NATGAS. The remaining 21 inputs and
forecasts are byte-identical to their preceding versions. This is a change
of economic exposure and sample history; a change in aggregate QS between
the two panels is not a forecast improvement on identical data.

`source/scripts/extension_20260831/commodity_scope.py` selects the current
paper, whose numerical inputs are in `artifacts/r8_commodity_etp/panel`.
The old `paper_scope.py` and `research/r8_native_panel/reaggregate.py`
retain the preceding selection experiment. Do not run the old display
instructions to infer that its commodity quotations are still the current
paper's data. All original daily arrays and fitted corrections are retained.

## Reconstruct the paper from stored numerical outputs

Use the recorded analysis environment (Python 3.13.9, NumPy 2.3.5,
pandas 2.3.3, SciPy 1.16.3, arch 8.0.0, pyarrow 21.0.0) and TeX Live.
The existing environment records also specify optimisation, plotting and
PDF dependencies. The commands below do not download data or refit models.
They verify the raw returns and native quantiles, stage the fixed panel,
recompute aggregates and intervals, and rebuild every paper display.

```sh
python research/r8_commodity_etp/validate.py
python research/r8_commodity_etp/validate_decision.py
python research/r8_commodity_etp/stage_panel.py
python research/r8_commodity_etp/analyse_panel.py
python research/r8_commodity_etp/diagnostics.py
python research/r8_commodity_etp/aggregate_panel.py --kind review
python research/r8_commodity_etp/aggregate_panel.py --kind decision
python research/r8_commodity_etp/validate_panel.py --replay
python source/scripts/extension_20260831/build_paper_outputs.py
python research/r8_integration/build.py
python research/r8_regime/build_paper.py
```

From `source`, build `main_R8.tex`, `supplement_R8.tex`, then each again
using `latexmk -g -pdf -interaction=nonstopmode -halt-on-error`.
From the project root run:

```sh
python source/scripts/extension_20260831/validate_r8.py
```

This includes daily-to-pair checks, exact aggregate and display replay,
deliberate corruption controls, and checks that the shipped PDFs match
clean builds. `panel/stage.json` binds every reused source and staged copy;
`aggregation_review.json` and `aggregation_decision.json` bind the mixed
21-old/3-new experiment before aggregation. Native outputs are traced to
their original inference archives, rather than relabelled or rerun under
an unrelated instrument name.

## Reproduce the replacement calculations

`PROTOCOL.md` fixes the instruments, endpoint, fitting interfaces and
calibration rules. `prepare.py` reconstructs returns from archived raw
Yahoo responses. `fetch.py` is for a separate live retrieval; do not replace
the archived response with a later vendor revision and call it an exact
replay. Source URLs, issuer corporate actions and failed Stooq format
evidence are retained. No credential is needed for the archived calculation.

The fitting commands are:

```sh
python source/scripts/extension_20260831/classical.py --root artifacts/r8_commodity_etp --assets USO GLD UNG --workers 3
python research/r8_commodity_etp/compute.py --kind classical-check
python research/r8_commodity_etp/replay_classical.py
python research/r8_commodity_etp/dynamic.py
python research/r8_commodity_etp/compute.py --kind evt
python research/r8_commodity_etp/compute.py --kind posthoc
python research/r8_commodity_etp/controlled_comparisons.py
python research/r8_commodity_etp/full_candidates.py
python research/r8_commodity_etp/decision.py
python research/r8_commodity_etp/decision.py --replay
```

Completed fitting artifacts have bindings and are checked before reuse.
`replay_classical.py` deliberately re-estimates all 15 classical series
and parameter histories in a fresh temporary directory. The decision
replay writes a separate set of all 21 pairs; its validator compares
every numerical output and perturbs future returns in three pairs.

For native inference use the model-specific environments and pinned
weights already recorded by the project's inference recipes. Call
`native.py --model moirai` or `native.py --model lagllama` with that
environment's Python, followed by the same command with `--replay`.
Moirai uses CPU; Lag-Llama uses MPS and actual calendar features. The
archive binds package versions, weights, batch seeds, input windows,
sample arrays and distribution parameters. Exact replay was checked on
the recorded hardware/backend; no cross-platform bitwise guarantee is
asserted. `run_native.py` is a convenience launcher for the original
machine's environment paths. For a restored environment, invoke
`native.py` directly with its interpreter.

`reduce.py` reconstructs forecast quantiles from the saved native arrays.
All 1,000 draws are retained for the sample-based models. Native
Chronos-2 and PatchTST forecasts were also produced and exactly replayed
for the three funds, without tail completion. They remain separate
candidate work and are not inputs to the seven-model article. Their
preceding full-universe comparisons refer to the earlier commodity
quotations and must not be presented as current-fund results.

To force any cached production fit to be calculated again, use a fresh
copy of the archive and retain the original outputs for comparison;
do not delete the sole recorded result. Stored seeds and floating-point
environment versions are part of the reconstruction specification.

## Editorial and numerical packages

`package_sources.py` creates the small current LaTeX ZIP and compares both
PDF texts with an isolated clean build. `package_release.py` creates a new
full numerical archive, retaining preceding inputs and adding this
replacement pipeline. These are local artifacts. Public deposit remains
a separate action; neither script submits or uploads anything.
