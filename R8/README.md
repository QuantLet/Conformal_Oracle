# R8 reproducible tables and figures

This package repairs the execution layout of the original 12 September deposit
and supplies the required intermediate numerical inputs. The scientific producer
code and the paper's numerical results are unchanged.

## Quick start

For exact output bytes, use the explicit macOS Apple Silicon environment. It
includes CPython 3.13.9 and the original graphics-library builds. From the
repository root:

```sh
conda create -y --prefix /tmp/r8-env --file R8/environment-osx-arm64.explicit.txt
/tmp/r8-env/bin/python -m pip install --no-deps -r R8/requirements-conda-overlay.txt
/tmp/r8-env/bin/python R8/reproduce.py verify
/tmp/r8-env/bin/python R8/reproduce.py replay --workdir /tmp/r8-replay --report /tmp/r8-replay.json
```

The `verify` mode checks file integrity only. The `replay` mode executes the
producers and verifies their outputs. A work directory must not already exist.
Omit `--workdir` to use a temporary directory removed when execution finishes.
No credentials, local research checkout or network data download is needed after
installing the dependencies.

## What is reproduced

Eleven producers rebuild **67 files**: 40 LaTeX tables/macro files,
13 PDF figures and seven figures in each of PNG and SVG formats. This includes
every generated numerical fragment and figure used by the article and supplement,
plus four supporting PDF figures also emitted by the same producers. Output
directories start empty; deposited reference outputs are never used as producer
inputs. Hash comparisons must match every generated file.

The replay then runs four existing statistical tests, the exact rational
count-law certificate, its independent matrix/quadrature check and the two
reported selection-error calculations. It verifies that missing outputs,
changed outputs and changed inputs are rejected. The recorded historical receipts
are checked as provenance inputs, not counted as freshly rerun experiments.

## Layout

| Path | Purpose |
|---|---|
| `reproduce.py` | Supported public entry point; creates a disposable working tree |
| `REPLAY_MANIFEST.json` | Hashes, producer order and expected output mapping |
| `environment-osx-arm64.explicit.txt` | Exact conda binary builds, including the graphics stack |
| `requirements-conda-overlay.txt` | Two pinned Python packages added to the conda environment |
| `requirements-replay.txt` | Python package version inventory; alone it does not fix rendering binaries |
| `source/scripts/extension_20260831/` | Original producers and statistical functions |
| `research/` | Research code, protocols and the remaining display producers |
| `artifacts/`, `results/` | Existing intermediate results and receipts used by replay |
| `data/generated_tables/`, `data/figures/` | Reference outputs, used only for comparison |
| `validation/` | Measured public-package validation records |
| `history/` | Superseded documentation and the unused historical tail-closure table |

The restored `source/` and `research/` arrangement preserves the relative paths
expected by the source-hashed producers. Do not move these directories. The
public entry point supplies the output directories and isolates each run.

## Scope and environment

This is regeneration **from archived intermediate results**. It does not claim
to rerun foundation-model inference, financial model fitting, all bootstrap
calculations or all simulation histories. The inputs include summary files,
supporting daily evaluation data and saved uncertainty results. They are sufficient
for the documented replay without the much larger original research archive.

The current sample definitions remain those in the manuscript: the market panel
ends in August 2026 and the external industry evaluation ends in July 2026.
`download_data.py` at the repository root retrieves the earlier `data-v1` archive;
it is not used here and cannot substitute for the R8 inputs.

Exact figure bytes depend on the Python libraries and rendering environment.
A separate pip-only environment reproduced all 34 LaTeX files but differed in
19 figure files: its Matplotlib wheel used FreeType 2.6.1 instead of the recorded
2.13.3. The explicit conda specification fixes that binary dependency; the replay
checks the renderer before starting. Validation records identify the tested
platform and versions. A mismatch causes
failure rather than silently weakening the output comparison. A different
operating system or library build is not certified by the macOS run.

The old research validators also bind historical manuscript snapshots and full
archives. They remain available for their documented historical scope; the
supported entry point for this public package is `R8/reproduce.py`.


## Extensions of 13 September 2026

The statements above about unchanged producers describe the 12 September
repair; the optimism-estimator study was amended on 13 September (see below).
Commits on `main` after `R8-2026-09-13-repair1` carry documentation and
regression-test maintenance only (the estimator's preflight factor check now
compares the true cross-validated loss with the independently specified 8/9
and rejects 4/5); every deposited output is hash-identical to the tag.

Three further studies are deposited in the same layout: `research/r8_power_analysis`
(nine prespecified contrasts on the main-panel loss difference),
`research/r8_optimism_estimator` (nuisance-free optimism estimators; the
amended-factor run in `artifacts/r8_optimism_estimator/` and the original
4/5 run in `artifacts/r8_optimism_estimator_original_4over5/`, see
`AMENDMENT.md`), and `research/r8_external2` (Developed ex-US 25 size/value
portfolios, 100 pairs; raw source response, admission, results and
provenance under `artifacts/r8_external2/devexus/`). Their `displays.py`
producers are steps 9–11 of the replay and regenerate the six display files
`numbers_power`, `tab_power`, `numbers_optest`, `tab_optest`, `numbers_extb`
and `tab_external2`. The studies' own computations read the stored 240-pair
daily losses, the synthetic histories and the external forecast paths, which
exceed this package and are supplied on request; each study folder has a
`--check` replay for use with the full archive.
