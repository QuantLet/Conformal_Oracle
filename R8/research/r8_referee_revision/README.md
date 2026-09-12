# Referee revision: reporting symmetry and a supervised forecast benchmark

The canonical manuscript remains R8. Read `PROTOCOL.md` for the eight-
contrast reporting family and `GBM_PROTOCOL.md` for the separate, fixed
direct-quantile forecasting check. Neither changes the old ten-model paths
or the external protocol. The previous six-contrast output is retained.

## Validate the saved numerical results

Use the existing analysis environment: Python 3.13.9, NumPy 2.3.5,
pandas 2.3.3, SciPy 1.16.3, arch 8.0.0, pyarrow 21.0.0,
LightGBM 4.6.0 and the recorded scikit-learn dependency. From the project root:

```sh
python research/r8_referee_revision/validate_inference.py
python research/r8_referee_revision/validate_gbm.py
python research/r8_ten_integration/build.py --check
python source/scripts/extension_20260831/validate_r8.py
```

The first validator independently reconstructs losses and all expanded
bands, including normalisation and exclusion of the two cryptocurrencies.
The second checks exact complete fresh-fit reproduction, independently
constructs every feature and traverses the fitted trees for every forecast,
and independently recomputes conformal ranks, metrics and all 22 paired
contrasts at both block lengths. These are calculation checks, not a
confidence statement about research specification choices.

## Regenerate in a disposable extracted copy

Completed output directories are immutable. Preserve the supplied archive.
In a disposable copy only, move `artifacts/r8_referee_revision/results`
and `artifacts/r8_referee_revision/gbm` outside that copy before running:

```sh
python research/r8_referee_revision/inference.py
python research/r8_referee_revision/validate_inference.py
python research/r8_referee_revision/gbm.py
python research/r8_referee_revision/gbm.py --output artifacts/r8_referee_revision/gbm/replay
python research/r8_referee_revision/evaluate_gbm.py
python research/r8_referee_revision/validate_gbm.py
python research/r8_ten_integration/build.py
```

`inference.py` also verifies exact equality of the underlying bootstrap
draws and point estimates with the old six-contrast run. It changes the
maximum statistic's family, not the realised resamples. GBM uses the
native-model comparison's existing paired calendar draws; its distinct
22-contrast family is never pooled with the correction family.

The two input bootstrap sequences have different archived seeds. Both
use 999 draws and calendar blocks of 20 and 60 days. Preserve each
sequence; the protocol fixes which study it belongs to.

To rerun earlier model or correction studies, follow their READMEs and
the pre-integration snapshot procedure in
`research/r8_ten_integration/README.md`. They protect their original
manuscript snapshots. Do not disable those checks to accommodate new prose.

## Build and package the documents

Run the four existing display builders listed in the integration README.
Compile main, supplement, main, supplement with `latexmk -g -pdf`, then
run `validate_r8.py`. `research/r8_ten_integration/package_sources.py`
checks a clean isolated source-package build against both current PDFs.
The new `package_release.py` writes a separate complete numerical archive
and verifies every archived byte by SHA-256; preceding release ZIPs are
preserved. Nothing is published by these commands.
