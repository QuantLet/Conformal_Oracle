# R8: training optimism integrated with its dependent-hit control

This stage integrates the validated expectation expansion and six-cell AR
control. It reads existing results; it does not fit a forecaster or generate
simulated paths. The failed nuisance-estimation admission remains failed.

Use the captured Python 3.13 environment at
`/private/tmp/irfa-r8-conda-clean/bin/python` on this host, or the dependencies
recorded in `research/r8_team_review/requirements-validation.txt` and
`artifacts/r8_team_review/environment.json`. TeX Live 2026 is used for builds.

## Check the current integration

```sh
python research/r8_optimism_integration/displays.py --check
python research/r8_optimism_integration/check_numbers.py --check-receipt
python research/r8_optimism_integration/validate.py
```

The final validator checks the existing studies' unchanged hashes and mtimes,
the previous mathematical statements and empirical displays, new numerical
and mathematical reviews, current document diagnostics, source-package
agreement and the PDF inspection receipt. It uses explicit corrupted-input
checks before accepting preservation, source bindings and whitespace checks.

## Rebuild after a deliberate source change

```sh
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
python source/scripts/extension_20260831/validate_r8.py
python research/r8_shape_integration/displays.py --check
python research/r8_optimism_integration/package_sources.py
```

A changed PDF requires a fresh visual inspection and updated source-bound
reviews. The 77-member portable source ZIP compiles after extraction and
reproduces normalised PDF text. It contains document sources and figures,
not the financial data or a full numerical release.

The immutable research stages retain their original manuscript bindings.
Their historical conservation statements describe the manuscript before
this authorised integration. Do not rewrite those bindings to accept a new
manuscript. The integration checks all 247 bound research artifacts without
changing them, and separately validates the current paper. Earlier complete
empirical archives remain available; this stage does not regenerate them.
