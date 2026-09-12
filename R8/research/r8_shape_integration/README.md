# R8: correction shape and estimation cost integrated

This release integrates the author's authorised shape-cost study into the
article and supplement. Proposition 4.4 compares two one-coefficient,
return-loss estimators under bounded known scale; Figure 3 tests its sign
predictions with 5,000 independent histories. The proof is in S.3.9 and the
simulation specification in S.4.2. The matched financial decomposition is
retrospective and descriptive; its two prespecified confidence-band families
are withheld because of empty state denominators.

The article is 44 pages and the supplement 36, using the previous typography.
Previous mathematics, empirical tables, figures, bibliography, forecasts and
financial inference families are preserved. No models or simulations were
rerun during integration. The preceding standalone study contains the new
simulation and its complete replay evidence, with its original source bindings.

## Verify the integrated release

Use Python 3.13.9 and the captured validation dependencies in
`research/r8_team_review/requirements-validation.txt` and
`artifacts/r8_team_review/environment.json`; study dependencies are recorded in
`research/r8_shape_cost/environment.json` and `requirements.txt`.
From the complete archive root run:

```sh
python research/r8_shape_integration/validate.py
```

This verifies preservation against the before-source snapshot, all original
portable figure/bibliography bytes, source-bound specialist reviews, immutable
study files, current-source agreement with the 73-member portable package,
PDF diagnostics and the visual receipt. It independently reconstructs all
33 new displayed macros and exactly regenerates the vector figure. It also
checks the recorded 57-display global validation and its four negative
controls. It does not rerun the costly financial panel or simulation.

To replay the complete numerical study, extract
`release/R8_20260911_shape_cost_study.zip` into a separate directory and use
its `research/r8_shape_cost/README.md`. Its original canonical snapshots must
remain in that extracted directory: its pre-integration validator intentionally
rejects later manuscript changes. This is historical version binding, not an
unresolved numerical exception. The integrated release checks that the study
archive and numerical inputs/outputs remain unchanged.

## Rebuild the documents

Extract `release/R8_LaTeX_sources.zip` and follow `BUILD.txt`. It contains the
complete TeX closure including the elsarticle class, but no numerical data or
Python validators. Repeated main/supplement builds resolve their reciprocal
references; text comparison is normalized, not cross-platform PDF-byte equality.

From the full project, after edits and with TeX on PATH:

```sh
python research/r8_shape_integration/displays.py --check
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
python source/scripts/extension_20260831/validate_r8.py
python research/r8_shape_integration/package_sources.py
```

The global validator compares the shipped PDFs with a rebuild; it does not
replace stale PDFs itself. A new PDF hash requires another visual inspection
before the final validator can pass. The final local status is recorded in
`artifacts/r8_shape_integration/final_validation.json`. Full-release completion
is certified only by its adjacent `full_release.json`, issued after all-member
hashing and archive-only validation replay.

The complete release inherits the preceding numerical archive with verified
hashes and retains its manifest under `historical/`. Public deposit, external
human review, independent-platform reproduction and journal acceptance are
not claimed by these local validation checks.
