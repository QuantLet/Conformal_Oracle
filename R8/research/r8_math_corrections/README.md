# R8: three local mathematical corrections

This implements the three replacements explicitly authorised after the fresh
mathematical audit. Only `source/sections_r8/risk_proofs.tex` changes in the
manuscript sources. The article is rebuilt to maintain cross-document
references; its normalized text is unchanged. The supplement remains 36 pages.

The original audit reports describe the pre-correction source and remain
immutable. Current completion is recorded in
`docs/IRFA_MATH_CORRECTIONS_20260911.md` and
`artifacts/r8_math_corrections/final_validation.json`.

## Verify the current project

Use the captured Python validation environment described in
`research/r8_team_review/requirements-validation.txt`. From the project root:

```sh
python source/scripts/extension_20260831/validate_r8.py
python research/r8_shape_integration/displays.py --check
python research/r8_math_corrections/validate.py
```

The first two commands are the existing numerical/display and document guards.
The third checks exact application of the authorised patch, preservation of
the other snapshotted inputs, figure and table bytes, current PDF/package
bindings, visual-inspection receipt and whitespace. It also checks the corrected
innovation-quantile identity for Normal and standardised Student-t(5) laws.
None of these commands reruns model inference or simulations.

## Rebuild the documents

With TeX Live on PATH, use the existing four-build order:

```sh
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
python research/r8_math_corrections/package_sources.py
```

The package builder reuses the established 73-member source closure and
independently compiles both PDFs after extraction. It writes
`release/R8_20260911_math_corrections_sources.zip` and the current alias
`release/R8_LaTeX_sources.zip`. Their BUILD.txt contains standalone instructions.
The source package contains no financial data or numerical-analysis environment.

A changed PDF hash requires a new visual inspection before its receipt can be
accepted. The prior complete empirical archive is preserved; this small
correction does not create or certify a new full empirical release.
