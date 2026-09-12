# R8 financial argument and literature positioning

Only the prose in `introduction.tex`, `results.tex` and `discussion.tex`
changes. Current scope, evidence and literature links are documented in
`docs/financial_argument_20260911/SUMMARY.md`. Formal mathematics, numeric
macros, tables, figures and estimator implementations are preserved.

Use the project's captured validation environment and TeX Live on PATH:

```sh
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/main_R8.tex
latexmk -cd -g -pdf -interaction=nonstopmode -halt-on-error source/supplement_R8.tex
python source/scripts/extension_20260831/validate_r8.py
python research/r8_shape_integration/displays.py --check
python research/r8_financial_argument/package_sources.py
python research/r8_financial_argument/validate.py
```

The package builder reuses the existing 73-member closure and independently
compiles after extraction. The final validator checks conservation, evidence
bindings, current guards, current source/PDF/package agreement, and the visual
receipt. A changed PDF requires a new visual inspection. These commands
perform stored-result checks, not model fitting or simulations.

Historical audit reports and version-specific validators retain their original
source bindings. The current validation receipt is
`artifacts/r8_financial_argument/final_validation.json`. The portable ZIP
contains document sources and figures, not the full empirical archive.
