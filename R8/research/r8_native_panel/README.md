# Current R8: forecasts without central-grid tail completion

The manuscript now uses seven base forecasters and 24 assets (168 pairs).
`source/scripts/extension_20260831/paper_scope.py` is the canonical panel
definition. The original nine-model archive and inference/fitting scripts
remain reproducible and are not silently relabelled as the current panel.

Run in the recorded analysis environment, with TeX Live on PATH:

```sh
python research/r8_native_panel/reaggregate.py
python research/r8_native_panel/reaggregate.py --check
python research/r8_native_panel/validate.py
python source/scripts/extension_20260831/build_paper_outputs.py
python research/r8_integration/build.py
python research/r8_regime/build_paper.py
```

Then rebuild `main_R8.tex`, `supplement_R8.tex`, and each once again from
`source`, using `latexmk -g -pdf -interaction=nonstopmode -halt-on-error`.
Finally run `python source/scripts/extension_20260831/validate_r8.py`.

Aggregation reads the original stored daily forecasts and fitted corrections;
it makes no new model fit and changes no pair-level statistic or test date.
It regenerates calendar-block uncertainty at the two original block lengths.
Its manifest binds 954 input files and 55 derived files. Exact replay and
a corrupted-output negative control are supplied. Empirical results are in
`artifacts/r8_native_panel/base/results` and
`artifacts/r8_native_panel/decision`. The 168-pair restriction was adopted
after seeing the broader panel and must be described as retrospective.

The prior forecast archive's validation remains evidence about its inputs;
current-paper validation is `artifacts/r8_native_panel/base/quality/r8_validation.json`.
The source ZIP is an editorial/build deliverable, not a full numerical
replication archive. Previous full release ZIPs remain immutable. The full
release producer has been updated to include the selected-panel dependencies,
but no new full release is claimed by this revision.

Candidate-model testing is separate in `research/r8_native_candidates`.
No candidate forecast is an input to this 168-pair manuscript build.
