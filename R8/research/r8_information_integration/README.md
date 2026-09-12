# Current R8: information limit integrated

The canonical article is `source/main_R8.tex`; the supplement is
`source/supplement_R8.tex`. Proposition 7.1 and its proof in S.3.8 give an
information lower bound for scalar correction in a constructed two-law
experiment. When the expected calibration-tail count stays finite, no
history-based scalar rule has negligible excess marginal loss relative
to the tail level under both laws. A separate transfer argument covers
static corrections in [0,1] on an increasing contiguous test block.

The article has 49 pages and the supplement 32. All preceding financial
results, displays, forecasts and numbered equations are preserved. Existing
theory numbering is unchanged; separated coverage remains Theorem 4.8.
The new Bayes identity is (S.13), and the partial-shift accounting moves to
(S.14). No new financial model or empirical superiority claim is added.

The bound uses classical testing and Bayes reductions. It is about a
constructed experiment, not an estimated information limit for financial
returns. Absolute excess loss can vanish. The contiguous extension keeps
one bounded correction fixed; it supplies no rolling coverage guarantee.

## Document and display reproduction

Use Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, arch 8.0.0,
pyarrow 21.0.0, Matplotlib 3.10.6, pypdf 6.14.2 and TeX Live. From the
project root:

```sh
python source/scripts/extension_20260831/build_paper_outputs.py
python research/r8_integration/build.py
python research/r8_regime/build_paper.py
python research/r8_ten_integration/build.py
python research/r8_partial_integration/build.py
python research/r8_information_integration/build.py
```

From `source`, compile `main_R8.tex`, then `supplement_R8.tex`, then both
again, each with `latexmk -g -pdf -interaction=nonstopmode -halt-on-error`.
Back at the project root:

```sh
python source/scripts/extension_20260831/validate_r8.py
python research/r8_ten_integration/package_sources.py
python research/r8_information_integration/validate.py --replay-study
```

The global suite checks 57 generated displays and corruption controls,
data/forecast structure, literal numbers, provenance, references and
compilation. The portable LaTeX package has 66 members; it is compiled in
isolation and both extracted PDF texts must match the canonical documents.
The integration validator independently recomputes the four new displayed
values, checks the preserved source scope, and replays the original study
validator from its immutable archive. The workspace lacks Git metadata;
source comparisons use `git diff --no-index --check` and a deliberate
whitespace negative control.

## Research reproduction

`release/R8_20260910_information_limit.zip` is a self-contained, immutable
study archive, included in the complete release. Extract it into a new
directory before following `research/r8_information_limit/README.md`.
Its original validator protects 109 pre-integration canonical files;
running it against the later manuscript should fail. The integration
validator extracts the study automatically and preserves these checks.

The independent study checks enumerate 18,504 full histories with rational
arithmetic, integrate risk directly, check 432 contiguous Markov identities
and verify all reported discrimination lengths. The deterministic primary
and fresh-process replay outputs agree exactly. No random paths, financial
fits, new observations or foundation-model forecasts were generated.

Earlier reproduction instructions remain in these directories:

- `research/r8_partial_integration/README.md` and the immutable
  `release/R8_20260910_partial_shift_study.zip`: selection-cost study.
- `research/r8_model_extension`: model inference and common support.
- `research/r8_ten_comparators/README.md`: comparator fits and replays.
- `research/r8_external/README.md`: external July industry evaluation.
- `research/r8_referee_revision/README.md`: simultaneous inference.
- `research/r8_horizon_bridge/README.md` and
  `research/r8_count_law/README.md`: contiguous expected loss and count laws.

Historical validators may protect their contemporary manuscript snapshots.
Use their documented extracted-copy procedure; do not weaken hash checks.
The present integration retains the preceding financial release without
reestimating its models. Main simultaneous bands still include zero; the
already inspected external sample is not new confirmation of this theory.

## Complete release

`package_release.py` overlays the current documents and new research on
the verified preceding full archive. It checks every member and repeats
the integration validation using only extracted archive bytes before
finalising `release/R8_20260910_information_integration.zip`. The receipt
is `artifacts/r8_information_integration/full_release.json` in the project.
The preceding release and the separate study archives are unchanged.

`release/R8_LaTeX_sources.zip` is the smaller document-only package. No
packaging operation publishes or submits the paper. The integration report
is `docs/IRFA_INFORMATION_INTEGRATION.md`.
