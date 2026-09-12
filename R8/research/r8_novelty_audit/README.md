# Current R8: novelty positioning and mathematical audit

The article is `source/main_R8.tex` (50 pages); the supplement is
`source/supplement_R8.tex` (32 pages). Related Literature now compares the
paper with seven additional primary references. The mathematical
statements, proofs, financial numbers, tables and figures are unchanged.
`docs/IRFA_NOVELTY_MATH_AUDIT.md` contains the source comparison, proof
audit and an alternative zero-positive-count proof of Proposition 7.1.

The audit found no fatal gap in the four reviewed loss/decision results.
It distinguishes specific correction-loss results from established
quantile asymptotics, shrinkage, pairwise-independent constructions and
testing reductions. It is performed by the same AI assistant with
independent code and derivations. It is not an external human review or
a certification of novelty or publication readiness. The prepared external
review brief is `docs/IRFA_EXTERNAL_MATH_REVIEW_BRIEF.md`; nothing was sent.

## Reproduce this audit

Use Python 3.13.9, NumPy 2.3.5, pandas 2.3.3 and SciPy 1.16.3. The
document validator also uses pypdf 6.14.2. From the project root:

```sh
python research/r8_novelty_audit/check.py
python research/r8_novelty_audit/validate.py
```

The first command uses closed renewal-block compositions and rational
integration, independently of the original recurrence. It also runs 180
nonreversible Markov transfer checks, 120 exact-uniform-moment/local-risk
checks and 36 rare-event zero-count checks, with four deliberate incorrect
expressions rejected. It imports no existing numerical producer. A fresh
process must reproduce its receipt and CSV files exactly. No random paths,
financial fits or foundation-model forecasts are generated.

The second checks the current source scope, equation preservation, appended
bibliography, document guards, portable source build receipt and visual
inspection hashes. It reruns the mathematics and preserves all previous
validation receipts. Git metadata is absent; whitespace validation uses
`git diff --no-index --check` with a deliberate failing control.

## Rebuild documents and displays

Use the pinned environment above plus arch 8.0.0, pyarrow 21.0.0,
Matplotlib 3.10.6 and TeX Live. Generate displays from the archived inputs:

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
From the project root:

```sh
python source/scripts/extension_20260831/validate_r8.py
python research/r8_ten_integration/package_sources.py
```

The global suite checks all 57 generated displays and corruption controls,
four document guards and their four negative controls. The 66-member
portable package is compiled in isolation and both PDF texts must match
the canonical documents. PDF hashes change on some rebuilds; after
rebuilding, repeat visual inspection and update its current receipt before
running the current audit validator. Do not alter historical receipts.

## Complete release and historical research

`package_release.py` overlays this audit and the current documents on the
verified preceding information-integration archive. Every member is
checked, including unchanged historical material. It then runs this audit
validator using only extracted archive files and requires exact agreement
before finalising `release/R8_20260910_novelty_audit.zip`. The receipt is
`artifacts/r8_novelty_audit/full_release.json` in the working project.

The immutable information-limit and partial-shift study archives remain
included. Their original validators bind contemporary manuscript snapshots;
follow their extraction/restoration instructions rather than disabling
those checks. `research/r8_information_integration/README.md` documents the
preceding research and financial reconstruction chain. Its source hashes
are historical after this literature edit.

The latest `release/R8_LaTeX_sources.zip` is the smaller document-only
package. Financial fits and inference are preserved from the preceding
validated release, not all rerun here. Main simultaneous bands still
include zero. No public deposit, submission or external message is made
by these packaging commands.
