# R8: focused reviewer revision

This release combines the current article and supplement with the inherited
numerical research archive. Its 100 focused AI reviews are indexed in
`docs/team_review_20260910/INDEX.md`; they are not external human endorsements.
`DECISIONS.md` records the changes adopted and the recommendations rejected.
Formal and bibliographic additions in `PROPOSED_SUBSTANTIVE_REVISION.md` are
proposals awaiting author authorization, not results integrated into the paper.

## Reproduce the documents

For a TeX-only build, extract `release/R8_LaTeX_sources.zip` into an empty
directory and follow its `BUILD.txt`. The source package includes the class,
bibliography and generated displays. It excludes the Python validators and
numerical archive. The validated toolchain is TeX Live 2026, pdfTeX 1.40.29
and latexmk 4.88. Four main/supplement/main/supplement passes resolve mutual
references. Portable compilation checks normalized PDF text and diagnostics;
it does not promise byte-identical PDFs on different machines.

## Validate the current complete archive

Run from the complete archive root, using Python 3.13.9 and the captured
`research/r8_team_review/requirements-validation.txt`. The actual package
versions and TeX executable versions are in
`artifacts/r8_team_review/environment.json`. This is a document and stored-result
validation environment, separate from the historical model-inference locks.
The root requirements files are not its dependency specification.

```sh
python -m pip install -r research/r8_team_review/requirements-validation.txt
python research/r8_team_review/validate.py
```

That verifies the archived current state. To rebuild the documents and replay
their display generation, use:

```sh
python source/scripts/extension_20260831/validate_r8.py
python research/r8_team_review/package_sources.py
```

Put the TeX executables on PATH first. The R8 wrapper selects current document
and producer scopes; standalone legacy guards retain historical defaults.
The rebuild sequence replays generated displays from stored results and
checks the current document build. It does not fit models or regenerate
financial observations. Source packaging also requires the included predecessor
`release/R8_20260910_commodity_sources.zip` for dependency closure.

The team validator checks all 100 report receipts, the before-source snapshot,
unchanged formal statements and generated results, exact current-source/package
agreement, document diagnostics, the recorded visual inspection, and an
independent replay of all 24 current EWMA forecast series. It checks that the
display producer's ownership repair changes neither its inputs nor its outputs.
The display and document checks include deliberate corruption controls.

The final receipt is `artifacts/r8_team_review/final_validation.json`.
`full_release.json`, issued alongside the completed archive, records its hash,
member verification and archive-only replay. A replay on the same host checks
relocation and dependency closure; it is not an independently repeated study on
another platform. The recorded visual inspection is human-facing inspection
by the same AI assistant, not automatically repeated by the numerical checker.
If rebuilding changes PDF bytes, the old visual receipt remains a record of
the archived PDFs. A new release needs a fresh visual inspection and receipt;
the validator deliberately rejects mixing newly built PDFs with that old hash.

## Numerical provenance and earlier versions

The new archive inherits numerical members from the immutable
`R8_20260910_novelty_audit.zip`; it overlays the current sources, PDFs,
validation code and receipts. The predecessor manifest is retained under
`historical/`. Earlier 50-page manuscript receipts remain accurate historical
records. Current status is in `docs/IRFA_REVIEW_STATE.md` and the team receipt;
do not run historical snapshot validators against newer prose and interpret
expected conservation failures as current numerical errors.

All archived observations, fits, forecast arrays, simulations and statistical
comparison families remain unchanged in this revision. The EWMA check recomputes
the stored recursion from its original inputs; it is not new model inference.
Fresh foundation-model inference and fresh financial fits were not performed
by this review. Public deposit remains pending; local packaging does not itself
publish data or establish the contents of the linked public repositories.
