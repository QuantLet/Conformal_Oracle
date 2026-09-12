# R8: contiguous marginal coverage integrated

This release integrates the author's 11 September authorization of the
prepared mathematical and bibliographic revision. The preceding 100-agent
review remains an immutable historical record. Its pending proposals are now
superseded by the current sources and `docs/contiguous_integration_20260911`.

## Scientific scope

Corollary 4.9 bounds marginal coverage of the full contiguous calibration
threshold under continuous stationary score marginals and summable beta
mixing. Under a geometric envelope its deficit is of order n^(-1/2).
The proof uses a lower-ranked prefix; deployment retains every calibration
score. The same lower bound holds at later deterministic static test dates.
The rolling consequence is for deterministic window size and evaluation date,
with an admissible rank. Its displayed remainder shrinks with window size,
not with calendar time alone. This does not establish conditional coverage,
validate empirical score assumptions, or extend the static loss expansion to
rolling forecasts. The prior separated theorem and existing proofs are intact.

Related Literature now cites Besbes and Mouchtaki (2023), distinguishing their
iid order-statistic costs and minimax expected relative regret from dependent
histories, contiguous static loss and the specified tail-normalised experiment.
All empirical results, base forecasts, simulation paths, comparison families
and generated financial displays are unchanged.

## Verify the archived state

Use Python 3.13.9 and the captured dependencies listed in
`research/r8_team_review/requirements-validation.txt`; the corresponding
record is `artifacts/r8_team_review/environment.json`. This validation
environment is distinct from historical fresh-inference environments.
From the complete archive root:

```sh
python research/r8_contiguous_integration/validate.py
```

It checks the before-source snapshot; conservation of existing mathematical
blocks, proofs and empirical displays; exactly one new bibliography entry;
the independent review's hashes of the new proof, statement and scope;
current-source/package agreement; PDF diagnostics and visual receipt; and
deterministic rank checks with ties, capping and a wrong-rank negative control.
Finite enumeration supports the algebraic audit, not a proof of general validity.

The current status is in `artifacts/r8_contiguous_integration/final_validation.json`.
The complete-release receipt `full_release.json` is issued alongside the archive
only after member hashing and an archive-only validation replay succeed.
Earlier validation receipts apply to their historical sources and should not
be rerun against changed prose as if they checked the current release.

## Rebuild documents and displays

For TeX alone, extract `release/R8_LaTeX_sources.zip` and follow `BUILD.txt`.
It contains 68 source/display members including the bundled elsarticle class;
it does not contain the numerical archive or Python validators. The tested
TeX toolchain is TeX Live 2026, pdfTeX 1.40.29 and latexmk 4.88.

For the complete project, put TeX executables on PATH and run:

```sh
python source/scripts/extension_20260831/validate_r8.py
python research/r8_contiguous_integration/package_sources.py
```

The first command replays 57 generated displays from stored results and runs
four document corruption controls. The second rebuilds the portable source
package and verifies normalized PDF-text agreement after extraction. The
predecessor commodity source ZIP is included for dependency closure.
If regenerated PDF bytes change, a fresh visual inspection/receipt is required
before sealing another release; the checker rejects stale visual hashes.

This revision did not rerun financial fits, foundation-model inference or
simulations. Full numerical members are inherited with verified hashes from
`R8_20260910_team_review.zip`, with its manifest preserved under `historical/`.
The archive-only replay checks relocation on the same host; it is not an
independent study on another platform. Public deposit remains pending, and
no external human review or publication outcome is claimed.
