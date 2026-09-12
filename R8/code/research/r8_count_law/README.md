# Exact finite-sample limitation of pairwise dependence diagnostics

10 September 2026. Proposition 4.3 and its proof in S.3.7 use a stationary
renewal construction. At a 1% tail and 125 calibration scores, two processes
have identical marginal and lagged pairwise distributions, but full
correction has opposite effects on expected independent-marginal pinball
loss. The example is deliberately near break-even; the effects are small.
It is an existence result, not an estimated financial decision rule.

`PROOF.md` explains the construction. `exact_witness.py` verifies the signs
using integer and rational arithmetic. It also checks simple analytic
cases. `validate.py` independently checks the cost using a killed Markov
kernel and numerical integration, and verifies source scope and the
current document build. Neither script generates random observations or
fits models.

## Reproduce the exact witness and current verification

Use the project's recorded Python environment (Python 3.13.9, NumPy 2.3.5,
SciPy 1.16.3, pandas 2.3.3 and pypdf 6.14.2). From the project root:

```sh
python research/r8_count_law/exact_witness.py
python research/r8_count_law/validate.py
```

The exact certificate is idempotent: an existing result must agree exactly.
The validator checks all 48 diagnostic rows against their fresh-process
replay, all recorded input hashes, the source snapshot and the unchanged
preceding validation receipt. It checks the two portable source ZIPs, PDF
hashes, title-page contents, absence of blank pages and the repaired
footnote destination. Rebuild instructions for both documents are in
`research/r8_ten_integration/README.md`.

## Independently replay the Normal-margin diagnostic

`PROTOCOL.md` fixes the complete 48-cell grid before its computation.
`calculate.py` uses a renewal count recursion and deterministic quadrature;
it does not draw paths. The Uniform-margin exact witness was derived
subsequently and is a separate mathematical construction. All diagnostic
cells remain available, including the small effects.

The calculation intentionally refuses to overwrite an existing output
directory. To repeat it without modifying the archive, make a disposable
project copy. In that copy only, rename `artifacts/r8_count_law_replay` to
`artifacts/r8_count_law_replay_saved`, then run:

```sh
python research/r8_count_law/calculate.py --replay
python research/r8_count_law/validate.py
```

The fresh process compares its entire output receipt with the original.
The original `artifacts/r8_count_law` directory must be retained: it holds
the comparison result and source snapshot. No financial-data retrieval,
foundation-model inference or empirical re-estimation is involved.

## Figure and release

`figure.py` creates a transparent PNG and SVG with the legend below the
plot. It is an explanatory artifact outside the journal PDFs; it adds no
manuscript pages. Set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable cache
directories if necessary. Its input and output hashes are in `figure.json`.

The current portable document package is `release/R8_LaTeX_sources.zip`.
`package_release.py` creates `release/R8_20260910_count_law.zip` by checking
and extending the preceding complete release. That archival construction
requires the predecessor ZIP on the maintainer's machine; numerical and
document reproduction from the completed archive do not require that ZIP.
The completed archive includes the small source-package input ZIP needed
by the existing portable-packaging command. Every member is hashed and
read back before the completion receipt is written to
`artifacts/r8_count_law/full_release.json`. Older releases are preserved.

See `docs/IRFA_COUNT_LAW_RESULT.md` for the literature comparison, numerical
checks, interpretation and remaining empirical limits. Historical validators
retain their original manuscript snapshots; use their documented disposable
copy procedure rather than relaxing their provenance checks.
