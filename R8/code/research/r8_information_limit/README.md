# Information limits: research study, R8

This directory contains a complete two-law analysis of the information
needed to choose a useful tail correction. The manuscript is unchanged.
Read `PROOF.md` for the propositions, proofs, range restrictions and
attribution. The binary result is sharp for equal-prior average regret;
the scalar result is an exact Bayes lower bound on minimax risk, not a
claim of minimax attainment. A labelled later addendum treats contiguous
static evaluation. No result covers rolling or per-date conformal coverage.

The fixed protocol uses 64 configurations and computes their likelihood
overlaps analytically. It generates no observations, simulated paths,
financial fits or foundation-model predictions. The current R8 sources,
PDFs and previous validation records remain protected by 109 hashes.

## Reproduce

Use Python 3.13.9, NumPy 2.3.5, pandas 2.3.3 and SciPy 1.16.3.
Figures use Matplotlib 3.10.6 and Pillow. From the project root:

```sh
python research/r8_information_limit/validate.py
```

The validator enumerates short full histories with exact rational
arithmetic, checks continuous risk integrals, verifies the contiguous
kernel identity using an independent transition-matrix calculation,
checks all minimum lengths and retains negative lower bounds.

To reproduce from scratch, extract the separate study archive to a new
directory, rename its `artifacts/r8_information_limit/run` and `replay`
directories to preserve them, then execute:

```sh
python research/r8_information_limit/run.py
python research/r8_information_limit/run.py --replay
python research/r8_information_limit/contiguous.py
python research/r8_information_limit/validate.py
python research/r8_information_limit/plot.py
python research/r8_information_limit/report.py
```

Production refuses to overwrite either numeric run directory. Every
configuration is reported. Exact computational reproducibility requires
the recorded environment; this does not imply identical bytes for every
other NumPy/SciPy version.

`package.py` creates a separate self-contained archive, verifies every
member, and repeats validation, contiguous calculations, figures and
report generation using only extracted bytes. Its receipt is
`artifacts/r8_information_limit/package.json` in the working project.
The archive is a research companion to the full financial release, not
its replacement. It carries historical canonical snapshots so their
integrity checks remain meaningful after a later manuscript integration.

## Scope

The constructed chain repeats its score exactly between refreshes. It
has bounded continuous margins and geometric beta-mixing, but is not
a fitted financial process. All candidate laws and the complete history
are supplied to the ideal selector. The count of independent refresh
values is observable almost surely and is ancillary to the tail mass.
Discrimination lengths are model-specific information benchmarks, not
recommended calibration windows. Classical testing/Bayes reductions
are attributed and no first-ever claim is made.

The primary protocol predates computation. The contiguous addendum was
derived after the primary calculation and before its own output was
computed. It is retained as a transparent theory extension, not described
as untouched empirical confirmation. The financial findings have not
been used to tune this grid.
