# The `conformal-oracle` package, as a reader would find it

Checked 2026-08-31 against PyPI and against the published sdist, not against the
working tree.

| where | version | bootstrap replicates |
|---|---|---|
| PyPI, current release | 0.3.2 | `np.quantile(boot_sample, 1 - alpha)` — plain empirical quantile |
| repo `python/` `__version__` and `pyproject.toml` | 0.3.3 | corrected |
| repo `python/CHANGELOG.md`, newest entry | 0.3.4 | the entry that records the correction |
| repo `conformal-oracle/` | 0.3.1 | a stale duplicate tree |

## Three things this shows, and one of them was in the manuscript

**The manuscript said the defect held "through version 0.3.1" and was
corrected.** Both halves were wrong. 0.3.2 is documentation-only and
behaviourally identical to 0.3.1 — verified by downloading the published sdist
and reading `src/conformal_oracle/conformal/bootstrap.py:28`, which is
`qv_boots[b] = np.quantile(boot_sample, 1 - alpha)`. The correction is dated
0.3.4 in the changelog. So the defect is present in everything a reader can
install, and "this is corrected" pointed at a release that does not exist.
Section 12 now states what is true and says the correction is unpublished.

**No result in the paper depends on it.** The manuscript's intervals come from
`scripts/qV_block_bootstrap.py`, which is declared ORDER_STATISTIC at both of
its sites in `QV_CONVENTION_SITES.tsv` and checked by
`scripts/audit_qv_convention.py`. The registry is what makes that answerable in
one line instead of by reading the script.

**The package's own version metadata is inconsistent.** `__version__` and
`pyproject.toml` say 0.3.3 while the changelog's newest entry is 0.3.4, so the
bump for the bootstrap correction was never made. Publishing 0.3.4 would make
the manuscript's sentence true and is the clean resolution, but a release is an
outward-facing action and is left to the author.

**There are two package trees.** `conformal-oracle/` is 0.3.1 and `python/` is
0.3.3; they have diverged. The paper describes 0.3.1 as the version carrying a
defect, and a copy of exactly that version sits in the replication material
beside the corrected one. Deleting a tree is not a decision this note takes.

## 0.3.4, prepared 2026-08-31

The version bump the changelog implied was missing: `pyproject.toml` and
`__version__` said 0.3.3 while the newest changelog entry was 0.3.4. Both now
say 0.3.4.

- test suite: `PYTHONPATH=src pytest` exits 0
- `python -m build`: `conformal_oracle-0.3.4.tar.gz` and the wheel
- `twine check dist/*`: PASSED on both
- the shipped sdist carries the fix, verified by unpacking it rather than by
  reading the working tree: `src/conformal_oracle/conformal/bootstrap.py:36` is
  `qv_boots[b] = conformal_quantile(boot_sample, alpha)`

**Not uploaded.** No PyPI token is present on this machine, and a release is an
irreversible outward action: a version number on PyPI can never be reused. The
upload is the author's to run:

    cd python && python -m twine upload dist/*

Until it runs, PyPI still serves 0.3.2 and Section 12's sentence — which says
the correction is unpublished — is correct as printed. When it runs, that
sentence should be revised to say the correction shipped in 0.3.4.

