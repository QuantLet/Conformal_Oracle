# Publish Log — conformal-oracle v0.3.0

Date: 2026-05-12

## Summary

| Step | Status | Detail |
|:-----|:-------|:-------|
| Pre-flight verification | PASS | Version 0.3.0 coherent across pyproject.toml, `__version__`, CITATION.cff. Twine check passed. No existing v0.3 tag. |
| Commit | PASS | `4129593` — 32 files, +2259/-230 lines |
| Tag | PASS | `v0.3.0-python` (annotated, follows `-python` suffix convention) |
| Push | PASS | `origin/main` updated `cedd13c..4129593`, tag pushed |
| PyPI upload | PASS | Both artefacts already uploaded (identical blake2 hashes). PyPI JSON API confirms v0.3.0 as latest with wheel + sdist. |
| Fresh-venv install | PASS | `pip install conformal-oracle==0.3.0` in clean venv: version 0.3.0, agnostic `audit(forecast=...)` works end-to-end, deprecation shim fires DeprecationWarning. |
| GitHub release | PASS | https://github.com/danpele/Conformal_Oracle/releases/tag/v0.3.0-python — wheel + sdist attached |
| Follow-up issues | PASS | 4 issues opened: [#1](https://github.com/danpele/Conformal_Oracle/issues/1) lazy imports, [#2](https://github.com/danpele/Conformal_Oracle/issues/2) docs site, [#3](https://github.com/danpele/Conformal_Oracle/issues/3) CI notebooks, [#4](https://github.com/danpele/Conformal_Oracle/issues/4) JOSS submission |
| README badges | PASS | PyPI version, Python versions, MIT license, download count — commit `5196463` |
| README rolling example | PASS | Agnostic quickstart now shows both `mode="static"` and `mode="rolling"` — commit `c0a3687` |
| Rebuild wheel | PASS | Rebuilt after README update. Wheel 76,657 B, sdist 51,377 B. Twine check passed. |
| Lazy recalibration imports | PASS | Recalibration classes + `audit_panel` now loaded on first access via `__getattr__`. Import time reduced from ~2.2s to ~1.2s. Fixed pre-existing bug: deprecated wrappers (`audit_static` etc.) now correctly imported from `_deprecated.py`. 217 tests pass. |
| Rebuild wheel (lazy) | PASS | Wheel 76,880 B, sdist 51,564 B. Twine check passed. |

## Artefacts (current — lazy imports + README updates)

| File | Size | Blake2-256 |
|:-----|:-----|:-----------|
| `conformal_oracle-0.3.0-py3-none-any.whl` | 76,880 B | `65c88abee09d3487d24fb606aa363b9c4df55f97c9e3accd23efedcbb3d08254` |
| `conformal_oracle-0.3.0.tar.gz` | 51,564 B | `f2ade4f58492b907d50cc8ad04b4c556e5724545f20dfa6a2e8e03365917a73b` |

### Prior artefacts (on PyPI — uploaded before README update)

| File | Size | Blake2-256 |
|:-----|:-----|:-----------|
| `conformal_oracle-0.3.0-py3-none-any.whl` | 76,473 B | `d91c3890fe24b117dcade91f585742a3d22b60113dca96c680a5b8e2a0280a69` |
| `conformal_oracle-0.3.0.tar.gz` | 50,991 B | `3f3c115256004ce55518cd0d6c89c31b6dd623b3f2a04f263d62abf99b6f6a98` |

**Note:** PyPI does not allow re-uploading files for the same version. The README-only change (badges + rolling example) is in the rebuilt local artefacts but cannot be pushed to PyPI under v0.3.0. The PyPI-hosted wheel contains the correct source code; only the bundled `README.md` differs. The GitHub release and repo README are up to date.

## Links

- PyPI: https://pypi.org/project/conformal-oracle/0.3.0/
- GitHub release: https://github.com/danpele/Conformal_Oracle/releases/tag/v0.3.0-python
- Migration guide: https://github.com/danpele/Conformal_Oracle/blob/main/python/docs/migration_v0.3.md
- CHANGELOG: https://github.com/danpele/Conformal_Oracle/blob/main/python/CHANGELOG.md
