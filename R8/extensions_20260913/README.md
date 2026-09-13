# R8 extensions of 13 September 2026 — code and data

Three prespecified studies added to the R8 revision. Each folder under
`code/` holds the protocol (`PROTOCOL.md`, fixed before computation), the
producer and a `--check` replay; each folder under `data/` holds the
outputs the manuscript displays are generated from, and `RESULTS.md`.

| Study | Code | Data | Manuscript |
|---|---|---|---|
| Why the pooled loss bands include zero (nine prespecified exploratory contrasts) | `code/r8_power_analysis/` | `data/power_analysis/` | Section 6.1, Supplement S.4.11 |
| Nuisance-free optimism estimators and shrinkage rule (synthetic admission; rule fails its value criterion) | `code/r8_optimism_estimator/` | `data/optimism_estimator/` | Section 5.1, Supplement S.4.10 |
| Second external universe: Developed ex-US 25 size/value portfolios, 100 pairs | `code/r8_external2/` | `data/external2/` (raw source response, admission, results, provenance) | Section 7.4, Supplement S.5 |

`data/generated_tables/` holds the macro and table files produced by each
study's `displays.py`. The power analysis reads the stored 240-pair daily
losses of the main panel and the optimism study reads the stored synthetic
histories; both inputs are part of the full research archive, supplied on
request. The second external test's per-pair forecast and correction paths
(about 600 MB) are likewise supplied on request; its results, per-pair
metrics, bootstrap intervals and validation receipts are included here.
