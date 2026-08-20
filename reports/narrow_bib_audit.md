# Bibliography audit — Phase 0

- Bibliography: `drafts/narrow_paper.bib`
- LaTeX sources scanned: `narrow_paper.md`
- Entries: **8**
- Verification: Crossref REST (`/works/{doi}`) for entries with a DOI; arXiv Atom API for preprints; entries with neither are **MANUAL** (a Crossref title search is run only to suggest a candidate DOI).
- **Nothing in the .bib has been modified by this script.**

## Status summary

| Status | Count | Meaning |
|---|---|---|
| MISMATCH | 0 | resolved but one or more fields disagree — needs a decision |
| UNRESOLVED | 0 | DOI/arXiv id present but did not resolve |
| MANUAL | 0 | no DOI or arXiv id — verify by hand |
| OK | 8 | resolved and all compared fields agree |

## All entries

| Status | Key | Our title | Source title | Our author/year/vol/pp | Source author/year/vol/pp | Flags |
|---|---|---|---|---|---|---|
| OK | `ansari2024chronos` | Chronos: Learning the Language of Time Series | Chronos: Learning the Language of Time Series | Ansari / 2024 / — / — | Ansari / 2024 / — / — |  |
| OK | `christoffersen1998evaluating` | Evaluating Interval Forecasts | Evaluating Interval Forecasts | Christoffersen / 1998 / 39 / 841-862 | Christoffersen / 1998 / 39 / 841 | Crossref records the first page only (841); our range 841-862 starts there — check the end page |
| OK | `engle2004caviar` | CAViaR: Conditional Autoregressive Value at Risk by Regression Quantil | CAViaR: Conditional Autoregressive Value at Risk by Regression Quantil | Engle / 2004 / 22 / 367-381 | Engle / 2004 / 22 / 367-381 |  |
| OK | `gibbs2021adaptive` | Adaptive Conformal Inference Under Distribution Shift | Adaptive Conformal Inference Under Distribution Shift | Gibbs / 2021 / 34 / 1660-1672 | Gibbs / 2021 / — / — |  |
| OK | `glosten1993relation` | On the Relation between the Expected Value and the Volatility of the N | On the Relation between the Expected Value and the Volatility of the N | Glosten / 1993 / 48 / 1779-1801 | GLOSTEN / 1993 / 48 / 1779-1801 |  |
| OK | `kupiec1995techniques` | Techniques for Verifying the Accuracy of Risk Measurement Models | Techniques for Verifying the Accuracy of Risk Measurement Models | Kupiec / 1995 / 3 / 73-84 | Kupiec / 1995 / 3 / 73-84 |  |
| OK | `lei2018distribution` | Distribution-Free Predictive Inference for Regression | Distribution-Free Predictive Inference for Regression | Lei / 2018 / 113 / 1094-1111 | Lei / 2018 / 113 / 1094-1111 |  |
| OK | `vovk2005algorithmic` | Algorithmic Learning in a Random World | Algorithmic Learning in a Random World | Vovk / 2005 / — / — | — / 2005 / — / — | container 'Springer' vs Crossref 'Springer-Verlag' — publisher/series naming, not an error |

## Citation cross-check

- Distinct `\cite*` keys found in the LaTeX sources: **0**
- Keys cited in text but **absent from the .bib**: **0**

- Entries present in the .bib but **never cited**: **8**

  - `ansari2024chronos`
  - `christoffersen1998evaluating`
  - `engle2004caviar`
  - `gibbs2021adaptive`
  - `glosten1993relation`
  - `kupiec1995techniques`
  - `lei2018distribution`
  - `vovk2005algorithmic`

---

Regenerate with `python scripts/audit_bib.py`. Responses are cached in `reports/.bib_audit_cache.json`; pass `--refresh` to re-query the APIs.
