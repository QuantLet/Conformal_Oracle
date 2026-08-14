# Bibliography audit — Phase 0

- Bibliography: `/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/Lppls-ai-ecosystem/paper/refs.bib`
- LaTeX sources scanned: `main.tex`, `preamble.tex`
- Entries: **21**
- Verification: Crossref REST (`/works/{doi}`) for entries with a DOI; arXiv Atom API for preprints; entries with neither are **MANUAL** (a Crossref title search is run only to suggest a candidate DOI).
- **Nothing in the .bib has been modified by this script.**

## Status summary

| Status | Count | Meaning |
|---|---|---|
| MISMATCH | 1 | resolved but one or more fields disagree — needs a decision |
| UNRESOLVED | 0 | DOI/arXiv id present but did not resolve |
| MANUAL | 5 | no DOI or arXiv id — verify by hand |
| OK | 15 | resolved and all compared fields agree |

## MISMATCH — full detail

### `sornette2015` (article, line 24)

Resolved via `arxiv:1509.00121`.

| Field | In our .bib | Returned by source |
|---|---|---|
| title ⚠️ | Real-Time Prediction and Post-Mortem Analysis of the Shanghai 2015 Stock Market Bubble and Crash | Tight fibered knots and band sums |
| author ⚠️ | Sornette | Baker |
| container ⚠️ | Journal of Investment Strategies | arXiv:1509.00121 |
| year | 2015 | 2015 |
| volume ⚠️ | 4 | — |
| pages ⚠️ | 77-95 | — |
| doi | — | — |

- **PROBLEM:** title mismatch vs arXiv (similarity 0.28)
- **PROBLEM:** first author 'Sornette' vs arXiv 'Baker'

## All entries

| Status | Key | Our title | Source title | Our author/year/vol/pp | Source author/year/vol/pp | Flags |
|---|---|---|---|---|---|---|
| MISMATCH | `sornette2015` | Real-Time Prediction and Post-Mortem Analysis of the Shanghai 2015 Sto | Tight fibered knots and band sums | Sornette / 2015 / 4 / 77-95 | Baker / 2015 / — / — | title mismatch vs arXiv (similarity 0.28); first author 'Sornette' vs arXiv 'Baker' |
| MANUAL | `boulder2024` | lppls: Python module for fitting the Log-Periodic Power Law Singularit | — | Technologies / 2024 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `goldmansachs2026` | Why AI Companies May Invest More than \$500 Billion in 2026 | — | Sachs / 2026 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `page1999` | The PageRank Citation Ranking: Bringing Order to the Web | — | Page / 1999 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `siliconangle2025` | OpenAI and Oracle Strike \$300B Cloud Computing Deal to Power AI | — | SiliconANGLE / 2025 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `sornette2003` | Why Stock Markets Crash: Critical Events in Complex Financial Systems | — | Sornette / 2003 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 2003 vs 2017 |
| OK | `acemoglu2012` | The Network Origins of Aggregate Fluctuations | The Network Origins of Aggregate Fluctuations | Acemoglu / 2012 / 80 / 1977-2016 | — / 2012 / 80 / 1977-2016 |  |
| OK | `battiston2012` | DebtRank: Too Central to Fail? Financial Networks, the FED and Systemi | DebtRank: Too Central to Fail? Financial Networks, the FED and Systemi | Battiston / 2012 / 2 / 541 | Battiston / 2012 / 2 / — |  |
| OK | `cao2025` | Identifying and Quantifying Financial Bubbles with the Hyped Log-Perio | Identifying and Quantifying Financial Bubbles with the Hyped Log-Perio | Cao / 2025 / — / — | Cao / 2025 / — / — |  |
| OK | `cont2013` | Running for the Exit: Distressed Selling and Endogenous Correlation in | RUNNING FOR THE EXIT: DISTRESSED SELLING AND ENDOGENOUS CORRELATION IN | Cont / 2013 / 23 / 718-741 | Cont / 2013 / 23 / 718-741 |  |
| OK | `demirer2018` | Estimating Global Bank Network Connectedness | Estimating global bank network connectedness | Demirer / 2018 / 33 / 1-15 | Demirer / 2018 / 33 / 1-15 |  |
| OK | `diebold2014` | On the Network Topology of Variance Decompositions: Measuring the Conn | On the network topology of variance decompositions: Measuring the conn | Diebold / 2014 / 182 / 119-134 | Diebold / 2014 / 182 / 119-134 |  |
| OK | `filimonov2013` | A Stable and Robust Calibration Scheme of the Log-Periodic Power Law M | A stable and robust calibration scheme of the log-periodic power law m | Filimonov / 2013 / 392 / 3698-3707 | Filimonov / 2013 / 392 / 3698-3707 |  |
| OK | `johansen2000` | Crashes as Critical Points | CRASHES AS CRITICAL POINTS | Johansen / 2000 / 3 / 219-255 | JOHANSEN / 2000 / 03 / 219-255 | volume 3 vs Crossref 03 (zero-padding only) |
| OK | `meng2026` | Artificial Intelligence and Systemic Risk: A Unified Model of Performa | Artificial Intelligence and Systemic Risk: A Unified Model of Performa | Meng / 2026 / — / — | Meng / 2026 / — / — |  |
| OK | `phillips2015` | Testing for Multiple Bubbles: Historical Episodes of Exuberance and Co | TESTING FOR MULTIPLE BUBBLES: HISTORICAL EPISODES OF EXUBERANCE AND CO | Phillips / 2015 / 56 / 1043-1078 | Phillips / 2015 / 56 / 1043-1078 |  |
| OK | `romano2019` | Conformalized Quantile Regression | Conformalized Quantile Regression | Romano / 2019 / 32 / — | Romano / 2019 / — / — |  |
| OK | `sarkar2026` | Is There an AI Bubble? Robust Date-Stamping for Periods of Exuberance | Is There an AI Bubble? Robust Date-Stamping for Periods of Exuberance | Sarkar / 2026 / — / — | Sarkar / 2026 / — / — |  |
| OK | `shu2020` | Detection of Chinese Stock Market Bubbles with LPPLS Confidence Indica | Detection of Chinese stock market bubbles with LPPLS confidence indica | Shu / 2020 / 557 / 124892 | Shu / 2020 / 557 / 124892 |  |
| OK | `sornette2014` | Financial Bubbles: Mechanisms and Diagnostics | Financial Bubbles: Mechanisms and Diagnostics | Sornette / 2015 / 2 / 279-305 | Sornette / 2015 / 2 / 279-305 |  |
| OK | `vovk2005` | Algorithmic Learning in a Random World | Algorithmic Learning in a Random World | Vovk / 2005 / — / — | — / 2005 / — / — | container 'Springer' vs Crossref 'Springer-Verlag' — publisher/series naming, not an error |

## MANUAL entries — candidate DOIs from Crossref title search

These are *suggestions only*. A candidate is listed when a Crossref title search returns a match with similarity ≥ 0.88. Confirm each before adding a DOI to the .bib.

| Key | Our entry | Candidate DOI | Candidate title | Candidate author/year/vol/pp | Sim | Disagreements |
|---|---|---|---|---|---|---|
| `boulder2024` | lppls: Python module for fitting the Log-Periodic Power | — | *no confident Crossref match* | — | — | verify by hand |
| `goldmansachs2026` | Why AI Companies May Invest More than \$500 Billion in  | — | *no confident Crossref match* | — | — | verify by hand |
| `page1999` | The PageRank Citation Ranking: Bringing Order to the We | — | *no confident Crossref match* | — | — | verify by hand |
| `siliconangle2025` | OpenAI and Oracle Strike \$300B Cloud Computing Deal to | — | *no confident Crossref match* | — | — | verify by hand |
| `sornette2003` | Why Stock Markets Crash: Critical Events in Complex Fin | `10.23943/princeton/9780691175959.001.0001` | Why Stock Markets Crash: Critical Events in Complex Fin | Sornette / 2017 / — / — | 1.0 | candidate match disagrees on: year 2003 vs 2017 |

## Citation cross-check

- Distinct `\cite*` keys found in the LaTeX sources: **0**
- Keys cited in text but **absent from the .bib**: **0**

- Entries present in the .bib but **never cited**: **21**

  - `acemoglu2012`
  - `battiston2012`
  - `boulder2024`
  - `cao2025`
  - `cont2013`
  - `demirer2018`
  - `diebold2014`
  - `filimonov2013`
  - `goldmansachs2026`
  - `johansen2000`
  - `meng2026`
  - `page1999`
  - `phillips2015`
  - `romano2019`
  - `sarkar2026`
  - `shu2020`
  - `siliconangle2025`
  - `sornette2003`
  - `sornette2014`
  - `sornette2015`
  - `vovk2005`

---

Regenerate with `python scripts/audit_bib.py`. Responses are cached in `reports/.bib_audit_cache.json`; pass `--refresh` to re-query the APIs.
