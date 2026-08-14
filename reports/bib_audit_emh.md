# Bibliography audit — Phase 0

- Bibliography: `/Users/danpele/Library/Mobile Documents/com~apple~CloudDocs/Documents/EMH/llm_emh_sim/paper/references.bib`
- LaTeX sources scanned: `paper.tex`
- Entries: **21**
- Verification: Crossref REST (`/works/{doi}`) for entries with a DOI; arXiv Atom API for preprints; entries with neither are **MANUAL** (a Crossref title search is run only to suggest a candidate DOI).
- **Nothing in the .bib has been modified by this script.**

## Status summary

| Status | Count | Meaning |
|---|---|---|
| MISMATCH | 0 | resolved but one or more fields disagree — needs a decision |
| UNRESOLVED | 0 | DOI/arXiv id present but did not resolve |
| MANUAL | 19 | no DOI or arXiv id — verify by hand |
| OK | 2 | resolved and all compared fields agree |

## All entries

| Status | Key | Our title | Source title | Our author/year/vol/pp | Source author/year/vol/pp | Flags |
|---|---|---|---|---|---|---|
| MANUAL | `brock1996simple` | A Test for Independence Based on the Correlation Dimension | — | Brock / 1996 / 15 / 197-235 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 1996 vs 2001; pages 197-235 vs 324-362 |
| MANUAL | `brogaard2014high` | High-Frequency Trading and Price Discovery | — | Brogaard / 2014 / 27 / 2267-2306 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 2014 vs 2012 |
| MANUAL | `chiarella2009impact` | The Impact of Heterogeneous Trading Rules on the Limit Order Book and  | — | Chiarella / 2009 / 33 / 525-537 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 2009 vs 2006 |
| MANUAL | `cont2001empirical` | Empirical Properties of Asset Returns: Stylized Facts and Statistical  | — | Cont / 2001 / 1 / 223-236 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `fama1970efficient` | Efficient Capital Markets: A Review of Theory and Empirical Work | — | Fama / 1970 / 25 / 383-417 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: pages 383-417 vs 383 |
| MANUAL | `farmer2002market` | The Price Dynamics of Common Trading Strategies | — | Farmer / 2002 / 49 / 149-171 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `grossman1980impossibility` | On the Impossibility of Informationally Efficient Markets | — | Grossman / 1980 / 70 / 393-408 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `hendershott2011does` | Does Algorithmic Trading Improve Liquidity? | — | Hendershott / 2011 / 66 / 1-33 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `hommes2006heterogeneous` | Heterogeneous Agent Models in Economics and Finance | — | Hommes / 2006 / 2 / 1109-1186 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `hong2007disagreement` | Disagreement and the Stock Market | — | Hong / 2007 / 21 / 109-128 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `kim2024financial` | Financial Statement Analysis with Large Language Models | — | Kim / 2024 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `kirilenko2017flash` | The Flash Crash: High-Frequency Trading in an Electronic Market | — | Kirilenko / 2017 / 72 / 967-998 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 2017 vs 2011 |
| MANUAL | `lebaron2006agent` | Agent-Based Computational Finance | — | LeBaron / 2006 / 2 / 1187-1233 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `lo1988stock` | Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple | — | Lo / 1988 / 1 / 41-66 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 1988 vs 1987 |
| MANUAL | `lo2004adaptive` | The Adaptive Markets Hypothesis | — | Lo / 2004 / 30 / 15-29 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `lux2000volatility` | Volatility Clustering in Financial Markets: A Microsimulation of Inter | — | Lux / 2000 / 3 / 675-702 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: volume 3 vs 03 |
| MANUAL | `park2023generative` | Generative Agents: Interactive Simulacra of Human Behavior | — | Park / 2023 / — / 1-22 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `storn1997differential` | Differential Evolution - A Simple and Efficient Heuristic for Global O | — | Storn / 1997 / 11 / 341-359 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `westerhoff2004multiasset` | Multiasset Market Dynamics | — | Westerhoff / 2004 / 8 / 596-616 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| OK | `li2023large` | Large Language Models in Finance: A Survey | Large Language Models in Finance: A Survey | Li / 2023 / — / — | Li / 2023 / — / — |  |
| OK | `lopez2023can` | Can ChatGPT Forecast Stock Price Movements? Return Predictability and  | Can ChatGPT Forecast Stock Price Movements? Return Predictability and  | Lopez-Lira / 2023 / — / — | Lopez-Lira / 2023 / — / — |  |

## MANUAL entries — candidate DOIs from Crossref title search

These are *suggestions only*. A candidate is listed when a Crossref title search returns a match with similarity ≥ 0.88. Confirm each before adding a DOI to the .bib.

| Key | Our entry | Candidate DOI | Candidate title | Candidate author/year/vol/pp | Sim | Disagreements |
|---|---|---|---|---|---|---|
| `brock1996simple` | A Test for Independence Based on the Correlation Dimens | `10.4337/9781782543046.00024` | A Test for Independence Based on the Correlation Dimens | Brock / 2001 / — / 324-362 | 1.0 | candidate match disagrees on: year 1996 vs 2001; pages 197-235 vs 324-362 |
| `brogaard2014high` | High-Frequency Trading and Price Discovery | `10.2139/ssrn.1938769` | High Frequency Trading and Price Discovery | Brogaard / 2012 / — / — | 1.0 | candidate match disagrees on: year 2014 vs 2012 |
| `chiarella2009impact` | The Impact of Heterogeneous Trading Rules on the Limit  | `10.2139/ssrn.893087` | The Impact of Heterogeneous Trading Rules on the Limit  | Chiarella / 2006 / — / — | 1.0 | candidate match disagrees on: year 2009 vs 2006 |
| `cont2001empirical` | Empirical Properties of Asset Returns: Stylized Facts a | `10.1080/713665670` | Empirical properties of asset returns: stylized facts a | Cont / 2001 / 1 / 223-236 | 1.0 | — |
| `fama1970efficient` | Efficient Capital Markets: A Review of Theory and Empir | `10.2307/2325486` | Efficient Capital Markets: A Review of Theory and Empir | Fama / 1970 / 25 / 383 | 1.0 | candidate match disagrees on: pages 383-417 vs 383 |
| `farmer2002market` | The Price Dynamics of Common Trading Strategies | `10.1016/s0167-2681(02)00065-3` | The price dynamics of common trading strategies | Farmer / 2002 / 49 / 149-171 | 1.0 | — |
| `grossman1980impossibility` | On the Impossibility of Informationally Efficient Marke | — | *no confident Crossref match* | — | — | verify by hand |
| `hendershott2011does` | Does Algorithmic Trading Improve Liquidity? | `10.1111/j.1540-6261.2010.01624.x` | Does Algorithmic Trading Improve Liquidity? | HENDERSHOTT / 2011 / 66 / 1-33 | 1.0 | — |
| `hommes2006heterogeneous` | Heterogeneous Agent Models in Economics and Finance | `10.1016/s1574-0021(05)02023-x` | Chapter 23 Heterogeneous Agent Models in Economics and  | Hommes / 2006 / — / 1109-1186 | 0.903 | — |
| `hong2007disagreement` | Disagreement and the Stock Market | `10.1257/jep.21.2.109` | Disagreement and the Stock Market | Hong / 2007 / 21 / 109-128 | 1.0 | — |
| `kim2024financial` | Financial Statement Analysis with Large Language Models | `10.2139/ssrn.4835311` | Financial Statement Analysis with Large Language Models | Kim / 2024 / — / — | 1.0 | — |
| `kirilenko2017flash` | The Flash Crash: High-Frequency Trading in an Electroni | `10.2139/ssrn.1686004` | The Flash Crash: High-Frequency Trading in an Electroni | Kirilenko / 2011 / — / — | 1.0 | candidate match disagrees on: year 2017 vs 2011 |
| `lebaron2006agent` | Agent-Based Computational Finance | — | *no confident Crossref match* | — | — | verify by hand |
| `lo1988stock` | Stock Market Prices Do Not Follow Random Walks: Evidenc | `10.3386/w2168` | Stock Market Prices Do Not Follow Random Walks: Evidenc | Lo / 1987 / — / — | 1.0 | candidate match disagrees on: year 1988 vs 1987 |
| `lo2004adaptive` | The Adaptive Markets Hypothesis | `10.3905/jpm.2004.442611` | The Adaptive Markets Hypothesis | Lo / 2004 / 30 / 15-29 | 1.0 | — |
| `lux2000volatility` | Volatility Clustering in Financial Markets: A Microsimu | `10.1142/s0219024900000826` | VOLATILITY CLUSTERING IN FINANCIAL MARKETS: A MICROSIMU | LUX / 2000 / 03 / 675-702 | 1.0 | candidate match disagrees on: volume 3 vs 03 |
| `park2023generative` | Generative Agents: Interactive Simulacra of Human Behav | `10.1145/3586183.3606763` | Generative Agents: Interactive Simulacra of Human Behav | Park / 2023 / — / 1-22 | 1.0 | — |
| `storn1997differential` | Differential Evolution - A Simple and Efficient Heurist | `10.1023/a:1008202821328` | Differential Evolution – A Simple and Efficient Heurist | Storn / 1997 / 11 / 341-359 | 1.0 | — |
| `westerhoff2004multiasset` | Multiasset Market Dynamics | `10.1017/s1365100504040040` | MULTIASSET MARKET DYNAMICS | WESTERHOFF / 2004 / 8 / 596-616 | 1.0 | — |

## Citation cross-check

- Distinct `\cite*` keys found in the LaTeX sources: **18**
- Keys cited in text but **absent from the .bib**: **0**

- Entries present in the .bib but **never cited**: **3**

  - `grossman1980impossibility`
  - `hommes2006heterogeneous`
  - `lux2000volatility`

---

Regenerate with `python scripts/audit_bib.py`. Responses are cached in `reports/.bib_audit_cache.json`; pass `--refresh` to re-query the APIs.
