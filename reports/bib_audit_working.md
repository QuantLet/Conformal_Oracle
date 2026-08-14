# Bibliography audit — Phase 0

- Bibliography: `calibrating_the_oracle.bib`
- LaTeX sources scanned: `20260517_Recalibrating_Tail_Event_Forecasts_under_Temporal_Dependence_revised_v22_anon_fixed.tex`
- Entries: **57**
- Verification: Crossref REST (`/works/{doi}`) for entries with a DOI; arXiv Atom API for preprints; entries with neither are **MANUAL** (a Crossref title search is run only to suggest a candidate DOI).
- **Nothing in the .bib has been modified by this script.**
- NOTE: run in `--no-network` mode; results come from the local cache.

## Status summary

| Status | Count | Meaning |
|---|---|---|
| MISMATCH | 1 | resolved but one or more fields disagree — needs a decision |
| UNRESOLVED | 0 | DOI/arXiv id present but did not resolve |
| MANUAL | 23 | no DOI or arXiv id — verify by hand |
| OK | 33 | resolved and all compared fields agree |

## MISMATCH — full detail

### `brehmer2021properification` (article, line 220)

Resolved via `crossref:10.1214/21-EJS1913`.

| Field | In our .bib | Returned by source |
|---|---|---|
| title ⚠️ | Properification--postprocessing of quantile and interval forecasts | Rate of estimation for the stationary distribution of jump-processes over anisotropic Holder classes |
| author ⚠️ | Brehmer | Amorino |
| container | Electronic Journal of Statistics | Electronic Journal of Statistics |
| year | 2021 | 2021 |
| volume | 15 | 15 |
| pages ⚠️ | 3692-3720 | — |
| doi | 10.1214/21-EJS1913 | 10.1214/21-ejs1913 |

- **PROBLEM:** TITLE MISMATCH (similarity 0.40): DOI resolves to a different work
- **PROBLEM:** first author 'Brehmer' vs Crossref 'Amorino'

## All entries

| Status | Key | Our title | Source title | Our author/year/vol/pp | Source author/year/vol/pp | Flags |
|---|---|---|---|---|---|---|
| MISMATCH | `brehmer2021properification` | Properification--postprocessing of quantile and interval forecasts | Rate of estimation for the stationary distribution of jump-processes o | Brehmer / 2021 / 15 / 3692-3720 | Amorino / 2021 / 15 / — | TITLE MISMATCH (similarity 0.40): DOI resolves to a different work; first author 'Brehmer' vs Crossref 'Amorino' |
| MANUAL | `acerbi2014back` | Back-Testing Expected Shortfall | — | Acerbi / 2014 / 27 / 76-81 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `angelopoulos2024conformal` | Conformal Risk Control | — | Angelopoulos / 2024 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable; arXiv preprint found by title: arXiv:2208.02814v4 (2022, similarity 1.0) |
| MANUAL | `angelopoulos2024pid` | Conformal PID Control for Time Series Prediction | — | Angelopoulos / 2024 / 36 / — | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 2024 vs 2023 |
| MANUAL | `ansari2024chronos` | Chronos: Learning the language of time series | — | Ansari / 2024 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable; arXiv preprint found by title: arXiv:2403.07815v3 (2024, similarity 1.0) |
| MANUAL | `basel1996` | Supervisory Framework for the Use of ``Backtesting'' in Conjunction wi | — | Supervision / 1996 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `bcbs2019frtb` | Minimum Capital Requirements for Market Risk | — | Supervision / 2019 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `bollerslev1986generalized` | Generalized autoregressive conditional heteroskedasticity | — | Bollerslev / 1986 / 31 / 307-327 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `das2024timesfm` | A decoder-only foundation model for time-series forecasting | — | Das / 2024 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable; arXiv preprint found by title: arXiv:2310.10688v4 (2023, similarity 1.0) |
| MANUAL | `gibbs2021adaptive` | Adaptive Conformal Inference Under Distribution Shift | — | Gibbs / 2021 / 34 / 1660-1672 | — | no DOI and no arXiv id in entry — not automatically verifiable; arXiv preprint found by title: arXiv:2106.00170v3 (2021, similarity 1.0) |
| MANUAL | `glosten1993relation` | On the relation between the expected value and the volatility of the n | — | Glosten / 1993 / 48 / 1779-1801 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `gneiting2007probabilistic` | Probabilistic forecasts, calibration and sharpness | — | Gneiting / 2007 / 69 / 243-268 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: year 2007 vs 2005 |
| MANUAL | `gneiting2014probabilistic` | Probabilistic forecasting | — | Gneiting / 2014 / 1 / 125-151 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `hazan2016introduction` | Introduction to Online Convex Optimization | — | Hazan / 2016 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `koenker2005quantile` | Quantile Regression | — | Koenker / 2005 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `lindner2009stationarity` | Stationarity, mixing, distributional properties and moments of GARCH(p | — | Lindner / 2009 / — / 43-69 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `newey1987simple` | A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelati | — | Newey / 1987 / 55 / 703-708 | — | no DOI and no arXiv id in entry — not automatically verifiable; candidate match disagrees on: pages 703-708 vs 703 |
| MANUAL | `politis1994stationary` | The Stationary Bootstrap | — | Politis / 1994 / 89 / 1303-1313 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `riskmetrics1996` | RiskMetrics -- Technical Document | — | J.P. Morgan/Reuters / 1996 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `romano2019conformalized` | Conformalized Quantile Regression | — | Romano / 2019 / 32 / — | — | no DOI and no arXiv id in entry — not automatically verifiable; arXiv preprint found by title: arXiv:1905.03222v1 (2019, similarity 1.0) |
| MANUAL | `woo2024moirai` | Unified training of universal time series forecasting transformers | — | Woo / 2024 / — / — | — | no DOI and no arXiv id in entry — not automatically verifiable; arXiv preprint found by title: arXiv:2402.02592v2 (2024, similarity 1.0) |
| MANUAL | `zaffran2022conformal` | Adaptive Conformal Predictions for Time Series | — | Zaffran / 2022 / 162 / 25834-25866 | — | no DOI and no arXiv id in entry — not automatically verifiable; arXiv preprint found by title: arXiv:2202.07282v1 (2022, similarity 1.0) |
| MANUAL | `zinkevich2003online` | Online Convex Programming and Generalized Infinitesimal Gradient Ascen | — | Zinkevich / 2003 / — / 928-936 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| MANUAL | `zinkevich2003online` | Online Convex Programming and Generalized Infinitesimal Gradient Ascen | — | Zinkevich / 2003 / — / 928-935 | — | no DOI and no arXiv id in entry — not automatically verifiable |
| OK | `ansari2025chronos2` | Chronos-2: From Univariate to Universal Forecasting | Chronos-2: From Univariate to Universal Forecasting | Ansari / 2025 / — / — | Ansari / 2025 / — / — |  |
| OK | `barber2023conformal` | Conformal Prediction Beyond Exchangeability | Conformal prediction beyond exchangeability | Barber / 2023 / 51 / 816-845 | Barber / 2023 / 51 / — |  |
| OK | `baroneadesi1999var` | VaR without Correlations for Portfolios of Derivative Securities | VaR without correlations for portfolios of derivative securities | Barone-Adesi / 1999 / 19 / 583-602 | Barone-Adesi / 1999 / 19 / 583-602 |  |
| OK | `bollerslev1987` | A Conditionally Heteroskedastic Time Series Model for Speculative Pric | A Conditionally Heteroskedastic Time Series Model for Speculative Pric | Bollerslev / 1987 / 69 / 542-547 | Bollerslev / 1987 / 69 / 542 | Crossref records the first page only (542); our range 542-547 starts there — check the end page |
| OK | `chernozhukov2018exact` | Exact and Robust Conformal Inference Methods for Predictive Machine Le | Exact and Robust Conformal Inference Methods for Predictive Machine Le | Chernozhukov / 2018 / 75 / 732-749 | Chernozhukov / 2018 / — / — |  |
| OK | `christoffersen1998` | Evaluating Interval Forecasts | Evaluating Interval Forecasts | Christoffersen / 1998 / 39 / 841-862 | Christoffersen / 1998 / 39 / 841 | Crossref records the first page only (841); our range 841-862 starts there — check the end page |
| OK | `diebold1995comparing` | Comparing predictive accuracy | Comparing Predictive Accuracy | Diebold / 1995 / 13 / 253-263 | Diebold / 1995 / 13 / 253-263 |  |
| OK | `driscoll1998consistent` | Consistent Covariance Matrix Estimation with Spatially Dependent Panel | Consistent Covariance Matrix Estimation with Spatially Dependent Panel | Driscoll / 1998 / 80 / 549-560 | Driscoll / 1998 / 80 / 549-560 |  |
| OK | `engle2004caviar` | CAViaR: Conditional Autoregressive Value at Risk by Regression Quantil | CAViaR: Conditional Autoregressive Value at Risk by Regression Quantil | Engle / 2004 / 22 / 367-381 | Engle / 2004 / 22 / 367-381 |  |
| OK | `fissler2016higher` | Higher Order Elicitability and Osband's Principle | Higher order elicitability and Osband’s principle | Fissler / 2016 / 44 / 1680-1707 | Fissler / 2016 / 44 / — |  |
| OK | `francq2019garch` | GARCH Models: Structure, Statistical Inference and Financial Applicati | GARCH Models: Structure, Statistical Inference and Financial Applicati | Francq / 2019 / — / — | Francq / 2019 / — / — | container 'John Wiley & Sons' vs Crossref 'Wiley' — publisher/series naming, not an error |
| OK | `gneiting2011making` | Making and Evaluating Point Forecasts | Making and Evaluating Point Forecasts | Gneiting / 2011 / 106 / 746-762 | Gneiting / 2011 / 106 / 746-762 |  |
| OK | `gneiting2013combining` | Combining predictive distributions | Combining predictive distributions | Gneiting / 2013 / 7 / 1747-1782 | Gneiting / 2013 / 7 / — |  |
| OK | `goel2024tsfmvar` | Time-series foundation model for Value-at-Risk forecasting | Time-Series Foundation AI Model for Value-at-Risk Forecasting | Goel / 2024 / — / — | Goel / 2024 / — / — |  |
| OK | `goel2025volatility` | Foundation Time-Series AI Model for Realized Volatility Forecasting | Foundation Time-Series AI Model for Realized Volatility Forecasting | Goel / 2025 / — / — | Goel / 2025 / — / — |  |
| OK | `hansen1994autoregressive` | Autoregressive Conditional Density Estimation | Autoregressive Conditional Density Estimation | Hansen / 1994 / 35 / 705-730 | Hansen / 1994 / 35 / 705 | Crossref records the first page only (705); our range 705-730 starts there — check the end page |
| OK | `harvey1997testing` | Testing the equality of prediction mean squared errors | Testing the equality of prediction mean squared errors | Harvey / 1997 / 13 / 281-291 | Harvey / 1997 / 13 / 281-291 |  |
| OK | `kupiec1995` | Techniques for Verifying the Accuracy of Risk Measurement Models | Techniques for Verifying the Accuracy of Risk Measurement Models | Kupiec / 1995 / 3 / 73-84 | Kupiec / 1995 / 3 / 73-84 |  |
| OK | `liu2024moirai2` | Moirai-MoE: Empowering time series foundation models with sparse mixtu | Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixtu | Liu / 2024 / — / — | Liu / 2024 / — / — |  |
| OK | `liu2024moiraimoe` | Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixtu | Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixtu | Liu / 2024 / — / — | Liu / 2024 / — / — |  |
| OK | `mcneil2000estimation` | Estimation of Tail-Related Risk Measures for Heteroscedastic Financial | Estimation of tail-related risk measures for heteroscedastic financial | McNeil / 2000 / 7 / 271-300 | McNeil / 2000 / 7 / 271-300 |  |
| OK | `newey1994automatic` | Automatic Lag Selection in Covariance Matrix Estimation | Automatic Lag Selection in Covariance Matrix Estimation | Newey / 1994 / 61 / 631-653 | Newey / 1994 / 61 / 631-653 |  |
| OK | `nolde2017elicitability` | Elicitability and Backtesting: Perspectives for Banking Regulation | Elicitability and backtesting: Perspectives for banking regulation | Nolde / 2017 / 11 / 1833-1874 | Nolde / 2017 / 11 / — |  |
| OK | `pele2025llmvar` | In the Beginning was the Word: LLM-VaR and LLM-ES | In the beginning was the Word: LLM-VaR and LLM-ES | Pele / 2026 / 295 / 128676 | Pele / 2026 / 295 / 128676 |  |
| OK | `rahimikia2025revisiting` | Re(Visiting) Time Series Foundation Models in Finance | Re(Visiting) Time Series Foundation Models in Finance | Rahimikia / 2025 / — / — | Rahimikia / 2025 / — / — | now published; journal DOI 10.2139/ssrn.577056 available — consider updating |
| OK | `rasul2024lagllama` | Lag-Llama: Towards foundation models for probabilistic time series for | Lag-Llama: Towards Foundation Models for Probabilistic Time Series For | Rasul / 2024 / — / — | Rasul / 2023 / — / — | year 2024 vs arXiv v1 posting 2023 |
| OK | `rigby2005generalized` | Generalized Additive Models for Location, Scale and Shape | Generalized Additive Models for Location, Scale and Shape | Rigby / 2005 / 54 / 507-554 | Rigby / 2005 / 54 / 507-554 |  |
| OK | `rio2017asymptotic` | Asymptotic Theory of Weakly Dependent Random Processes | Asymptotic Theory of Weakly Dependent Random Processes | Rio / 2017 / — / — | Rio / 2017 / — / — | container 'Springer' vs Crossref 'Probability Theory and Stochastic Modelling' — publisher/series naming, not an error |
| OK | `taillardat2016calibrated` | Calibrated ensemble forecasts using quantile regression forests and en | Calibrated Ensemble Forecasts Using Quantile Regression Forests and En | Taillardat / 2016 / 144 / 2375-2393 | Taillardat / 2016 / 144 / 2375-2393 |  |
| OK | `tamingtailrisk2026` | Taming Tail Risk in Financial Markets: Conformal Risk Control for Nons | Taming Tail Risk in Financial Markets: Conformal Calibration for Nonst | Schmitt / 2026 / — / — | Schmitt / 2026 / — / — |  |
| OK | `vovk2005algorithmic` | Algorithmic Learning in a Random World | Algorithmic Learning in a Random World | Vovk / 2005 / — / — | — / 2005 / — / — | container 'Springer' vs Crossref 'Springer-Verlag' — publisher/series naming, not an error |
| OK | `xu2024conformal` | Conformal Prediction for Time Series | Conformal Prediction for Time Series | Xu / 2023 / 45 / 11575-11587 | Xu / 2023 / 45 / 11575-11587 |  |
| OK | `yu1994rates` | Rates of convergence for empirical processes of stationary mixing sequ | Rates of Convergence for Empirical Processes of Stationary Mixing Sequ | Yu / 1994 / 22 / 94-116 | Yu / 1994 / 22 / — |  |

## MANUAL entries — candidate DOIs from Crossref title search

These are *suggestions only*. A candidate is listed when a Crossref title search returns a match with similarity ≥ 0.88. Confirm each before adding a DOI to the .bib.

| Key | Our entry | Candidate DOI | Candidate title | Candidate author/year/vol/pp | Sim | Disagreements |
|---|---|---|---|---|---|---|
| `acerbi2014back` | Back-Testing Expected Shortfall | — | *no confident Crossref match* | — | — | verify by hand |
| `angelopoulos2024conformal` | Conformal Risk Control | — | *no confident Crossref match* | — | — | verify by hand |
| `angelopoulos2024pid` | Conformal PID Control for Time Series Prediction | `10.52202/075280-1000` | Conformal PID Control for Time Series Prediction | Angelopoulos / 2023 / — / 23047-23074 | 1.0 | candidate match disagrees on: year 2024 vs 2023 |
| `ansari2024chronos` | Chronos: Learning the language of time series | — | *no confident Crossref match* | — | — | verify by hand |
| `basel1996` | Supervisory Framework for the Use of ``Backtesting'' in | — | *no confident Crossref match* | — | — | verify by hand |
| `bcbs2019frtb` | Minimum Capital Requirements for Market Risk | — | *no confident Crossref match* | — | — | verify by hand |
| `bollerslev1986generalized` | Generalized autoregressive conditional heteroskedastici | `10.1016/0304-4076(86)90063-1` | Generalized autoregressive conditional heteroskedastici | Bollerslev / 1986 / 31 / 307-327 | 1.0 | — |
| `das2024timesfm` | A decoder-only foundation model for time-series forecas | — | *no confident Crossref match* | — | — | verify by hand |
| `gibbs2021adaptive` | Adaptive Conformal Inference Under Distribution Shift | — | *no confident Crossref match* | — | — | verify by hand |
| `glosten1993relation` | On the relation between the expected value and the vola | `10.1111/j.1540-6261.1993.tb05128.x` | On the Relation between the Expected Value and the Vola | GLOSTEN / 1993 / 48 / 1779-1801 | 1.0 | — |
| `gneiting2007probabilistic` | Probabilistic forecasts, calibration and sharpness | `10.21236/ada454827` | Probabilistic Forecasts, Calibration and Sharpness | Gneiting / 2005 / — / — | 1.0 | candidate match disagrees on: year 2007 vs 2005 |
| `gneiting2014probabilistic` | Probabilistic forecasting | `10.1146/annurev-statistics-062713-085831` | Probabilistic Forecasting | Gneiting / 2014 / 1 / 125-151 | 1.0 | — |
| `hazan2016introduction` | Introduction to Online Convex Optimization | `10.1561/9781680831719` | Introduction to Online Convex Optimization | Hazan / 2016 / — / — | 1.0 | — |
| `koenker2005quantile` | Quantile Regression | `10.1017/cbo9780511754098` | Quantile Regression | Koenker / 2005 / — / — | 1.0 | — |
| `lindner2009stationarity` | Stationarity, mixing, distributional properties and mom | `10.1007/978-3-540-71297-8_2` | Stationarity, Mixing, Distributional Properties and Mom | Lindner / 2009 / — / 43-69 | 1.0 | — |
| `newey1987simple` | A Simple, Positive Semi-Definite, Heteroskedasticity an | `10.2307/1913610` | A Simple, Positive Semi-Definite, Heteroskedasticity an | Newey / 1987 / 55 / 703 | 1.0 | candidate match disagrees on: pages 703-708 vs 703 |
| `politis1994stationary` | The Stationary Bootstrap | `10.1080/01621459.1994.10476870` | The Stationary Bootstrap | Politis / 1994 / 89 / 1303-1313 | 1.0 | — |
| `riskmetrics1996` | RiskMetrics -- Technical Document | — | *no confident Crossref match* | — | — | verify by hand |
| `romano2019conformalized` | Conformalized Quantile Regression | — | *no confident Crossref match* | — | — | verify by hand |
| `woo2024moirai` | Unified training of universal time series forecasting t | — | *no confident Crossref match* | — | — | verify by hand |
| `zaffran2022conformal` | Adaptive Conformal Predictions for Time Series | — | *no confident Crossref match* | — | — | verify by hand |
| `zinkevich2003online` | Online Convex Programming and Generalized Infinitesimal | — | *no confident Crossref match* | — | — | verify by hand |
| `zinkevich2003online` | Online Convex Programming and Generalized Infinitesimal | — | *no confident Crossref match* | — | — | verify by hand |

## Citation cross-check

- Distinct `\cite*` keys found in the LaTeX sources: **50**
- Keys cited in text but **absent from the .bib**: **0**

- Entries present in the .bib but **never cited**: **6**

  - `angelopoulos2024conformal`
  - `hazan2016introduction`
  - `koenker2005quantile`
  - `liu2024moiraimoe`
  - `newey1987simple`
  - `politis1994stationary`

---

Regenerate with `python scripts/audit_bib.py`. Responses are cached in `reports/.bib_audit_cache.json`; pass `--refresh` to re-query the APIs.
