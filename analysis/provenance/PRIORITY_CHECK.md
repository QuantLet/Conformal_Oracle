# Priority check on Theorem 4.5 and Corollary 4.6

Run 2026-08-31. Claims below were verified by downloading the primary PDFs and
reading the theorem statements, not from search snippets or an agent's summary.

## The finding

**Theorem 4.5 is preceded.** Oliveira, Orenstein, Ramos & Romano, "Split
Conformal Prediction and Non-Exchangeable Data", JMLR 25(225):1--38, 2024,
Section 3.3 and Theorem 4. Verified verbatim from
<https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf>:

> **Theorem 4 (Marginal coverage: stationary β-mixing processes)** Suppose the
> sample (X_i,Y_i) is stationary β-mixing. Then given α ∈ (0,1) and δ_cal > 0,
> for i ∈ I_test, P[Y_i ∈ C_{1−α}(X_i)] ≥ 1 − α − η, with η = ε_cal + ε_train +
> δ_cal

with ε_cal at their eq. (12) carrying a `sqrt(4/(n_cal − r + 1) · log(...))`
term plus `(r−1)/n_cal`, and ε_train a β-coefficient evaluated at a separation
between the training block and the evaluated point. That is the shape of our
remainder. They also name our data class explicitly:

> This class of non-exchangeable data is broad enough to cover many important
> applications, such as hidden Markov models and Markov chains as well as ARMA
> and **GARCH** models (Carrasco and Chen, 2002; Mokkadem, 1988)

**And the rate is superseded.** Barber & Pananjady, "Predictive inference for
time series: why is split conformal effective despite temporal dependence?",
arXiv:2510.02471, ALT 2026. Verified verbatim from
<https://arxiv.org/pdf/2510.02471>, their own comparison paragraph:

> Let us compare again with the result of Oliveira et al. (2024, Theorem 4), who
> show that ... coverage loss for split conformal prediction on a β-mixing
> process is bounded ... by a term of the order min_{τ,τ*}{√(τ/n)+√(τ*/n)+2β(τ)
> +2β(τ*)}. As before, comparing with Corollary 2 above, note that our bound on
> the coverage loss is tighter, scaling linearly in τ/n and τ*/n.

Under geometric mixing with τ = c log n their remainder is O(log n / n) against
our O(√(log n / n)).

**Corollary 4.6's rate and constant are preceded.** Zheng & Proutiere,
"Conformal Predictions under Markovian Data", ICML 2024, PMLR 235:61470--61497,
Theorem 5.1 gives K* = O(ln n / ln(1/ρ)) under geometric ergodicity with rate ρ
— the same rate and the same constant as g_n = ⌈c log n⌉, c = 1/|log ρ|. Not
verified by me from the primary PDF; reported from the research pass, and to be
checked before anything is written on it. The GARCH-to-geometric-β-mixing step
is Carrasco & Chen, Econometric Theory 18(1):17--39, 2002 — a citation, not a
result.

## What none of them do

Full-text searches of the four principal prior-art papers return zero hits for
Value-at-Risk, Basel, one-sided, or volatility modelling. The one-sided
regulatory specialization, the GARCH parameterization, the Basel traffic-light
identification result of Propositions 5.1--5.3, and the empirical panel are not
touched by this prior art.

## What the manuscript currently cites

`barber2023conformal`, `gibbs2021adaptive`, `zaffran2022conformal`. It does
**not** cite Oliveira et al. (2024), Zheng & Proutiere (2024), or Barber &
Pananjady (2026). Those three are the nearest prior art to Section 4 and their
absence is the finding.

## The four "unfound" identifiers are all real, and two of them matter

The research pass reported four arXiv identifiers it could not retrieve, and was
right to refuse to cite them. Resolving them directly: all four return HTTP 200
with exactly the reported titles. The agents had exhausted their fetch budget,
not invented the papers.

| id | author, date | why it matters |
|---|---|---|
| 2603.22569 | Zhong, Tenghan, 2026-03-23 | *Proxy-Reliance Control in Conformal Recalibration of One-Sided Value-at-Risk.* The closest published work to this manuscript's own framing. Already in `analysis/litsurvey/SURVEY.md`; not cited in the manuscript. |
| 2606.18199 | Cuonzo & Deliu, 2026-06-16 | *Conformal Prediction Intervals with Tail-Specific Guarantees.* One-sided split conformal with marginal validity, exchangeable (finite-sample) and **non-exchangeable (asymptotic)**, demonstrated on a financial left-tail application. |
| 2602.03903 | Schmitt, 2026-02-03 | Already cited as `tamingtailrisk2026`. Sequential one-sided VaR calibration; coverage bounds for data-driven weights under regime drift. |
| 2507.05470 | Aich, Aich & Jain, 2025-07-07 | *Temporal Conformal Prediction.* Rolling split-conformal calibration benchmarked against GARCH, Historical Simulation, QR and ACI on S&P 500, Bitcoin and Gold. |

### Both read in full, 2026-08-31. Neither preempts the theory.

**Zhong (2603.22569), 44 pages.** A proxy-reliance parameter that interpolates
between a constant shift and a fully volatility-proxy-scaled shift in one-sided
VaR recalibration, with a six-ETF panel and VIX-linked state variables. Full-text
counts: `mixing` 0, `split conformal` 0, `coverage guarantee` 0, `order
statistic` 0, `exchangeab` 1, `traffic light` 0. Its Proposition 4.1 is an
elasticity-and-invariance result about how the adjustment responds to proxy
rescaling — a design-sensitivity statement, not a coverage bound. The 47 `Basel`
hits are references to the Committee's documents, not a traffic-light analysis.

*Verdict: closely related work, must be cited, preempts nothing.* It is in fact
the natural citation for why this paper uses a constant shift: Zhong studies
precisely the constant-versus-scaled choice and finds the constant end
competitive in stressed states.

**Cuonzo & Deliu (2606.18199), 52 pages.** One-sided split-conformal intervals
with separately calibrated tail coverage, plus their intersection. Their
exchangeable result (Propositions 1 and 5) is finite-sample and, in their own
words, "follows verbatim the theorems in Lei et al. (2018)". Their
non-exchangeable results are Proposition 2 and Theorem 4.2 "under the same
conditions as Gibbs & Candes (2021)" and Proposition 3 under DtACI, all of the
form

> |(1/N) Σ err_i − α| ≤ (max{α_1, 1−α_1} + γ) / (Nγ)

which is a **time-averaged empirical miscoverage** bound over an online horizon,
not marginal coverage at a designated test point. The two `mixing` hits refer to
the DtACI algorithm's own mixing sequences, not to a mixing condition on the
data. GARCH(1,1)-t appears as the benchmark forecaster in their stock
application, not as a data-generating process carrying a mixing rate. `Basel`: 0.

*Verdict: closely related work, must be cited, preempts nothing here.* Their
non-exchangeable case is the ACI family, which finding [5] of the research pass
already established does not preempt marginal coverage under β-mixing.

**Correction to the warning written above.** Read from the abstract, "guarantees
are asymptotic" in the non-exchangeable setting looked like it could overlap
Theorem 4.5. It does not. What these two papers do narrow is a different and
weaker claim — that nobody has applied one-sided conformal calibration to VaR.
Two people have, in 2026, and both must be cited. The coverage theorem's
positioning is unaffected, because it is already attributed to Oliveira et al.

## Not citable

(Superseded: the four identifiers below were resolved and are recorded above.
Kept for the record of how they were handled.)

Four arXiv identifiers surfaced in search metadata and could not be retrieved at
a real URL by any verifier: 2507.05470, 2602.03903, 2603.22569, 2606.18199. The
last three have titles close enough to this manuscript's framing that they may
be search noise or model-generated. They are recorded here as unfound leads and
must not be cited. This repository has retracted one fabricated citation
already.

## Unanswered

The research pass returned nothing verified on time-series foundation models for
financial risk (whether anyone has documented the Chronos `top_k` truncation
defect) and nothing on IRFA author requirements. Those two questions are open.
