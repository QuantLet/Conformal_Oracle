17 August 2026

The Editors
*International Review of Financial Analysis*

Dear Editors,

We submit our manuscript, **"Recalibrating Tail Risk Forecasts under Temporal
Dependence"**, for consideration in the *International Review of Financial
Analysis*.

**What the paper reports.** It is, to our knowledge, the first systematic
1%-tail audit of zero-shot time-series foundation models (TSFMs) used as
Value-at-Risk forecasters, covering 13 forecasters and 24 assets across
equities, bonds, commodities, cryptocurrencies and FX. Its central finding is
that the *predictive interface* — whether a model emits Monte Carlo samples or a
coarse grid of quantiles — rather than its architecture or pretraining corpus,
governs extreme-tail calibration. The cleanest evidence is a within-family
control: Moirai 1.1 and Moirai 2.0 share an architecture and a closely related
pretraining design and differ in output format. At the 1% level the sample-based
release violates on 1.5% of days; the quantile-grid release violates on 98.8%.
The same division holds across the wider set.

The practical implication is direct. A nine-decile grid whose lowest point is
the 10% quantile does not contain a 1% quantile; it must be supplied by an
external tail-completion step, and that step, not the pretrained model, then
determines tail accuracy. A risk manager can identify this exposure *before*
deployment, from the output format alone, without running a backtest.

**What the paper does not claim.** Our instrument is a one-parameter conformal
shift, retained as a signed, continuous statistic *R* measuring the correction's
size relative to the reported VaR. We claim marginal coverage and diagnosis
only. Three results delimit it, and we report each as a finding rather than a
caveat:

1. *Recalibration is an intervention with an indication, not a universal
   improvement.* On base forecasts that already pass a Kupiec test, the
   single-split correction is statistically indistinguishable from doing nothing,
   while the rolling variant degrades 93% of them. Of the degradations it causes,
   60% buy no change of Basel zone at all. Gating the correction on a failed raw
   backtest removes half the damage while retaining every one of the 176 zone
   upgrades — zero cost on the regulatory axis.
2. *It reaches only part of the problem.* In the score decomposition of Gneiting
   and Resin (2023), a constant shift addresses a median 35% of total
   miscalibration on the usable forecasters; the remainder is conditional.
3. *It does not restore violation independence.* Where the Christoffersen test
   is defined it rejects on 62% of pairs after correction, and duration-based
   tests flag clustering on a further 47 pairs where that test is undefined by
   construction.

We also state plainly that the statistic is not new: for the quantile loss, the
score-minimising constant shift is the empirical residual quantile, whose score
consequence is the unconditional miscalibration term of Gneiting and Resin
(2023). What *R* contributes is a scale, not a quantity — it is expressed in
units directly comparable to the VaR it corrects, so it can be read against a
Basel zone and a Quantile Score on the same axis.

**Fit with IRFA.** The paper is a risk-management study rather than a
forecasting-methodology one. Its outputs are the quantities a risk function
acts on: Basel Traffic Light zones, capital implications of threshold widening,
and an explicit deployment rule with its cost tabulated. It evaluates the
model class that banks and asset managers are currently being offered as a
drop-in replacement for per-asset volatility models, and it reports a concrete,
checkable reason why one large subclass of them cannot be used for regulatory
tail risk without an additional modelling choice that the vendor does not
supply. The benchmark set includes CAViaR (both specifications) and a
score-driven GAS model alongside GJR-GARCH, GARCH-N, EWMA and Historical
Simulation; notably, CAViaR requires essentially no correction (*R* = 0.001),
which is itself the sharpest confirmation of our main delimiting result.

**Reproducibility and a disclosure.** The full replication pipeline — data
retrieval, model inference, recalibration, backtesting, robustness analysis and
figure generation — is available through the Quantlet repository, together with
machine-readable result matrices for every model–asset pair. A Python package,
`conformal-oracle`, implements the audit.

An earlier version of this work was reviewed elsewhere, and one reviewer
identified an incorrect bibliographic reference. In response we audited the
entire bibliography programmatically rather than that entry alone: every
reference was resolved against the Crossref REST API and, for preprints, the
arXiv API, and the returned title, first author, container, year, volume and
page range were compared against our entry. The audit found eight defective
entries out of fifty-six — unresolvable or misdirected DOIs, an incorrect venue,
and incorrect author lists — all of which have been corrected against the
authoritative records. The audit script is included in the replication package
(`scripts/audit_bib.py`), so the check can be re-run independently. We extended
the same procedure to a provenance audit of every table and figure in the
manuscript, verifying that each is emitted by a script in the package and that
the emitted values match those printed; that manifest is also included.

We believe the manuscript is well suited to IRFA's readership in risk
measurement and financial econometrics. It has not been published previously and
is not under consideration elsewhere. All authors have approved the submission
and declare no conflicts of interest.

Thank you for your consideration.

Sincerely,

Daniel Traian Pele (corresponding author)
Bucharest University of Economic Studies
danpele@ase.ro

*on behalf of the co-authors*
