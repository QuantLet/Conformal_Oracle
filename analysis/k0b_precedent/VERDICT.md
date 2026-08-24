# K0b — verdict: the 2012 precedent is specific, not general

Four papers read against the pre-registration in this directory. Full text of
Escanciano & Pei (2012) and Gordy & McNeil (2020) extracted to `papers/`.

---

## Escanciano & Pei (2012), *Pitfalls in backtesting Historical Simulation VaR models*, JBF 36(8) 2233–2244

**Theorem 1, quoted.** "Assume that $m_\alpha(I_{t-1},\theta_0)$ is given by a HS
or FHS model, and let Assumption A1 in Appendix A hold. Then
$\Pi(\tau,\alpha) = 2\Phi\!\left(z_{\tau/2}\sqrt{\alpha(1-\alpha)}/\sigma_K\right)$,"
where $\sigma_K^2 = \alpha(1-\alpha) + 2\sum_{j\ge1}\mathrm{Cov}(h_{t,\alpha},h_{t-j,\alpha})$.
The consequence: $\Pi(\tau,\alpha) < 1$ — Kupiec's test is inconsistent.

**Answering the pre-registered question, in three parts.**

1. **It is specific to the estimator.** The hypothesis of Theorem 1 is that the
   forecast *is* a Historical Simulation or Filtered Historical Simulation
   quantile. The mechanism is that HS and FHS are unconditionally calibrated by
   construction: "the corresponding hit sequence has a mean equal to $\alpha$
   under the alternative hypothesis." Nothing in the theorem holds for a
   forecaster that is not unconditionally calibrated by construction — which is
   every forecaster in this manuscript's panel except Historical Simulation.

2. **It is specific to one test.** The result is about the unconditional backtest
   ("the K-test"). Their own Section 3 then constructs weighted backtests
   $K_{P,w}$ with $\mathrm{E}[w(I_{t-1})(h_{t,\alpha}-\alpha)]$ as the moment, and
   **Lemma 1 exhibits a consistent one**. So the 2012 paper's message is that a
   test *inside* the exceedance class recovers the power the unconditional test
   loses. The manuscript's Proposition 6.1 says the opposite thing about a
   different set of alternatives: for alternatives agreeing on the whole process
   $\{u_t\}$, *no* test measurable with respect to the exceedance sequence has
   power, weighted backtests included. There is no conflict — Escanciano & Pei's
   alternatives agree with the null only on the *mean* $\mathrm{E}[u_t]=\alpha$,
   not on the process — and there is no priority either.

3. **It is about estimation risk and long-run variance, not identification.**
   The quantity doing the work is $\sigma_K$, the long-run variance of the hit
   sequence. Nowhere does the paper state what the exceedance sequence identifies.

**The one place the two papers touch the same object.** Escanciano & Pei's
Lemma 1 gives the optimal weight
$w^*(I_{t-1}) := F_{I_{t-1}}(m_\alpha(I_{t-1},\theta_0))$ — the conditional
distribution function of the return evaluated at the reported quantile. **That is
exactly $u_t$**, the manuscript's realised tail probability, equation (11). The
same functional appears in both papers with opposite uses: they weight by it to
*gain* power, the manuscript shows that it *exhausts* what the exceedance path
carries. This is the sentence Section 2 owes the reader, and it is a strong one:
the object the manuscript proves is identified is the object the 2012 literature
had already found to be the sufficient weight.

**Multiplication factors.** Their empirical application on three US stock
portfolios (1999–2009) has the D-test rejecting HS and FHS while
"the multiplication factors ... are all set at the lowest level of 3", the green
zone. Their conclusion: "in the current regulatory framework the multiplication
factors are underestimated for the cases where HS or FHS provide inefficient
forecasts." A regulatory capital multiplier sitting at its floor while a
consistent test rejects the model is the precedent for the manuscript's Basel
sentence, and it is fourteen years old. Section 2 must say so.

---

## Gordy & McNeil (2020), *Spectral backtests of forecast distributions*, JBF 116

Their tests read the PIT value $P_t$, not the binary indicator: $W_t = G_\nu(P_t)$.
The design premise is stated explicitly — "**What is essential to our contribution
is that the regulator does not observe the entire distribution $\hat F_t$, but
does observe more than the VaR exception indicator** $1\{L_t \ge \widehat{VaR}_{\alpha,t}\}$"
— and the closing discussion says why: "**Until recently, regulators effectively
observed only a sequence of VaR exceedance event indicators at a single level
$\alpha$, and therefore backtests were designed to take such data as input.**"

So the poverty of the single-level exceedance indicator is **recognised and acted
on** in this literature. What is not there is the characterisation: Gordy & McNeil
do not state what the exceedance sequence identifies, and they do not size the
gap. They escape it by requiring a richer input, which a regulator can demand and
a user of a third-party forecast series cannot.

---

## Kratz, Lok & McNeil (2018), *Multinomial VaR backtests*, JBF 88

Same structural move, different instrument: exceptions at $N \ge 4$ VaR levels
rather than one, shown to be "much more powerful at detecting misspecifications of
trading book loss models than standard binomial exception tests corresponding to
the case $N=1$." Again the limitation of the single-level indicator is treated as
a premise to design around, not as a result to state.

---

## Cont, Deguest & Scandolo (2010), *Robustness and sensitivity analysis of risk measurement procedures*, Quantitative Finance 10(6) 593–606

Not about backtesting. It matters here for a different reason, and it is the
attribution this manuscript most conspicuously lacks. They define the **risk
measurement procedure** as the pair (estimation step, risk measure) and show that
"the same risk measure may exhibit quite different sensitivities depending on the
estimation procedure used." That is the manuscript's central framing — that the
configuration of the reduction step is part of the estimator, as a bandwidth is
part of a kernel estimator — stated in 2010 for the estimation step. The
manuscript's contribution relative to it is narrower and should be stated
narrowly: an instance in which the procedure-dependence is a *documented software
default* rather than an estimator choice, and is invisible to the diagnostics the
domain uses.

---

# Decision

**The intermediate branch of the pre-registration, resolved toward "specific".**

Proposition 6.1 stands as stated. The 2012 result covers one estimator class and
one test; it does not state the identification result, and its own Lemma 1 runs in
the opposite direction. But the *situation* — exceedance-based backtesting is
informationally impoverished at a single level, and regulatory capital inherits
that — is established, by three of the four papers, and the manuscript currently
reads as though it is not.

**Consequences, item by item.**

1. Section 8 **keeps $\delta^\star$** and keeps Table `tab:gate_residual`. The
   pre-registered condition for moving them to the supplement was that the
   precedent be general. It is not.
2. Section 8 **loses any claim of priority** over the observation that
   exceedance-based tests are weak. The claim it keeps is the characterisation
   and the magnitude.
3. Section 2 gains the delimitation paragraph with four specific attributions:
   Escanciano & Pei on the inconsistency of the unconditional backtest for
   unconditionally-calibrated estimators and on the multiplication factors;
   Gordy & McNeil and Kratz–Lok–McNeil as the class of backtests that read more
   than the exceedance, together with what they require of the reporting regime;
   Cont–Deguest–Scandolo on the estimation procedure as part of the risk measure.
4. $u_t$ is introduced in Section 6 **as Escanciano & Pei's optimal weight
   $w^*$**, with the reference. This costs nothing and is the strongest available
   evidence that the functional is the right one.
5. The abstract does not change.
