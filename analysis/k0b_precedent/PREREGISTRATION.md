# K0b — how much of the identification result is already in the literature

Written before any of the four papers was retrieved.

## The claim at risk

Proposition 6.1 of the manuscript ("What a backtest can see"): the law of the
exceedance sequence depends on the joint law of forecast and return only through
the law of {u_t}, u_t = F_t(q̂^lo_t); hence any test measurable with respect to
sigma(V_1..V_T), and any test of a predictable moment restriction on V_t − alpha,
has power equal to its size against alternatives agreeing on {u_t}.

Proposition 6.2 attaches a magnitude to it: delta*, the largest truncation depth
an indistinguishable alternative can reach, and the VaR understatement it permits
(49.4% under unimodality, 30.9% after the structural checks).

## What is being decided

Whether the four papers below already establish the first proposition, and in what
generality.

- **Escanciano & Pei (2012)** — read in full, as the closest precedent.
- **Gordy & McNeil (2020)**, **Kratz, Lok & McNeil (2018)**,
  **Cont, Deguest & Scandolo (2010)** — read for what they establish about the
  informational content of exceedance-based backtests and about robustness of risk
  measures to the underlying law.

## Interpretations, fixed in advance

**If the 2012 precedent is general** — that is, if it states the non-identification
for exceedance-based tests as such, rather than for one estimator or one parametric
family — then Proposition 6.1 is not new and must not be presented as new. Section 8
of the rebuilt manuscript keeps two paragraphs: the identification constraint,
attributed, as the reason structural validation precedes correction; and the panel
fact that a truncated series reaches pi-hat = 0.0108 and 19/24 green after
correction. **delta* and Table `tab:gate_residual` move to the supplement**, since
their role was to size a gap the paper claimed to have identified. The abstract
does not change: it already states the identification result as a premise of the
ordering, not as a contribution.

**If the precedent is specific** — to an estimator, to a parametric VaR family, or
to estimation risk rather than to the identified functional — then Proposition 6.1
stands as stated, the attribution paragraph in Section 2 states precisely what the
2012 result covers and where this one goes beyond it, and Section 8 keeps delta*
at its current length.

**The intermediate case is the likely one and is decided the same way.** If the
2012 paper establishes a related result under assumptions this paper does not
make (a correctly specified parametric family, an estimated parameter, i.i.d.
innovations), the correct treatment is attribution plus a stated delta, not
deletion, and Section 8 keeps delta* but loses the claim of priority.

## What is not at stake

No number changes under any branch. This decides placement and attribution only.
