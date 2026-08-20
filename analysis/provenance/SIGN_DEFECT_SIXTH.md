# A sixth instance of the sign defect, in a robustness script

Found 20 August 2026 while regenerating the tail-closure tables.

`scripts/tail_completion_robustness.py` and
`Quantlets/CO_robustness_inner7/run_inner7_tail_closure.py` reconstruct the
nine-decile grid from the stored Student-$t$ parameters, refit a closure rule to
it, and read the 1% quantile. Both then stored

    var_series.iloc[t] = -q_alpha

The $\alpha$-quantile of a return distribution is already negative and is itself
the Value-at-Risk threshold. Negating it points the threshold at the wrong tail.
This is the same defect that corrupted the stored TimesFM 2.5 and Moirai 2.0
series, reproduced independently in the two scripts that were supposed to
establish that the results do not depend on the closure rule.

## The check

On Moirai-2.0 / SP500, row 100 of the promoted series:

| quantity | value |
|---|---|
| fitted $\nu$, $\mu$, $\sigma$ | 34.36, $-0.001484$, $0.012397$ |
| $q_{0.01}$ from those parameters | $-0.031733$ |
| what the script stored | $+0.031733$ |
| `VaR_0.01` in the promoted series | $-0.031733$ |

The reconstruction reproduces the promoted series exactly once the negation is
removed, which both identifies the defect and validates the reconstruction.

## What it invalidated

`tab_tail_closure.tex` and `tab_tail_closure_extended.tex`, and with them the
manuscript's claim that the ordering by $\bar R$ is invariant across Student-$t$,
Gaussian and linear closure rules. The pre-fix run reported $\hat\pi = 0.075$ and
$R = 1.77$ for Moirai-2.0 on NATGAS; the corrected series runs at $\hat\pi
\approx 0.018$ with $R \approx 0.16$.

Both tables are regenerated after the fix, and the invariance claim is
withdrawn rather than restated. Across three assets the same grids now give
R between 0.005 and 1.70 depending on the closure rule -- a factor of 3.3 to 76
within a single model-asset pair -- and on BTC the choice of rule moves both
models from the Basel yellow zone to the red one.

A second thing came out of the rebuild. The inner-seven-deciles refit, presented
in earlier drafts as the more demanding check, reproduces the full fit to six
decimal places. It cannot do otherwise: the deciles are reconstructed from the
stored Student-t parameters, so fitting a Student-t to any subset of them
recovers those parameters exactly. The check was vacuous by construction and the
appendix now says so.

## Why it survived

The same reason as the others: every number downstream of it was computed after
conformal recalibration, which restores coverage whatever the input. A closure
rule fitted to an inverted grid produced an inverted threshold, the correction
absorbed it, and the resulting Basel zones looked ordinary. Nothing in the
pipeline compared the reconstruction against the series it was reconstructing --
which is the check that found it, and which costs one line.
