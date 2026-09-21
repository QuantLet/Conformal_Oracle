# Amendment to the optimism-estimator protocol — 13 September 2026

Dated after the original run, which is preserved unchanged in
`artifacts/r8_optimism_estimator_original_4over5/` (its `admission.json`
already showed both estimators passing criterion 1 and failing criterion 2).
Prompted by the independent Codex review (`docs/content_improvement_20260913/
CODEX_PROTOCOL_REVIEW.md`, `CODEX_FINAL_EXTENSION_REVIEW.md`).

## The error

The protocol rescaled the blocked cross-validation difference by $(K-1)/K$,
arguing that it converts the penalty from calibration size $n(1-1/K)$ to
$n$. That is the wrong target. With $A_n=a/n$ the first-order cost and
$m=n(K-1)/K$ the fold training size, the loss expansion gives
$E[L_{cv}]=-\mathcal B+A_m+o(1/n)$ and $E[I_n]=-\mathcal B-A_n+o(1/n)$, so
$E[L_{cv}-I_n]=A_m+A_n+o(1/n)=A_n\{K/(K-1)+1\}$. The factor that maps this
to $2A_n$ is $2(K-1)/(2K-1)$, which is $8/9$ for $K=5$. The original $4/5$
targets $1.8A_n$, a 10% downward bias at first order in the iid reference
case. This is an iid leading-order justification; it is not a finite-sample
result for dependent blocks or disjoint training complements.

## What changes

`engine.py` uses `CV_FACTOR = 2(K-1)/(2K-1)`; E2, the shrinkage rule, the
admission criteria, the bands and every other step are unchanged. The run is
repeated on the same stored histories with the same seeds; outputs overwrite
`artifacts/r8_optimism_estimator/`. The admission decision is re-evaluated
under the unchanged criteria. The manuscript displays are regenerated and
the supplement states the amendment.
