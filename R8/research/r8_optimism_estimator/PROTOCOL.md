# Nuisance-free optimism estimator and feasible shrinkage — protocol

Fixed 13 September 2026 before any calculation. Purpose: replace the failed
kernel-density/Bartlett route (median density error 25–29% against a 15%
criterion) by estimators of the optimism penalty $2A_{0,n}=\Omega/(nf_*)$
that need no density or long-run-variance estimate, and derive from them a
feasible shrinkage factor.

## Estimators (both computed; neither may be dropped after results)

Scores $S_1,\dots,S_n$; conformal shift $C_n=S_{(k_n)}$,
$k_n=\min\{n,\lceil(n+1)p\rceil\}$, $p=0.99$; training loss change
$I_n(C_n)=n^{-1}\sum_t D(C_n,S_t)$ with $D(c,s)=\rho_\alpha(c-s)-\rho_\alpha(-s)$.

E1. Blocked cross-validation optimism. Split the calibration index into
$K=5$ contiguous blocks. For block $j$, fit $C_{-j}$ as the conformal shift
of the remaining $n-n/K$ scores (rank rule applied to that size) and
evaluate $L_j=|B_j|^{-1}\sum_{t\in B_j}D(C_{-j},S_t)$. Put
$L_{cv}=K^{-1}\sum_j L_j$ and $\hat O_{cv}=\frac{K-1}{K}\{L_{cv}-I_n(C_n)\}$.
The factor $(K-1)/K$ rescales the penalty from calibration size $n(1-1/K)$
to $n$.

E2. Block-bootstrap optimism (Efron-type). Draw $R=200$ circular block
bootstrap samples of the score sequence with block length
$b=\lceil n^{1/3}\rceil$. For sample $r$ fit $C^{*}_r$ and put
$\hat O_{boot}=R^{-1}\sum_r\{n^{-1}\sum_t D(C^*_r,S_t)-I^*_r(C^*_r)\}$,
where $I^*_r$ is the training loss change on the bootstrap sample.

Shrinkage. With $\hat A=\hat O/2$ and $\hat{\mathcal B}=-I_n(C_n)-\hat A$,
the first-order optimal factor $\lambda_*=\mathcal B/(\mathcal B+A_0)$ is
estimated by $\hat\lambda=\operatorname{clip}\{\hat{\mathcal B}/(\hat{\mathcal B}+\hat A),0,1\}$;
$\hat\lambda=0$ when $\hat{\mathcal B}+\hat A\le0$. One rule per estimator.

## Synthetic admission (stored data only)

Inputs: `results/theory_loop/synthetic/calibration_{normal,t5}.npz`
(500 histories × 2000 scores), `truth.csv` ($\Omega=0.0099$, $f_*$), the
Chebyshev reference risk in `mixture_integration_*.npz` used by
`research/r8_theory_loop_v3/engine.py` to evaluate $R(c)$ exactly, and the
bias translations $b\in\{0, 0.25\sqrt{10^{-5}/0.05}\}$ of the v3 protocol.
Sizes $n\in\{250,500,700,1000,2000\}$, calibration prefix $[0:n]$.

Criteria, fixed now:
1. Penalty accuracy: for each law and $n\ge700$, $|\text{mean}_h\hat O/(2A_{0,n})-1|\le0.15$
   (mean over 500 histories; $2A_{0,n}=\Omega/(nf_*)$ from truth.csv). Report
   all $n$; the criterion applies at 700, 1000, 2000.
2. Rule value: for $b>0$ and $n\ge700$, the exact expected loss of
   $\hat\lambda C_n$ (mean over histories of $R(\hat\lambda C_n)-R(0)$ from the
   reference risk) must be below that of full correction $C_n$ and below raw
   (zero) for both laws. For $b=0$ it must be below full correction.
3. Uncertainty: one simultaneous family per estimator over all (law, n, b)
   cells, using the saved 999 history-bootstrap rows
   (`history_bootstrap_indices.npy`), max-standardised-deviation rule.

An estimator passes if 1 and 2 hold. If neither passes, the financial
application is NOT_RUN and the negative result is reported.

## Financial application (only for a passing estimator)

Panel: the 240 pairs in `artifacts/r8_ten_comparators/pairs/*/daily.parquet`
(read-only). For each pair, compute $\hat\lambda$ from the calibration
scores of the primary 70/30 split (same calibration block as Shift-CP) and
the policy $q_t^{lo}-\hat\lambda C_n$ on the test dates. Evaluate QS,
violation rate and Kupiec at 1%. Contrast: shrunken minus Shift-CP and
shrunken minus raw, with the existing 999 common-calendar bootstrap draws
at 20 and 60 calendar days (reuse the resampling convention of
`research/r8_ten_comparators/aggregate.py`), as a separate two-contrast
simultaneous family. This is a retrospective extension of an inspected
panel; report it as such.

## Outputs

`artifacts/r8_optimism_estimator/`: synthetic tables (per law, n, b:
mean and MCSE of $\hat O$, ratio to $2A_{0,n}$, band, mean $\hat\lambda$,
expected loss of raw/full/shrunken), admission.json with the criteria and
their pass/fail, and, if run, the financial table and bands. A
`RESULTS.md` with every number and the decision. Code with a `--check`
replay. No R8 file is modified.
