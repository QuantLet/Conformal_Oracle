# Information needed to recognise a useful tail correction

10 September 2026. Fixed before computing this extension. The author has
requested further scientific development toward a stronger paper. This
stage develops proofs and deterministic calculations, without model
inference, new simulated paths or changes to the validated manuscript.

## Question and admission criterion

Can selection difficulty be shown for every decision rule in an explicit
dependent score model, rather than inferred from failure of one gate?
The result must have a complete proof, a calculation independently checked
by finite enumeration, and a rare-event limit with n*alpha bounded.
It must distinguish a sharp two-action selection result from a lower
bound for arbitrary scalar estimators. Established testing/Bayes tools
are attributed as such. No claim of unique priority is made.

The study is potentially useful if it links tail sparsity, dependence
and indication through an exact experiment, and remains informative
after giving the selector the whole history and the candidate laws.
It is not evidence that financial scores follow the constructed law,
nor that existing financial policy failures attain the lower bound.

## Fixed experiment

The marginal score density is (1-theta) on [-1,0] and theta on (0,1].
At each time retain the previous score with probability r, otherwise
draw independently from that marginal law. Start at stationarity.
Continuous refresh values make the refresh runs observable, almost surely.
Their number is K=1+Binomial(n-1,1-r); signs of their distinct draws
have J|K ~ Binomial(K,theta). Conditional values given signs carry no
theta information. The full history is available to every procedure.

For two fixed actions, raw c=0 and correction c=1/2, let
theta_star=alpha/(1-c/2), theta_minus=theta_star*(1-epsilon),
theta_plus=theta_star*(1+epsilon). Their loss differences have equal
opposite magnitude alpha*c*epsilon. All specified theta_minus exceed
alpha, so both population optimal corrections are interior to (0,1).

- alpha: 0.01, 0.05.
- n: 125, 250, 500, 1000.
- r: 0, 0.25, 0.5, 0.75.
- epsilon: 0.1, 0.2.
- All 64 configurations are retained with equal status.

Primary output: the exact best equal-prior binary decision error,
one half of the mixture of overlaps of the two binomial count laws.
Its loss counterpart multiplies by the common loss difference.
Also compute the exact equal-prior Bayes lower bound for the expected
excess pinball loss of any real-valued scalar correction, relative to
each law's population optimum. This is a minimax lower bound, not an
assertion that the equal-prior rule is minimax.

Use the exact binomial-CDF expression to find the first n with best
average error at most 10% and 5%, for every alpha/r/epsilon combination.
These are discrimination lengths in the specified model, not recommended
financial calibration windows or a sufficiency certificate for uniform
error control. Refuse a silent finite-search cap.

Evaluate the Poisson rare-event limit at tau=n*alpha in {1.25,2.5,5,10},
for the same r and epsilon, and compare finite alpha=0.01,0.001,0.0001
at those intensities. This is an analytic limit check, not simulated data.

## Verification and preservation

- Integrate the piecewise uniform pinball risk independently.
- Enumerate all refresh/sign histories for small n with exact rational
  arithmetic; compare binary Bayes error and continuous Bayes risk.
- Check likelihood-ratio CDF error against direct binomial overlap sums.
- Check the observable-run sufficient statistic and full-history count
  accounting through independent enumeration of refresh patterns.
- Verify Hellinger affinity and its error lower bound independently.
- Check that Bayes error/risk decrease with additional observations;
  minimum sample sizes must pass and their predecessor must fail.
- Recompute outputs in a fresh process and compare every numeric file.
- Bind producer, protocol, outputs, environment and all current canonical
  source/PDF/validation files by hashes. Keep previous releases immutable.

Report bounds as properties of the two-law experiment. Do not imply a
universal ban on useful recalibration, a guarantee for selected conformal
shifts, or that this standard testing reduction is itself a new theorem.
