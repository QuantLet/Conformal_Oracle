# Pre-registration — Chronos sampling-configuration test

Written and committed **before** any inference. Third mechanism proposed for the
Chronos anomaly; one has already been proposed and retracted. Treated as a
hypothesis to falsify.

## Established before running (Task 1)

Both checkpoints resolve identical sampling defaults, and the notebook passes
none of them, so all three fell through:

    top_k = 50   top_p = 1.0   temperature = 1.0   n_tokens = 4096

`amazon/chronos-t5-small` @ a971ba21945c4f1796b17a91fe69214b5f4ad472
`amazon/chronos-t5-mini`  @ bd6a4fde8403b8469acd0abd52852b29dbe61c7b
chronos-forecasting 2.2.2, transformers 4.57.6.

`num_samples` was overridden to 1000 by the notebook (config default is 20), so
the sample count is not the issue. `CONTEXT = 512` matches the trained context
length exactly.

## The anomaly to be explained

Chronos predictive std / realised sigma = 0.117 (Small), 0.109 (Mini), against
1.29 for Moirai 1.1 and 0.80 for Lag-Llama. Quantile extraction is internally
sound: standardised by the model's own predictive distribution the four levels
come out at -1.78, -1.67, -1.54, -1.32 against normal references -2.33, -1.96,
-1.64, -1.28. So the defect is in the width of the sampled distribution, not in
how quantiles are read off it.

## Backend caveat, recorded in advance

The published run used an A30 (CUDA). This runs on Apple MPS. Sampling under a
different backend is not the same experiment: kernel-level differences in
softmax and multinomial sampling can shift results. Any conclusion here is
indicative and must be confirmed on the A30 before it enters the paper. The
purpose is to establish direction and rough magnitude, not final numbers.

## Branches, fixed now

**B1 — hypothesis holds.** Predictive dispersion rises monotonically with
`top_k` and approaches realised sigma at large `top_k`, and pi-hat at
alpha = 0.01 falls toward nominal. The anomaly is a sampling-configuration
artefact, not a model property.
*Consequence:* the finding becomes that the checkpoint default truncates the
predictive support to 50 of 4094 bins, and that every zero-shot Chronos
tail-risk number computed under defaults measures the configuration rather than
the model. That is a methodological result about how these models are used, it
applies beyond this paper, and it makes the sampling-parameter survey (Task 7)
load-bearing rather than incidental.

**B2 — partial.** Dispersion widens but Chronos remains materially
miscalibrated at alpha = 0.01 (say, pi-hat still above 5%).
*Consequence:* part configuration, part model. The residual needs its own test
and the paper reports the decomposition without claiming to have explained the
remainder.

**B3 — no movement.** Dispersion is flat in `top_k`.
*Consequence:* hypothesis falsified; the 50 retained tokens already carry
essentially all the probability mass and the narrowness is the model's genuine
belief. Reopen. Do not propose a fourth mechanism in the same pass. The
candidate that would then move to the front is the interaction between
`MeanScaleUniformBins` and a series whose mean absolute value is ~0.008.

## Design

One asset (SP500), 20 dates, both variants, identical contexts and seed across
cells. Dose-response on `top_k` in {50, 200, 1000, 4094} at `top_p` = 1.0 and
`temperature` = 1.0. Then, at `top_k` = 50, vary `top_p` in {0.9, 0.99, 1.0} and
`temperature` in {0.5, 1.0, 2.0} one at a time, to establish that `top_k` is the
operative parameter rather than a proxy.

Reported per cell: predictive std / realised sigma; the four quantiles
standardised by the predictive distribution; pi-hat at each alpha; and the
fraction of sampled values at the boundary of the retained set.

Raw sample paths retained this time.
