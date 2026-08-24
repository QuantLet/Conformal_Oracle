# K1 — independent verification: results

Run against `PREREGISTRATION.md` in this directory. Every check carried a negative
control in the same process. One of them agreed when it was built to disagree,
and that is the finding of this block.

---

## K1a — the analytic Chronos estimator is off by one quantisation bin

**FAILS. A defect, in our own estimator, of the same class as the five the paper audits.**

The estimator was rebuilt from the five numbered steps of Section 4.4 and from the
`chronos` package's own tokenizer source, with no file in `scripts/`, `pipeline/`
or `Quantlets/` opened. Run on 200 contiguous mid-series SP500 dates
(2013-09-24 to 2014-07-10) against `chronos_small_analytic/SP500.parquet`:

| comparison | median \|rel\| | max \|rel\| | verdict |
|---|---|---|---|
| VaR_0.01, rebuilt vs stored | 2.506e-03 | 3.096e-03 | **disagree** |
| VaR_0.10, rebuilt vs stored | 4.975e-03 | 6.667e-03 | **disagree** |
| **negative control: stored shifted by one bin width** | **2.39e-09** | **8.09e-08** | **AGREE — the control fired** |

The two differ by **exactly one quantisation bin**: stored − rebuilt = 0.999999
bin widths, sd 1e-16, on SP500, GOLD and EURUSD for `chronos-t5-small` and on
SP500 for `chronos-t5-mini`. Four assets, two checkpoints, the same offset.

**Which of the two is right, settled deterministically.** The library's decoder
`MeanScaleUniformBins.output_transform` maps token id $j$ to
`centres[j − n_special − 1]`. Sampling at 2000 draws cannot adjudicate a one-bin
gap — its own error is about five bins — so the mapping was forced instead:
`top_k = 1` makes every draw the arg-max token, and the value the library decodes
it to is a fact with no Monte Carlo error in it. On **12 of 12** cells across
three assets the library's own decoder agrees with the rebuild and sits one full
bin below the stored series. The shipped analytic series pairs each bin's
probability with the **next bin up**.

The direction matters: a threshold one bin too high is one bin too shallow, so the
shipped series over-reports violations.

**What it moves.** The offset is deterministic — stored = correct + binwidth ×
scale$_t$, and scale$_t$ is the mean absolute value of the 512-day context, which
needs no model. The reconstruction rule was checked against the 200 re-run dates
(max relative error 8.1e-08) before being applied to all 24 assets:

| | published | corrected |
|---|---|---|
| Chronos-Small-A, $\hat\pi$ at α=0.01 | 0.0175 | **0.0173** |
| Chronos-Mini-A, $\hat\pi$ at α=0.01 | 0.0178 | **0.0177** |
| Chronos-Small-A, ratio $\hat\pi/\alpha$ | 1.750 | **1.733** |
| Chronos-Mini-A, ratio $\hat\pi/\alpha$ | 1.781 | **1.772** |
| Chronos-Small-A, Kupiec passes at α=0.01 | 8/24 | 8/24 |
| Chronos-Mini-A, Kupiec passes at α=0.01 | 8/24 | 8/24 |
| Chronos-Mini-A, Kupiec passes at α=0.10 | 22/24 | **20/24** |
| Chronos-Small-A, Kupiec passes at α=0.10 | 20/24 | 20/24 |
| Quantile score, both, α=0.01 | — | changes by −0.09% |

**Nothing qualitative moves. Three printed numbers do**, and one of them
(`\nKupMiniAnalyticTen`) is a count. The paper's arguments are untouched: the
analytic route still removes the sampler, the residual is still a tail deficiency
that shrinks with α, and 1.73× nominal is the same finding as 1.75×.

**But the defect is in the estimator the paper offers as the fix**, and the
paper's own validation was **blind to it, not merely coarse**. Section 4.4 reports
two agreements against full-vocabulary sampling on 40 SP500 dates:

- *predictive standard deviations to within 0.3%* — but the defect is a uniform
  translation of the support, which leaves the second central moment **exactly**
  invariant. That is not a loose tolerance; it is a tolerance on a quantity the
  defect does not move, and no tighter choice would have helped.
- *violation rates at all four levels agreeing to four decimal places* — on 40
  dates at α = 0.01, where the expected count is 0.4 and the rate lives on a grid
  of 1/40 = 0.025. The offset is 0.249% of VaR₀.₀₁ and 0.591% of the predictive
  standard deviation. Neither is resolvable on that grid.

Comparing the per-date quantiles — the object rather than two summaries of it —
would have found it in one line. Registered as **R14** with the pattern named,
and `PROTOCOL.md` Rule 2 gains a subsection: every validation tolerance is
declared together with the size of the defect it cannot see, and with whether the
statistic being compared is one the failure mode moves at all.

**Recommendation.** Regenerate both analytic panels with the corrected map before
submission. The correction is exact and needs no model re-run; the model re-run is
nonetheless the cleaner path and costs about ten minutes per checkpoint.

---

## K1b1 — violation rates for TimesFM 2.5 and Moirai 2.0: VERIFIED

Recomputed from returns and stored series alone, with the 70/30 split applied
independently. Eight series, agreement to six decimals with `all_results.csv`:

| series | recomputed cell mean | stored | printed |
|---|---|---|---|
| TimesFM-2.5 | 0.014337 | 0.014337 | 0.0143 |
| Moirai-2.0 | 0.017832 | 0.017832 | 0.0178 |
| Moirai-1.1 | 0.015413 | 0.015413 | 0.0154 |
| Lag-Llama | 0.029380 | 0.029380 | 0.0294 |
| Chronos-Small-A | 0.017501 | 0.017501 | 0.0175 |
| Chronos-Mini-A | 0.017809 | 0.017809 | 0.0178 |

**One reconciliation point.** The printed figures are the **cell mean** — the mean
across 24 assets — not the pooled panel rate. Pooled, TimesFM-2.5 is 0.014732 and
Moirai-2.0 is 0.018203. The manuscript nowhere says which reading it prints. It
must, once, in the experimental-design subsection.

**Negative controls.** Reversing the inequality gives 0.9857 — fires. Computing on
the full sample instead of the test window gives 0.014052 against 0.0143: it fails
the pre-registered rule (agreement to the fourth decimal) but only just, and it
would have passed a 5e-4 tolerance. Recorded as a **weak control**: at the printed
precision, this number does not by itself identify which window it was computed on.

---

## K1b2 — 1000 draws under the default contain exactly 50 distinct values: VERIFIED on 40 cells

Pre-registered as a 40-cell subsample of the manuscript's 1600. A subsample can
refute an exact equality; it cannot re-establish it over 1600, and this does not.

Two checkpoints x two assets (SP500, EURUSD) x ten dates, 1000 draws, seed 0.
The packaged defaults read off the checkpoints are `top_k = 50`, `top_p = 1.0`,
`temperature = 1.0`, `num_samples = 20` -- so the manuscript's 1000 draws are an
override of `num_samples` and not of anything else, which is what it claims.

| configuration | distinct values in 1000 draws | cells |
|---|---|---|
| **default** | **exactly 50, on every cell** | 40/40 |
| `top_k = 10` | exactly 10, on every cell | 40/40 |
| `top_k = 4094` | 331 to 648 | 40/40 |

Both negative controls fire: the counter returns 10 when the support is 10, and
several hundred when the vocabulary is not truncated. The claim holds where it was
checked, and the arithmetic character of the diagnosis holds with it.

---

## K1b3 — the degeneracy percentages are right and the stated reason is wrong

**Percentages VERIFIED. The mechanism stated in the manuscript is FALSE.**

Recomputed from returns and stored series, with the conformal shift rebuilt from
equation (8), for all 312 pairs at α = 0.01:

| reading | (A) n₁₁ = n₁₀ = 0 | (B) n₁₁ = 0 < n₁₀ | n₁₁ = 0 | (C) n₀₁ = n₁₁ = 0 | stored NaN | printed |
|---|---|---|---|---|---|---|
| raw | **0** | 108 | 108 (34.6%) | 0 | 108 (34.6%) | 34.6% |
| corrected | **0** | 167 | 167 (53.5%) | 0 | 167 (53.5%) | 53.5% |

The arithmetic reproduces exactly. But **not one of the 312 pairs has
n₁₁ = n₁₀ = 0.** Every degenerate pair is case (B): exceedances occur, and never
consecutively. The transition table is populated in three of its four cells.

The manuscript (Section 1, and again in Section 6.3) says the test is degenerate
"because a 1% tail generates too few exceedances to populate a transition table".
That is not what happens on this panel. The table is populated; what is empty is
the single cell n₁₁, and $\hat\pi_{11}$ is therefore estimated at the boundary
$0$ — which is the *other* half of the manuscript's own sentence, and that half
is correct. `analysis/cc_column/MEMO.md` carries the same error in its
parenthetical "(n₁₁ = n₁₀ = 0)".

**This converts a sparsity argument into a structural one, and the manuscript
must say so in those terms.** The two readings differ in what they imply:

| | the published mechanism | what the panel shows |
|---|---|---|
| why the test is undefined | too few exceedances to fill the table | π̂₁₁ is at the boundary because no exceedance is followed by another |
| what the degenerate pairs look like | sparse, short windows | 20-odd exceedances in ~1,500 days, well spaced |
| what fixes it | a longer window, or a higher α | **nothing about the sample.** A longer window from the same process adds exceedances that are still not consecutive |
| what a "pass" means | absence of evidence from scarcity | absence of evidence from the estimator hitting its own boundary |

The sentence to write: *the Christoffersen independence test returns no verdict
precisely on the series whose exceedances look most independent, because the
transition probability it estimates is then at the boundary of the parameter
space where its χ²₁ calibration fails — and this is a property of the test at
α = 0.01, not of the sample size.* That is a structural limitation of the
instrument, which is the paper's argument, and not a data limitation, which is a
weaker and different claim.

It also sharpens the α-monotonicity already reported: the share falls to 0.0% at
α = 0.10 not because there are more exceedances in some incidental sense, but
because at a tenth of the tail depth consecutive exceedances become common enough
for π̂₁₁ to leave the boundary.

**Negative controls.** A constructed sequence with n₁₁ > 0 is not flagged; an
all-zero sequence is; a spread-out sequence with n₁₁ = 0 < n₁₀ is flagged. All fire.

---

## K1c — CAViaR 15/24: VERIFIED, and the vector agrees, not only the count

`analysis/phase3_dynamic/` already contains a second implementation
(Engle–Manganelli recursions written directly, Powell optimiser, different
starting values). The task's premise that this is unverified is out of date. The
pre-registered comparison is the per-asset pass/fail vector:

| model | original | second implementation | common window | vector agreement | max \|Δπ̂\| |
|---|---|---|---|---|---|
| CAViaR-AS | 15/24 | **15/24** | 15/24 | **24/24** | 7.9e-04 |
| CAViaR-SAV | 14/24 | 14/24 | 16/24 | 24/24 (22/24 common) | 1.2e-03 |

CAViaR-AS is confirmed on the count, on the vector, and on both windows. The two
CAViaR-SAV common-window disagreements are STOXX (p 0.0028 → 0.0535) and EURUSD
(0.0265 → 0.0608), both pairs straddling the 0.05 threshold on a shortened window;
neither touches a printed number.

**Negative control.** Kupiec applied to series constructed to violate at 5%
against a nominal 1% rejects on 24/24. Fires.

---

## K1d — evidence grade for reconstruction-backed material

No computation. A classification, to be carried into whatever section inherits
these objects.

**Graded WEAKER — reconstruction, not retrieved artefact:**

1. **The recalibration-baselines comparison** (`tab_baselines`, Table S.25 in the
   current supplement; Table 3 of the submitted version). Its two input CSVs were
   untracked by a repo cleanup and recovered from commit `ae79321`. With both
   restored the generator runs but emits 77 numeric tokens against the submitted
   79, misaligned from the first, and prints *"Regenerated table differs from
   committed version"*. The recovered intermediates carry 9 forecasters and 216
   pairs; the published table reflects ten. Status **NOT_EMITTED, not an
   erratum**: there is no evidence the printed values are wrong, and no artefact
   that produces them. (`analysis/provenance/RECOVERED_INTERMEDIATES.md`)

2. **The six TSFM series themselves**, on which everything else rests. They came
   from GPU inference on an A30 and sampling is not bit-reproducible across
   backends: **UNVERIFIABLE_HERE**. That is a statement about this machine, not a
   pass. By contrast the four parametric benchmarks reproduce exactly or to
   round-off (`analysis/provenance/PRODUCER_VERIFICATION.md`), except
   GARCH-N, where the upstream dividend-adjusted histories were restated after the
   forecasts were written — 24 of 24 assets **data revised**, max absolute
   difference 1.8e-01. That series cannot be reconciled without the original data
   vintage and should be graded WEAKER too.

3. **The analytic Chronos panels**, now, on the strength of K1a above: they carry
   a known one-bin defect and must be regenerated before they can be graded
   anything else.

**BLOCKED — cannot be graded as asked.** "Table I.6" has no referent in the
current document. In the submitted version's Appendix I (Robustness Details), I.6
falls between `tab:regime_stability` (I.5) and `tab:tail_closure_extended` (I.7)
and carries no `\label`, so it cannot be resolved from the `.aux` file. The two
candidates by position are the Monte Carlo simulation table
(`tab_simulation_extended`) and the calibration-fraction sensitivity table
(`tab_h15_fc_sensitivity`). **Both are panel-independent** — they generate their
own GARCH paths and read nothing from `cfp_ijf_data/`
(`analysis/provenance/STALE_TABLES.md`) — so under either reading the object is
*not* reconstruction-backed and the grade requested does not apply. Recorded and
passed over; if a specific table was meant, name it and it will be graded.
