*Draft for Daniel to send. I cannot send email. Addressed to Wolfgang Härdle,
Stefan Lessmann, and the co-authors on IJF-D-26-00531. Adjust tone and salutation
as you see fit — the content below is what I would want to know if I were on the
author list.*

---

**Subject: Errors found in our TSFM VaR pipeline — the headline results do not stand**

Dear all,

I have audited the forecast pipeline behind our tail-risk paper and found errors
that change its main empirical results. I am writing before deciding anything
about resubmission, because the findings affect work of ours that is already
public and I do not want any of you to learn this from a referee.

**The short version.** Four of the numbers the paper is built on were artefacts
of our own code, not properties of the models we evaluated.

| | as submitted | corrected |
|---|---|---|
| Moirai 2.0, violation rate at 1% | 0.988 | 0.017 |
| TimesFM 2.5, violation rate at 1% | 0.990 | 0.014 |
| Chronos-Small, violation rate at 1% | 0.388 | 0.018 |
| GJR-GARCH, violation rate at 1% | 0.004 | 0.019 |

**What went wrong, in each case.**

*Moirai 2.0 and TimesFM 2.5.* Our code stored the Value-at-Risk as
`-ppf(alpha, ...)` rather than `ppf(alpha, ...)`, so the threshold pointed at the
wrong tail. This is verifiable arithmetic rather than an inference: the stored
column is reproduced by the negated quantile to ten decimal places from
parameters saved in the same file, and the empirical exceedance rate at the
stored threshold reproduces our published 0.988 and 0.990 to four decimals. Eight
of our ten forecast directories store the correct sign; these two do not.

*Chronos.* The Chronos sampling routine defaults to `top_k = 50`, which truncates
the model's predictive distribution to 50 of its 4093 bins before any quantile is
computed. Our 0.388 was a property of that default. Read from the model's own
predictive distribution instead of sampling it, the rate is 0.018.

*GJR-GARCH.* Our benchmark applied `t.ppf(alpha, 5)` — degrees of freedom
hard-coded at five, and the raw rather than standardised quantile — inflating
every threshold by about 45%. That is why it sat in the Basel green zone on all
24 assets. With the Gaussian innovation the paper actually describes, it runs at
0.019 and fails unconditional coverage on most assets.

**Why it survived our checks.** Conformal recalibration restores near-nominal
coverage regardless of the forecaster it is applied to — that is what the
guarantee says. Applied to the sign-inverted series it produced 19 of 24 assets
in the Basel green zone. Every backtest in the submitted paper was computed after
recalibration, so every one of them passed. The errors were invisible precisely
because the method worked.

I would also point out something uncomfortable. Our submitted Table 1 partitions
the panel on |q_V|/|VaR_raw| > 1, and Panel B — the four models we labelled
"effective replacement" — is exactly the four defective series. The statistic
separated them correctly. We read the separation as a property of the models.

**What this means for the paper.** The empirical contribution is much weaker than
we claimed. Corrected, the foundation models are neither catastrophically
miscalibrated nor competitive: they sit alongside GARCH at the 1% level, and a
per-asset CAViaR fitted the ordinary way beats all of them. Two further checks
have come back negative — the q_V diagnostic turns out to be a rank-preserving
transform of the violation rate rather than independent information, and adaptive
conformal inference outperforms our static shift on coverage. The theorem is
untouched; the empirical case for the method is not what we wrote.

**What I think we should do.**

1. Post a correction notice on the SSRN preprint (6757685), which currently
   carries the pre-audit numbers publicly. I can draft it.
2. Correct or withdraw the affected Quantinar material.
3. Decide together what, if anything, we submit. My current view is that the
   defensible paper is a narrow one about the sampling default — it reproduces
   from a public checkpoint without our data, and it is the one finding that does
   not depend on our panel being right.

Everything is reproducible: corrected series, the diagnostic scripts, and a
validation procedure that catches all three defect classes without needing the
original hardware. I am happy to walk any of you through it.

I am sorry to be bringing this rather than a revision. I would rather we correct
it ourselves than defend it.

Daniel
