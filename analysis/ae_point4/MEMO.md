# Phase 1 — AE point 4: does recalibration help everywhere?

**No. And where it fails is exactly where the AE said it would.**

Everything below comes from `run_ae_point4.py` over 960 cells
(10 models × 24 assets × 4 α). Single-split figures reproduce the paper's
`all_results.csv` to ~1e-5; rolling figures are re-scored from the stored
forecast series, since `rolling_vs_static.csv` stores coverage but not QS. No
model inference was re-run.

---

## 1. The headline

At α = 0.01, out of 240 model–asset pairs:

| Estimator | pairs worse (ΔQS < 0) |
|---|---|
| single split | **27 / 240** (11%) |
| rolling | **84 / 240** (35%) |

The pattern holds at every α. Deterioration is not noise around a positive mean:
it is concentrated, systematic, and it separates the two estimators sharply.

## 2. The AE's hypothesis, tested directly

Restricting to pairs whose **raw** forecast already passes Kupiec on the test set
(p > 0.05) — "already well calibrated", in the AE's words:

| α | estimator | n | worse | mean ΔQS | median | t-test p |
|---|---|---|---|---|---|---|
| 0.01 | single split | 44 | 20 | −0.38% | +0.16% | 0.43 |
| 0.01 | **rolling** | 44 | **41** | **−6.81%** | −5.99% | **3.9e-12** |
| 0.025 | single split | 50 | 21 | −0.25% | +0.09% | 0.17 |
| 0.025 | **rolling** | 50 | **46** | **−2.76%** | −2.29% | **5.4e-10** |
| 0.05 | single split | 73 | 29 | −0.01% | +0.05% | 0.71 |
| 0.05 | **rolling** | 73 | **71** | **−1.69%** | −1.45% | **1.5e-14** |
| 0.10 | single split | 70 | 36 | −0.03% | −0.00% | 0.28 |
| 0.10 | **rolling** | 70 | **64** | **−1.01%** | −0.86% | **9.0e-14** |

Two different answers, and the distinction is the finding:

- **Single split**: a wash. The mean change on well-calibrated bases is
  statistically indistinguishable from zero at every α (p = 0.17–0.71), the
  median is positive at three of four levels, and losses are individually small.
  The correction is close to free when it is not needed.
- **Rolling**: a systematic, highly significant loss. 41 of 44 well-calibrated
  pairs get worse at α = 0.01, mean −6.8%, p ≈ 4e-12 — and the same at every α.

The cross-tabulation says it without the test: at α = 0.01 the single-split
deterioration rate is 45.5% among raw-not-rejected pairs versus 3.6% among
rejected ones; for rolling it is **93.2% versus 21.9%**.

## 3. Regression

`rel_ΔQS = a + b · |π̂_raw − α|`, restricted to the region where the base
forecast is usable (|π̂_raw − α| < 0.05, grid-interface failures excluded):

| α | estimator | n | slope b | s.e. | intercept a | R² |
|---|---|---|---|---|---|---|
| 0.01 | single split | 144 | 6.95 | 0.39 | **−0.0247** | 0.69 |
| 0.01 | rolling | 144 | 9.01 | 0.60 | **−0.0947** | 0.61 |
| 0.025 | single split | 141 | 3.07 | 0.11 | −0.0162 | 0.84 |
| 0.025 | rolling | 141 | 3.66 | 0.17 | −0.0436 | 0.76 |
| 0.05 | single split | 122 | 1.33 | 0.08 | −0.0068 | 0.70 |
| 0.05 | rolling | 122 | 1.79 | 0.13 | −0.0256 | 0.61 |
| 0.10 | single split | 118 | 0.56 | 0.04 | −0.0040 | 0.69 |
| 0.10 | rolling | 118 | 0.79 | 0.06 | −0.0155 | 0.62 |

The intercept is the answer to the AE: it is the expected change in QS for a base
forecast that is *perfectly* calibrated. It is negative everywhere, and three to
four times more negative for rolling than for single split.

**A caveat that must be stated in the paper.** Run on the pooled sample with the
99%-violation models included, the same regression gives slope ≈ 1.03 and
R² ≈ 0.98. That is close to an identity, not evidence: when a model violates on
99% of days, QS_raw is dominated by the very gap that defines |π̂_raw − α|. Those
rows are in `regression_results.csv` for completeness and should not be quoted.
The restricted rows above are the informative ones.

## 4. Where the deteriorations live

At α = 0.01, by model (24 assets each):

| Model | worse (split) | worse (rolling) | median ΔQS split | median rolling | raw pairs passing Kupiec |
|---|---|---|---|---|---|
| Chronos-Mini | 0 | 0 | +86.3% | +86.7% | 0 |
| Chronos-Small | 0 | 0 | +84.9% | +85.5% | 0 |
| TimesFM-2.5 | 0 | 0 | +97.2% | +98.1% | 0 |
| Moirai-2.0 | 0 | 0 | +97.0% | +97.7% | 0 |
| Lag-Llama | 1 | 2 | +11.3% | +8.6% | 0 |
| GJR-GARCH | 4 | 13 | +3.4% | −1.2% | 7 |
| GARCH-N | 6 | 14 | +3.5% | −2.7% | 8 |
| EWMA | 3 | 16 | +4.3% | −1.4% | 5 |
| **Hist-Sim** | 4 | **24** | +1.8% | **−8.8%** | 11 |
| **Moirai-1.1** | **9** | 15 | +0.7% | −2.3% | **13** |

The split is perfectly clean: **every deterioration is in a model that was
already roughly calibrated**, and no deterioration occurs in any model whose raw
forecast is broken. Under the rolling estimator, Historical Simulation loses on
**all 24 assets**, median −8.8%.

## 5. Two consequences beyond AE point 4

**(a) This independently confirms Referee 1's point vii.** The rolling estimator
is the paper's strongest empirical headline and its weakest theoretical object.
It now also carries a measurable cost: it buys coverage by systematically
degrading the quantile score precisely where the base forecast needed no help.
Presenting rolling as the recommended default is no longer defensible; it should
be presented as the option that trades sharpness for coverage, with this table as
the price list.

**(b) It sharpens the Phase 5 interface story.** Moirai-1.1 and Moirai-2.0 are
the within-family control, and they sit at opposite extremes of this table:

- Moirai-**1.1** (sample-based interface): 13 of 24 raw pairs already pass
  Kupiec — more than any other model — and the correction's median gain is
  **+0.7%**, with 9 pairs made worse.
- Moirai-**2.0** (quantile-grid interface): 0 of 24 raw pairs pass, and the
  correction's median gain is **+97.0%**, with 0 pairs made worse.

Same family, same pretraining corpus, opposite behaviour. The value of the
correction is a function of the *interface*, not the architecture — which is the
repositioned contribution, now visible in the ΔQS distribution rather than
asserted.

## 6. How to write it up

Not as a limitation. The correction has a domain of validity, and this measures
it: it is a repair for forecasts whose tail interface is broken, and it is close
to free (single split) or actively harmful (rolling) when applied to a forecast
that is already calibrated. That is a deployment rule — monitor R and Kupiec on
the raw forecast; apply the correction when the raw forecast fails, not by
default — and it replaces the discarded binary classification with something
measured.

## 7. One reconciliation needed before Phase 5

The cross-check reproduces `all_results.csv` exactly for all nine main models
(qV relative difference 0.000000). **Moirai-1.1 is the exception**: its legacy
`moirai11_full_results.csv` uses a test sample one observation shorter than the
same loader gives for every other model, in 48 of 96 cells. At α = 0.01 that is
not cosmetic — qV is an extreme order statistic, so one observation moves it by
up to 28% on some assets (e.g. ICLN 0.00428 vs 0.00333).

The numbers in this memo use a single consistent computation across all ten
models. But since Moirai-1.1 is about to become the paper's within-family
control, its legacy pipeline must be reconciled with the main one before Phase 5
quotes any of its figures. This is worth doing anyway — it is itself a small
demonstration of the tail-sparsity sensitivity the paper argues for.

---

## 8. Does the QS loss at least buy a Basel zone change?

Asked of the 84 pairs the rolling estimator degrades at α = 0.01. Rows are the
raw zone, columns the zone after correction:

```
          → Green   → Yellow
Green        48        2
Yellow       33        1
```

| Outcome | Pairs | Reading |
|---|---|---|
| Yellow → Green | **33** | A defensible trade: QS paid for a zone upgrade. |
| Green → Green | **48** | **Pure loss.** The pair was already Green; the correction degraded QS and bought nothing. |
| Green → Yellow | **2** | **Strictly worse on both axes.** |
| Yellow → Yellow | 1 | Loss, no upgrade. |

So **50 of 84 degradations (60%) are pure loss**, and 48 of those 50 were pairs
that were already in the Green zone before any correction was applied. Only 39%
of the degradation is a trade a risk manager could knowingly accept.

This is the sharpest available statement of the result: on those 50 pairs the
correction is not a trade-off, it is simply the wrong intervention.

## 9. The decision rule, evaluated

The natural gate follows directly: **apply the correction only when the raw
forecast actually fails a backtest** (Basel zone worse than Green, or Kupiec
rejected). At α = 0.01 that gate applies to 196 of 240 pairs and skips 44.

| α | estimator | applied | degraded when applied | skipped | degradations avoided | zone upgrades kept / total |
|---|---|---|---|---|---|---|
| 0.01 | single split | 196 | 7 | 44 | **20 of 27** | 160 / 160 |
| 0.01 | **rolling** | 196 | 43 | 44 | **41 of 84** | **176 / 176** |
| 0.025 | rolling | 239 | 74 | 1 | 1 | 148 / 148 |
| 0.05 | rolling | 240 | 84 | 0 | 0 | 1 / 1 |
| 0.10 | rolling | 240 | 75 | 0 | 0 | 0 / 0 |

At α = 0.01 the gate removes **half the rolling estimator's damage (41 of 84)
while retaining every single one of the 176 Basel zone upgrades** — the cost on
the regulatory axis is exactly zero. The three skipped pairs that would have
gained are worth a median of +0.16%; the 41 avoided losses are worth a median of
−5.99%.

**A limitation to state plainly.** The gate only bites at α = 0.01. At
α ≥ 0.025 nearly every raw forecast already fails its backtest, so the gate
applies to almost everything and avoids nothing. The decision rule is meaningful
precisely in the extreme tail — which is where this paper lives, but the claim
should be scoped to it rather than stated generally.

---

### Files

| File | Contents |
|---|---|
| `pairs_long.csv` | 960 cells, all quantities |
| `deteriorating_pairs.csv` | the 326 cells with ΔQS < 0 under either estimator |
| `well_calibrated_test.csv` | §2 |
| `regression_results.csv` / `.txt` | §3, all specifications |
| `tab_deterioration.tex` | LaTeX table of deteriorating pairs |
| `tab_crosstab_static.tex`, `tab_crosstab_roll.tex` | §2 cross-tabulation |
| `fig_dqs_scatter.png` | **the figure for the paper** — usable region |
| `fig_dqs_scatter_full.png` | full range, appendix |
| `summary.md` | machine-generated numbers behind this memo |
