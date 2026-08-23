# Dose–response report

Separate from the panel report by design, so that the readings can be seen to
have been fixed before the panel was run. Four assets, 200 dates, 8,000 cells in
the leaf arm and 5,600 in the knob arm. **No figure here is a panel number and
none enters the manuscript as one.**

Readings (i)–(iii) were fixed in `drafts/prereg_ml.md` before the leaf arm ran.
Reading (iv) was fixed in Amendment 1, written after the knob arm had produced
output but before it was read, and bound to the file's SHA-256.

---

## The result

| reading | outcome |
|---|---|
| (i) monotone response to the leaf-size parameter | **holds from the second grid point up**, not across the whole grid |
| (ii) the documented knobs do not move the tail | **fails on all three** |
| (iii) the centre is unchanged across the grid | **holds** |
| (iv) `num_leaves`, a second granularity parameter, moves the tail the same way | **contradicted** |

Two of four as predicted. The two failures are more informative than the two
successes, and together they move the result out of Section 4's category.

## Reading (i): monotone, with the fine end excepted

LightGBM, pooled over the four assets:

| `min_data_in_leaf` | 1 | 5 | **20 (library default)** | 100 | 500 |
|---|---|---|---|---|---|
| π̂ | 0.0588 | 0.0600 | **0.0300** | 0.0162 | 0.0062 |
| π̂/α | 5.9 | 6.0 | **3.0** | 1.6 | 0.6 |

From leaf 5 upward the response is monotone and spans a factor of **9.6**.
Between leaf 1 and leaf 5 it is not: π̂ rises from 0.0588 to 0.0600. The
difference is smaller than one standard error on 800 cells (≈0.008), but it is
recorded as a non-monotonicity rather than dissolved by that observation. The
manuscript will not say "all readings hold".

Quantile random forest spans 0.0062 to 0.0100 across the same grid — a factor of
1.6, and no trend.

## Reading (iii): the centre does not move

Median over training standard deviation ranges 0.0358–0.0481 for LightGBM across
the whole grid, a spread of 0.012 standard deviations, while the tail moves by a
factor of nearly ten. The structural signature of Section 4 — a tail that moves
while the centre does not — is present.

## Reading (ii): fails, on all three control knobs

Each knob moved alone, leaf size held at the library default of 20. π̂ at that
default is 0.0300.

| knob | swept range | π̂ range | factor | verdict at the 1.5 threshold |
|---|---|---|---|---|
| `learning_rate` | 0.03 → 0.30 | 0.0162 → 0.0538 | **3.31** | fails |
| `n_estimators` | 50 → 400 | 0.0213 → 0.0538 | **2.53** | fails |
| `max_depth` | 3 → 6 → none | 0.0188 → 0.0300 | **1.60** | fails |

This is the reverse of the Chronos case. There, temperature across a fourfold
range moved predictive dispersion by 0.001 and nucleus sampling by 0.006, so an
analyst sweeping the documented parameters would correctly report the model as
robust while the tail was destroyed by a parameter nobody varies. Here **every
documented parameter moves the tail**, and by factors between 1.6 and 3.3.

There is therefore no configuration trap in this family. An analyst sweeping the
ordinary knobs would find the 1% tail unstable immediately.

## Reading (iv): contradicted, and it identifies the mechanism

`num_leaves` raised from its default of 31 to 127 leaves π̂ at **0.0300** — the
default value, ratio 1.00 to three decimals.

The pre-registered prediction was that a second granularity parameter would move
the tail in the same direction as making leaves finer. It does not move it at
all. Taken with reading (i), this separates two things the original design
conflated:

- `min_data_in_leaf` sets **how many observations back each leaf estimate**. It
  moves the tail by a factor of 9.6.
- `num_leaves` sets **how many leaves there are**. It moves the tail not at all.

The parameter that matters is the one that controls the sample size behind the
tail estimate, not the one that controls granularity. The mechanism is
**estimability under sparsity**, not resolution. At α = 0.01 on a
1,000-observation window there are about ten training points below the
conditional 1% quantile; what governs the estimate is how many of them stand
behind each leaf.

## What this does to the placement

Under P7 the default reaches 3.0× nominal, which is the threshold for the
positive reading. But the positive reading in P7 was written for a *configuration
trap*: a documented default that silently breaks the tail while a sensitivity
analysis over the swept parameters looks flat. Reading (ii) removes that
description. Nothing is silent here.

**The result is therefore not a second instance of Section 4's mechanism, and
placing it beside Section 4 would describe a mechanism that was not demonstrated.**
It is a direct instance of the tail-sparsity constraint of Remark 3.1, which the
paper states for the recalibration layer and which this shows binding on an
estimator fitted to the tail. That is where it goes, and the placement is now
forced by the data rather than chosen.

The bounded claim of P7's negative branch survives in modified form and is worth
stating as the finding: **a library default reaches the tail only when it governs
the support or the effective sample size of the predictive object, and it
constitutes a trap only when the parameters an analyst would sweep leave the tail
alone.** Chronos satisfies both conditions. LightGBM satisfies the first and
fails the second.

## What is not claimed

Four assets. No panel number. Nothing here about where gradient boosting ranks
against GARCH or against a foundation model, and the full panel has not been run.
