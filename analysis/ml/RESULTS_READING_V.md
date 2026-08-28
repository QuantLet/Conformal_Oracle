# Reading (v) and the intermediate range — results

Run 2026-08-28 against `analysis/ml/PREREG_READING_V.md`. Unit: one cell is one
estimator x one asset x one `min_data_in_leaf`, at alpha = 0.01 over 200 dates.
40 cells = 2 x 4 x 5. **Not** the 312-cell sequence panel; no count here is
pooled with a count from there.

All 40 cells carry 200 finite dates. Nothing dropped, so the exercise is not
blocked on its own falsification condition.

## 1. Reading (v): BLOCKED, one direction settled

The pre-registered object is the 24-asset panel and it does not exist.
`analysis/ml/series/lgbm_default/` is empty; only the 4-asset dose-response ran.

Most negative scale ratio, median over dates of $\hat q^{lo}_{0.01}/\hat\sigma_t$,
over the 40 cells: **−2.871** (quantile forest, SP500, `min_data_in_leaf = 500`).
The lower scale edge is −3.500, so the margin is **0.629** and no ML cell binds it.

A subset can only settle this one way. Finding a cell below −3.500 would have
been an existence claim and would have made the four S7.2 passages false
immediately. Not finding one on 4 of 24 assets is not the pre-registered band 3,
and reading (v) stays **BLOCKED**. `blind_gate.py` is not re-run on the partial
panel: its band logic is written against the 24-asset object, and reporting its
verdict on 4 assets would be R3 again — a correct calculation on the wrong unit.

## 2. The intermediate range: the third pre-registered outcome, a split

The two estimators land on opposite sides.

| estimator | cells | scale-ratio range | in the gap $(-1.947,\,-0.283)$ | outside the upper edge $-1.800$ |
|---|---|---|---|---|
| LightGBM | 20 | [−2.748, −1.264] | **15** | **15** |
| quantile forest | 20 | [−2.871, −1.997] | **0** | **0** |

The pre-registration named this and said what to do with it: a split is reported
per estimator, and the margin paragraph is false for the family that lands in the
gap regardless of what the other does, because one counterexample empties a claim
that a range contains nothing.

**So S7.2's margin paragraph is false as written.** "The separation here is clean
because this panel contains no series that is *moderately* misspecified" holds
for the 312-cell sequence panel and does not survive the LightGBM family: 15 of
its 20 cells sit between the worst well-specified cell at −1.947 and the best
truncated cell at −0.283, at leaf settings of 1, 5, 20 and 100. The quantile
forest is not a counterexample — every one of its cells sits inside the band,
below −1.997 — which is why the claim has to be stated per estimator rather than
per family.

## 3. Table 2's fourth row: recomputed, and unchanged

`CONDITIONAL_PASSAGES.md` calls this the row a referee would find, on the ground
that a family in the intermediate range "changes the false-positive count at
every candidate edge".

Recomputed, it does not. **Zero of the 40 cells lie between −1.940 and −1.800**,
the strip the tightening adds:

- the 15 LightGBM cells that fail are at −1.782 and above, already outside the
  current edge of −1.800, so the tightened edge blocks nothing the standing edge
  does not;
- every quantile-forest cell, and the 5 remaining LightGBM cells, sit at −1.996
  or below, so both edges pass them.

No ML cell changes verdict between the two edges, so the false-positive count at
−1.940 is unchanged and the 0.007 margin above the worst well-specified cell is
untouched by this family. The row stands. The reason it stands is worth stating,
because it is not the reason it was expected to: the ML cells are not *near* the
tightened edge, they are well clear of it on one side or the other.

## 4. A resolution finding, and it is the R14 class again

`CONDITIONAL_PASSAGES.md` describes the quantile forest as sitting "at 0.6x to
1.0x nominal on four assets". At alpha = 0.01 over **200 dates** the expected
violation count is **2**, so $\hat\pi/\alpha$ lives on a grid of $1/200/0.01$,
that is **multiples of 0.5**. The observed values are exactly 0.0, 0.5, 1.0 and
1.5 and nothing else. **0.6x is not a value this panel can produce.**

Measured ranges on that grid: quantile forest 0.0 to 1.5, LightGBM 0.0 to 9.5.

This is the second mode again, in the ML exhibit: a coverage statistic quoted to
a precision its own sample size cannot carry, exactly as Section 4.4's violation
rates were quoted "to four decimal places" on 40 dates. The scale ratio does not
have the problem — it is a median over 200 dates of a continuous quantity — which
is why the split in section 2 above is reportable and the "0.6x to 1.0x" phrasing
is not. Any coverage claim about the ML exhibit is stated in multiples of 0.5
nominal, or it is stated on a longer window.

## Consequences, to be applied before the sections are written

| location | verdict |
|---|---|
| S7.2 "blocks 0 of 312 cells" | stands; the ML cells do not reach the lower edge, and reading (v) is BLOCKED not resolved |
| S7.2 "the scale check is one-sided" | stands, and is now **sharper**: the ML family exercises the upper edge 15 times, so the check is one-sided in *which edge binds*, not in whether the check ever fires |
| S7.2 "whether a lower edge is needed cannot be determined from these data" | stands |
| S7.2 "nine checks exercised and one that was not" | stands |
| S7.2 margin paragraph, "no series that is moderately misspecified" | **FALSE as written.** Revise per estimator: 15 of 20 LightGBM cells in the gap, 0 of 20 quantile-forest cells |
| S7.1 Table 2 row 4, tightened edge −1.940 | **recomputed, unchanged.** 0 of 40 cells lie in the added strip |
| S6 Proposition 6.2 discussion, $\delta^\star$ | unaffected, as declared — analytic, not panel-derived |
