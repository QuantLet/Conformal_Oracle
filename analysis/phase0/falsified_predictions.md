# Falsified pre-registered predictions

A different object from `retracted_hypotheses.md`, and kept separate on purpose.

That file records **defects**: a claim asserted, then found wrong, with the
evidence that killed it. Three of its entries share one mechanism — correct
arithmetic on the wrong object (R3, a rectangular matrix read as symmetric; R6, a
grid too narrow to carry the constraint; R9, series medians where the gate uses
cells). The value of that list is the coherence of the pattern it documents.

This file records something else: a prediction written down **before** the data
were seen, which the data then contradicted. Nothing was done wrong. The design
worked exactly as designs are supposed to work, and the entry exists so that the
prediction cannot be quietly reinterpreted after the fact.

Mixing the two would dilute the pattern in the first file and inflate the
retraction count in the second, where the cover letter's table of withdrawn
claims does its work through coherence.

## F1 — the direction of the leaf-size effect

**Pre-registered (drafts/prereg_ml.md, P3).** The paired configurations were
declared as "library default" against "leaf size lowered", on the reasoning that
finer leaves give a higher-resolution predictive object and therefore a better
1% tail. The sharper prediction that followed was that LightGBM (default
`min_data_in_leaf = 20`, the coarser setting) and the quantile forest (default
`min_samples_leaf = 1`, the finest) would move in **opposite directions** from
their defaults.

**Falsified by the run.** For LightGBM the tail improves as leaf size *rises*, not
falls: pi-hat over the grid {1, 5, 20, 100, 500} is 0.0588, 0.0600, 0.0300,
0.0162, 0.0062. Lowering the parameter from its default of 20 makes coverage
twice as bad; raising it to 500 brings pi-hat to 0.62 of nominal. The direction
written into P3 is wrong.

**Why, and it is not a detail.** At alpha = 0.01 with a 1,000-observation window
there are about ten training points below the conditional 1% quantile. Fine
leaves fit that handful of points; the resulting quantile is an artefact of a
few observations rather than an estimate. Coarse leaves pool, which at this
level helps. This is the tail-sparsity constraint of Remark 3.1 appearing in a
learner rather than in a recalibration layer, and it means the leaf parameter
governs tail *estimability*, not merely tail *resolution*.

**What this does to the design.** The pair remains the object and the single
varied parameter is unchanged; only the label of which member is the "tail
configuration" flips. No threshold in P7 is moved: the default still has to reach
twice nominal for the positive reading, and it does (3.0x). Recorded here rather
than edited into the pre-registration, per its own rule.
