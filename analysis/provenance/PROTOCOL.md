# Computation protocol

Standing rules for this project, each added because it was broken.

## Rule 1 — name the object before the calculation

**Every computation declares its unit of analysis in writing before it is run.**
The four units in this project are confusable and have already produced three
retractions:

| unit | what one row is | count at alpha = 0.01 |
|---|---|---|
| **cell** | one series on one asset | 312 (13 x 24), or 384 for the main panel |
| **series** | one forecaster, aggregated across assets | 13 sequence, 16 main |
| **pair** | two forecasters compared, on common dates | 30 benchmark-vs-TSFM, 55 all-pairs |
| **panel** | one forecaster pooled across assets and dates | 36,588 obs (foundation), 38,473 (benchmark) |

Aggregating before testing changes the answer. A statistic computed on series
medians is not the same statistic computed on cells, and the gate operates on
cells.

### The three retractions this rule exists for

All three were **correct arithmetic on the wrong object**. In each case I checked
*how* I was computing and not *what* I was computing it over.

- **R3** — read the Diebold--Mariano matrix as symmetric and took its upper
  triangle. It is rectangular, 5 benchmarks x 6 foundation-model series. Unit
  confusion: pair-as-unordered versus pair-as-ordered. Gave 12 of 15 instead of
  18 of 30.
- **R6** — declared the unimodal construction infeasible on a candidate support
  capped at |x| <= 4. The object was the grid, not the model, and the grid was
  too narrow to carry the compensating mass. Gave INFEASIBLE for a feasible
  programme.
- **R9** — calibrated the gate's scale band against per-**series** medians when
  the gate evaluates per-**cell**. The well-specified population runs to -2.085
  by series and -1.947 by cell. Recommended an edge that blocks 11 well-specified
  cells while reporting zero false positives.

### R6 is the case the row count does not catch

R3 and R9 are caught by declaring the expected number of rows: an ordered-pair
matrix has 30 cells and an unordered one 15; a cell panel has 312 rows and a
series panel 13. R6 is not. The linear programme had the same number of
decision variables under a grid capped at 4 and a grid capped at 32; only the
*range* of the object differed, and an infeasibility verdict came back from a
correct model on a malformed support.

**A second field is therefore mandatory beside the row count: what varies between
rows, and over what range.** Declaring "the candidate support, spanning |x| <= 4"
makes the defect visible before the solver is called, because the question "is 4
wide enough to hold the mass this constraint needs?" is then on the page.

### What the rule requires

1. Write the unit in the pre-registration, in the words of the table above,
   before running anything.
1a. Write, beside it, **what varies from row to row and over what range** ---
   the parameter swept, the grid, the support, the window. A row count alone does
   not distinguish a correct object from a truncated one.
2. State the expected row count for that unit, and check it against the data
   before reading any result.
3. When a result is aggregated, say at which step: cells -> series -> panel is
   not the same as cells -> panel.
4. A figure quoted in the manuscript carries its unit or it is not checkable.
   `paper_numbers.py` enforces this through macro naming: `Main`, `Seq`, `Gap`
   prefixes. Extend the convention rather than adding an untagged macro.

## Rule 2 — no check reports a pass until it has failed on a case built to fail

See `scripts/build_guards.py`. Each guard runs its negative control first and
reports BROKEN if the control does not fail. Three checks in this project
returned false passes before this rule: the substring provenance screen,
`audit_prose_numbers.py`, and the undefined-reference check.

A check that cannot fail informatively is not evidence when it passes. This
applies to the project's own instruments, and it has already caught one: the
lower edge of the gate's scale band blocks 0 of 312 cells and therefore carries
no evidence when a series passes it. Reported in Section 7 rather than quietly
adjusted.

## Rule 3 — pre-register, then run

Write down the expected result and what would falsify it, before computing. Files
named `PREREGISTRATION.md` under `analysis/*/`. A miss is recorded in
`analysis/phase0/retracted_hypotheses.md` with the evidence that killed it.

## Rule 4 — never assert a number not recomputed in the current session

If a figure is carried forward from earlier context, mark it as such.

## Rule 5 — read the auxiliaries before deleting them

`latexmk -C` destroyed the one log that would have settled how a stale-`.aux`
build shipped with four unresolved references. Auxiliaries are evidence until the
question they bear on is closed.
