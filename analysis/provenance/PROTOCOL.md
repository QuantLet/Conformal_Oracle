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

### The tolerance is part of the check

Rule 2 is not satisfied by a check that *can* fail. It is satisfied by a check
that can fail **at the resolution of the defect it exists to exclude**. A
tolerance chosen loosely enough turns a real guard into a false one without ever
looking broken, because its negative control still fires — on a defect larger than
the one that matters.

**Every validation tolerance is therefore declared together with the size of the
defect it cannot see.** Not the tolerance alone: the tolerance and the smallest
error that would pass it, in the units of the quantity being checked.

Two cases in this project, and they are the same failure at two scales.

- **The gate's lower band edge** at −3.500 blocks 0 of 312 cells. Its resolution
  is not merely coarse, it is infinite: no series in this panel could fail it.
  Caught by asking how many cells the check rejects.
- **The analytic estimator's validation** (R14). Section 4.4 validates the
  closed-form quantiles against full-vocabulary sampling on 40 SP500 dates and
  reports two agreements: predictive standard deviations "to within 0.3%", and
  violation rates at all four levels "to four decimal places". The defect present
  in the estimator was a one-bin offset — a **uniform translation of the support**.
  Neither announced agreement could have seen it, and for two different reasons:

  - A translation leaves the second central moment **exactly** unchanged. The
    dispersion check has not a coarse resolution against this defect but a null
    one. The 0.3% figure is not a tolerance that was too loose; it is a tolerance
    on a quantity the defect does not touch.
  - The violation-rate check ran on 40 dates at α = 0.01, where the expected
    count is 0.4 and the rate lives on a grid of 1/40 = 0.025. "Agreement to four
    decimal places" there means both routes returned the same integer, almost
    always zero. The offset is 0.249% of VaR₀.₀₁ and 0.591% of the predictive
    standard deviation; nothing on a 0.025 grid resolves it.

  The check ran, passed, and was incapable of seeing what it was written to
  exclude. Not caught by asking how often it fails; caught only by rebuilding the
  estimator from the prose.

The second case is the one the row-count discipline of Rule 1 does not reach and
the negative-control discipline of Rule 2 does not reach either. Two implementations
of the same estimator differed by one quantisation bin, and every aggregate the
project computed from them — dispersion ratios, violation rates, Kupiec counts —
agreed to the precision at which those aggregates are printed.

**What the rule requires.** Beside every reported agreement, state (i) the
tolerance, (ii) the smallest defect that would survive it, and (iii) whether the
statistic being compared is one the failure mode moves at all. Point (iii) is the
one R14 adds: a tolerance is meaningless on a quantity that is invariant to the
defect, and no choice of tolerance repairs it. If (ii) exceeds the scale of a
plausible failure mode, or (iii) fails, the check is recorded as non-informative
and reported as such, exactly as the −3.500 edge is.

### One registry is the cross-document guard

A quantity reported in both documents must not resolve to two values. The
manuscript's Section 5.3 and the supplement's S.4.1 both gave the resolution of
the same reproduction gate, in the same sentence construction, and disagreed ---
2.1e-4 against 1.3e-4. It survived a deliberate comparison of about twenty
numeric chains between the two documents, because two identical sentences
carrying different digits read as a restatement rather than a contradiction. The
$Z_2$ contradiction found in the same pass differed in wording as well as value,
which is why that one was visible.

The obvious remedy is a guard that compares, for every quantity used in both
documents, the values they carry. It was written and then removed, for two
reasons worth recording.

**It fires on the convention it is supposed to respect.** `\nMainForecasters`
is 16 and `\nSeqForecasters` is 13 because they are different panels, which is
what the panel prefix of Rule 1 exists to say. Any rule that strips the prefix to
compare stems deletes the distinction that makes the two correct, and reports
`MainPairs` against `SeqPairs` as a contradiction.

**It would not have caught the case that motivated it.** The supplement carried a
macro; the manuscript carried a *typed literal*. A comparison over the registry
sees one value and no disagreement, because the second value was never in the
registry to compare.

What prevents the class is the invariant, not a comparison: **every figure in
either document is a macro from one registry.** Two documents cannot then
disagree about a quantity, because there is one definition of it. The defect was
possible only because guard 2 read two of the four files the manuscript is
assembled from, and the figure sat in one of the two it did not read. Extending
that guard closed it; a value-comparison guard on top would add nothing and would
fail on correct work.

### Round once, at emission

Three instances, so it is a class and not an accident:

| where | printed | from the unrounded values |
|---|---|---|
| a Monte Carlo control mean, S.4.1 | 0.00071 | **0.00070** |
| a rule failure rate quoted from a display | 0.000705 read as 0.00071 | the same figure rounded twice |
| a variance ratio, Section 5.1 | 4.80, from 0.00307 / 0.00064 | **4.83** |

Each was produced the same way: a quantity was computed from figures that had
already been rounded for display, rather than from the values behind them. The
error is small every time and it is not random --- it inherits the direction of
the first rounding, so it does not average away across a table.

**A derived quantity is computed from unrounded values and rounded once, at the
point of emission.** A ratio, a difference or a percentage that a reader could
reconstruct from two printed numbers is emitted by the same script that prints
them, not left for the prose to assemble. Where the reconstruction and the
emitted value differ, the emitted one is right and the difference is the
measure of how much rounding the printed figures carry.

`scripts/paper_numbers.py` asserts this for the ratios it emits: a ratio whose
operands are also emitted is checked against the quotient of their *rounded*
displays, and a disagreement beyond the last printed place is reported, because
that is exactly the gap a reader dividing the table would fall into.

### A rate carries the grid it lives on

Three instances of one shape, which is the count at which this project turns a
note into a check.

| where | the claim | the grid it was on |
|---|---|---|
| Section 4.4 | violation rates agree "to four decimal places" | 40 dates at alpha = 0.01: steps of 0.025 |
| Section 4.4 | predictive dispersion agrees "to within 0.3%" | a statistic the defect leaves exactly invariant |
| `CONDITIONAL_PASSAGES.md` | a family sits "at 0.6x to 1.0x nominal" | 200 dates: steps of 0.5x, and 0.6 is not on it |

A rate computed from N Bernoulli trials takes values on a grid of 1/N. A figure
printed with a step finer than that claims resolution the data does not carry,
and the claim is invisible because nothing about the printed digits says how many
observations produced them.

**Every rate the manuscript prints is emitted together with the number of
observations it summarises**, by `scripts/paper_numbers.py`'s `rate()`, and
guard 6 fails the build when a printed step is finer than 1/N or when a rate
reaches the registry without an N. On its first run it caught one figure written
in the same session --- an ML exceedance rate at four decimals on 800
observations, grid 0.00125 --- and one generated table carrying the same defect,
which now prints exact counts rather than rounded rates.

The rule has a corollary about presentation, not only precision. Where rounding
to the grid would destroy the comparison a table is making, the answer is the
**count out of N**, which is exact and loses nothing, rather than a rate rounded
until the dose response disappears.

The corollary for validating an estimator against a second route: compare the
**object**, not a summary of it. Comparing per-date quantiles would have found
this in one line; comparing dispersion and coverage could not have found it at
any sample size.

### Where the negative control is planted

Rule 2 says a check must be seen to fail. It does not say *on what*, and for a
year that omission was invisible because every control happened to be planted in
the case the check reads best.

**A negative control is planted in the region the check covers worst, not in the
region it covers most typically.**

Two instruments failed this on the same day, in the same shape.

- **Guard 2**, prose literals. Its control planted `0.7391` in running prose. The
  guard replaced `\begin{tabular}...\end{tabular}` wholesale before extracting,
  so it was blind to every table written by hand in either document -- and a
  prose-only control cannot fail at the resolution of a table-only defect. It
  reported "no bare decimal literals in prose" over `0.990` and `0.988`, the
  violation rates under the inverted-sign defect, for as long as it existed.
- **Check 5** of `audit_structural_claims.py`, "N of M" claims. Its control
  planted two typed numbers, `7 of 99`. The check skipped any claim in which
  *either* side was a macro, on the reasoning that one side then came from an
  artefact. Eleven claims in the two documents had exactly that shape and none
  was ever read. Two of the eleven were wrong.

The common configuration is the point, and it is the worst one in a document:
**a literal standing beside a macro.** It reads as verified to any reader,
because half of it is; it read as verified to both instruments, for the same
reason; and it is where a stale number survives longest, because everything
around it is being regenerated. A control planted in the fully typed case cannot
see it, and the fully typed case is the one a control naturally reaches for.

The remedy is a question asked when the control is written, not when it is run:
*which inputs does this check skip, ignore, or strip before it looks?* Whatever
the answer names is where the control belongs. Guard 2's control now plants a
literal inside a tabular whose column specification carries lengths; check 5's
plants a typed numerator against a macro denominator.

This is a rule about controls, so it stands beside the four modes rather than
inside them. It is how a check ends up in the second mode --- "cannot see" ---
without anyone noticing, because a check with a control in the wrong place looks
exactly like a check that works.

### A check states its own coverage

The rule about where a control is planted has a companion, and it was found by
asking the same question of an instrument that was passing: *what does this check
skip, strip, or ignore before it looks?*

Guard 1 reads two things --- the `.log` LaTeX writes, and the text of the PDF that
shipped. Where poppler is absent it reads the PDF with pypdf, and pypdf prints
`Exceeded 5000 form XObject invocations while extracting text; further form
content is skipped`. That message says content was dropped and says nothing about
how much, so a guard that prints `pass` beneath it is reporting a verdict over an
unstated fraction of the document.

**Measured rather than assumed.** pypdf recovers text from **76 of 76** pages of
`main_R2.pdf` (158,978 characters) and **59 of 59** of `supplement.pdf`
(103,440), with no page empty or thin. The skipped content is form XObject
content, which in these documents is included graphics, not the page streams that
carry body text. A `??` was then planted twice in the real document and rebuilt:
once in the abstract, and once inside the Conclusion at page 72 of 76, well past
the point where the budget is exceeded. Both were detected, by both limbs.

What remains outside coverage is therefore narrow and worth stating exactly: a
`??` rendered *inside an included graphic*. No `\ref` in these documents can
produce one, because a reference in the source is typeset into the page stream;
it would require a figure whose own source carried an unresolved reference when
the figure was produced.

**What the rule requires.** A check that reads a document, a panel or an archive
**prints how much of it was read**, in the units of the thing --- pages,
cells, rows, files --- beside the verdict. Guard 1 now prints its page and
character coverage on every run. A pass over an unstated fraction is the same
defect as a pass at an unstated tolerance, and it is invisible in exactly the
same way: nothing in the output distinguishes a check that read everything from
one that read half.

### The four ways a check stops being evidence

Rule 2 was written against one of these and has since met three more. They are
distinct failures with distinct remedies, and collapsing them into "the check was
weak" loses the remedy each time.

| mode | what happens | instance | remedy |
|---|---|---|---|
| **cannot fail** | no case in the population would trip it | the gate's lower band edge at −3.500, blocking 0 of 312 cells | state how many cases the check rejects |
| **cannot see** | it fails informatively, but on a statistic the defect leaves invariant | R14: a support translation against a dispersion tolerance | state the smallest defect that survives, and whether the statistic moves at all |
| **cannot run** | the check exists only where it was written | three of the four audits `MIGRATION.md` requires, and `build_guards.py` itself, matched by the `/scripts/*` glob and never re-included | the build fails when the written discipline names a file git does not carry — guard 4 |
| **verdict outlives its state** | it ran, it passed, and the thing it described changed underneath | `MANIFEST.md` grading `tab_regime_sensitivity.tex` `OK` before the 2026-08-17 sign correction, still reading as a live guarantee in the working tree it does not examine | a passing verdict carries the state it was computed against, or it is re-run |

### A fifth mode, and it is not a defect of a check

The four above are ways a check fails to be evidence. This one is a way a
*number* fails, and it is the worst thing in this register because it cannot be
caught by re-running anything.

The manuscript reported a constructed pair of forecasters whose thresholds
"differ by half", one at 2.61 sigma and the other at 1.32 sigma. The construction
in `analysis/phase2/pair.npz` gives 1.46 sigma: the alternative holds 56 percent
of the honest capital, not half. The printed 1.32 is the value at which the
sentence's own word --- "half" --- comes out exact.

Every other defect in this register is a pipeline fault: a stale vintage, a
missing producer, a convention reimplemented four ways, an off-by-one in a token
map. Each was found by recomputing the object. This one would survive any amount
of recomputation of the pipeline, because the pipeline never produced it. The
number was fitted to the prose.

**There are now two, so it is not an accident.** The second is quieter and its
mechanism is the same. Supplement S.5 reported the tuned GBM-QR ablation as
"5/9 Kupiec rejections (vs. 0/9), and 88.9% Green (vs. 100%)", over what it
called "the nine series carried in the tuning ablation". The shipped grid is
8 configurations x **13** models, the summary records `n_pairs = 13` on every
row, and the emitted table on the same page prints `8/13` and `84.6%`. 88.9% is
8/9 exactly, and 8/9 is a ratio no count in the archive produces.

The two instances differ in what the number was fitted to and agree in what
made them survive:

| | the constructed pair | the tuned ablation |
|---|---|---|
| printed | 1.32 sigma, "differing by half" | 88.9% Green, over "the nine series" |
| artefact | 1.46 sigma, 56% of the honest threshold | 84.6%, over thirteen |
| what the figure was fitted to | the sentence's own word, *half* | the panel size the sentence names, *nine* |
| what made it survive | no artefact ever produced it | an artefact was re-run and its prose was not |

The second adds a mechanism the first did not have: **a clause that names the
object can be the false part.** "Those counts are over the nine series carried in
the tuning ablation, not the full panel" reads as the careful qualification a
referee looks for, and it is what made the numbers beside it look accounted for.
A wrong count with a stated object is harder to catch than a wrong count with
none, because the statement of the object is itself the reassurance.

**What the rule requires.** A figure that appears in the text and in no artefact
is not a rounding of an artefact until that has been checked in the direction the
rounding would have to go. A clause naming the object a figure was computed over
is checked against the artefact like any other claim; it is evidence about the
author's intent and none at all about the number. Where a number and a word agree suspiciously well ---
"half", "double", "an order of magnitude", "a third" --- the word is written from
the number, never the number from the word, and the artefact is named beside it.
The remedy is `PRODUCERS.tsv` and guard 5 for tables, `DECLARED_CONSTANTS.md` and
guard 2 for literals; for this class the only instrument is the discipline of
computing the figure before writing the sentence that characterises it.

The third is the one that reaches outside the machine. A rule that names its own
enforcement harness, in a repository published on Quantlet and distributed on
PyPI, while that harness is excluded from the distribution, is a rule nobody but
its author can apply — not a referee reproducing the result, and not the same
author after the migration `MIGRATION.md` describes. The discipline was written
down and shipped without the instrument.

The fourth is the reason `build_manifest.py` did not raise any of the ten
divergences in `L1_TABLE_REPRODUCTION.md`. Its verdicts were true when recorded.
It compares against the frozen submission rather than the working tree, so drift
introduced after the submission is outside its field of view by construction, and
nothing re-runs it when an input is corrected. A snapshot with no expiry is read
as a guarantee.

### The defect is in the instrument more often than in the object

Counted, not asserted, and against a stated criterion: **the census counts a
defect that reached a stored artefact or a claim in the manuscript.** A mistake
caught while writing --- a wrong column read from a table, a pattern that matched
the wrong thing on the first attempt --- is not a defect of the project; it is
the work. Without that line the count is unfalsifiable, which is the wrong
property for a number in a section about honest accounting.

Eight defects found; **five were in the checking apparatus and three in the
thing being checked.**

| where | what | how it failed |
|---|---|---|
| instrument | guard 2's `DOCS` list omitted the `\input` section files | **scope** |
| instrument | guard 2 stripped every hand-authored tabular before reading | scope |
| instrument | check 5 skipped any "N of M" with a macro on either side | scope |
| instrument | `build_manifest.py` compares against the frozen submission, not the working tree | scope |
| instrument | `paper_numbers.py --check` regenerated the artefact it checks against | circularity |
| object | thirteen supplement literals that did not reproduce | --- |
| object | two half-macro claims that did not reproduce | --- |
| object | a variance ratio printed as 4.80, obtained by dividing two already-rounded figures | double rounding |

**Two candidates rejected by the criterion, and they are worth naming so the
count can be reproduced.** While clearing the section literals, a macro for the
corrected-rate deviation was written over all cells when the sentence says "from
$T = 1{,}000$ onward", and a macro for the yellow-zone boundary used $9.5/250$
where Proposition 5.1 uses $9/250$. In both the prose was right and the new macro
wrong, and both were caught before the macro reached the document. They are the
work, not defects of it. The variance ratio is different: 4.80 was printed in
Section 5.1, so it reached a claim, and it counts.

It is classified as an **object** defect rather than an instrument one. Nothing
checked it wrongly; there was no check, because the ratio was computed by hand
from two printed figures instead of by the emitter from the values behind them.
That is the same failure as a hand-carried literal, which is what the registry
exists to prevent.

The first is the clearest of them and it is the one worth generalising. Guard 2
had no defective logic: its pattern was right, its tolerance was right, its
negative control fired. It read two files and the manuscript is four, so
Section 4 --- the section carrying the paper's thesis --- and the whole of
Section 5 had never been read by any literal check in the project's history. A
stale $\hat\rho = 0.67$ survived there through four rounds in which the same
figure was corrected everywhere else.

**The failure mode is scoping, not calibration.** Four of the five instrument
defects are that shape: a check that is correct about what it examines and
examines less than the object. That is the same failure the paper reports in its
own gate, where tail reach returns *inapplicable* on all thirteen series and the
block count is nine checks exercised and one that was not --- an instrument
whose verdict is silent about the part of the object it never reached. A wrong
threshold announces itself the first time it fires; a wrong scope never fires at
all, and its silence is indistinguishable from a pass.

This is why the coverage rule above is stated in the units of the thing read:
pages, cells, rows, files. A check that prints how much of the object it read
cannot fail this way undetected.

The circularity case was introduced and caught inside one session.
Reading a grid spacing out of `delta_by_class.py` by importing it re-ran the
module, and the module writes `delta_by_class.json` at module level. So
`--check`, whose entire function is to fail when a stored number has gone stale,
refreshed the stored number first. It could not have failed on the defect it
exists to detect, and it would have reported "numbers.tex is current" forever.
Fixed by reading the defaults with `ast` and verifying, across a `--check`, that
the JSON is byte-identical.

**What the rule requires.** A check reads its reference; it never writes,
regenerates, or imports anything that writes. Where a check needs a value that
lives in code, it parses the source rather than executing it. And when a defect
turns up, the instrument that should have caught it is examined in the same pass
as the object -- on this project's record that is where the defect is more often
than not, and an instrument nobody audits is the one place a defect can sit
indefinitely while every report says the work is clean.

### Groupings come from a column, not from a reading

The rule this project needed while writing new prose around existing figures,
and it was earned on an error caught at the moment of writing rather than at
audit. Replacing a wrong range in Supplement S.12, the sentence "the three
series with $\bar R$ above 0.35" was written -- a cut chosen after seeing the
numbers -- and guarded with an assertion that the top three separate from the
rest. The assertion fired: Lag-Llama sits at 0.357 against 0.184 below it, a
factor of 1.9, while the two truncated Chronos are at 17 and 24. The
two-order-of-magnitude gap is *inside* the proposed group, not below it. The
sentence was rewritten on the master table's own `kind` column.

**Every grouping, threshold, cut or subset that appears in the prose is a column
of an artefact, or it does not appear.** "The eight foundation-model series" is a
column. "The series with $\bar R$ above 0.35" is a reading of the data dressed as
a category, and a reader cannot tell the two apart -- which is the same property
that makes the fifth pattern's second instance hard to catch.

Where no column exists and the grouping is genuinely needed, it is a declared
constant with the choice stated at the point of use, exactly as
`DECLARED_CONSTANTS.md` admits the detection severity cut at 0.005. What is not
allowed is the third option, which is the one that happens by itself: a cut
inferred from the numbers, written as though it were a property of the objects.

This matters most where a lot of new prose is written around figures that already
exist, because that is the configuration in which both of the half-macro defects
survived. A sentence composed after looking at a table will land on the grouping
the table happens to show.

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
