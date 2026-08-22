# Contribution statement

## The thesis

A coverage backtest cannot identify the source of the miscoverage it detects, so
a forecast series must be validated structurally before it is calibrated or
evaluated.

## The three results that support it

1. **A formal non-identification result.** There exist a correctly specified
   forecaster and a structurally defective one that induce identically
   distributed exceedance processes, so no test measurable with respect to the
   natural filtration of that process — Kupiec, both Christoffersen components,
   the Basel traffic light — has power exceeding its size against the
   alternative, at any sample size. *(Phase 2. Not yet built. If the construction
   does not go through, this document is void and the paper reverts to T1.)*

2. **A controlled exhibit that the failure is reachable in practice.** Holding
   weights, contexts, seeds and dates fixed and varying only the sampling
   configuration of a public Chronos-T5 checkpoint, predictive dispersion
   recovers monotonically in `top_k` (0.121, 0.331, 0.958, 1.087) and 1000 draws
   under the shipped default contain exactly 50 distinct values in all 1600
   cells, while the 1% violation rate moves from 0.3884 to 0.0175 and the median
   barely moves. *(Artefacts: `Quantlets/CO_chronos_sampling/tab_dose_response.csv`,
   `analysis/chronos_sampling/coverage_SP500.csv`. Recomputed from raw parquet
   this session.)*

3. **The identified complement.** What the exceedance process cannot separate,
   the predictive object can. Ten checks partition into what is identified from
   the series and returns alone and what requires the predictive object; the gate
   blocks 4 of 13 series here. *(Artefact:
   `analysis/provenance/PROMOTION_GATE.csv`, `scripts/promotion_gate.py`.)*

## Argument on the recommendation

**I accept the recommendation, with one amendment and one condition.**

### Why non-identification is the right thesis

The decisive property is that it is the only claim in the paper that does not
carry "in our panel" as an implicit qualifier. T1 is a fact about two checkpoints
and 24 assets; T2 is a fact about 16 forecasters and 24 assets. Both are
falsifiable by someone else's data and neither generalises by construction.
Non-identification, if it can be proved, is a property of the exceedance process
and holds for every panel anyone will ever run.

It also fixes the weakest thing about the current paper. The ten structural
checks are, as written, a checklist — ten items a practitioner would endorse,
with seven declared bands, defended by the argument that "the alternative
provably cannot do this job". That defence is currently an assertion. Under the
thesis it becomes a derivation: the checks are what remains identified once the
proposition has removed the exceedance process from consideration. The gate stops
being ten good ideas and becomes the complement of a proved blind spot. Section 7
of the brief's own test — that the practical rule should follow from the
proposition rather than sit beside it — is satisfied for the gate only under this
framing.

And it is the framing that survives the rejection. The paper was rejected for
doing three things at once; the three things are three *claims*, and the reason
they cannot be merged is that they have different scopes. Choosing the one with
the widest scope and demoting the others to exhibit and consequence is the only
arrangement in which they stop competing.

### Amendment: T3 does not currently follow, and must be re-derived

The recommendation calls T3 "the practical consequence". As the paper now stands
it is not a consequence of anything; it is a separate empirical finding about
when recalibration helps, reported next to the identification argument rather
than derived from it. On the brief's own test that is a failure.

It can be made to follow, and this is the amendment I propose. The conformal
shift targets coverage. Applying it is justified only when miscoverage is the
actual defect. The instrument that would establish that is a coverage backtest —
which the proposition says cannot identify the source. So the indication problem
is not an empirical discovery about 312 pairs; it is a corollary. Recalibration
is an intervention whose indication is exactly the quantity the proposition
proves unidentifiable, and the measured cost (44 of 174 deteriorations avoided,
20 of 205 Basel upgrades lost, against an oracle that avoids 89) becomes the
empirical size of a gap the proposition predicts must be non-zero. Stated that
way T3 belongs. Stated as it is now, it does not.

**Condition.** If Phase 2 fails — if the construction needs an assumption this
panel violates, or the equivalence is only approximate with a remainder that is
not small — then T3 loses its derivation and drops to the supplement with T2, and
the paper reverts to T1 as the thesis: a configuration paper with an exhibit, a
gate, and no theorem. I would rather ship that than a vague proposition.

### Where I disagree with the framing, mildly

Calling T1 "the exhibit" understates it. It is the only result in the paper a
reader can reproduce without our data, from a public checkpoint and published
code, and that property is doing more work than its position in the argument
suggests — it is what makes the non-identification result concrete rather than
philosophical. I would keep it in the main text at close to its current length
(Section 4, four pages) rather than compressing it as "the exhibit" invites.

I also note that non-identification stated informally is close to a tautology: a
test of one scalar functional cannot distinguish causes that produce the same
value of that functional. A referee will say so. The contribution therefore
cannot be the claim; it has to be the **construction** — an explicit pair, an
explicit test class, an explicit boundary showing which tests escape it. That is
what Phase 2 must deliver, and it is why an approximate proposition with a stated
remainder is acceptable while a hand-waved one is worthless.

## Cut list

Nothing here is deleted without a destination. `SUPP` = supplement, `SEP` =
separate paper, `REC` = internal record only, `DEL` = deleted with reason.

### Moves to the supplement

| item | current location | destination | reason |
|---|---|---|---|
| The ordering by R-bar, and R as a statistic | S5.2, S5.2.1 | SUPP | R is a rank-preserving transform of the violation rate (Spearman 0.991). It carries no information the adjacent column lacks, and it is a recalibration-magnitude story, not an identification one. |
| Static recalibration results | S5.3 | SUPP, except the concealment result | Green-zone recovery is a property of the guarantee, not a finding. The one part that serves the thesis — that the guarantee is indifferent to the forecaster — moves into the formal section as the corollary that recalibration destroys the evidence. |
| Static versus rolling | S5.5 | SUPP | An estimator comparison. Nothing in it bears on identification. |
| Multi-level and forecast comparison | S5.6 | SUPP | Except the alpha-response contrast, which is a gate check and moves to Section 7. |
| Alternative recalibration methods | S5.7 | SUPP | Ten baselines answer "which recalibrator is best", a question the paper no longer asks. |
| Simulation and robustness | S5.8 | SUPP | Already mostly there. |
| T2 in full: the benchmark ranking, DM matrix, the GARCH-vs-TSFM comparison | S5.6, S8.4 | SUPP as a scope statement | Retained in the main text as one sentence: correctly configured, these models are neither catastrophically miscalibrated nor superior. Its function is to stop a misreading, and that needs a sentence, not a section. |
| Capital arithmetic | S8.3 | SUPP | Explicitly illustrative, FRTB-adjacent, and unconnected to the thesis. |
| Operational failure modes, COVID lag | S8.2 | SUPP | Deployment guidance. |
| Methodological limitations on conditional recalibration | S8.1 | SUPP | Except the sentence that a scalar shift cannot reshape conditional dynamics, which the formal section needs. |

### Moves into the formal section (Phase 2)

| item | current location | why it moves |
|---|---|---|
| Christoffersen degeneracy rates by level | S6.1 | Becomes the empirical companion to the proposition: the test is not merely powerless against the alternative, it is frequently undefined. |
| Kupiec rejection rates and the discrimination exercise | S6.2 | Becomes the finite-sample illustration of a result now proved at the population level. The in-sample objection loses its force because the proposition does not depend on the labelled set. |
| The concealment result | S5.3, S6.3 | Becomes a corollary. |
| ES material: Z2, Fissler--Ziegel, the ES construction | S3.1.2, SUPP S.1 | **Earns its place back.** These are the tests that use exceedance magnitude rather than occurrence, so they lie outside the proposition's test class. They are the boundary of the result, which is where the paper must be most careful. |

### Deleted, with reason

| item | reason |
|---|---|
| "121,923 dates per checkpoint" | Not emitted by any artefact (Phase 0, N1). The cost argument survives on "one forward pass per date against 1,000 sampled paths". |
| "roughly ten minutes per checkpoint on a laptop GPU" | Unreproducible hardware timing (N2). |
| "1.041 against 1.038 in units of realised volatility" | The normalisation is not stored (N3). The agreement claim, 0.3%, reproduces at 0.27% and stays. |
| "36,600 panel observations / 366 expected violations" and "36,000 observations" | Two different wrong values for one quantity; recomputed 37,313 and 373 (D2). Neither sentence needs the figure. |
| The 216-pair ACI figures 0.0750 / 0.0443 | Quoted from an artefact that no document prints, on an undeclared panel, and the comparison reverses on the declared one (P1). Replaced with the 312-pair figures already in the printed table. |

### Kept in the main text

Introduction; related literature, trimmed; the framework, reduced to what the
proposition and the conformal shift need; Section 4 in full as the exhibit; the
formal section; the structural gate as the identified complement; the practical
rule as a corollary; conclusion. Table 1 (failure modes) stays as the source of
the labelled set and the motivation.

## Open, and deliberately unresolved at this stage

1. **Which package tree is on PyPI.** `conformal/quantile.py` is byte-identical
   across 0.3.1 and 0.3.2, so the property the Data and Code section asserts holds
   in both and the assertion is safe either way. But the sentence should name the
   property rather than the version, and the docstring defect (C1, third site)
   should be fixed in whichever tree ships.
2. **Whether the EVT-POT and FHS rows should be re-run.** Their producer reads
   only `cfp_ijf_data/returns` and never touches a forecast series, so the April
   vintage is not contaminated by the pipeline corrections. They are, however,
   aggregated over assets rather than pairs and printed in a column of pair
   counts. Since the section moves to the supplement, marking the aggregation is
   sufficient; re-running is not required. This is the AE-5 question and the
   answer is no.
