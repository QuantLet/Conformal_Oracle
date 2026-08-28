# Phase 1 — the four items that were truncated in chat

## 1. The doubling: which published numbers passed through a doubled panel?

**None, and not because of the guard.** The chain, with the evidence:

- The five scripts that concatenated the stale `moirai11_*results.csv` onto
  `all_results.csv` were: `rebuild_master_table.py`, `run_qV_ranking.py` (two
  copies), `run_frontier.py`, `run_violation_rates.py`.
- **Four of the five emit only figures** — no `.csv`, no `.tex`. Checked by
  grepping each for `to_csv`/`write_text`: only `savefig` calls.
- Of those figures, exactly one is `\includegraphics`'d by a document:
  `fig_frontier_killer` in the supplement. `fig_qV_ranking` and
  `fig_violation_rates` are built by `make.sh` and included nowhere.
- The fifth, `rebuild_master_table.py`, writes `tab_master_results_rebuilt.*`,
  which no document inputs.
- **`fig_frontier_killer.pdf` is dated 5 June.** `all_results.csv` began
  carrying Moirai-1.1 on 17 August. So the shipped figure was produced while the
  concatenation was still correct — Moirai-1.1 lived only in the separate file.

So the defect was **latent**: introduced on 17 August by a change to an input,
never exercised because nothing regenerated those figures afterwards. It would
have fired on the next `make.sh figures`.

**What it would have produced, measured rather than asserted.** The stale file's
width columns are the exact negatives of the live ones — `max |live + stale| =
0.0` across all 24 assets. The mean over the doubled panel is therefore 2.9e-19,
and the frontier plots `abs(mean)`: Moirai-1.1 would have been drawn at width
zero instead of 0.0387. A point at the origin of a width axis is not a subtle
error, which is the user's "exact 2x is an arithmetic marker" applied to this
case: the failure would have been visible on the face of the figure.

**A second finding from the same trace.** There are two copies of
`all_results.csv`: the canonical untracked one under `cfp_ijf_data/` and a
tracked copy under `Quantlets/CO_full_evaluation/results/`. The tracked copy had
**no Moirai-1.1 rows at all** as of its last commit (20 August) and was refreshed
only in this session. Any reader reproducing from the repository alone was
working from a twelve-model panel.

**Erratum-adjacent, separate from the doubling.** `fig_frontier_killer` is a
5 June vintage, so it predates both the 17 August sign correction and R14. It has
been regenerated; the change belongs in the final list of moved numbers.

## 2. What "the decomposition became 103" means

`main_R2.tex` §7 reads: *of the 174 pairs the rolling estimator degrades, N move
from Yellow to Green, while M obtain no zone improvement: A were already Green,
B was not and stays, and C move from Green to Yellow.*

Before: M = 102 was a macro (`\nDegradedRollNoChange`); A = 99, B = one, C = 2
were prose literals. After R14, `zone_tradeoff.csv` gives zone_up 72 → 71 and
zone_same 100 → 101, so M = zone_same + zone_down = 101 + 2 = **103**, and the
parts are **A = 100, B = 1, C = 2**. The macro would have moved to 103 while the
literals stayed at 99 + 1 + 2 = 102, printing a decomposition that does not sum
to its own total. Three macros were added (`AlreadyGreen`, `SameNotGreen`,
`ZoneDown`) and the literals removed.

## 3. What footnote 3 counts correctly now

Old text: "Every series in Table 1 has been through the scripted validation gate."

Table 1 has **16 rows**; `PROMOTION_GATE.csv` has **13**. The three missing are
CAViaR-AS, CAViaR-SAV and GAS-t — estimated per asset by a separate pipeline that
stores no forecast series, so the gate has no object to run on. They are
**ungated, not passed**. The footnote now says so and counts 13 via
`\nGateSeries`, which is itself now computed from the artefact rather than
supplied by a literal fallback.

## 4. Guard 5 and the manifest

**Guard 5 — every `\input` has a declared producer.** Live and passing: 30
`\input` targets in `main_R2.tex`, `supplement.tex` and `sections/*.tex`, each
declared in `analysis/provenance/PRODUCERS.tsv` as `generated` with a producer
that exists, or `authored` for the two hand-written section files. Its negative
control plants an `\input` of an undeclared table and is caught.

**Manifest expiry.** `build_manifest.py` gains `input_stamp()`, which records the
SHA-256 of each verdict's producer and of every canonical table that producer
names, and `--check-stale`, which re-hashes them and reports STALE for any verdict
whose inputs moved. Stamps are written by `--run`. With no stamp file it exits 2
and says so rather than reporting freshness it has not measured.
