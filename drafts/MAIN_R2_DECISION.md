# What becomes of `main_R2.tex`

*Decision memo, 20 August 2026. Every status below is read from a file in this
tree; the file is named so the verdict can be checked rather than taken.*

The manuscript cannot be repaired sentence by sentence, because the sentence it
is organised around is an artefact of our own code. This memo separates what
died from what survived, prices the three ways forward, and recommends one.

---

## 1. The abstract, claim by claim

| # | Claim as printed | Verdict | Evidence |
|---|---|---|---|
| C1 | "13 forecasters and 24 assets" | **STALE** — the panel is now 16 × 24 | `Quantlets/CO_full_evaluation/tab_master_results_r2.csv` (16 rows, rebuilt 17 Aug) |
| C2 | "the *predictive interface* … rather than architecture or pretraining … drives extreme-tail calibration failure" | **DEAD** | see §2 |
| C3 | "Moirai … two versions share an architecture and a closely related pretraining design" | **FALSE, independently of any defect** | arXiv:2511.11698: 2.0 is decoder-only, single-patch, quantile-loss, new 36M-series corpus. Inventory item #1 |
| C4 | "$R$ spans four orders of magnitude, from 0.001 for CAViaR to 23.5 for Chronos-Mini" | **HOLLOW** | the 17.3 and 23.5 are the `top_k` defect. Read analytically, Chronos sits at 0.145 and 0.161. Excluding misconfigured series, $R$ spans 0.001–0.36 |
| C5 | recalibration is "an intervention with an indication"; gating on a failed raw backtest | **SURVIVES**, numbers to refresh | `analysis/ae_point4/`, re-run 18 Aug on the corrected panel |
| C6 | "the scalar shift addresses a median 35% of total miscalibration" | **NOT RE-RUN** | `analysis/umcb/run_umcb.py` edited 17 Aug 17:38; `MEMO.md`, `umcb_pairs.csv`, `fig_umcb_qv.png` all still 15 Aug 20:42 — the magnitude is from the defective panel |

## 2. Why the central finding is gone, and how completely

The interface claim rested on one contrast: Moirai 1.1 (samples) at 1.5% against
Moirai 2.0 (grid) at 98.8%. The 98.8% was our own sign error — the stored VaR was
written as $-F^{-1}(\alpha)$, so the threshold pointed at the wrong tail
(`analysis/recompute/RECOMPUTE.md`, reproduced to ten decimals from the stored
parameters).

On the corrected panel, at $\alpha = 0.01$:

| forecaster | interface | raw $\hat\pi$ |
|---|---|---|
| TimesFM-2.5 | **grid** | **0.0143** |
| Moirai-1.1 | sample | 0.0154 |
| Chronos-Small-A | sample (analytic) | 0.0175 |
| Moirai-2.0 | **grid** | **0.0178** |
| Chronos-Mini-A | sample (analytic) | 0.0178 |
| Lag-Llama | sample | 0.0294 |

The within-family gap is 0.24 percentage points, and the best raw TSFM in the
panel is a grid model. The ordering the paper claims is not merely weakened; it
does not hold in sign across the panel. There is no version of this table from
which the abstract's second sentence can be recovered.

C3 compounds it. Even had the numbers held, the "within-family control" was not
a control: the two Moirai releases differ in architecture, corpus and output
parameterisation. The one comparison the paper called clean was confounded on
three axes.

## 3. What survives, and is worth publishing

1. **The corrected panel.** 16 forecasters × 24 assets × 4 levels, raw and
   recalibrated, each series past a promotion gate. Its message is deflationary
   and defensible: correctly configured zero-shot TSFMs land in the same band as
   the GARCH family, and per-asset CAViaR (raw $\hat\pi$ = 0.0110, 15/24 Kupiec)
   beats every one of them.
2. **The indication rule** (`analysis/ae_point4/`). At $\alpha = 0.01$ under the
   rolling estimator, 94 pairs pass Kupiec raw; recalibrating them degrades 89
   (94.7%), and 102 of the 174 degradations panel-wide buy no Basel zone change.
   Skipping those 94 removes 51% of the damage, forgoes 5 gains, and keeps all
   205 zone upgrades. Under the single split it removes 58 of 66 degradations
   and keeps all 182 upgrades.
3. **The structural gate** (`scripts/promotion_gate.py`, 10 checks). Four
   defects passed every backtest in the submitted paper; the gate blocks 4 of 13
   series. This is the methodological contribution that the audit actually
   produced.
4. **The `top_k` mechanism** — established by controlled dose–response, and the
   only finding reproducible from a public checkpoint without our data.
5. **The theorem, and the marginal/conditional evidence**: corr(VaR_cp, σ_t)
   = +0.530 on the inverted input against −0.530 on the corrected one, unanimous
   across 24 assets, marginal coverage attained either way.

Two things do *not* survive as instruments. $R$ is a rank-preserving transform of
the raw violation rate — Spearman 0.9912 across the well-specified forecasters —
so the audit-statistic framing goes (`analysis/detection/VERDICT.md`, C3). And
the retrospective observation that the submitted Table 1's Panel B was exactly
the four defective series remains true and remains striking, but it would have
partitioned identically on the violation rate printed in the adjacent column.

## 4. The three ways forward

**A — Rebuild `main_R2` as one paper around configuration and the gate.**
The abstract and introduction already exist as `drafts/config_trap_abstract_intro.md`
("What Backtests Cannot Detect"). Remaining: re-run uMCB and the AE-7 window
sweep for GJR, rewrite §3.3, §4.2, §4.3 and the conclusion, regenerate every
dependent table and figure, and reconcile a ~60% overlap with the narrow draft.
Cost: high. Risk: one manuscript carrying both a mechanism and a panel, where the
mechanism is the strong half and a referee will ask what the panel adds.

**B — Two papers, mechanism first.** Ship `drafts/narrow_paper.md` (§1–§6 drafted,
bibliography verified 8/8) as the primary result. Rebuild the panel material
separately as a shorter paper on conformal recalibration: theorem, indication
rule, marginal-vs-conditional, corrected benchmark table. Cost: moderate now,
moderate later. The narrow paper's §4–§6 already use panel numbers, so the split
line has to be drawn deliberately — mechanism and validation in one, the
recalibration decision problem in the other.

**C — Archive `main_R2`, publish the narrow paper only.** Post the SSRN
correction, correct the Quantinar material, and let the panel work stand as a
replication package. Cost: lowest. Loses the indication rule, which is the one
finding that answers a referee question we were actually asked.

## 5. Recommendation

**B**, with `main_R2.tex` retired as a manuscript rather than corrected.

The reason is not that the corrections are too many. It is that the paper's
organising claim and its instrument both dissolved, and what is left — a
deflationary benchmark table, a rule about when not to recalibrate, and a
structural gate — is not the paper that title, abstract and §4 were built to
argue. Editing it into shape would produce a manuscript whose history is
invisible to a referee and whose framing survives only because it was already
typeset.

Retire it, and the same material makes two honest papers, one of which is
already drafted.

## 6. Immediate consequences, whichever option is chosen

- **Do not send `IRFA_cover_letter.md`.** Its second and third paragraphs sell
  the interface finding and the 1.5%-vs-98.8% contrast as the headline.
- The co-author message (`drafts/coauthor_message.md`) precedes any submission
  decision, and its table is consistent with everything above.
- `analysis/umcb/` must be re-run before any number of the form "the shift
  addresses X% of miscalibration" appears in any draft.
- The AE-7 window sweep must be re-run for the GJR rows
  (`analysis/phase3_windows/`); GARCH-N, EWMA and Hist-Sim rows are unaffected.
