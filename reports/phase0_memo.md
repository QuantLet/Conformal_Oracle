# Phase 0 memo — bibliography audit and the properification framing

**Status: complete. Nothing in the .bib has been modified.** All findings below are
for your decision. Full machine-generated table: `reports/bib_audit.md`
(+ `reports/bib_audit.json`). Regenerate with `python scripts/audit_bib.py`.

---

## 0. Two path corrections before anything else

The prompt refers to files that do not exist under those names:

| Prompt says | Actually is |
|---|---|
| `reports/referee_1.pdf`, `reports/ae_report.pdf`, `reports/editor_letter.docx` | `submission_IJF/20260812 Review/IJF_paper.pdf`, `.../Report_AE_IJF-D-26-00531.pdf`, `.../editor.docx` |
| `main_R1.bib` | `submission_IJF/calibrating_the_oracle.bib` (56 entries; the root copy has 57 — it additionally contains `zinkevich2003online`) |

`reports/` did not exist; I created it. All three reports were read before any
other work. Note that **Referee 2's report is not a separate file** — it is
embedded in the body of `editor.docx`, and it is the one positive report
(it likes the drift diagnostic $\hat\delta_w(t)$ and calls $R$ "one of the
manuscript's most interesting contributions", while asking exactly the question
Phase 4 answers by removing the threshold).

---

## 1. Audit result: 56 entries

| Status | Count |
|---|---|
| OK (resolved, all compared fields agree) | 30 |
| MISMATCH | 4 |
| UNRESOLVED (identifier does not resolve) | 3 |
| MANUAL (no DOI/arXiv id; not automatically verifiable) | 19 |

**Seven entries have real defects.** They fall into three classes, and the class
matters for how you present this if anyone ever asks.

### Class A — the DOI points at a different paper (1 entry)

**`brehmer2021properification`** — the AE is exactly right, and it is worse than a
typo. Crossref confirms `10.1214/21-EJS1913` resolves to:

> Amorino, *Rate of estimation for the stationary distribution of jump-processes
> over anisotropic Hölder classes*, EJS 15, 2021.

Our entry claims title "Properification — postprocessing of quantile and interval
forecasts", EJS 15, 3692–3720. **No such paper exists.** The title is a splice of
the two real Brehmer–Gneiting papers; the journal/volume/year were taken from the
DOI's actual target. Verified replacements (both confirmed against Crossref):

```bibtex
@article{brehmer2020properization,
	author  = {Brehmer, Jonas R. and Gneiting, Tilmann},
	title   = {{Properization: constructing proper scoring rules via Bayes acts}},
	journal = {Annals of the Institute of Statistical Mathematics},
	volume  = {72}, number = {3}, pages = {659--673}, year = {2020},
	doi     = {10.1007/s10463-019-00705-7}
}

@article{brehmer2021scoring,
	author  = {Brehmer, Jonas R. and Gneiting, Tilmann},
	title   = {{Scoring interval forecasts: equal-tailed, shortest, and modal interval}},
	journal = {Bernoulli},
	volume  = {27}, number = {3}, pages = {1993--2010}, year = {2021},
	doi     = {10.3150/20-BEJ1298}
}
```

Caveat on the second: Crossref confirms authors, title, journal, volume 27,
issue 3, year 2021, but **deposits no page range** for Bernoulli. The
1993–2010 range is from your prompt and is not machine-verified — check it
against Project Euclid before it enters the .bib.

My recommendation (§3) is that **neither is needed**, so this may be moot.

### Class B — the DOI does not exist at all (3 entries)

These return "DOI Not Found" from doi.org itself, not merely from Crossref. Same
failure signature as Class A: a plausible-looking identifier for a real paper.

| Key | Cited as | Verified truth |
|---|---|---|
| `xu2024conformal` | TPAMI **2024**, **46(12)**, **9788–9804**, doi `10.1109/TPAMI.2024.3443853` | TPAMI **2023**, **45(10)**, **11575–11587**, doi `10.1109/tpami.2023.3272339`. Year, volume, issue, pages **and** DOI are all wrong; only authors and title are right. |
| `angelopoulos2024conformal` | *Conformal Risk Control*, **JASA** 2024, doi `10.1080/01621459.2024.2316667` | Published at **ICLR 2024** (DBLP-confirmed), not JASA. There is no journal version. **This entry is also never cited** — simplest fix is to delete it. |
| `baroneadesi1999var` | doi `...19:5<583::AID-**FUT6**>...` | Real DOI is `10.1002/(SICI)1096-9934(199908)19:5<583::AID-**FUT5**>3.0.CO;2-S`. One character. Volume, pages, year, authors, journal all correct. This one is a plain typo, not a fabrication. |

### Class C — invented or garbled author lists (3 entries)

Crossref cannot catch these (conference/TMLR papers aren't deposited), so the
script falls back to an arXiv title search and diffs the author lists.

| Key | Problem |
|---|---|
| `das2024timesfm` | Our entry lists **Leber, Andrew** and **Mathews, Rajat** — neither exists. Real author list (arXiv:2310.10688) is Das, Kong, **Sen**, **Zhou** — four authors, not five. "Mathews, Rajat" appears to be a scramble of "Rajat Sen"; "Zhou, Yichen" is missing. Venue ICML 2024 is **correct** (DBLP-confirmed). |
| `rasul2024lagllama` | Our entry invents **Bhatt, Rishika** (real: Rishika *Bhagwatkar*), **Thiele, Sahil** (real: Sahil *Garg*), and **Gasthaus, Johannes** (not an author at all). Nine real authors are missing. Also `Hassen, Nadhir Vincent` → real is `Nadhir Hassen`. **Separately, the venue is also wrong**: DBLP lists Lag-Llama only as CoRR `abs/2310.08278` (2023, "Informal and Other Publications") with no ICML entry, and its arXiv record carries no `journal_ref`. Our entry types it `@inproceedings` in "Proceedings of the 41st ICML (2024)". It is a **preprint** and should be typed as one. |
| `ansari2024chronos` | `Shen, Huishuai` → the record says **Huibin Shen**. Single given-name error; everything else checks out. Venue (TMLR 2024) and the `others` truncation are fine. |

### What is *not* wrong

The 30 OK entries include every classical reference a finance referee will check
(Kupiec, Christoffersen, Glosten et al., Bollerslev, Engle–Manganelli,
McNeil–Frey, Newey–West, Hansen, Gneiting–Resin's neighbours). Three earlier
"mismatches" were artifacts of Crossref conventions, not errors, and the script
now classifies them correctly:

- Crossref splits `Main title: subtitle` across two fields (CAViaR, Francq–Zakoïan);
- JSTOR-sourced records deposit only the **first** page (Christoffersen 1998,
  Bollerslev 1987, Hansen 1994) — our full ranges are right;
- for books the .bib holds the publisher while Crossref's `container-title` is
  the *series* (Rio 2017 → "Probability Theory and Stochastic Modelling").

I mention this because the same three would reappear as false alarms in any
future run of a naïve checker.

### Citation cross-check

- 50 distinct `\cite*` keys in `main_R1.tex` / `main_R1_anon.tex`.
- **0 keys cited in text but missing from the .bib** — no broken references.
- **6 entries present but never cited**: `angelopoulos2024conformal` (fabricated
  DOI — delete), `hazan2016introduction`, `koenker2005quantile`,
  `liu2024moiraimoe`, `newey1987simple`, `politis1994stationary`. The last four
  are legitimate works that simply lost their citation site during R1 edits;
  `politis1994stationary` is worth *re-citing* rather than dropping, since the
  stationary bootstrap underpins the block-bootstrap CIs in `CO_robustness`.

### Pattern

Six of the seven defects sit in the **2019–2024 ML/conformal/TSFM block**. Every
pre-2018 econometrics reference is clean. The defective entries share a
signature: correct title, correct first author, invented identifier or invented
co-authors. That is characteristic of LLM-generated BibTeX, which is presumably
why your constraint list ends with the rule about `scripts/audit_bib.py`. Worth
being aware of if a future editor asks how it happened.

---

## 2. Guarding the constraint

`scripts/audit_bib.py` is the gate for the "no new LLM-sourced references" rule.
It takes `--bib` so a candidate entry can be checked before it is merged:

```bash
python scripts/audit_bib.py --bib /tmp/new_entries.bib --out /tmp/check.md
```

Responses are cached in `reports/.bib_audit_cache.json`, so re-runs are offline
and the audit is reproducible; `--refresh` re-queries. Exit is always 0 — it
reports, it never edits.

---

## 3. Replacing the "restricted properification map" framing

`brehmer2021properification` is cited in exactly **two** places in `main_R1.tex`
(and the same two in `main_R1_anon.tex`). Both drafts below drop the reference
entirely rather than swapping in the corrected one.

### Site 1 — Section 3.2, line 296 (the substantive one)

**Current:**

> The one-sided design targets the loss-side exceedances that drive Basel III/FRTB
> capital and Yellow/Red zone triggers (Section~\ref{sec:economic}); it is a
> restricted properification map \citep{brehmer2021properification}, with more
> flexible alternatives evaluated in Section~\ref{sec:baselines}. Operational
> safeguards, including the non-negativity constraint $\qVstat \geq 0$, are
> discussed in Section~\ref{sec:failure}.

**Draft replacement:**

> The one-sided design is dictated by the regulatory loss function, not by a
> symmetry consideration. Basel III/FRTB capital and the Yellow/Red zone triggers
> are functions of loss-side exceedances alone (Section~\ref{sec:economic}): a day
> on which the realised return falls below the reported VaR is counted, a day on
> which it comfortably exceeds it is not. A correction applied to the upper tail
> would therefore alter no capital charge and no zone assignment, while widening
> the reported interval and degrading sharpness. We accordingly shift only
> $\hat q_t^{lo}$ and leave the remainder of the predictive distribution
> untouched; two-sided and distributional alternatives are evaluated in
> Section~\ref{sec:baselines}. Operational safeguards, including the
> non-negativity constraint $\qVstat \geq 0$, are discussed in
> Section~\ref{sec:failure}.

This justifies the design on the regulatory objective alone. It is also *stronger*
for the target journals — a finance referee cares that the asymmetry matches the
capital rule, not that it is a special case of a post-processing taxonomy. It has
the incidental benefit of answering AE point 1 ("one-sided coverage is a weak
guarantee") by making the one-sidedness a deliberate match to the loss function
rather than a limitation of the method.

### Site 2 — Section 2, line 181 (the literature-review list)

**Current:** `\citep{gneiting2013combining,brehmer2021properification,taillardat2016calibrated}`

**Draft replacement:** `\citep{gneiting2013combining,taillardat2016calibrated}`

The sentence ("Post-hoc calibration adjusts the output of a fixed forecaster
without retraining") is fully supported by the two survivors; the third was
carrying no load. If you want a third for balance, `brehmer2020properization`
(the verified AISM paper) genuinely is about post-hoc construction of proper
scoring rules and would be defensible — but nothing in the sentence requires it,
and adding a Gneiting-coauthored citation to a paper Gneiting just rejected is
not a signal worth sending.

---

## 4. What I have not done

- Not corrected the .bib (per instruction).
- Not touched `main_R1.tex`; the drafts above are text for you to place in
  `main_R2.tex` when Phase 5 starts.
- Not re-typed `rasul2024lagllama` from `@inproceedings` to `@misc`/`@article`,
  though the evidence says it should be.

## Recommended decisions before Phase 1

1. Delete `angelopoulos2024conformal` (uncited, fabricated DOI).
2. Fix `baroneadesi1999var` FUT6 → FUT5, and `ansari2024chronos` Huishuai → Huibin.
3. Replace `xu2024conformal` wholesale with the verified 2023 TPAMI record.
4. Rewrite the author lists of `das2024timesfm` and `rasul2024lagllama` from
   their arXiv records, and re-type Lag-Llama as a preprint.
5. Drop `brehmer2021properification` and adopt the §3 replacement text.
6. Decide whether to re-cite `politis1994stationary` (recommended) or drop it.
