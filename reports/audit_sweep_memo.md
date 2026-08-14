# Bibliography sweep across everything in circulation

Follow-up to `reports/phase0_memo.md`. Same method, applied to the rest of the
pipeline's output. **Nothing was modified anywhere.**

## Headline

**The published Mathematics paper is clean.** No fabricated identifier, no
identifier pointing at the wrong work, in any of its 28 references. There is no
erratum decision to make.

Two defects exist elsewhere, both fixable before submission, neither published:

| Document | Refs | Hard defects | Status |
|---|---|---|---|
| **Mathematics 2026, 14(13), 2316** (published) | 28 | **0** | clean |
| DELPHI — project description | 61 | **1** | not yet submitted |
| SPARC — project description | 36 | **0** | clean |
| AIDA MSCA — Application Package | 4 + 5 inline | 0 | clean |
| AIDA MSCA — Concept Note | 10 | 0 | clean |
| Stablecoin governance (draft) | 10 | 0 | clean |
| MESA — paper1 (Nature Comp Sci draft) | 12 | 0 | clean |
| ING ROBOR — expert report (draft) | 20 | 0 | clean |
| LPPLS — `Lppls-ai-ecosystem/paper/refs.bib` | 21 | **1** | draft |
| EMH — `llm_emh_sim/paper/references.bib` | 21 | 0 | draft |
| **IJF paper** (`calibrating_the_oracle.bib`) | 56 | **7** | rejected, being reworked |

Reports: `reports/refs_audit.md` (Mathematics, DELPHI, SPARC),
`reports/refs_audit_tier3.md` (AIDA, stablecoin, MESA, ING ROBOR),
`reports/bib_audit_lppls.md`, `reports/bib_audit_emh.md`, `reports/bib_audit.md`.

## The two defects outside the IJF paper

**DELPHI, reference [12]** — cites `10.2478/picbe-2026-0002` for Jheng, Pele, Tak
& Găman, *A multimodal vision-language framework for financial anomaly
detection*. That DOI is the proceedings **front matter** ("Contents", pp. i–xl).
The article's real DOI is **`10.2478/picbe-2026-0020`**, pp. 191–201 — which
matches the page range the reference already states. A digit transposition
(0002 / 0020), not a fabrication, and it is your own paper being mis-cited.

**LPPLS, `sornette2015`** — carries `note = {arXiv:1509.00121}`. That ID is
*Tight fibered knots and band sums* by Baker — a knot-theory paper. I could not
find any arXiv version of the Sornette et al. Shanghai bubble paper, so the fix
is to **delete the note**, not to replace the ID. The journal reference itself
(J. Investment Strategies 4(4), 77–95) is not in Crossref and needs a manual
check.

## Why this differs from the IJF paper's seven

The signature there was *invented* — DOIs for papers that exist under different
identifiers, co-authors who are not on the papers. Nothing of that kind appears
anywhere else in the sweep. What the rest of the corpus has is ordinary citation
error: one transposed digit, one stale note. That is a meaningful distinction if
the question ever comes up — the failure was concentrated in one document, not
distributed across the pipeline.

The Mathematics paper is worth a second look for the opposite reason: it is the
most identifier-dense document of the set (24 of 28 references carry a DOI or
arXiv ID) and every one of them resolves to the right work.

## Coverage caveats — where a clean result means less

- **EMH** (`llm_emh_sim`): 19 of 21 entries carry no DOI or arXiv ID, so only 2
  were machine-verifiable. "0 defects" mostly means "not checkable", not
  "checked and clean". Worth adding DOIs before that paper goes out.
- **AIDA Application Package**: has no bibliography; the 5 DOIs it cites inline
  sit in partner-supplied publication lists. Those resolve, but the surrounding
  text is a context window, so the tool reports them as needing a manual look
  rather than passing them.
- I could not locate **joint VaR–ES** or **minimax ES** as separate documents.
  The Mathematics paper *is* the ES-precision work; if the other two are separate
  manuscripts, point me at them.
- **Redispatch** exists here only as `TSA/Redispatch_2_0_Slides.pdf` (slides, no
  reference list). If there is a manuscript, it is not on this machine.
- The **ASUS machine** is not reachable from this session — no Claude session is
  running on it, and it is not in Google Drive's index. Both scripts are
  self-contained and need only Python 3 plus network: copy `scripts/audit_bib.py`
  and `scripts/audit_refs.py` there and run them, or start a session on it.

## What the sweep cost in tooling

`scripts/audit_refs.py` is new — `audit_bib.py` only reads `.bib` files, and most
of the corpus has no `.bib`. It reads `.md`, `.docx`, `.pdf`, LaTeX `\bibitem`
lists and Google Drive text blobs, and applies the same three checks.

Six false-positive classes surfaced during the sweep and are now handled, each
documented at its check. They matter because every one of them would have
reported a fabrication where there was none — on a grant proposal:

1. PDF-to-markdown splits a DOI across two links, leaving a truncated stub in the
   visible text (`10.1093/jjfinec/`); the href holds the whole DOI. **This alone
   produced 5 phantom defects in the published Mathematics paper.**
2. Old Elsevier DOIs contain parentheses — `10.1016/S0378-4266(02)00283-2`.
3. Line wrapping splits a DOI with a space (`10.1016/j.rser.2017. 05.234`); the
   repair can over-join a trailing page number, so candidates are tried in order
   and whichever resolves wins.
4. PDF extraction drops inter-word spaces
   (`Makingandevaluatingpointforecasts`), defeating word-overlap title matching.
5. References cite a main title without its subtitle.
6. Markdown emphasis markers glued to a DOI (`...02.037.**`).

A **coverage warning** now fires when the document mentions materially more DOIs
than the parsed reference list accounts for. It caught the AIDA package, where
the bibliography-shaped section was not the bibliography — without it, that
document would have reported "4 references, 0 defects" and looked fine.

## Suggested order of action

1. Fix DELPHI [12] before submission to UEFISCDI (one character).
2. Fix the IJF paper's seven, per `reports/phase0_memo.md`.
3. Delete the bad `note` in LPPLS `sornette2015` whenever that paper moves.
4. Add DOIs to the EMH bibliography so it becomes checkable at all.
5. Run both scripts on the ASUS machine.
