# Revision Log: Anonymisation Fix for IJF Resubmission

**Date:** 2026-05-17  
**Source:** `20260515_Recalibrating_Tail_Event_Forecasts_under_Temporal_Dependence_revised_v22_anon.tex`  
**Output:** `20260517_Recalibrating_Tail_Event_Forecasts_under_Temporal_Dependence_revised_v22_anon_fixed.tex`  
**Purpose:** Remove all author-identifying content for double-blind review per EiC Pierre Pinson's instructions.

---

## Issue 1 — Acknowledgements (lines 617-621)

**Action:** Replaced entire funding paragraph (IDA, AI4EFin, PNRR, MSCA DIGITAL, MA'AT/POCIDIF) with:
```
[Funding and project acknowledgements omitted for anonymous review. Will be added upon acceptance.]
```

**Retained:** The "Declaration of generative AI use" paragraph (line 623-624) — contains no identifying information.

## Issue 2 — Data and Code Availability (lines 608-614)

**Action:** Replaced section content. Removed:
- `\quantlet{https://github.com/QuantLet/Conformal_Oracle/}` (QuantLet brand)
- `\url{https://github.com/QuantLet/Conformal_Oracle}` (identifies research group)
- PyPI link to `conformal-oracle` package

Replaced with anonymised wording referencing "an anonymised repository" with canonical URL to be made public upon acceptance.

**Note:** The anonymised repository URL (anonymous.4open.science) requires manual creation by the author. The text currently refers to "an anonymised repository" without a specific URL.

## Issue 3 — PDF Metadata (preamble, after line 14)

**Action:** Added `\hypersetup{...}` block with `pdfauthor={}` (empty).

**Verification:** PDF Properties show Author="Anonymous;" (from `\author{Anonymous}`), Title = manuscript title, no identifying metadata.

## Issue 4 — First-Person Self-References

**Searches performed:**
- `our earlier`, `our previous`, `our work`, `we have shown`, `we showed`, `our prior`, `we previously` → **0 matches**
- `in our` → 2 matches (lines 437, 1191): both are internal cross-references ("In our panel", "In our implementation"), not self-citations. **No changes needed.**
- All citations to `pele2025llmvar` are in third person (`\citet{pele2025llmvar}`, `\citep{...pele2025llmvar}`). **No changes needed.**

## Issue 5 — Final Sweep

### 5a. QuantLet/Quantinar brand
- Line 127: Removed `Data and code accompanying this paper are available via the Quantlet platform...` paragraph. Replaced with comment.
- `\quantlet{}` and `\quantinar{}` macro definitions retained in preamble (harmless; no longer invoked).
- Post-edit: **0 remaining `\quantlet{` or `\quantinar{` invocations** in body.

### 5b. Institution names
- Searched: `bucharest`, `humboldt`, `ase`, `tomate`, `romania` → **0 matches**

### 5c. URLs
- Searched: `github.com`, `gitlab`, `pypi.org`, `huggingface.co/[a-z]` → **0 matches** (HuggingFace model references are in the .bib, not inline)

### 5d. Identifiers
- Searched: `orcid`, `@`, `twitter`, `mastodon` → **0 matches** (no email/social handles in manuscript)

### 5e. Author commands
- `\author{Anonymous}` (line 54) — correct for blind review
- No `\affiliation{}`, `\thanks{}`, or `\date{}` commands present

---

## Verification

| Check | Result |
|-------|--------|
| Compiles clean | Yes (1 pre-existing double-subscript error, same as original) |
| Page count | 65 (original: 66 — shorter due to acknowledgements placeholder) |
| PDF Author field | "Anonymous;" |
| PDF Title field | "Recalibrating Tail Risk Forecasts under Temporal Dependence" |
| Full-text search for identifying terms | Only "Pele" in bibliography references (third-person citations) |
| Cross-references | All resolve (same as original) |
| No figures/tables lost | Confirmed |
| Original file preserved | Yes — `20260515_...anon.tex` unchanged |

## Action Required by Author

1. **Create anonymised repository** at https://anonymous.4open.science/ by submitting `https://github.com/QuantLet/Conformal_Oracle`. Insert the resulting URL into the Data and Code Availability section.
2. Upload `20260517_...anon_fixed.pdf` via Editorial Manager.
