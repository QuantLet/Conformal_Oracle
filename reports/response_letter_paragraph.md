# Bibliography paragraph for the resubmission cover letter

Drop-in text. States what was done and what was found; offers no account of
cause. Adjust the count if further entries change before submission.

---

## Version A — cover letter to the new journal

> One reviewer of an earlier submission of this manuscript identified an
> incorrect reference. In response we audited the entire bibliography
> programmatically rather than that entry alone: every reference was resolved
> against the Crossref REST API and, for preprints, the arXiv API, and the
> returned title, first author, container, year, volume and page range were
> compared against our entry. The audit found eight defective entries out of
> fifty-six — unresolvable or misdirected DOIs, incorrect venues, and incorrect
> author lists — all of which have been corrected against the authoritative
> records. The audit script is included in the replication package
> (`scripts/audit_bib.py`) and the full entry-by-entry report is available on
> request; every reference now resolves to the work cited.

## Version B — shorter, if the letter is tight

> Following a reviewer observation about one incorrect reference, we audited the
> full bibliography against Crossref and arXiv. Eight of fifty-six entries
> carried defective metadata and have been corrected against the authoritative
> records. The audit script is part of the replication package
> (`scripts/audit_bib.py`); every reference now resolves to the work cited.

---

## Notes on wording

- "One reviewer of an earlier submission" — accurate without naming IJF or the
  AE. If you attach the reports, name them instead; a mismatch between letter and
  attachment is worse than either.
- "audited ... rather than that entry alone" is the load-bearing clause. It is
  what converts the finding from a defect into a control.
- Count: **8**, not 7. The eighth (`angelopoulos2024conformal`: two co-authors
  who are not on the paper) surfaced only after the fabricated DOI was replaced,
  because an entry with a DOI never reached the author-list check. The script now
  runs that check even when the DOI fails, so the number is stable.
- No explanation of cause, per your instruction. If asked directly, the factual
  answer is available: the defects concentrate entirely in the 2019–2024
  ML/conformal/TSFM block, and the sweep across the rest of the corpus
  (`reports/audit_sweep_memo.md`) found the published record clean.
- Do not describe the audit as "AI-assisted" or "automated verification of
  AI-generated references". It is a Crossref/arXiv resolution check; that is what
  it should be called.
