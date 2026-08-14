#!/usr/bin/env python3
"""Reference-list audit for documents that have no .bib file.

Companion to `audit_bib.py`, for grant proposals, Word manuscripts and published
PDFs whose bibliography exists only as formatted text. Same purpose: find
references whose identifier does not resolve, or resolves to a different work.

Input is any of: .txt/.md, .docx, .pdf (needs `pdftotext`), or the JSON blob
produced by the Google Drive reader (`{"fileContent": ...}`).

For every numbered reference it:
  1. verifies a stated DOI against Crossref, and against doi.org when Crossref
     404s, so a *fabricated* DOI is distinguished from one merely absent from
     Crossref;
  2. checks that the title Crossref returns for that DOI actually appears in the
     reference text — this is what catches a real DOI attached to the wrong work;
  3. for references with no identifier, runs a Crossref bibliographic query and
     reports whether the work can be found at all.

Nothing is auto-corrected.

Usage:
    python scripts/audit_refs.py DOC [DOC ...] --out reports/refs_audit.md
    python scripts/audit_refs.py DOC --label "DELPHI proposal"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bib import (  # noqa: E402
    Fetcher, collapse, cr_field, cr_first_author, cr_full_title, cr_year,
    de_latex, norm_text, sim,
)

# A reference is "found" if Crossref returns something this close to its text.
MATCH_OK = 0.62
# Old Elsevier DOIs embed parentheses, e.g. 10.1016/0167-7152(95)00163-8, so
# they cannot simply be excluded; unbalanced trailing ones are trimmed below.
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\]}>\"']+")
ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})(v\d+)?", re.I)
# The heading may carry a section number ("B.2.5. Bibliography") and/or markdown
# emphasis ("**References**", "## References") once converted from PDF/Word.
BIB_HEADING = re.compile(
    r"(?im)^[^\S\n]*[#*_]{0,3}[^\S\n]*(?:[A-Z]?[\d.]{0,8}\.?\s*)?\\?\[?\s*"
    r"(references|bibliography|referin[tț]e|bibliografie|works cited|literature)"
    r"\s*\]?[^\S\n]*[#*_]{0,3}[^\S\n]*:?[^\S\n]*$")


# --------------------------------------------------------------------------- #
# text extraction
# --------------------------------------------------------------------------- #

def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            out = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                 capture_output=True, timeout=180)
            if out.returncode == 0:
                return out.stdout.decode("utf-8", "replace")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        raise SystemExit(
            f"cannot extract text from {path.name}: install poppler "
            "(`brew install poppler`) or convert the file first")
    if suffix == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        xml = re.sub(r"</w:p>", "\n", xml)
        xml = re.sub(r"<[^>]+>", "", xml)
        import html
        return html.unescape(xml)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json" or text.lstrip().startswith('{"fileContent"'):
        try:
            return json.loads(text)["fileContent"]
        except (json.JSONDecodeError, KeyError):
            pass
    return text


def clean(text: str) -> str:
    """Undo the markdown escaping and hard wrapping that PDF/Drive text carries."""
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # Join lines split mid-reference, but keep blank-line structure.
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text


# --------------------------------------------------------------------------- #
# reference splitting
# --------------------------------------------------------------------------- #

def find_bibliography(text: str) -> str:
    """Return the tail of the document starting at the last bibliography heading."""
    matches = list(BIB_HEADING.finditer(text))
    if matches:
        return text[matches[-1].end():]
    # No heading: fall back to the first numbered entry that is followed by
    # several more, which is where a reference list normally begins.
    m = re.search(r"\n\s*\[1\]\s", text)
    return text[m.start():] if m else text


def trim_entry(entry: str) -> str:
    """Cut anything the converter appended after the last reference.

    Word-to-text conversion often dumps the document's hyperlink targets as a
    block of bare `<https://...>` lines at the end; without this the final entry
    swallows them and picks up an unrelated DOI."""
    m = re.search(r"\s<https?://", entry)
    if m:
        entry = entry[:m.start()]
    return collapse(entry)[:1500]


def split_refs(bib: str) -> list[str]:
    """Split a numbered reference list into individual entries.

    Handles `\\bibitem{key} ...`, `[1] ...` and `1. ...` numbering (the last
    often arrives markdown-escaped as `1\\.` from converted Word documents)."""
    if r"\bibitem" in bib:
        parts = re.split(r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", bib)[1:]
        refs = []
        for p in parts:
            p = re.split(r"\\end\s*\{thebibliography\}", p)[0]
            p = collapse(de_latex(p))
            if len(p) > 25:
                refs.append(p[:1500])
        if refs:
            return refs
    bib = re.sub(r"\n\n", " \n", bib)
    bib = re.sub(r"(?m)^(\d{1,3})\\\.", r"\1.", bib)
    # `N.` entries are often run together on one line by the docx/PDF converter,
    # so the number is matched anywhere; the consecutive-run filter below is what
    # keeps sentence-internal numbers ("...499-513. 2. Mihoci...") from matching.
    for pattern in (r"\[(\d{1,3})\]\s", r"(?:^|\s)(\d{1,3})\.\s+(?=[A-ZÄÖÜŠČ])"):
        positions = [(m.start(), int(m.group(1)))
                     for m in re.finditer(pattern, bib)]
        # Keep only a run of consecutive numbers starting at the first entry.
        kept, expected = [], None
        for pos, num in positions:
            if expected is None:
                expected = num
            if num == expected:
                kept.append(pos)
                expected += 1
        refs = []
        for i, pos in enumerate(kept):
            end = kept[i + 1] if i + 1 < len(kept) else len(bib)
            entry = trim_entry(collapse(bib[pos:end]))
            if len(entry) > 25:
                refs.append(entry)
        if len(refs) > 2:
            return refs
    # Unnumbered list: one reference per paragraph that contains a year.
    return [collapse(p) for p in re.split(r"\n\s*\n", bib)
            if len(collapse(p)) > 60 and re.search(r"\b(19|20)\d{2}\b", p)]


def strip_identifiers(entry: str) -> str:
    """Reference text with DOI/arXiv/URL removed, for title matching."""
    s = DOI_RE.sub(" ", entry)
    s = ARXIV_RE.sub(" ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\bdoi\s*:?", " ", s, flags=re.I)
    s = re.sub(r"^\[\d+\]", " ", s)
    return collapse(s)


def repair_doi(entry: str) -> str:
    """Rejoin DOIs that line-wrapping split with a space.

    PDF extraction turns `10.1016/j.rser.2017.05.234` into
    `10.1016/j.rser.2017. 05.234`. Only a token starting with a digit or an
    opening paren immediately after a DOI-looking prefix is rejoined, so ordinary
    sentence breaks after a complete DOI are left alone."""
    prev = None
    while prev != entry:
        prev = entry
        entry = re.sub(r"(10\.\d{4,9}/\S*?)\s+(?=[\d(])", r"\1", entry, count=1)
    return entry


def clean_doi(raw: str) -> str:
    """Trim trailing punctuation, markdown emphasis and unbalanced parens."""
    doi = raw.strip()
    while doi and doi[-1] in ".,;:*_`'\"\\":
        doi = doi[:-1]
    while doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1]
    return doi


def doi_candidates(entry: str) -> list[str]:
    """Plausible readings of the DOI in a reference, best guess first.

    Line-wrap repair can over-join — `...2634827. 12` glues on a stray page
    number — so the un-repaired reading and a trailing-number-stripped reading
    are offered too, and the caller keeps whichever actually resolves."""
    out: list[str] = []

    def add(doi: str) -> None:
        doi = clean_doi(doi.replace("\\", ""))
        if doi and doi not in out:
            out.append(doi)

    # A doi.org link target is the most reliable source: PDF-to-markdown
    # conversion splits the *visible* DOI across two links, leaving a truncated
    # stub in the running text, while the href stays whole.
    # Greedy up to the closing paren: DOIs may themselves contain parens
    # (10.1016/S0378-4266(02)00283-2), so stopping at the first ")" truncates.
    for m in re.finditer(r"\]\(\s*https?://(?:dx\.)?doi\.org/([^\s]+)\)", entry):
        add(m.group(1))
    for m in re.finditer(r"https?://(?:dx\.)?doi\.org/([^\s\]<>]+)", entry):
        add(m.group(1))
    for text in (repair_doi(entry), entry):
        m = DOI_RE.search(text)
        if m:
            add(m.group(0))
            trimmed = re.sub(r"\.\d{1,3}$", "", clean_doi(m.group(0)))
            add(trimmed)
    return out


def _squash(s: str) -> str:
    return norm_text(s).replace(" ", "")


def title_is_present(title: str, entry: str) -> float:
    """How well does a Crossref title match some span of the reference text?

    Word-overlap, because the entry also holds authors, journal and pages. Two
    corrections matter in practice:
      * PDF extraction often drops the spaces between words
        ("Makingandevaluatingpointforecasts"), so a space-free containment test
        is run as well;
      * references routinely cite the main title without its subtitle, so the
        part before the colon is scored separately.
    """
    body = strip_identifiers(entry)
    best = 0.0
    variants = [title]
    if ":" in title:
        variants.append(title.split(":", 1)[0])
    for variant in variants:
        t_words = [w for w in norm_text(variant).split() if len(w) > 3]
        if not t_words:
            continue
        e_words = set(norm_text(body).split())
        best = max(best, sum(1 for w in t_words if w in e_words) / len(t_words))
        # Space-insensitive containment, for text with the spaces stripped out.
        sq_t, sq_e = _squash(variant), _squash(body)
        if len(sq_t) >= 12 and sq_t in sq_e:
            best = 1.0
    return best


# --------------------------------------------------------------------------- #
# per-reference audit
# --------------------------------------------------------------------------- #

def audit_ref(entry: str, fetch: Fetcher) -> dict:
    rec = {"entry": entry, "status": "OK", "problems": [], "notes": [],
           "doi": None, "arxiv": None, "remote": {}}

    candidates = doi_candidates(entry)
    doi = candidates[0] if candidates else None
    rec["doi"] = doi
    am = ARXIV_RE.search(entry)
    rec["arxiv"] = am.group(1) if am else None

    if doi:
        msg = {"__error__": "not attempted"}
        for cand in candidates:
            msg = fetch.crossref_doi(cand)
            if "__error__" not in msg:
                doi, rec["doi"] = cand, cand
                break
        if "__error__" in msg:
            exists = any(fetch.doi_exists(c) for c in candidates)
            if exists is False:
                rec["status"] = "DOI_NOT_FOUND"
                rec["problems"].append(
                    f"stated DOI `{doi}` does not exist — Crossref has no record "
                    "and doi.org returns 'DOI Not Found'"
                    + (f" (also tried: {', '.join(candidates[1:])})"
                       if len(candidates) > 1 else ""))
            elif exists is True:
                rec["status"] = "CHECK"
                rec["notes"].append(
                    f"DOI `{doi}` resolves but is not in Crossref (another "
                    "registration agency) — verify by hand")
            else:
                rec["status"] = "CHECK"
                rec["notes"].append(f"DOI `{doi}` could not be checked")
            return rec
        remote = {
            "title": cr_full_title(msg),
            "author": cr_first_author(msg),
            "container": cr_field(msg, "container-title"),
            "year": cr_year(msg),
            "doi": collapse(str(msg.get("DOI", "") or "")),
        }
        rec["remote"] = remote
        overlap = title_is_present(remote["title"], entry)
        rec["title_overlap"] = round(overlap, 2)
        if overlap < 0.45:
            rec["status"] = "WRONG_WORK"
            rec["problems"].append(
                f"DOI `{doi}` resolves to \"{remote['title']}\" "
                f"({remote['author']}, {remote['year']}), which does not match "
                "the reference text")
        else:
            if remote["author"] and norm_text(remote["author"]) not in \
                    norm_text(strip_identifiers(entry)):
                rec["status"] = "CHECK"
                rec["notes"].append(
                    f"first author on the DOI record is '{remote['author']}', "
                    "which does not appear in the reference")
            years = re.findall(r"\b(19|20)\d{2}\b", strip_identifiers(entry))
            stated = {y for y in re.findall(r"\b((?:19|20)\d{2})\b",
                                            strip_identifiers(entry))}
            if remote["year"] and stated and remote["year"] not in stated:
                rec["status"] = "CHECK" if rec["status"] == "OK" else rec["status"]
                rec["notes"].append(
                    f"year on the DOI record is {remote['year']}; the reference "
                    f"states {'/'.join(sorted(stated))}")
        return rec

    if rec["arxiv"]:
        res = fetch.arxiv(rec["arxiv"])
        if "__error__" in res:
            rec["status"] = "ARXIV_NOT_FOUND"
            rec["problems"].append(
                f"arXiv id {rec['arxiv']} does not resolve ({res['__error__']})")
            return rec
        rec["remote"] = {"title": res["title"], "year": res["year"],
                         "author": res["authors"][0] if res["authors"] else ""}
        overlap = title_is_present(res["title"], entry)
        rec["title_overlap"] = round(overlap, 2)
        if overlap < 0.45:
            rec["status"] = "WRONG_WORK"
            rec["problems"].append(
                f"arXiv:{rec['arxiv']} is \"{res['title']}\", which does not "
                "match the reference text")
        return rec

    # No identifier at all: can Crossref find the work from the reference text?
    text = strip_identifiers(entry)
    hits = fetch.crossref_search(text[:300], "", "")
    best, best_ov = None, 0.0
    for h in hits[:5]:
        ov = title_is_present(cr_full_title(h), entry)
        if ov > best_ov:
            best_ov, best = ov, h
    if best and best_ov >= MATCH_OK:
        rec["status"] = "OK_NO_DOI"
        rec["remote"] = {
            "title": cr_full_title(best), "author": cr_first_author(best),
            "year": cr_year(best), "doi": collapse(str(best.get("DOI", ""))),
        }
        rec["notes"].append(
            f"no DOI in the reference; Crossref match `{rec['remote']['doi']}` "
            f"(overlap {best_ov:.2f})")
    else:
        rec["status"] = "UNVERIFIED"
        rec["notes"].append(
            "no DOI and no confident Crossref match — may be a book, report, "
            "working paper or conference item, or may not exist; verify by hand")
        if best:
            rec["notes"].append(
                f"closest Crossref hit (overlap {best_ov:.2f}): "
                f"\"{cr_full_title(best)}\"")
    return rec


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

SEVERITY = {"DOI_NOT_FOUND": 0, "WRONG_WORK": 1, "ARXIV_NOT_FOUND": 2,
            "UNVERIFIED": 3, "CHECK": 4, "OK_NO_DOI": 5, "OK": 6}

LABELS = {
    "DOI_NOT_FOUND": "stated DOI does not exist anywhere",
    "WRONG_WORK": "identifier resolves to a different work",
    "ARXIV_NOT_FOUND": "arXiv id does not resolve",
    "UNVERIFIED": "no identifier and no Crossref match — check by hand",
    "CHECK": "resolves, but a field disagrees",
    "OK_NO_DOI": "no DOI stated, but the work was found",
    "OK": "identifier resolves and matches the reference",
}


def esc(s: str) -> str:
    return s.replace("|", r"\|").replace("\n", " ")


def inline_entries(text: str, refs: list[dict]) -> list[str]:
    """Pseudo-references for DOIs cited inline, outside any reference list.

    Some proposals carry no bibliography at all and put the DOI next to the claim
    it supports; those still need checking, so a context window around each such
    DOI is treated as the reference text."""
    seen = {r["doi"] for r in refs if r.get("doi")}
    seen |= {c for r in refs for c in doi_candidates(r["entry"])}
    out, used = [], set()
    for m in DOI_RE.finditer(text):
        doi = clean_doi(m.group(0).replace("\\", ""))
        if not doi or len(doi) <= 12 or doi in seen or doi in used:
            continue
        used.add(doi)
        out.append(collapse(text[max(0, m.start() - 320): m.end() + 120]))
    return out


def coverage_warning(text: str, refs: list[dict]) -> str | None:
    """Guard against a silently missed bibliography.

    If the document as a whole mentions far more DOIs than the parsed references
    account for, the reference list was probably not found — which would make a
    clean result meaningless."""
    doc_dois = {clean_doi(d.replace("\\", "")) for d in DOI_RE.findall(text)}
    seen = {r["doi"] for r in refs if r.get("doi")}
    seen |= {c for r in refs for c in doi_candidates(r["entry"])}
    missed = {d for d in doc_dois if d and d not in seen and len(d) > 12}
    if doc_dois and len(missed) > max(3, 0.4 * len(doc_dois)):
        return (f"{len(missed)} of {len(doc_dois)} DOIs in this document are "
                "outside the parsed reference list — the bibliography may not "
                "have been located correctly. Check before trusting this result.")
    return None


def report_doc(name: str, refs: list[dict], lines: list[str],
               warning: str | None = None) -> dict:
    counts: dict[str, int] = {}
    for r in refs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append(f"## {name}")
    lines.append("")
    lines.append(f"{len(refs)} references parsed.")
    lines.append("")
    if warning:
        lines.append(f"> ⚠️ **Coverage warning.** {warning}")
        lines.append("")
    lines.append("| Status | Count | Meaning |")
    lines.append("|---|---|---|")
    for st in sorted(counts, key=lambda s: SEVERITY[s]):
        lines.append(f"| {st} | {counts[st]} | {LABELS[st]} |")
    lines.append("")

    bad = [r for r in refs if r["problems"]]
    if bad:
        lines.append(f"### Defects ({len(bad)})")
        lines.append("")
        for r in bad:
            lines.append(f"- **{r['status']}** — {esc(r['entry'][:230])}")
            for p in r["problems"]:
                lines.append(f"  - {p}")
            lines.append("")
    else:
        lines.append("**No fabricated or mismatched identifiers found.**")
        lines.append("")

    soft = [r for r in refs if not r["problems"] and r["status"] in
            ("CHECK", "UNVERIFIED")]
    if soft:
        lines.append(f"<details><summary>Needs a manual look ({len(soft)})</summary>")
        lines.append("")
        for r in soft:
            lines.append(f"- {esc(r['entry'][:190])}")
            for n in r["notes"]:
                lines.append(f"  - {n}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return counts


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docs", type=Path, nargs="+")
    ap.add_argument("--label", nargs="*", default=None,
                    help="display name per document, in the same order")
    ap.add_argument("--out", type=Path, default=root / "reports" / "refs_audit.md")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    fetch = Fetcher(args.out.parent / ".bib_audit_cache.json",
                    offline=args.no_network, refresh=args.refresh)

    lines = ["# Reference-list audit", "",
             "Documents without a `.bib` file, checked against Crossref / arXiv / "
             "doi.org. Companion to `reports/bib_audit.md`. Nothing was modified.",
             ""]
    summary, payload = [], []
    for i, doc in enumerate(args.docs):
        label = (args.label[i] if args.label and i < len(args.label) else doc.name)
        print(f"\n=== {label} ({doc})", file=sys.stderr)
        if not doc.exists():
            lines += [f"## {label}", "", f"**File not found:** `{doc}`", ""]
            continue
        text = clean(load_text(doc))
        refs = split_refs(find_bibliography(text))
        print(f"  {len(refs)} references", file=sys.stderr)
        recs = []
        for j, entry in enumerate(refs, 1):
            rec = audit_ref(entry, fetch)
            recs.append(rec)
            print(f"  [{j:>3}/{len(refs)}] {rec['status']}", file=sys.stderr)
        extra = inline_entries(text, recs)
        if extra:
            print(f"  + {len(extra)} inline-cited DOIs outside the reference list",
                  file=sys.stderr)
            for entry in extra:
                rec = audit_ref(entry, fetch)
                rec["inline"] = True
                # The "entry" here is an arbitrary context window around the DOI,
                # so a title mismatch may just mean the window straddles two
                # citations. Never let that stand as a hard defect.
                if rec["status"] == "WRONG_WORK":
                    rec["status"] = "CHECK"
                    rec["notes"] += [p + " (inline citation: the surrounding "
                                     "text is a context window, so verify "
                                     "manually)" for p in rec["problems"]]
                    rec["problems"] = []
                recs.append(rec)
        counts = report_doc(label, recs, lines, coverage_warning(text, recs))
        defects = sum(c for s, c in counts.items()
                      if s in ("DOI_NOT_FOUND", "WRONG_WORK", "ARXIV_NOT_FOUND"))
        summary.append((label, len(refs), defects))
        payload.append({"label": label, "path": str(doc), "counts": counts,
                        "records": recs})
        fetch.save()

    head = ["| Document | References | Hard defects |", "|---|---|---|"]
    for label, n, d in summary:
        head.append(f"| {label} | {n} | {'**' + str(d) + '**' if d else '0'} |")
    lines[3:3] = head + [""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=1),
                                             encoding="utf-8")
    print(f"\nwrote {args.out}", file=sys.stderr)
    for label, n, d in summary:
        print(f"  {label}: {n} refs, {d} hard defects", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
