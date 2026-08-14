#!/usr/bin/env python3
"""Bibliography audit: verify every .bib entry against Crossref / arXiv.

Phase 0 of the post-rejection rework (IJF-D-26-00531). The AE flagged that
`Brehmer and Gneiting (2021)` does not exist in the cited form; this script
checks every entry the same way instead of only that one.

For each entry:
  * DOI present      -> resolve https://api.crossref.org/works/{doi} and compare
                        title, first-author surname, container-title, year,
                        volume and pages against the local entry.
  * arXiv id present -> resolve the arXiv API and compare title/first author.
  * neither          -> a Crossref bibliographic query is attempted to *suggest*
                        a DOI; nothing is ever written back.

Nothing is auto-corrected. Output is reports/bib_audit.md plus a machine-readable
reports/bib_audit.json.

KNOWN CROSSREF CONVENTIONS — do NOT "fix" the .bib to match these
----------------------------------------------------------------
A naive checker raises all of the following on a correct bibliography. Each is a
quirk of how Crossref stores metadata, not an error in our entry, and each is
handled at the comparison that would otherwise flag it. They are listed here so a
future reader does not re-open them:

  1. Title/subtitle split. Crossref stores "CAViaR" in `title` and "Conditional
     Autoregressive Value at Risk by Regression Quantiles" in `subtitle`. Same
     for Francq & Zakoian's "GARCH Models". -> cr_full_title() rejoins them.
  2. First page only. JSTOR-sourced records deposit `page: "841"` for an article
     that runs 841-862 (Christoffersen 1998, Bollerslev 1987, Hansen 1994). Our
     full ranges are the correct ones. -> compare_crossref() reports a note.
  3. Book container. For a monograph the .bib carries the publisher while
     Crossref's `container-title` is the *series* ("Probability Theory and
     Stochastic Modelling" for Rio 2017), and publisher strings differ by imprint
     ("Springer" / "Springer-Verlag", "John Wiley & Sons" / "Wiley").
     -> compared against `publisher` too, and demoted to a note for book types.
  4. Zero-padded volumes ("03" for volume 3) at some journals.
  5. Online-first year drift of +/-1 between `published-print` and `issued`.

Usage:
    python scripts/audit_bib.py [--bib PATH] [--tex PATH ...] [--out PATH]
                                [--no-network] [--refresh]

Responses are cached in reports/.bib_audit_cache.json so re-runs are offline and
the audit is reproducible.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CROSSREF_WORK = "https://api.crossref.org/works/{doi}"
CROSSREF_QUERY = "https://api.crossref.org/works"
DOI_ORG = "https://doi.org/{doi}"
ARXIV_QUERY = "http://export.arxiv.org/api/query"
MAILTO = "danpele@ase.ro"
USER_AGENT = f"cfp-bib-audit/1.0 (mailto:{MAILTO})"

# Title similarity bands.
TITLE_OK = 0.90
TITLE_WARN = 0.72
# Minimum similarity for an untrusted title-search hit to be reported as a candidate.
SUGGEST_MIN = 0.88

REQUEST_PAUSE = 0.4  # polite pool: stay well under Crossref's rate limits


# --------------------------------------------------------------------------- #
# .bib parsing (self-contained; no bibtexparser dependency)
# --------------------------------------------------------------------------- #

def strip_comments(text: str) -> str:
    """Drop whole-line % comments (the .bib uses them as section rules)."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%"))


def parse_bib(path: Path) -> list[dict]:
    """Return [{key, type, fields, lineno}] for every @entry in the file."""
    raw = path.read_text(encoding="utf-8")
    text = strip_comments(raw)
    entries: list[dict] = []
    for m in re.finditer(r"@(\w+)\s*\{", text):
        etype = m.group(1).lower()
        if etype in {"comment", "preamble", "string"}:
            continue
        # Walk braces to find the end of the entry.
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[m.end(): i - 1]
        key, _, rest = body.partition(",")
        entries.append({
            "key": key.strip(),
            "type": etype,
            "fields": parse_fields(rest),
            "lineno": text[: m.start()].count("\n") + 1,
        })
    return entries


def parse_fields(body: str) -> dict[str, str]:
    """Split an entry body into {field: value}, respecting nested braces."""
    fields: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        m = re.compile(r"\s*(\w+)\s*=\s*").match(body, i)
        if not m:
            break
        name = m.group(1).lower()
        i = m.end()
        if i < n and body[i] == "{":
            depth, start = 1, i + 1
            i += 1
            while i < n and depth:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            value = body[start: i - 1]
        elif i < n and body[i] == '"':
            start = i + 1
            i += 1
            while i < n and body[i] != '"':
                i += 1
            value = body[start:i]
            i += 1
        else:  # bare token (numeric year etc.)
            start = i
            while i < n and body[i] not in ",\n":
                i += 1
            value = body[start:i]
        value = collapse(value)
        if name in fields and fields[name] != value:
            fields.setdefault("__duplicate_fields__", "")
            fields["__duplicate_fields__"] += f"{name}; "
        fields[name] = value
        while i < n and body[i] in " ,\n\t":
            i += 1
    return fields


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# normalisation helpers
# --------------------------------------------------------------------------- #

LATEX_ACCENT = re.compile(r'\\[\'"`^~=.]\s*\{?(\w)\}?')


def de_latex(s: str) -> str:
    s = LATEX_ACCENT.sub(r"\1", s)
    s = s.replace("\\&", "&").replace("--", "-").replace("---", "-")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return s.replace("{", "").replace("}", "")


def norm_text(s: str) -> str:
    s = de_latex(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return collapse(s)


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def first_author_surname(author_field: str) -> str:
    """Surname of the first author, handling 'Last, First' and 'First Last'."""
    if not author_field:
        return ""
    first = re.split(r"\s+and\s+", de_latex(author_field))[0].strip()
    if first.startswith("{") or "/" in first:  # corporate author, e.g. J.P. Morgan
        return collapse(first.strip("{}"))
    if "," in first:
        return collapse(first.split(",")[0])
    parts = first.split()
    return parts[-1] if parts else ""


def author_surnames(author_field: str) -> list[str]:
    """All author surnames in a BibTeX author field, in order."""
    if not author_field:
        return []
    out = []
    for a in re.split(r"\s+and\s+", de_latex(author_field)):
        a = a.strip()
        if not a or a.lower() == "others":
            continue
        out.append(first_author_surname(a))
    return out


def split_name(s: str) -> tuple[str, str]:
    """(surname, given names) from either 'Last, First' or 'First Last'."""
    s = collapse(de_latex(s))
    if "," in s:
        last, _, given = s.partition(",")
        return collapse(last), collapse(given)
    parts = s.split()
    return (parts[-1], " ".join(parts[:-1])) if parts else ("", "")


def check_given_names(ours_raw: str, theirs_raw: list[str], rec: dict) -> None:
    """For authors whose surname matches, check the given names agree.
    Catches 'Shen, Huishuai' where the record says 'Huibin Shen'."""
    theirs = {}
    for t in theirs_raw:
        sur, giv = split_name(t)
        theirs.setdefault(norm_text(sur), giv)
    for a in re.split(r"\s+and\s+", de_latex(ours_raw)):
        a = a.strip()
        if not a or a.lower() == "others":
            continue
        sur, giv = split_name(a)
        their_giv = theirs.get(norm_text(sur))
        if not giv or not their_giv:
            continue
        og, tg = norm_text(giv).split(), norm_text(their_giv).split()
        if not og or not tg:
            continue
        # Compare only the first forename, and ignore pure initials.
        if len(og[0]) > 1 and len(tg[0]) > 1 and og[0] != tg[0]:
            rec["problems"].append(
                f"given name for {sur}: our '{giv}' vs source '{their_giv}'")


def check_author_list(ours: list[str], theirs: list[str], rec: dict) -> None:
    """Compare two author-surname lists and record any invented or dropped names."""
    if not ours or not theirs:
        return
    n_ours = {norm_text(x) for x in ours if x}
    n_theirs = {norm_text(x) for x in theirs if x}
    invented = [o for o in ours if norm_text(o) not in n_theirs]
    missing = [t for t in theirs if norm_text(t) not in n_ours]
    truncated = "others" in rec["local"].get("author_raw", "").lower()
    if invented:
        rec["problems"].append(
            "author(s) in our entry not on the source record: "
            + ", ".join(invented))
    if missing and not truncated:
        rec["notes"].append(
            "author(s) on the source record missing from our entry: "
            + ", ".join(missing))
    if ours and theirs and norm_text(ours[0]) != norm_text(theirs[0]):
        rec["problems"].append(
            f"first author '{ours[0]}' vs source '{theirs[0]}'")


def norm_pages(p: str) -> str:
    p = de_latex(p).replace("--", "-").replace("–", "-")
    return re.sub(r"\s+", "", p)


def extract_arxiv_id(fields: dict[str, str]) -> str | None:
    for f in ("eprint", "archiveprefix", "journal", "note", "url", "howpublished"):
        v = fields.get(f, "")
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", v)
        if m:
            return m.group(1)
        m = re.search(r"arxiv[:/ ]\s*([a-z\-]+/\d{7})", v, re.I)
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# network layer (cached)
# --------------------------------------------------------------------------- #

class Fetcher:
    def __init__(self, cache_path: Path, offline: bool = False, refresh: bool = False):
        self.cache_path = cache_path
        self.offline = offline
        self.cache: dict[str, object] = {}
        if cache_path.exists() and not refresh:
            try:
                self.cache = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                self.cache = {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))

    def _get(self, url: str, parse: str) -> object:
        if url in self.cache:
            return self.cache[url]
        if self.offline:
            return {"__error__": "offline; not cached"}
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=45) as fh:
                payload = fh.read().decode("utf-8", "replace")
            out = json.loads(payload) if parse == "json" else {"__xml__": payload}
        except Exception as exc:  # network error, 404, malformed DOI ...
            out = {"__error__": f"{type(exc).__name__}: {exc}"}
        self.cache[url] = out
        time.sleep(REQUEST_PAUSE)
        return out

    def crossref_doi(self, doi: str) -> dict:
        url = CROSSREF_WORK.format(doi=urllib.parse.quote(doi, safe="/"))
        res = self._get(url, "json")
        if "__error__" in res:
            return res
        return res.get("message", {})

    def doi_exists(self, doi: str) -> bool | None:
        """Does the DOI resolve at doi.org at all? Distinguishes a fabricated DOI
        from one that is real but absent from Crossref (DataCite, mEDRA, ...)."""
        url = DOI_ORG.format(doi=urllib.parse.quote(doi, safe="/"))
        key = f"__exists__{url}"
        if key in self.cache:
            return self.cache[key]
        if self.offline:
            return None
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT,
                          "Accept": "application/vnd.citationstyles.csl+json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as fh:
                body = fh.read(4096).decode("utf-8", "replace")
            out = "DOI Not Found" not in body
        except urllib.error.HTTPError as exc:
            out = exc.code not in (404, 410)
        except Exception:
            out = None
        self.cache[key] = out
        time.sleep(REQUEST_PAUSE)
        return out

    def crossref_search(self, title: str, author: str, year: str) -> list[dict]:
        params = {
            "query.bibliographic": f"{title} {year}".strip(),
            "rows": "5",
            "mailto": MAILTO,
        }
        if author:
            params["query.author"] = author
        url = f"{CROSSREF_QUERY}?{urllib.parse.urlencode(params)}"
        res = self._get(url, "json")
        if "__error__" in res:
            return []
        return res.get("message", {}).get("items", [])

    def arxiv_search(self, title: str) -> dict:
        """Find a preprint by title. Conference/TMLR papers are absent from
        Crossref but almost always have an arXiv version, which is the only way
        to check the author list of such an entry."""
        query = f'ti:"{re.sub(chr(34), "", title)}"'
        url = f"{ARXIV_QUERY}?{urllib.parse.urlencode({'search_query': query, 'max_results': '3'})}"
        res = self._get(url, "xml")
        if "__error__" in res:
            return res
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(res["__xml__"])
        except ET.ParseError as exc:
            return {"__error__": f"arXiv XML parse: {exc}"}
        best, best_sim = None, 0.0
        for entry in root.findall("a:entry", ns):
            t = collapse(entry.findtext("a:title", default="", namespaces=ns) or "")
            s = sim(title, t)
            if s > best_sim:
                idu = entry.findtext("a:id", default="", namespaces=ns) or ""
                best_sim, best = s, {
                    "title": t,
                    "authors": [
                        (a.findtext("a:name", default="", namespaces=ns) or "").strip()
                        for a in entry.findall("a:author", ns)],
                    "year": (entry.findtext("a:published", default="",
                                            namespaces=ns) or "")[:4],
                    "arxiv_id": idu.rsplit("/", 1)[-1],
                    "similarity": round(s, 3),
                }
        return best or {"__error__": "arXiv: no title match"}

    def arxiv(self, arxiv_id: str) -> dict:
        url = f"{ARXIV_QUERY}?{urllib.parse.urlencode({'id_list': arxiv_id, 'max_results': '1'})}"
        res = self._get(url, "xml")
        if "__error__" in res:
            return res
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(res["__xml__"])
        except ET.ParseError as exc:
            return {"__error__": f"arXiv XML parse: {exc}"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return {"__error__": "arXiv: no entry returned"}
        title_el = entry.find("a:title", ns)
        if title_el is None or not (title_el.text or "").strip():
            return {"__error__": "arXiv: id not found"}
        authors = [
            (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        ]
        published = (entry.findtext("a:published", default="", namespaces=ns) or "")
        doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
        return {
            "title": collapse(title_el.text or ""),
            "authors": authors,
            "year": published[:4],
            "published_doi": (doi_el.text if doi_el is not None else None),
        }


def cr_field(msg: dict, name: str) -> str:
    v = msg.get(name)
    if isinstance(v, list):
        return collapse(v[0]) if v else ""
    return collapse(str(v)) if v is not None else ""


def cr_full_title(msg: dict) -> str:
    """Crossref splits 'Main title: subtitle' across `title` and `subtitle`
    (e.g. 'CAViaR' + 'Conditional Autoregressive Value at Risk ...'). Rejoin them
    so the comparison is not a false positive."""
    title = cr_field(msg, "title")
    subtitle = cr_field(msg, "subtitle")
    if subtitle and norm_text(subtitle) not in norm_text(title):
        return f"{title}: {subtitle}"
    return title


def cr_year(msg: dict) -> str:
    for key in ("published-print", "published", "issued", "published-online", "created"):
        parts = (msg.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return ""


def cr_first_author(msg: dict) -> str:
    for a in msg.get("author") or []:
        if a.get("family"):
            return collapse(a["family"])
        if a.get("name"):
            return collapse(a["name"])
    return ""


# --------------------------------------------------------------------------- #
# per-entry audit
# --------------------------------------------------------------------------- #

def check_against_arxiv(local: dict, f: dict, fetch: Fetcher, rec: dict) -> None:
    """Find the work on arXiv by title and diff the author list against ours.

    Conference proceedings and TMLR papers are not deposited in Crossref, so this
    is the only way to check their author lists; it is also run when a stated DOI
    fails to resolve, because a fabricated identifier and a fabricated co-author
    tend to travel together."""
    ax = fetch.arxiv_search(local["title"])
    if "__error__" in ax or ax.get("similarity", 0) < SUGGEST_MIN:
        return
    rec["arxiv_match"] = ax
    rec["notes"].append(
        f"arXiv preprint found by title: arXiv:{ax['arxiv_id']} "
        f"({ax['year']}, similarity {ax['similarity']})")
    ours = author_surnames(f.get("author", ""))
    theirs = [first_author_surname(a) for a in ax["authors"]]
    check_author_list(ours, theirs, rec)
    check_given_names(f.get("author", ""), ax["authors"], rec)


def audit_entry(entry: dict, fetch: Fetcher) -> dict:
    f = entry["fields"]
    local = {
        "title": collapse(de_latex(f.get("title", ""))),
        "author": first_author_surname(f.get("author", "")),
        "author_raw": collapse(de_latex(f.get("author", ""))),
        "container": collapse(de_latex(
            f.get("journal") or f.get("booktitle") or f.get("institution")
            or f.get("publisher") or "")),
        "year": collapse(f.get("year", "")),
        "volume": collapse(f.get("volume", "")),
        "pages": norm_pages(f.get("pages", "")),
        "doi": collapse(f.get("doi", "")),
    }
    rec = {"key": entry["key"], "type": entry["type"], "lineno": entry["lineno"],
           "local": local, "remote": {}, "source": None, "status": "MANUAL",
           "problems": [], "notes": [], "candidates": []}
    if f.get("__duplicate_fields__"):
        rec["problems"].append(
            "entry declares the same field twice: "
            + f["__duplicate_fields__"].strip("; ")
            + " (BibTeX silently keeps one)")

    doi = local["doi"]
    arxiv_id = extract_arxiv_id(f)

    if doi:
        rec["source"] = f"crossref:{doi}"
        msg = fetch.crossref_doi(doi)
        if "__error__" in msg:
            rec["status"] = "UNRESOLVED"
            exists = fetch.doi_exists(doi)
            rec["doi_exists"] = exists
            if exists is False:
                rec["problems"].append(
                    "**DOI DOES NOT EXIST** — not registered at Crossref and "
                    "doi.org returns 'DOI Not Found'. The cited object cannot be "
                    "reached by this identifier.")
            elif exists is True:
                rec["problems"].append(
                    "DOI resolves at doi.org but is not in the Crossref works API "
                    "(registered with another agency, e.g. DataCite) — verify by hand.")
            else:
                rec["problems"].append(
                    f"DOI did not resolve at Crossref ({msg['__error__']}); "
                    "doi.org check inconclusive.")
            # A bad DOI must not buy the rest of the entry a free pass: an entry
            # with a fabricated identifier often has fabricated co-authors too,
            # so still try to find the work and diff the author list.
            check_against_arxiv(local, f, fetch, rec)
            return rec
        remote = {
            "title": cr_full_title(msg),
            "author": cr_first_author(msg),
            "container": cr_field(msg, "container-title"),
            "year": cr_year(msg),
            "volume": collapse(str(msg.get("volume", "") or "")),
            "pages": norm_pages(str(msg.get("page", "") or "")),
            "doi": collapse(str(msg.get("DOI", "") or "")),
            "type": msg.get("type", ""),
        }
        remote["publisher"] = collapse(str(msg.get("publisher", "") or ""))
        rec["remote"] = remote
        compare_crossref(local, remote, rec)
        rec["status"] = "MISMATCH" if rec["problems"] else "OK"
        return rec

    if arxiv_id:
        rec["source"] = f"arxiv:{arxiv_id}"
        res = fetch.arxiv(arxiv_id)
        if "__error__" in res:
            rec["status"] = "UNRESOLVED"
            rec["problems"].append(f"arXiv id {arxiv_id} did not resolve ({res['__error__']})")
            return rec
        remote = {
            "title": res["title"],
            "author": first_author_surname(res["authors"][0]) if res["authors"] else "",
            "container": f"arXiv:{arxiv_id}",
            "year": res["year"],
            "volume": "", "pages": "", "doi": res.get("published_doi") or "",
        }
        rec["remote"] = remote
        t = sim(local["title"], remote["title"])
        if t < TITLE_WARN:
            rec["problems"].append(f"title mismatch vs arXiv (similarity {t:.2f})")
        elif t < TITLE_OK:
            rec["notes"].append(f"title differs in wording (similarity {t:.2f})")
        if remote["author"] and norm_text(local["author"]) != norm_text(remote["author"]):
            rec["problems"].append(
                f"first author '{local['author']}' vs arXiv '{remote['author']}'")
        if local["year"] and remote["year"] and local["year"] != remote["year"]:
            rec["notes"].append(
                f"year {local['year']} vs arXiv v1 posting {remote['year']}")
        if remote["doi"]:
            rec["notes"].append(
                f"now published; journal DOI {remote['doi']} available — consider updating")
        rec["status"] = "MISMATCH" if rec["problems"] else "OK"
        return rec

    # No DOI and no arXiv id: cannot be verified automatically. Try a title
    # search only to surface a candidate DOI for manual checking.
    rec["status"] = "MANUAL"
    rec["notes"].append("no DOI and no arXiv id in entry — not automatically verifiable")
    hits = fetch.crossref_search(local["title"], local["author"], local["year"])
    for h in hits[:5]:
        s = sim(local["title"], cr_full_title(h))
        if s >= SUGGEST_MIN:
            rec["candidates"].append({
                "similarity": round(s, 3),
                "doi": collapse(str(h.get("DOI", ""))),
                "title": cr_full_title(h),
                "author": cr_first_author(h),
                "container": cr_field(h, "container-title"),
                "year": cr_year(h),
                "volume": collapse(str(h.get("volume", "") or "")),
                "pages": norm_pages(str(h.get("page", "") or "")),
            })
    if not rec["candidates"]:
        check_against_arxiv(local, f, fetch, rec)

    if rec["candidates"]:
        best = rec["candidates"][0]
        flags = []
        if best["author"] and norm_text(best["author"]) != norm_text(local["author"]):
            flags.append(f"author {local['author']} vs {best['author']}")
        if best["year"] and local["year"] and best["year"] != local["year"]:
            flags.append(f"year {local['year']} vs {best['year']}")
        if best["volume"] and local["volume"] and best["volume"] != local["volume"]:
            flags.append(f"volume {local['volume']} vs {best['volume']}")
        if best["pages"] and local["pages"] and best["pages"] != local["pages"]:
            flags.append(f"pages {local['pages']} vs {best['pages']}")
        if flags:
            rec["notes"].append("candidate match disagrees on: " + "; ".join(flags))
    if rec["problems"]:
        rec["status"] = "MISMATCH"
    return rec


def compare_crossref(local: dict, remote: dict, rec: dict) -> None:
    t = sim(local["title"], remote["title"])
    rec["title_similarity"] = round(t, 3)
    if t < TITLE_WARN:
        rec["problems"].append(
            f"TITLE MISMATCH (similarity {t:.2f}): DOI resolves to a different work")
    elif t < TITLE_OK:
        rec["notes"].append(f"title wording differs (similarity {t:.2f})")

    if remote["author"] and norm_text(local["author"]) != norm_text(remote["author"]):
        rec["problems"].append(
            f"first author '{local['author']}' vs Crossref '{remote['author']}'")

    if local["year"] and remote["year"] and local["year"] != remote["year"]:
        # Online-first publication routinely shifts the year by one.
        if abs(int(local["year"]) - int(remote["year"])) == 1:
            rec["notes"].append(f"year {local['year']} vs Crossref {remote['year']} (±1)")
        else:
            rec["problems"].append(f"year {local['year']} vs Crossref {remote['year']}")

    if local["volume"] and remote["volume"] and local["volume"] != remote["volume"]:
        # Some journals deposit zero-padded volumes ("03" for volume 3).
        if local["volume"].lstrip("0") == remote["volume"].lstrip("0"):
            rec["notes"].append(
                f"volume {local['volume']} vs Crossref {remote['volume']} "
                "(zero-padding only)")
        else:
            rec["problems"].append(
                f"volume {local['volume']} vs Crossref {remote['volume']}")

    if local["pages"] and remote["pages"] and local["pages"] != remote["pages"]:
        # JSTOR-sourced records routinely deposit only the first page.
        if "-" not in remote["pages"] and local["pages"].startswith(remote["pages"] + "-"):
            rec["notes"].append(
                f"Crossref records the first page only ({remote['pages']}); "
                f"our range {local['pages']} starts there — check the end page")
        else:
            rec["problems"].append(f"pages {local['pages']} vs Crossref {remote['pages']}")

    if local["container"]:
        # For a monograph the .bib holds the publisher while Crossref's
        # container-title is the book series, so accept either.
        cands = [remote.get("container", ""), remote.get("publisher", "")]
        c = max((sim(local["container"], x) for x in cands if x), default=0.0)
        best = max((x for x in cands if x),
                   key=lambda x: sim(local["container"], x), default="")
        if not best:
            pass
        elif c < TITLE_WARN:
            msg = (f"container '{local['container']}' vs Crossref "
                   f"'{remote.get('container') or remote.get('publisher')}'")
            # For a book the .bib field is the publisher imprint, which Crossref
            # records inconsistently ('Springer' / 'Springer-Verlag' / series
            # name). That is a style difference, not a wrong citation.
            if rec["type"] in {"book", "inbook", "incollection", "techreport", "manual"}:
                rec["notes"].append(msg + " — publisher/series naming, not an error")
            else:
                rec["problems"].append(msg)
        elif c < TITLE_OK:
            rec["notes"].append(
                f"container wording differs: '{local['container']}' vs '{best}'")

    if local["doi"] and remote["doi"] and local["doi"].lower() != remote["doi"].lower():
        rec["problems"].append(f"DOI {local['doi']} normalised to {remote['doi']}")


# --------------------------------------------------------------------------- #
# citation cross-check
# --------------------------------------------------------------------------- #

CITE_RE = re.compile(r"\\(?:no)?cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")


def cited_keys(tex_paths: list[Path]) -> dict[str, list[str]]:
    """Return {key: [files it is cited in]}."""
    out: dict[str, list[str]] = {}
    for p in tex_paths:
        if not p.exists():
            continue
        text = "\n".join(
            ln for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if not ln.lstrip().startswith("%")
        )
        for m in CITE_RE.finditer(text):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    out.setdefault(k, [])
                    if p.name not in out[k]:
                        out[k].append(p.name)
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def md_escape(s: str) -> str:
    return s.replace("|", r"\|").replace("\n", " ")


def write_report(records: list[dict], cites: dict[str, list[str]], bib: Path,
                 tex_paths: list[Path], out: Path, offline: bool) -> None:
    keys = {r["key"] for r in records}
    missing = sorted(k for k in cites if k not in keys)
    uncited = sorted(k for k in keys if k not in cites)

    by_status: dict[str, list[dict]] = {}
    for r in records:
        by_status.setdefault(r["status"], []).append(r)

    L: list[str] = []
    L.append("# Bibliography audit — Phase 0")
    L.append("")
    L.append(f"- Bibliography: `{bib}`")
    L.append(f"- LaTeX sources scanned: {', '.join(f'`{p.name}`' for p in tex_paths)}")
    L.append(f"- Entries: **{len(records)}**")
    L.append("- Verification: Crossref REST (`/works/{doi}`) for entries with a DOI; "
             "arXiv Atom API for preprints; entries with neither are **MANUAL** "
             "(a Crossref title search is run only to suggest a candidate DOI).")
    L.append("- **Nothing in the .bib has been modified by this script.**")
    if offline:
        L.append("- NOTE: run in `--no-network` mode; results come from the local cache.")
    L.append("")
    L.append("## Status summary")
    L.append("")
    L.append("| Status | Count | Meaning |")
    L.append("|---|---|---|")
    meanings = {
        "OK": "resolved and all compared fields agree",
        "MISMATCH": "resolved but one or more fields disagree — needs a decision",
        "UNRESOLVED": "DOI/arXiv id present but did not resolve",
        "MANUAL": "no DOI or arXiv id — verify by hand",
    }
    for st in ("MISMATCH", "UNRESOLVED", "MANUAL", "OK"):
        L.append(f"| {st} | {len(by_status.get(st, []))} | {meanings[st]} |")
    L.append("")

    # Problem entries first, in full detail.
    for st in ("MISMATCH", "UNRESOLVED"):
        group = by_status.get(st, [])
        if not group:
            continue
        L.append(f"## {st} — full detail")
        L.append("")
        for r in group:
            L.append(f"### `{r['key']}` ({r['type']}, line {r['lineno']})")
            L.append("")
            L.append(f"Resolved via `{r['source']}`.")
            L.append("")
            if r["remote"]:
                L.append("| Field | In our .bib | Returned by source |")
                L.append("|---|---|---|")
                for fld in ("title", "author", "container", "year", "volume", "pages", "doi"):
                    lv = md_escape(r["local"].get(fld, "") or "—")
                    rv = md_escape(r["remote"].get(fld, "") or "—")
                    mark = "" if norm_text(lv) == norm_text(rv) else " ⚠️"
                    L.append(f"| {fld}{mark} | {lv} | {rv} |")
                L.append("")
            for p in r["problems"]:
                L.append(f"- **PROBLEM:** {p}")
            for n in r["notes"]:
                L.append(f"- note: {n}")
            L.append("")

    # Full one-row-per-entry table.
    L.append("## All entries")
    L.append("")
    L.append("| Status | Key | Our title | Source title | Our author/year/vol/pp | "
             "Source author/year/vol/pp | Flags |")
    L.append("|---|---|---|---|---|---|---|")
    order = {"MISMATCH": 0, "UNRESOLVED": 1, "MANUAL": 2, "OK": 3}
    for r in sorted(records, key=lambda x: (order[x["status"]], x["key"])):
        lo, re_ = r["local"], r["remote"]

        def trip(d):
            return "{} / {} / {} / {}".format(
                md_escape(d.get("author") or "—"), d.get("year") or "—",
                d.get("volume") or "—", md_escape(d.get("pages") or "—"))

        flags = "; ".join(md_escape(p) for p in r["problems"]) or \
                "; ".join(md_escape(n) for n in r["notes"]) or ""
        L.append("| {} | `{}` | {} | {} | {} | {} | {} |".format(
            r["status"], r["key"], md_escape(lo["title"])[:70],
            md_escape(re_.get("title", "") or "—")[:70],
            trip(lo), trip(re_) if re_ else "—", flags[:180]))
    L.append("")

    # MANUAL entries with candidate DOIs.
    manual = by_status.get("MANUAL", [])
    if manual:
        L.append("## MANUAL entries — candidate DOIs from Crossref title search")
        L.append("")
        L.append("These are *suggestions only*. A candidate is listed when a Crossref "
                 f"title search returns a match with similarity ≥ {SUGGEST_MIN}. "
                 "Confirm each before adding a DOI to the .bib.")
        L.append("")
        L.append("| Key | Our entry | Candidate DOI | Candidate title | Candidate author/year/vol/pp | Sim | Disagreements |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sorted(manual, key=lambda x: x["key"]):
            if r["candidates"]:
                c = r["candidates"][0]
                dis = "; ".join(md_escape(n) for n in r["notes"] if "disagrees" in n)
                L.append("| `{}` | {} | `{}` | {} | {} / {} / {} / {} | {} | {} |".format(
                    r["key"], md_escape(r["local"]["title"])[:55], c["doi"],
                    md_escape(c["title"])[:55], md_escape(c["author"]), c["year"],
                    c["volume"] or "—", md_escape(c["pages"] or "—"),
                    c["similarity"], dis or "—"))
            else:
                L.append("| `{}` | {} | — | *no confident Crossref match* | — | — | "
                         "verify by hand |".format(
                             r["key"], md_escape(r["local"]["title"])[:55]))
        L.append("")

    # Citation cross-check.
    L.append("## Citation cross-check")
    L.append("")
    L.append(f"- Distinct `\\cite*` keys found in the LaTeX sources: **{len(cites)}**")
    L.append(f"- Keys cited in text but **absent from the .bib**: **{len(missing)}**")
    if missing:
        L.append("")
        L.append("| Missing key | Cited in |")
        L.append("|---|---|")
        for k in missing:
            L.append(f"| `{k}` | {', '.join(cites[k])} |")
    L.append("")
    L.append(f"- Entries present in the .bib but **never cited**: **{len(uncited)}**")
    if uncited:
        L.append("")
        for k in uncited:
            L.append(f"  - `{k}`")
    L.append("")
    L.append("---")
    L.append("")
    L.append("Regenerate with `python scripts/audit_bib.py`. "
             "Responses are cached in `reports/.bib_audit_cache.json`; "
             "pass `--refresh` to re-query the APIs.")
    L.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")

    payload = {
        "bib": str(bib),
        "tex": [str(p) for p in tex_paths],
        "n_entries": len(records),
        "counts": {k: len(v) for k, v in by_status.items()},
        "records": records,
        "cited_not_in_bib": missing,
        "in_bib_not_cited": uncited,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bib", type=Path,
                    default=root / "submission_IJF" / "calibrating_the_oracle.bib")
    ap.add_argument("--tex", type=Path, nargs="*", default=[
        root / "submission_IJF" / "main_R1.tex",
        root / "submission_IJF" / "main_R1_anon.tex",
    ])
    ap.add_argument("--out", type=Path, default=root / "reports" / "bib_audit.md")
    ap.add_argument("--no-network", action="store_true",
                    help="use only the cached responses")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)

    if not args.bib.exists():
        print(f"bib not found: {args.bib}", file=sys.stderr)
        return 2

    entries = parse_bib(args.bib)
    print(f"parsed {len(entries)} entries from {args.bib.name}", file=sys.stderr)

    fetch = Fetcher(args.out.parent / ".bib_audit_cache.json",
                    offline=args.no_network, refresh=args.refresh)
    records = []
    for i, e in enumerate(entries, 1):
        rec = audit_entry(e, fetch)
        records.append(rec)
        print(f"  [{i:>2}/{len(entries)}] {rec['status']:<10} {rec['key']}", file=sys.stderr)
    fetch.save()

    cites = cited_keys(list(args.tex))
    write_report(records, cites, args.bib, list(args.tex), args.out, args.no_network)

    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nwrote {args.out}", file=sys.stderr)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
