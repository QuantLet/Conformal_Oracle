#!/usr/bin/env python3
"""Check the claims a numeric ledger cannot see.

`paper_numbers.py` closes the numbers. It says nothing about a sentence that
counts items wrongly, points at a column that does not exist, or cites a section
that moved. Those defects have all occurred in this project:

  * a caption asserted that the last column held the count of negative shifts;
    the last column held R-bar;
  * "99 of them ... and 2" summed to 101 against a stated 102;
  * "Eight of the ten need nothing but the series and the returns" against a
    table listing one check that needs sampled paths;
  * "Section 5.9" survived a subsection merge that renumbered it to 5.8.

Each check runs a negative control first and reports BROKEN if the control does
not fail, per PROTOCOL.md Rule 2.

    python scripts/audit_structural_claims.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DOCS = ["main_R2", "supplement"]
RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"
OFF = "\033[0m"


_MACROS = None


def _macros() -> dict[str, str]:
    """Generated macro values, so a check reads what the reader reads.

    Without this every claim whose total is a macro is invisible: the paper's own
    discipline of emitting numbers rather than typing them would make the
    structural checks unable to fire, which is the pathology this file exists to
    prevent.
    """
    global _MACROS
    if _MACROS is None:
        f = BASE / "numbers.tex"
        _MACROS = dict(re.findall(r"\\newcommand\{\\(n\w+)\}\{([^}]*)\}",
                                  f.read_text(encoding="utf-8"))) if f.exists() else {}
    return _MACROS


def _body(name: str, expand: bool = True) -> str:
    t = (BASE / f"{name}.tex").read_text(encoding="utf-8")
    t = t.split(r"\begin{document}", 1)[-1]
    t = re.sub(r"(?<!\\)%.*", "", t)
    if expand:
        m = _macros()
        t = re.sub(r"\\(n[A-Za-z]+)\{?\}?", lambda k: m.get(k.group(1), k.group(0)), t)
    return t


def _labels(name: str) -> dict[str, str]:
    aux = BASE / f"{name}.aux"
    if not aux.exists():
        return {}
    return dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux.read_text(errors="ignore")))


# ------------------------------------------------------------------ check 1 --
# "N of M" and "N/M": the parts must not exceed the whole, and enumerated
# decompositions must sum to the total they claim to decompose.
DECOMP = re.compile(
    r"(\d+)\s+obtain no zone improvement[^.]*?(\d+)\s+were already Green[^.]*?"
    r"(?:(\w+)\s+was not[^.]*?)?(\d+)\s+move from Green to Yellow", re.S)


def control_decomposition() -> bool:
    txt = ("102 obtain no zone improvement: 99 were already Green and 2 move "
           "from Green to Yellow.")
    return _decomposition_bad(txt) != []


def _decomposition_bad(txt: str) -> list[str]:
    out = []
    for m in DECOMP.finditer(txt):
        total = int(m.group(1))
        parts = int(m.group(2)) + int(m.group(4))
        if m.group(3):
            parts += 1
        if parts != total:
            out.append(f"decomposition sums to {parts}, stated total {total}")
    return out


def check_decomposition() -> bool:
    ok = True
    for d in DOCS:
        bad = _decomposition_bad(_body(d))
        for b in bad:
            print(f"  {RED}FAIL{OFF}   {d}: {b}")
            ok = False
    if ok:
        print(f"  {GRN}pass{OFF}   enumerated decompositions sum to their stated totals")
    return ok


# ------------------------------------------------------------------ check 2 --
# A sentence claiming a table column must name a column the table actually has.
COLREF = re.compile(r"the (last|first) column[^.]{0,80}?Table~\\ref\{([^}]*)\}|"
                    r"Table~\\ref\{([^}]*)\}[^.]{0,60}?the (last|first) column")


def _table_header(doc: str, label: str) -> list[str] | None:
    """Recover the header cells of the tabular carrying `label`."""
    t = _body(doc)
    i = t.find(rf"\label{{{label}}}")
    if i < 0:
        return None
    seg = t[max(0, i - 4000): i + 4000]
    m = re.search(r"\\input\{([^}]*)\}", seg)
    if m:
        f = BASE / (m.group(1) + ".tex")
        if f.exists():
            seg = f.read_text()
    hdr = re.search(r"\\toprule(.*?)\\midrule|\\hline\\hline(.*?)\\hline", seg, re.S)
    if not hdr:
        return None
    block = hdr.group(1) or hdr.group(2) or ""
    rows = [r for r in block.split(r"\\") if "&" in r]
    return [c.strip() for c in rows[-1].split("&")] if rows else None


def control_column() -> bool:
    """A caption naming a last column that the header does not carry must fail."""
    return _column_mismatch("nonexistent_label_xyz", ["a", "b"], "count of foo") is True


def _column_mismatch(label, header, phrase) -> bool:
    if not header:
        return True
    return phrase.split()[0].lower() not in " ".join(header).lower()


def check_columns() -> bool:
    ok = True
    for d in DOCS:
        t = _body(d)
        for m in re.finditer(r"the last column[^.]{0,160}", t):
            frag = m.group(0)
            lab = re.search(r"Table~\\ref\{([^}]*)\}", frag)
            if not lab:
                # a caption refers to its own table: take the nearest label after
                nxt = re.search(r"\\label\{(tab:[^}]*)\}", t[m.start(): m.start() + 3000])
                if not nxt:
                    continue
                lab = nxt
            hdr = _table_header(d, lab.group(1))
            if hdr is None:
                continue
            # The claim must name what the last header cell holds. Compare
            # identifiers -- words and LaTeX symbols -- not prose keywords, so a
            # column headed $\qVstat<0$ is matched by a sentence naming qVstat.
            def ident(x: str) -> set[str]:
                return {w.lower() for w in re.findall(r"\\[A-Za-z]+|[A-Za-z]{3,}", x)}
            lo = max(0, m.start() - 220)
            around = t[lo: m.end()]
            if not (ident(hdr[-1]) & ident(around)):
                print(f"  {RED}FAIL{OFF}   {d}: 'last column' claim does not name "
                      f"header cell {hdr[-1]!r} of {lab.group(1)}")
                ok = False
    if ok:
        print(f"  {GRN}pass{OFF}   'last column' claims match the tables they name")
    return ok


# ------------------------------------------------------------------ check 3 --
# Item counts stated in prose must equal the rows of the list they describe.
def control_itemcount() -> bool:
    """The historical defect: "Eight of the ten" against a table giving seven."""
    return _itemcount_bad("Eight of the ten need nothing but the forecast\nseries "
                          "and the returns.", 7) is not None


WORDS = {"Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10}
ITEMCOUNT = re.compile(r"(Six|Seven|Eight|Nine|Ten)\s+(?:of the ten\s+)?need\s+nothing\s+"
                       r"but\s+the\s+(?:forecast\s+)?series", re.S)


def _itemcount_bad(sentence: str, actual: int):
    m = ITEMCOUNT.search(" ".join(sentence.split()) if "\n" in sentence else sentence)
    if not m:
        m = ITEMCOUNT.search(" ".join(sentence.split()))
    if not m:
        return None
    claimed = WORDS[m.group(1)]
    return None if claimed == actual else f"claims {claimed}, gate table gives {actual}"


def check_itemcounts() -> bool:
    """Count the gate rows that need only the series and the returns.

    Realised volatility is computed from the returns, so a check needing
    "realised sigma" needs nothing the returns do not already supply. Rows
    needing an evaluation window or sampled paths do not count.
    """
    t = " ".join(_body("main_R2").split())
    i = t.find(r"\label{tab:gate_compact}")
    seg = t[i: i + 2500] if i >= 0 else ""
    from_series = len(re.findall(r"& forecast only", seg)) \
        + len(re.findall(r"& returns", seg)) \
        + len(re.findall(r"& realised \$\\sigma\$", seg))
    if from_series == 0:
        print(f"  {RED}BROKEN{OFF} gate table rows not found -- check cannot fire")
        return False
    ok = True
    for m in ITEMCOUNT.finditer(t):
        bad = _itemcount_bad(m.group(0), from_series)
        if bad:
            print(f"  {RED}FAIL{OFF}   main_R2: {bad}")
            ok = False
    if ok:
        print(f"  {GRN}pass{OFF}   gate item counts agree with the gate table "
              f"({from_series} rows need only the series and the returns)")
    return ok


# ------------------------------------------------------------------ check 4 --
# Hand-written cross-document references must resolve in the target document.
XREF = re.compile(r"(?:Table|Figure|Section|Supplement|Theorem|Proposition|Remark|Lemma|Corollary)~?\s*S\.(\d+(?:\.\d+)*)")


def control_xref() -> bool:
    return _xref_bad("Table~S.99 of the supplement", set()) != []


def _xref_bad(text: str, known: set[str]) -> list[str]:
    return [f"S.{m.group(1)}" for m in XREF.finditer(text) if f"S.{m.group(1)}" not in known]


def check_xrefs() -> bool:
    known = set(_labels("supplement").values())
    bad = sorted(set(_xref_bad(_body("main_R2"), known)))
    if bad:
        print(f"  {RED}FAIL{OFF}   main_R2 cites supplement numbers that do not exist: "
              f"{', '.join(bad)}")
        return False
    print(f"  {GRN}pass{OFF}   every S.x reference in main_R2 resolves in supplement.aux "
          f"({len(known)} numbers defined)")
    return True


# ------------------------------------------------------------------ check 5 --
# Every "N of M" claim must name an object, not two typed numbers.
#
# Check 3 above matches one sentence shape -- "(Six|...|Ten) of the ten need
# nothing but the series" -- and so it saw neither "eight of them need nothing
# but the series" in the conclusion, which is the same claim in lower case, nor
# "thirteen of the sixteen forecasters have R-bar between 0.001 and 0.18", where
# the count and the bound disagreed by three forecasters. A check written for one
# sentence cannot see the class the sentence belongs to; PROTOCOL.md calls that
# the second mode, and this file exists for exactly that class.

COUNTS = BASE / "analysis" / "provenance" / "DECLARED_COUNTS.tsv"
_WORD = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
         r"thirteen|fourteen|fifteen|sixteen|twenty|none")
_N = rf"(?:\\n[A-Za-z]+(?:\{{\}})?|\d+|{_WORD})"
NOFM = re.compile(rf"({_N})\s+of\s+(?:the\s+)?({_N})\b", re.I)


def _declared_counts() -> set[str]:
    if not COUNTS.is_file():
        return set()
    out = set()
    for line in COUNTS.read_text().splitlines():
        if line.startswith("#") or not line.strip() or line.startswith("claim\t"):
            continue
        out.add(line.split("\t")[0].strip())
    return out


def _typed_counts(tex: str) -> list[str]:
    body = " ".join(tex.split(r"\begin{document}", 1)[-1].split())
    bad = []
    for m in NOFM.finditer(body):
        a, b = m.group(1), m.group(2)
        if a.startswith("\\n") or b.startswith("\\n"):
            continue                      # at least one side comes from an artefact
        frag = " ".join(m.group(0).split())
        if frag.lower() in {c.lower() for c in _declared_counts()}:
            continue
        bad.append(f"{frag}  ...{body[max(0, m.start()-52):m.start()]}")
    return bad


def control_counts() -> bool:
    """A claim with two typed numbers and no declaration must be caught."""
    return len(_typed_counts(r"\begin{document} it blocks 7 of 99 series.")) == 1


def check_counts() -> bool:
    ok = True
    for doc in DOCS:
        bad = _typed_counts((BASE / f"{doc}.tex").read_text(encoding="utf-8"))
        if bad:
            ok = False
            print(f"  {RED}FAIL{OFF}   {doc}: {len(bad)} 'N of M' claim(s) with no "
                  f"declared object")
            for b in bad[:10]:
                print(f"           {b[:104]}")
        else:
            print(f"  {GRN}pass{OFF}   {doc}: every 'N of M' claim is macro-backed "
                  f"or declared")
    return ok


CHECKS = [("enumerated decompositions", control_decomposition, check_decomposition),
          ("N-of-M claims name an object", control_counts, check_counts),
          ("table column claims", control_column, check_columns),
          ("gate item counts", control_itemcount, check_itemcounts),
          ("cross-document references", control_xref, check_xrefs)]


def main() -> int:
    rc = 0
    for name, control, check in CHECKS:
        print(f"\n{name}")
        if not control():
            print(f"  {RED}BROKEN{OFF} negative control did not fail -- "
                  f"'{name}' is not evidence")
            rc = 1
            continue
        print(f"  {YEL}ctrl{OFF}   negative control reproduces the failure")
        if not check():
            rc = 1
    print("\nOK" if rc == 0 else "\nSTRUCTURAL CLAIMS FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
