#!/usr/bin/env python3
"""Every number in the prose must be traceable to a generated artefact.

The provenance manifest compares generated `.tex` *table files* against their
submitted copies. Numbers written inline in the prose of the manuscript have
never been checked by anything. Both errors found there so far -- the 205/240
claim, which no code path produces, and the commodity 64%/61% erratum -- were
caught by reading. Reading does not scale and did not generalise.

This script extracts numeric literals from the prose (tables excluded: the
manifest owns those) and asks whether each appears in any generated artefact.

    SOURCED    the literal occurs in some emitted .tex/.csv/.md under the
               searched trees
    UNSOURCED  it does not -- either hand-entered, stale, or derived by an
               arithmetic step no script performs

A match is necessary, not sufficient: a number can coincide with an unrelated
value elsewhere. This is a screen that bounds the hand-entered surface, not a
proof of provenance. It is deliberately biased toward false SOURCED rather than
false UNSOURCED, so the UNSOURCED list is short enough to work through by hand
and everything on it is worth checking.

Ignored by construction: years, section/table/figure/equation cross-references,
citation keys, LaTeX lengths and font sizes, the nominal alpha levels, and
integers below a threshold that are almost always counts of sections or
footnotes rather than results.

Usage:
    python scripts/audit_prose_numbers.py [--tex main_R2.tex] [--min-decimals 2]
Output: analysis/provenance/PROSE_NUMBERS.md and .csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "analysis" / "provenance"

# Trees that hold *generated* artefacts. Prose documentation is deliberately
# excluded: a number is not sourced because a narrative .md repeats it, and
# including this repository's own provenance write-ups would make the check
# circular -- they quote the very numbers under audit. Scripts are excluded for
# the same reason: a literal hard-coded in a script is the thing being looked
# for, not evidence for it.
SEARCH_TREES = ["Quantlets", "analysis", "paper_outputs", "pipeline/results"]
SEARCH_SUFFIXES = {".tex", ".csv"}
SKIP_DIRS = {"quarantine", "superseded", "recovered", "__pycache__"}
# This script's own output lists every literal it is about to look for. Left in
# the haystack it reports 0 unsourced on the second run regardless of the truth,
# and the tell is that shrinking the haystack lowered the unsourced count, which
# is arithmetically impossible.
SKIP_STEMS = {"PROSE_NUMBERS"}

# environments the manifest already owns
DROP_ENVS = ["tabular", "tabularx", "table", "table*", "longtable", "sidewaystable"]
# macros whose arguments are never results
DROP_MACROS = ["cite", "citep", "citet", "citeauthor", "ref", "eqref", "label",
               "autoref", "cref", "Cref", "includegraphics", "input", "include",
               "bibliography", "usepackage", "documentclass", "hspace", "vspace",
               "setlength", "caption", "url", "href", "footnotemark"]

# literals that are structural, not empirical
ALPHA_LEVELS = {"0.01", "0.025", "0.05", "0.10", "0.1"}


def strip_tex(src: str) -> str:
    """Remove comments, table environments and reference-bearing macros."""
    src = re.sub(r"(?<!\\)%.*", "", src)
    for env in DROP_ENVS:
        src = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
                     " ", src, flags=re.S)
    for m in DROP_MACROS:
        # macro plus its optional and mandatory arguments
        src = re.sub(rf"\\{m}\s*(\[[^\]]*\])?\s*(\{{[^{{}}]*\}})*", " ", src)
    return src


def extract(src: str, min_decimals: int) -> list[tuple[str, str]]:
    """(literal, surrounding context) for each candidate numeric claim."""
    out, seen = [], set()
    pat = re.compile(rf"(?<![\w.]) (\d{{1,3}}(?:,\d{{3}})* | \d+ ) "
                     rf"(?: \. (\d{{{min_decimals},}}) )? (?![\w.])",
                     re.X)
    for m in pat.finditer(src):
        lit = m.group(0).strip()
        if m.group(2) is None:          # bare integer
            v = m.group(1).replace(",", "")
            if len(v) == 4 and v.startswith(("19", "20")):
                continue                # a year
            if int(v) < 100:
                continue                # section counts, footnotes, small ints
        if lit in ALPHA_LEVELS:
            continue
        ctx = re.sub(r"\s+", " ", src[max(0, m.start() - 90):m.end() + 90]).strip()
        key = (lit, ctx[:40])
        if key in seen:
            continue
        seen.add(key)
        out.append((lit, ctx))
    return out


def build_index(trees, suffixes) -> str:
    """One big haystack of every generated artefact."""
    chunks = []
    for t in trees:
        p = BASE / t
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if SKIP_DIRS & set(f.parts) or f.stem in SKIP_STEMS:
                continue
            if f.suffix.lower() in suffixes and f.is_file():
                try:
                    chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="main_R2.tex")
    ap.add_argument("--min-decimals", type=int, default=2)
    a = ap.parse_args()

    tex = BASE / a.tex
    if not tex.exists():
        print(f"no such file: {tex}", file=sys.stderr)
        return 1
    prose = strip_tex(tex.read_text(encoding="utf-8", errors="ignore"))
    cands = extract(prose, a.min_decimals)

    hay = build_index(SEARCH_TREES, SEARCH_SUFFIXES)
    print(f"haystack {len(hay) / 1e6:.1f} MB | {len(cands)} candidate literals",
          file=sys.stderr)

    rows = []
    for lit, ctx in cands:
        plain = lit.replace(",", "")
        rows.append({"literal": lit, "sourced": (lit in hay) or (plain in hay),
                     "context": ctx})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "PROSE_NUMBERS.csv", index=False)

    uns = df[~df["sourced"]]
    L = [f"# Prose numbers in `{a.tex}`", "",
         "The manifest owns the generated table files. This covers the numbers "
         "written inline in the prose, which nothing checked before.", "",
         f"- candidates examined: **{len(df)}**",
         f"- found in a generated artefact: **{int(df['sourced'].sum())}**",
         f"- **not found: {len(uns)}**", "",
         "A match is necessary, not sufficient — a literal can coincide with an "
         "unrelated value. This bounds the hand-entered surface; it does not "
         "prove provenance. Years, cross-references, citation keys, LaTeX "
         "lengths, the nominal α levels and integers below 100 are excluded by "
         "construction.", ""]
    if len(uns):
        L += ["## Not found in any generated artefact", "",
              "| literal | context |", "|---|---|"]
        for _, r in uns.iterrows():
            c = r["context"].replace("|", "\\|")
            L.append(f"| `{r['literal']}` | …{c}… |")
    else:
        L.append("No unsourced literals.")
    L.append("")
    (OUT / "PROSE_NUMBERS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:14]))
    print(f"\nfull list: {OUT / 'PROSE_NUMBERS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
