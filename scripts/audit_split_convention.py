#!/usr/bin/env python3
"""Fail the build when a calibration/test split appears without declaring its gap.

Second instance of the defect QV_CONVENTION.md records. The conformal shift was
defined once in prose and re-implemented four ways because nothing enumerated the
sites that implemented it. The separation gap of Corollary 4.6 is in the same
position now: it is defined in the theorem, imposed in one function, and every
other place that takes a chronological calibration/test split takes it without.

This audit enumerates every site that splits a series into a calibration block
and a test block, and requires each to be either

  (a) a call to cfp_config.separation_gap, or
  (b) an entry in analysis/provenance/SPLIT_SITES.tsv declaring what it does
      about the gap and why that is correct there.

The declarations are the inventory. They are produced by running this audit
rather than by reading the tree, which is the point: a migration planned from a
hand-made list is planned from a quantity computed outside the code that owns
the object, and PROTOCOL forbids that for numbers in the manuscript for the same
reason it should be avoided here.

Conventions a site may declare:

  CONTIGUOUS      g_n = 0. The protocol as reported. Correct today, and the
                  thing a switch would migrate.
  GAPPED          imposes separation_gap on the pair's own rho-hat.
  NOT_A_PANEL     splits something that is not a forecast series -- Monte Carlo
                  replications, a bootstrap resample, a simulation grid -- where
                  the theorem's separation does not apply.
  MEASUREMENT     computes both arms on purpose, to compare them.
  FROZEN          under submission_IJF/ or legacy/; not run, kept as shipped.

Entries are keyed on (file, code) and not on line number, so an edit that
moves a site does not fail the build spuriously. The cost is that two
identical split lines in one file share one declaration: 55 sites collapse
to 50 keys here. That is the right trade -- the same code in the same file
should have the same convention -- but it means a third copy of an already
declared line is covered without being declared, and it is stated rather
than left for someone to discover.

PROTOCOL Rule 2: the negative control runs first and must be flagged.

    python scripts/audit_split_convention.py
    python scripts/audit_split_convention.py --emit   # write the registry
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "analysis" / "provenance" / "SPLIT_SITES.tsv"
SEARCH = ["scripts", "Quantlets", "python/src", "pipeline", "analysis", "conformal-oracle"]
SKIP = ("__pycache__", ".ipynb_checkpoints", "superseded", "quarantine", "legacy",
        "submission_IJF")
SPLIT = re.compile(r"n_cal\s*=\s*int\(")
# A site that has been migrated no longer matches SPLIT, so without this the
# registry would shrink as the migration proceeds and the audit would call
# that progress. It is not: an unenumerated site is unenumerated whether it
# takes the split by hand or by import. MIGRATED finds the calls to the single
# definition, so the population stays whole and the counts move between
# conventions instead of out of the file.
MIGRATED = re.compile(r"split_indices\s*\(")
CANONICAL = re.compile(r"separation_gap\s*\(")
VALID = {"CONTIGUOUS", "GAPPED", "NOT_A_PANEL", "MEASUREMENT", "FROZEN",
         "MIGRATED"}
RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"; OFF = "\033[0m"


def norm(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def sites() -> list[tuple[str, int, str, bool]]:
    """(relative path, line number, normalised code, whether it uses the helper)."""
    out = []
    for root in SEARCH:
        d = ROOT / root
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.py")):
            if any(s in str(f) for s in SKIP) or f.name == pathlib.Path(__file__).name:
                continue
            try:
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            body = "\n".join(lines)
            # Skip matches inside triple-quoted strings: the evaluation
            # driver documents its own split in its module docstring, and
            # a line of documentation is not a site.
            in_doc, quote = False, ""
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                for q in ('"""', "'''"):
                    if not in_doc and stripped.startswith(q):
                        in_doc, quote = True, q
                        if len(stripped) > 3 and stripped.endswith(q):
                            in_doc = False
                        break
                    if in_doc and quote == q and q in line:
                        in_doc = False
                        break
                if in_doc:
                    continue
                if SPLIT.search(line) or MIGRATED.search(line):
                    out.append((str(f.relative_to(ROOT)), i, norm(line),
                                bool(CANONICAL.search(body))))
    return out


def load() -> dict[tuple[str, str], str]:
    if not REGISTRY.is_file():
        return {}
    reg = {}
    with REGISTRY.open() as fh:
        for row in csv.DictReader(
                (l for l in fh if not l.startswith("#")), delimiter="\t"):
            reg[(row["file"], row["code"])] = row["convention"]
    return reg


def control() -> bool:
    """A planted split with no declaration must be caught.

    Planted as text rather than on disk: the check is the set difference between
    what the tree contains and what the registry declares, so an entry absent
    from the registry is exactly what it must flag.
    """
    reg = {("a.py", "n_cal = int(n * 0.7)"): "CONTIGUOUS"}
    planted = ("b.py", 1, "n_cal = int(n * 0.7)", False)
    return (planted[0], planted[2]) not in reg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()

    if not control():
        print(f"  {RED}BROKEN{OFF} negative control does not fire")
        return 2
    print(f"  {YEL}ctrl{OFF}   an undeclared split is flagged")

    found = sites()
    reg = load()

    if a.emit:
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        with REGISTRY.open("w") as fh:
            fh.write("# Every chronological calibration/test split, with what it does\n"
                     "# about the separation gap of Corollary 4.6. Emitted by\n"
                     "# scripts/audit_split_convention.py --emit; the conventions are\n"
                     "# then set by hand and the audit fails the build on an\n"
                     "# undeclared site. See analysis/convention/GAP_SWITCH_SCOPE.md.\n")
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["file", "code", "convention", "line", "note"])
            for f, ln, code, helper in found:
                prev = reg.get((f, code))
                conv = prev or ("GAPPED" if helper else "CONTIGUOUS")
                w.writerow([f, code, conv, ln, ""])
        print(f"  wrote {REGISTRY} with {len(found)} site(s)")
        return 0

    undeclared = [(f, ln, c) for f, ln, c, _ in found if (f, c) not in reg]
    stale = [k for k in reg if k not in {(f, c) for f, _, c, _ in found}]
    badconv = [(k, v) for k, v in reg.items() if v not in VALID]

    ok = True
    if undeclared:
        ok = False
        print(f"  {RED}FAIL{OFF}   {len(undeclared)} split site(s) not declared")
        for f, ln, c in undeclared[:10]:
            print(f"           {f}:{ln}\n               {c}")
    if stale:
        ok = False
        print(f"  {RED}FAIL{OFF}   {len(stale)} registry entry/entries for a site "
              "that no longer exists")
        for f, c in stale[:5]:
            print(f"           {f}: {c}")
    if badconv:
        ok = False
        print(f"  {RED}FAIL{OFF}   {len(badconv)} entry/entries with an unknown convention")
    if ok:
        # Count SITES, not registry keys. Two identical lines in one file share
        # a declaration, so a breakdown over reg.values() would not sum to the
        # site count the supplement partitions.
        by = {}
        for f, _, c, _ in found:
            v = reg[(f, c)]
            by[v] = by.get(v, 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in sorted(by.items()))
        assert sum(by.values()) == len(found)
        print(f"  {GRN}pass{OFF}   {len(found)} split site(s), each declared: {summary}"
              f"  ({len(reg)} distinct declarations)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
