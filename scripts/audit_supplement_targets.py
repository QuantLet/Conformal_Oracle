#!/usr/bin/env python3
"""Resolve every Supplement S.x reference in the body to the heading it lands on.

`audit_structural_claims.py` checks that each S.x number is *defined* in
supplement.aux. That check passes on a reference that resolves to the wrong
subsection, which is exactly what happened when the proofs were renumbered. This
one prints the target heading beside the citing sentence so a wrong target is
visible, and fails when a number resolves to nothing.

PROTOCOL Rule 2: the negative control runs first.
"""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

def supplement_headings():
    """Map S.x number -> heading text, by replaying \\section/\\subsection order."""
    tex = (ROOT / "supplement.tex").read_text()
    tex = re.sub(r"(?m)^\s*%.*$", "", tex)
    out, sec, sub = {}, 0, 0
    for m in re.finditer(r"\\(section|subsection)\*?\{(.+?)\}\s*$", tex, re.M):
        kind, title = m.group(1), m.group(2)
        title = re.sub(r"\\[a-zA-Z]+\s*", "", title).replace("{", "").replace("}", "").strip()
        if kind == "section":
            sec += 1; sub = 0; out[f"S.{sec}"] = title
        else:
            sub += 1; out[f"S.{sec}.{sub}"] = title
    return out

def body_references():
    refs = []
    for f in [ROOT / "main_R2.tex"] + sorted((ROOT / "sections").glob("*.tex")):
        for i, line in enumerate(f.read_text().split("\n"), 1):
            for m in re.finditer(r"Supplement~(S\.\d+(?:\.\d+)?)", line):
                start = max(0, m.start() - 90)
                refs.append((f.name, i, m.group(1), line[start:m.end() + 40].strip()))
    return refs

def main():
    head = supplement_headings()
    refs = body_references()
    bad = [r for r in refs if r[2] not in head]

    print("negative control: a reference to a subsection that does not exist")
    fake = "S.99.99"
    print(f"  {fake} -> {'RESOLVES (control broken)' if fake in head else 'unresolved, as required'}")
    if fake in head:
        sys.exit("control failed")

    print(f"\n{len(refs)} Supplement references in the body, {len(head)} headings in the supplement\n")
    for name, ln, num, ctx in refs:
        target = head.get(num, "*** UNRESOLVED ***")
        print(f"  {name}:{ln:<5} {num:<8} -> {target}")
        print(f"      ...{ctx}")
    if bad:
        print(f"\n{len(bad)} unresolved reference(s)")
        sys.exit(1)
    print("\nevery reference resolves; the target headings are printed above for reading.")

if __name__ == "__main__":
    main()
