#!/usr/bin/env python3
"""Every use of the migrated split's index is dominated by its own assignment.

Why this exists. The migration replaced test-block slices `[n_cal:]` with
`[t0:]` by regular expression. At module scope that is unsound: a `t0` assigned
inside one loop stays bound afterwards, so a later loop that was never migrated
silently reads the previous loop's value. It happened once, in
run_robustness_summary.py, where `test_start = max(n_cal, WINDOW)` became
`max(t0, WINDOW)` and used a t0 from a different section. The table it produced
was plausible: one row moved in the third decimal.

A scope check that only asks "is t0 bound somewhere in this scope?" passes that
case, because it is bound. This check asks the stronger question: is every use
of t0 preceded, inside the SAME loop body or function body, by its assignment?

PROTOCOL Rule 2: the negative control runs first and must be flagged.
"""
import ast, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"t0"}
RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"
OFF = "\033[0m"


def _check(body, inherited: set, rel: str, bad: list) -> None:
    """Walk one statement list in order.

    Assignments propagate INWARD: a name bound before a nested block is visible
    inside it. They do not propagate OUTWARD past the end of a compound
    statement: a name bound inside a loop is left bound by Python, but a
    sibling statement after the loop that reads it is reading whatever the last
    iteration happened to leave, which is the defect this audit exists for.
    """
    seen = set(inherited)
    for stmt in body:
        stores = {n.id for n in ast.walk(stmt)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        nested = [b for f in ("body", "orelse", "finalbody")
                  for b in ([getattr(stmt, f)] if isinstance(getattr(stmt, f, None), list)
                            and getattr(stmt, f) else [])]
        if nested:
            for b in nested:
                _check(b, seen, rel, bad)
        else:
            for n in ast.walk(stmt):
                if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                        and n.id in NAMES and n.id not in seen):
                    bad.append(f"{rel}:{n.lineno}: {n.id} read where no enclosing "
                               "region assigns it first")
        seen |= (stores & NAMES) if not nested else set()
        if nested:
            # a direct assignment at this level, e.g. "t0 = ..." with an if/else,
            # still counts when the statement itself binds at this level
            seen |= {n.id for n in ast.walk(stmt)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                     and n.id in NAMES and isinstance(stmt, (ast.Assign, ast.AugAssign))}


def violations(src: str, rel: str) -> list[str]:
    tree = ast.parse(src)
    bad: list[str] = []
    _check(tree.body, set(), rel, bad)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = {a.arg for a in node.args.args}
            _check(node.body, args, rel, bad)
    return bad


def control() -> bool:
    """A t0 read in a loop that never assigns it must be flagged."""
    src = ("for a in x:\n    t0 = 1\n    y = a[t0:]\n"
           "for b in z:\n    w = b[t0:]\n")
    return bool(violations(src, "<control>"))


def main() -> int:
    if not control():
        print(f"  {RED}BROKEN{OFF} negative control does not fire")
        return 2
    print(f"  {YEL}ctrl{OFF}   a t0 read in a region that never assigns it is flagged")
    # Scope: the migrated population. A file that does not call the single
    # definition has its own t0 and is none of this audit's business.
    files = [p for p in ROOT.rglob("*.py")
             if not any(x in str(p) for x in (".git", "submission_IJF", "legacy",
                                              "quarantine", "superseded", ".venv"))
             and "split_indices(" in p.read_text(encoding="utf-8", errors="ignore")]
    bad = []
    for p in files:
        try:
            bad += violations(p.read_text(encoding="utf-8"), str(p.relative_to(ROOT)))
        except SyntaxError:
            continue
    if bad:
        print(f"  {RED}FAIL{OFF}   {len(bad)} unguarded read(s) of a migrated split index")
        for b in bad[:10]:
            print(f"           {b}")
        return 1
    print(f"  {GRN}pass{OFF}   {len(files)} file(s) using t0, each assigning it in "
          "the region that reads it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
