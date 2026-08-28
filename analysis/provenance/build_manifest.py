#!/usr/bin/env python3
"""Provenance manifest: for every table and figure in the paper, is there a
script that emits it, and does the emitted value match what was printed?

Three defects surfaced in three consecutive checks (a stale results CSV, a
divergent table generator, an undocumented CC convention). That rate says the
last one has not been found. This is the systematic pass.

Every entry lands in exactly one of three categories, which must not be merged:

  OK          a generator exists and reproduces the submitted artefact
  DIFFERS     a generator exists and its output differs  -> erratum
  NOT_EMITTED no generator produces the artefact         -> reproducibility gap

DIFFERS and NOT_EMITTED are different problems. Only the first is an erratum;
the second is what a data-availability editor asks about.

Method: each artefact is regenerated in the working tree and compared against
the frozen copy under submission_IJF/, which is what was actually submitted. The
working tree is restored afterwards, so the run is non-destructive.

Usage:
    python analysis/provenance/build_manifest.py [--run] [--only PATTERN]

Without --run, the inventory and generator mapping are produced but no generator
is executed (fast, safe). With --run, each generator is executed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import hashlib
import json
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
PAPER = BASE / "submission_IJF" / "main_R1.tex"
FROZEN = BASE / "submission_IJF"
OUT = Path(__file__).resolve().parent

# Verified by hand earlier in this audit; carried so the manifest is complete.
KNOWN = {
    "Quantlets/CO_full_evaluation/tab_master_results.tex":
        ("DIFFERS", "rebuild_master_table.py reproduces 109/110 cells; "
                    "Moirai 1.1 W/GJR printed 1.00 vs computed 0.99. The "
                    "shipped run_master_table.py does not emit this table at "
                    "all (9 models, wrong panels) and is marked SUPERSEDED."),
    "Quantlets/CO_multi_quantile_panel/tab_multiquantile.tex":
        ("DIFFERS", "Moirai 1.1 at alpha=0.01: 10/24 rejections printed, 9/24 "
                    "correct. Built from the stale moirai11_full_results.csv, "
                    "now replaced."),
}


# ---------------------------------------------------------------------------
# Verdict freshness. A verdict is a statement about a state, and this file used
# to record it without recording the state: `tab_regime_sensitivity.tex` was
# graded OK, the sign defect was corrected on 2026-08-17, its inputs moved, and
# the OK stayed on the page reading as a live guarantee. PROTOCOL.md calls this
# the fourth way a check stops being evidence -- the verdict outliving its state.
#
# Each verdict is therefore stamped with the SHA-256 of its producer and of every
# canonical table that producer reads. `--check-stale` re-hashes them and reports
# STALE for any verdict whose inputs have moved since it was recorded.
TABLES_DIR = BASE / "cfp_ijf_data" / "paper_outputs" / "tables"
STAMPS = OUT / "MANIFEST_STAMPS.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "absent"


def input_stamp(generators) -> dict:
    """Producer hashes, plus the canonical tables those producers name."""
    st = {}
    for g in generators:
        gp = Path(g) if Path(g).is_absolute() else BASE / g
        if not gp.is_file():
            # the manifest records generators by basename, not by path
            hits = sorted(BASE.glob(f"Quantlets/*/{g}")) + sorted(BASE.glob(f"scripts/{g}"))
            gp = hits[0] if hits else gp
        if not gp.is_file():
            st[str(g)] = "absent"
            continue
        st[str(g)] = _sha(gp)
        text = gp.read_text(encoding="utf-8", errors="replace")
        for t in sorted(TABLES_DIR.glob("*.csv")):
            if t.name in text:
                st[f"tables/{t.name}"] = _sha(t)
    return st


def check_stale() -> int:
    if not STAMPS.is_file():
        print("no MANIFEST_STAMPS.json; run build_manifest.py --run to record one")
        return 2
    rec = json.loads(STAMPS.read_text())
    stale = []
    for art, e in rec.get("artefacts", {}).items():
        now = input_stamp(e.get("generators", []))
        moved = [k for k in set(now) | set(e.get("inputs", {}))
                 if now.get(k) != e.get("inputs", {}).get(k)]
        if moved:
            stale.append((art, e.get("status", "?"), moved))
    for art, status, moved in stale:
        print(f"  STALE  {art}  (recorded {status})  inputs moved: {', '.join(sorted(moved)[:4])}")
    if stale:
        print(f"\n{len(stale)} verdict(s) describe a state that has changed since they were recorded.")
        return 1
    print(f"{len(rec.get('artefacts', {}))} verdict(s), none stale")
    return 0


def inventory() -> tuple[list[str], list[str]]:
    text = PAPER.read_text(encoding="utf-8", errors="replace")
    text = "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("%"))
    inputs = []
    for m in re.finditer(r"\\input\{([^}]*)\}", text):
        t = m.group(1)
        inputs.append(t if t.endswith(".tex") else t + ".tex")
    figs = [m.group(1) for m in
            re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text)]
    return sorted(set(inputs)), sorted(set(figs))


def find_generator(target: Path) -> list[Path]:
    """Scripts in the artefact's directory that mention its filename."""
    d = BASE / target.parent
    if not d.is_dir():
        return []
    hits = []
    for py in sorted(d.glob("*.py")):
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if target.name in src or target.stem in src:
            hits.append(py)
    return hits


def run_generator(script: Path, timeout: int = 1800) -> tuple[bool, str]:
    try:
        p = subprocess.run([sys.executable, script.name], cwd=script.parent,
                           capture_output=True, timeout=timeout)
        return p.returncode == 0, (p.stderr.decode("utf-8", "replace")[-400:]
                                   if p.returncode else "")
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as exc:
        return False, str(exc)


def normalise(text: str) -> str:
    """Ignore whitespace-only and macro-rename differences when comparing."""
    text = text.replace(r"\qVstat", r"\qV")
    return "\n".join(ln.rstrip() for ln in text.splitlines() if ln.strip())


NUM = re.compile(r"-?(?:\d+\.\d+|\.\d+|\d+)")


def numbers(text: str) -> list[str]:
    """Every numeric token in the table, formatting stripped.

    Earlier this anchored the body at the first \\hline or \\midrule. That is
    wrong when the two files use different rule macros: a booktabs version
    starts at \\toprule and a plain one at \\hline\\hline, so the two get
    truncated at different offsets and every subsequent token misaligns --
    reporting a purely cosmetic table as a numeric erratum. Strip the rule and
    layout macros instead, and compare what is left.
    """
    t = text
    t = re.sub(r"\\(?:top|mid|bottom)rule|\\hline", " ", t)
    t = re.sub(r"\\cmidrule\s*(?:\([^)]*\))?\{[^}]*\}", " ", t)
    t = re.sub(r"\\multicolumn\s*\{[^}]*\}\s*\{[^}]*\}", " ", t)
    t = re.sub(r"\\begin\{tabular\}\s*\{[^}]*\}", " ", t)
    t = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", " ", t)
    t = re.sub(r"\\(?:label|ref|cite\w*)\{[^}]*\}", " ", t)
    return NUM.findall(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--check-stale", action="store_true",
                    help="re-hash each verdict's inputs and report those that moved")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    if args.check_stale:
        return check_stale()

    inputs, figs = inventory()
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="manifest_"))

    for target in inputs:
        if args.only and args.only not in target:
            continue
        tpath = Path(target)
        live = BASE / tpath
        frozen = FROZEN / tpath
        gens = find_generator(tpath)
        rec = {"artefact": target, "generators": [g.name for g in gens],
               "status": "", "detail": ""}

        if target in KNOWN:
            rec["status"], rec["detail"] = KNOWN[target]
            rows.append(rec)
            continue
        if not gens:
            rec["status"] = "NOT_EMITTED"
            rec["detail"] = "no script in the artefact's directory writes it"
            rows.append(rec)
            continue
        if not frozen.exists():
            rec["status"] = "NO_BASELINE"
            rec["detail"] = "no frozen copy under submission_IJF/ to compare to"
            rows.append(rec)
            continue
        if not args.run:
            rec["status"] = "PENDING"
            rec["detail"] = f"generator found ({gens[0].name}); rerun with --run"
            rows.append(rec)
            continue

        backup = tmp / tpath.name
        before = None
        if live.exists():
            shutil.copy2(live, backup)
            before = (live.stat().st_mtime_ns,
                      hashlib.sha256(live.read_bytes()).hexdigest())
        ok, err = run_generator(gens[0])
        # A generator can exit 0 without touching its target -- wrong output
        # path, an early return, a silently skipped branch. The pre-existing
        # file would then be compared against the baseline and could be
        # reported OK, turning a failure into a pass. Require positive evidence
        # that THIS run wrote the file.
        wrote = False
        if live.exists():
            after = (live.stat().st_mtime_ns,
                     hashlib.sha256(live.read_bytes()).hexdigest())
            wrote = before is None or after[0] != before[0] or after[1] != before[1]
        if not ok:
            rec["status"] = "RUN_FAILED"
            rec["detail"] = f"{gens[0].name}: {err.splitlines()[-1] if err else '?'}"
        elif not live.exists():
            rec["status"] = "NOT_EMITTED"
            rec["detail"] = f"{gens[0].name} ran but did not write {tpath.name}"
        elif not wrote:
            rec["status"] = "NOT_WRITTEN"
            rec["detail"] = (f"{gens[0].name} exited 0 but left {tpath.name} "
                             "untouched (mtime and hash unchanged) -- the file "
                             "on disk predates this run, so no verdict is possible")
        else:
            lt, ft = live.read_text(errors="replace"), frozen.read_text(errors="replace")
            same_text = normalise(lt) == normalise(ft)
            same_nums = numbers(lt) == numbers(ft)
            if same_text:
                rec["status"], rec["detail"] = "OK", f"reproduced by {gens[0].name}"
            elif same_nums:
                rec["status"] = "COSMETIC"
                rec["detail"] = (f"{gens[0].name}: every reported value is "
                                 "identical; only formatting differs")
            else:
                ln, fn = numbers(lt), numbers(ft)
                n_diff = sum(1 for a, b in zip(ln, fn) if a != b) + abs(len(ln) - len(fn))
                rec["status"] = "DIFFERS"
                rec["detail"] = (f"{gens[0].name}: {n_diff} numeric token(s) "
                                 f"differ from the submitted copy "
                                 f"({len(ln)} vs {len(fn)} tokens)")
        if backup.exists():
            shutil.copy2(backup, live)
        rows.append(rec)

    for f in figs:
        rows.append({"artefact": f, "generators": [], "status": "FIGURE",
                     "detail": "figure; checked separately"})

    order = {"DIFFERS": 0, "NOT_EMITTED": 1, "NOT_WRITTEN": 2, "RUN_FAILED": 3,
             "NO_BASELINE": 4, "COSMETIC": 5, "PENDING": 6, "OK": 7, "FIGURE": 8}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["artefact"]))

    L = ["# Provenance manifest", "",
         "For every table and figure in `main_R1.tex`: does a script emit it, "
         "and does the emitted value match what was submitted?", "",
         "| Status | Meaning |", "|---|---|",
         "| `OK` | a generator exists and reproduces the submitted artefact |",
         "| `DIFFERS` | a generator exists, output differs — **erratum** |",
         "| `NOT_EMITTED` | no generator — **reproducibility gap**, not an erratum |",
         "| `RUN_FAILED` | generator exists but does not execute |",
         "| `COSMETIC` | regenerates with identical values; only formatting differs |",
         "| `NOT_WRITTEN` | generator exited 0 but did not touch the file — **no verdict** |",
         "| `PENDING` | generator found, not yet executed |", ""]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    L += ["## Summary", "", "| Status | Count |", "|---|---|"]
    for s, c in sorted(counts.items(), key=lambda kv: order.get(kv[0], 9)):
        L.append(f"| {s} | {c} |")
    L += ["", "## Detail", "",
          "| Artefact | Status | Generator | Note |", "|---|---|---|---|"]
    for r in rows:
        if r["status"] == "FIGURE":
            continue
        g = ", ".join(r["generators"]) or "—"
        L.append(f"| `{r['artefact']}` | **{r['status']}** | {g} | {r['detail']} |")
    L += ["", "## Figures", "",
          "| Artefact | Note |", "|---|---|"]
    for r in rows:
        if r["status"] == "FIGURE":
            L.append(f"| `{r['artefact']}` | {r['detail']} |")
    L.append("")

    (OUT / "MANIFEST.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    STAMPS.write_text(json.dumps(
        {"artefacts": {r["artefact"]: {"status": r["status"],
                                       "generators": list(r["generators"]),
                                       "inputs": input_stamp(r["generators"])}
                       for r in rows}}, indent=2), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(rows)} artefacts; " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"wrote {OUT}/MANIFEST.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
