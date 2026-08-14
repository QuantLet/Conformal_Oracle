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
    """Every numeric token in the table body.

    A table can differ from the submitted copy in rule style (\toprule vs
    \hline\hline) or macro names while every reported value is identical. That
    is a formatting drift, not an erratum, and conflating the two would bury the
    cases that matter. Only a difference in this sequence is an erratum."""
    body = text
    for marker in (r"\midrule", r"\hline"):
        i = body.find(marker)
        if i != -1:
            body = body[i:]
            break
    body = re.sub(r"\\(?:cmidrule|multicolumn)\s*(?:\([^)]*\))?\{[^}]*\}", " ", body)
    return NUM.findall(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

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
        if live.exists():
            shutil.copy2(live, backup)
        ok, err = run_generator(gens[0])
        if not ok:
            rec["status"] = "RUN_FAILED"
            rec["detail"] = f"{gens[0].name}: {err.splitlines()[-1] if err else '?'}"
        elif not live.exists():
            rec["status"] = "NOT_EMITTED"
            rec["detail"] = f"{gens[0].name} ran but did not write {tpath.name}"
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

    order = {"DIFFERS": 0, "NOT_EMITTED": 1, "RUN_FAILED": 2, "NO_BASELINE": 3,
             "COSMETIC": 4, "PENDING": 5, "OK": 6, "FIGURE": 7}
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
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(rows)} artefacts; " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"wrote {OUT}/MANIFEST.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
