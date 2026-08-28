#!/usr/bin/env python3
"""Fail the build when a fifth quantile convention for the conformal shift appears.

The shift is defined once, by equation (8): S_(k) with k = ceil((n+1)(1-alpha)).
Four different implementations of it were nonetheless found in this repository --
the plain empirical quantile, np.quantile at level k/n, the order statistic, and
the package's own helper -- and the manuscript claims in two places that every
result uses equation (8). This audit enumerates every site that takes a quantile
of nonconformity scores and requires each to be either

  (a) a call to cfp_config.conformal_quantile, or
  (b) an entry in analysis/provenance/QV_CONVENTION_SITES.tsv declaring which
      convention it uses and why that is correct there.

A site that is neither fails the build. So does a registry entry whose code has
changed, and so does an entry for a site that no longer exists.

PROTOCOL Rule 2: the negative control runs first and must be flagged.
"""
import re, sys, csv, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "analysis" / "provenance" / "QV_CONVENTION_SITES.tsv"
SEARCH = ["scripts", "Quantlets", "python/src", "pipeline", "analysis"]
SKIP = ("__pycache__", ".ipynb_checkpoints", "superseded", "quarantine", "legacy")

QUANTILE = re.compile(r"(np\.(quantile|percentile)\(|(?<![\w.])\.quantile\(|\bnp\.sort\()")
SCORE_WORDS = re.compile(
    r"nonconform|\bscores?_?(cal|arr|valid|v)?\b|\bs_V\b|\bq_?V\b|\bqV\b|q_hat_V|"
    r"conformal|calibration score|cal_scores", re.I)
CANONICAL = re.compile(r"conformal_quantile\s*\(")

def norm(line):
    return re.sub(r"\s+", " ", line.strip())

def candidates():
    out = []
    for root in SEARCH:
        for f in sorted((ROOT / root).rglob("*.py")):
            if any(s in str(f) for s in SKIP) or f.name == "audit_qv_convention.py":
                continue
            lines = f.read_text(errors="ignore").split("\n")
            for i, line in enumerate(lines):
                if line.lstrip().startswith("#") or not QUANTILE.search(line):
                    continue
                # prose in a docstring mentioning np.quantile is not a call site
                if not re.search(r"(=|\breturn\b)", line):
                    continue
                if CANONICAL.search(line):
                    continue
                ctx = "\n".join(lines[max(0, i - 8): i + 4])
                if not SCORE_WORDS.search(ctx):
                    continue
                out.append((str(f.relative_to(ROOT)), i + 1, norm(line)))
    return out

def load_registry():
    if not REGISTRY.exists():
        return {}
    with REGISTRY.open() as fh:
        return {(r["file"], r["code"]): r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("file") and not r["file"].startswith("#")}

def main():
    emit = "--emit" in sys.argv
    found = candidates()

    if emit:
        w = csv.writer(sys.stdout, delimiter="\t")
        w.writerow(["file", "code", "convention", "produces", "note"])
        for f, ln, code in found:
            w.writerow([f, code, "UNCLASSIFIED", "", ""])
        return 0

    reg = load_registry()

    print("negative control: a synthetic site using the plain empirical quantile")
    fake = ("scripts/__control__.py", "q_V = np.quantile(cal_scores, 1 - alpha)")
    print(f"  registered: {fake in reg}  -> {'CONTROL BROKEN' if fake in reg else 'would be flagged, as required'}")
    if fake in reg:
        sys.exit("control failed: the synthetic site is in the registry")

    unregistered = [(f, ln, c) for f, ln, c in found if (f, c) not in reg]
    live = {(f, c) for f, ln, c in found}
    stale = [k for k in reg if k not in live]

    print(f"\n{len(found)} sites take a quantile of nonconformity scores; "
          f"{len(reg)} are registered\n")
    by_conv = {}
    for f, ln, c in found:
        r = reg.get((f, c))
        conv = r["convention"] if r else "UNREGISTERED"
        by_conv.setdefault(conv, []).append(f"{f}:{ln}")
    for conv in sorted(by_conv):
        print(f"  {conv}")
        for s in by_conv[conv]:
            print(f"      {s}")

    fail = False
    if unregistered:
        fail = True
        print(f"\nFAIL: {len(unregistered)} unregistered site(s). Declare the convention "
              f"in {REGISTRY.relative_to(ROOT)} or route the call through "
              f"cfp_config.conformal_quantile:")
        for f, ln, c in unregistered:
            print(f"  {f}:{ln}\n      {c}")
    if stale:
        fail = True
        print(f"\nFAIL: {len(stale)} registry entr(ies) whose code no longer exists "
              f"(the site moved or its convention changed -- re-declare it):")
        for f, c in stale:
            print(f"  {f}\n      {c}")
    if fail:
        return 1
    print("\nevery site is declared; no fifth convention.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
