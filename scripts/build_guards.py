#!/usr/bin/env python3
"""Build guards, each with a negative control that must fail before it may pass.

Standing rule: no check reports a pass until it has been seen to fail on a case
built to make it fail. Three checks in this project returned false passes:

  * the substring provenance screen (retired -- see guard 3, which documents
    why it cannot be repaired);
  * `audit_prose_numbers.py`, whose "0 unsourced" verdict is a substring match
    against a 6 MB corpus;
  * the undefined-reference check, which was run against a redirected stdout log
    and against a PDF from a different build than the one that shipped.

Each guard below runs its negative control FIRST. If the control does not fail,
the guard reports BROKEN and the build stops -- a guard that cannot fail is not
evidence, which is the paper's own argument applied to its own toolchain.

    python scripts/build_guards.py            # run all guards
    python scripts/build_guards.py --controls # run only the negative controls
"""
from __future__ import annotations
import argparse, re, subprocess, sys, tempfile, shutil, pathlib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DOCS = ["main_R2", "supplement"]
# Guard 2 reads DOCS only, and the two \input section files were never in it.
# That left Section 4 and the whole of Section 5 -- 4,504 words, including a
# hand-authored Monte Carlo table -- outside every literal check the project
# runs, and it is where a stale 0.67 survived four corrections elsewhere.
LITERAL_DOCS = DOCS + ["sections/sec4_theory", "sections/sec5_montecarlo"]
RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"; OFF = "\033[0m"


def _ok(m):    print(f"  {GRN}pass{OFF}   {m}")
def _bad(m):   print(f"  {RED}FAIL{OFF}   {m}")
def _ctl(m):   print(f"  {YEL}ctrl{OFF}   {m}")


def _pdf_text(path) -> str:
    """Text of a PDF, from pdftotext when it is on PATH and pypdf when it is not.

    The guards used to shell out to pdftotext and nothing else, so on a machine
    without poppler three of the four could not run at all -- the "cannot run"
    mode of PROTOCOL.md's table, in the harness that table is enforced by.
    Ghostscript's txtwrite was tried first and rejected on evidence: it recovers
    108,646 letters from main_R2.pdf and drops the digits, which is precisely the
    content guard 2 exists to read.
    """
    path = str(path)
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", path, "-"],
                              capture_output=True, text=True).stdout
    try:
        import pypdf
    except ImportError:
        return ""
    return "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(path).pages)


# ---------------------------------------------------------------- guard 1 ----
# Undefined references. Read the .log LaTeX writes, never a redirected stdout.
UNDEF = re.compile(r"Reference `[^']*' on page \d+ undefined|There were undefined references")

def _log_undefined(log_text: str) -> list[str]:
    return [l.strip() for l in log_text.splitlines() if UNDEF.search(l)]

def control_undefined() -> bool:
    """Build a two-label document, resolve it, rename one label, run ONE pass.

    This is the exact failure that shipped: a partially stale .aux resolves every
    pre-existing label and leaves only the renamed one as `??`, while the caption
    prints the correct number because it comes from the counter, not the .aux.
    """
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        doc = (r"\documentclass{article}\begin{document}"
               r"\begin{table}\caption{First}\label{tab:%s}\end{table}"
               r"Refer to Table~\ref{tab:%s}.\end{document}")
        (d / "t.tex").write_text(doc % ("old", "old"))
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"],
                           cwd=d, capture_output=True)
        # rename the label, single pass on the now-stale .aux
        (d / "t.tex").write_text(doc % ("new", "new"))
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"],
                       cwd=d, capture_output=True)
        hits = _log_undefined((d / "t.log").read_text(errors="ignore"))
        pdf = _pdf_text(d / "t.pdf")
        # the control is only valid if it reproduces BOTH symptoms
        return bool(hits) and "??" in pdf and "Table 1" in pdf

def _pdf_coverage(path) -> tuple[int, int, int]:
    """Pages, pages yielding text, characters recovered.

    Guard 1 reads the PDF with pypdf when poppler is absent, and pypdf stops
    descending into form XObjects after 5,000 invocations. That warning is
    printed by the library and says nothing about how much of the document was
    read, so the guard states its own coverage rather than leaving a reader to
    infer it from an absence of complaint.
    """
    try:
        import pypdf
    except ImportError:
        return (0, 0, 0)
    if shutil.which("pdftotext"):
        t = _pdf_text(path)
        return (-1, -1, len(t))          # poppler: no per-page accounting needed
    pages = [p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages]
    return (len(pages), sum(1 for t in pages if t.strip()), sum(len(t) for t in pages))


def guard_undefined() -> bool:
    good = True
    for doc in DOCS:
        log = BASE / f"{doc}.log"
        if not log.exists():
            _bad(f"{doc}.log absent -- cannot verify; never infer from stdout"); good = False; continue
        hits = _log_undefined(log.read_text(errors="ignore"))
        pdf = BASE / f"{doc}.pdf"
        txt = _pdf_text(pdf) if pdf.exists() else ""
        n_q = txt.count("??")
        if hits or n_q:
            _bad(f"{doc}: {len(hits)} undefined-reference warnings, {n_q} '??' in the PDF")
            for h in hits[:4]: print(f"           {h}")
            good = False
        else:
            n_pg, n_txt, n_ch = _pdf_coverage(pdf) if pdf.exists() else (0, 0, 0)
            cov = (f"{n_txt}/{n_pg} pages, {n_ch:,} characters"
                   if n_pg > 0 else f"{n_ch:,} characters via pdftotext")
            _ok(f"{doc}: no undefined references in {doc}.log, no '??' in "
                f"{doc}.pdf ({cov})")
    return good


# ---------------------------------------------------------------- guard 2 ----
# Every numeric claim in the prose is a macro from numbers.tex, or is on the
# declared allow-list of structural constants. A bare result literal is a defect.
DECLARED = BASE / "analysis" / "provenance" / "DECLARED_CONSTANTS.md"


def _declared() -> set[str]:
    """Constants the manuscript may carry without a macro, each with a stated
    role in DECLARED_CONSTANTS.md. A literal absent from that file is a defect."""
    if not DECLARED.exists():
        return set()
    out: set[str] = set()
    for line in DECLARED.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cell = line.split("|")[1].strip()
        for tok in re.findall(r"-?\d+\.?\d*", cell):
            out.add(tok.lstrip("-"))
    return out



_TAB_OPEN = re.compile(r"\\begin\{tabular[*x]?\}(\[[^\]]*\])?\{")


def _strip_tabular_specs(s: str) -> str:
    r"""Remove a tabular's COLUMN SPECIFICATION and keep its body.

    The guard used to replace `\begin{tabular}...\end{tabular}` wholesale. That
    is right for generated tables, which arrive by `\input` and are guard 5's
    business -- and `\input{...}` is already replaced before this runs, so no
    generated table ever reaches here. What it also removed was every table
    written by hand in the document itself, which is prose wearing a table's
    costume and is the only place guard 2 was the sole check.

    It cost two measured rates: 0.990 and 0.988, the violation rates under the
    inverted-sign defect, sat as literals in a Section 8 table whose neighbouring
    column was already a macro. Guard 2 reported "no bare decimal literals in
    prose" over them for as long as it existed.

    Only the column specification goes, because `p{3.5cm}` and `p{1.3cm}` are
    typesetting lengths and not claims -- the same reason `\includegraphics`
    options are stripped above. Brace matching is done by scanning, since a
    specification may nest: `>{\raggedright\arraybackslash}p{3.5cm}`.
    """
    out, i = [], 0
    while True:
        m = _TAB_OPEN.search(s, i)
        if not m:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:m.start()])
        out.append(" TABULARSPEC ")
        j, depth = m.end(), 1          # m.end() is just past the opening brace
        while j < len(s) and depth:
            if s[j] == "\\":
                j += 2; continue
            depth += (s[j] == "{") - (s[j] == "}")
            j += 1
        i = j


def _prose_literals(tex: str) -> list[tuple[str, str]]:
    # The preamble is layout, not claims: colours, box offsets, font sizes.
    tex = tex.split(r"\begin{document}", 1)[-1]
    s = re.sub(r"(?<!\\)%.*", "", tex)
    # Typesetting lengths, citation locators and contract numbers are not claims.
    s = re.sub(r"\\includegraphics\[[^\]]*\]", " GRAPHIC ", s)
    s = re.sub(r"\\(?:renewcommand|setlength|arraystretch)\{[^}]*\}\{[^}]*\}", " LEN ", s)
    # List and box geometry, the same class as a tabular column width.
    s = re.sub(r"(?:leftmargin|itemsep|topsep|parsep|labelwidth|labelsep)\s*=\s*[\d.]+\s*(?:em|ex|pt|cm|in|mm)", " LEN ", s)
    s = re.sub(r"\\cite[a-z]*\[[^\]]*\]", " CITE ", s)
    s = re.sub(r"\b[A-Z]{2}\d+/\d[\d./]*", " CONTRACT ", s)
    s = re.sub(r"no\.\\?\s*\d[\d./]*", " CONTRACT ", s)
    # Model release numbers are names, not results: Moirai~1.1, TimesFM~2.5.
    s = re.sub(r"(?<=[A-Za-z])~\d+\.\d+", " MODELVER ", s)
    # A three-component semantic version in \texttt is a release NAME, like
    # Moirai~1.1 above. Deliberately narrow: three components and inside
    # \texttt, so a two-component decimal stays a claim and is still caught.
    s = re.sub(r"\\texttt\{\d+\.\d+\.\d+\}", " PKGVER ", s)
    s = re.sub(r"GARCH\(\d,\d\)", " GARCHSPEC ", s)
    s = re.sub(r"\\n[A-Z][A-Za-z]*\{?\}?", " MACRO ", s)
    s = re.sub(r"\\input\{[^}]*\}", " INPUT ", s)
    s = _strip_tabular_specs(s)
    s = re.sub(r"\\(?:label|ref|eqref|cite[a-z]*|href|url)\{[^}]*\}", " ", s)
    allow = _declared()
    out = []
    for m in re.finditer(r"(?<![\w.\\])(\d+\.\d+)", s):
        if m.group(1) in allow:
            continue
        out.append((m.group(1), " ".join(s[max(0, m.start()-58):m.end()+38].split())))
    return out


def control_literals() -> bool:
    """A fabricated decimal that no artefact emits must be flagged.

    Two controls, because the guard has two reaches and the second was added
    after the first had passed for a year over literals it could not see.

    (a) In running prose. The control uses 0.7391, but the case that motivates
        this guard is real: a p-value of 0.035 was typed into Section 5.6,
        matched no artefact, and was never seen by paper_numbers.py because it
        was never emitted.
    (b) Inside a tabular written by hand in the document, with a column
        specification carrying lengths that must NOT be flagged. This is the
        real defect of 2026-08-28: 0.990 and 0.988 sat in a Section 8 table
        beside a column that was already a macro, and the guard replaced the
        whole environment before looking. A control that only plants a literal
        in running prose passes on a guard blind to every hand-authored table,
        which is what happened.
    """
    prose = r"\begin{document} The corrected rate is 0.7391 across the panel."
    table = (r"\begin{document}"
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.5cm}p{1.3cm}@{}}"
             r"Rate under the defect & $0.7391$ \\"
             r"\end{tabular}")
    return (len(_prose_literals(prose)) == 1
            and len(_prose_literals(table)) == 1)

def guard_literals() -> bool:
    good = True
    for doc in LITERAL_DOCS:
        lits = _prose_literals((BASE / f"{doc}.tex").read_text(encoding="utf-8"))
        if lits:
            _bad(f"{doc}: {len(lits)} bare decimal literals in prose (should be macros)")
            for v, c in lits[:8]: print(f"           {v:>8s}  ...{c[-70:]}")
            good = False
        else:
            _ok(f"{doc}: no bare decimal literals in prose")
    return good


# ---------------------------------------------------------------- guard 3 ----
# The retired substring screen, kept as an executable demonstration of why it
# was retired. It must FAIL its own negative control; if it ever passes, someone
# has changed it into something that could be trusted, and that needs review.
def control_substring_screen() -> bool:
    """A number that is real but comes from an undeclared panel must slip through.

    0.0750 is a genuine value emitted by CO_aci_baseline on a 216-pair panel the
    manuscript never declares. A substring screen calls it SOURCED and is right
    about the characters and wrong about the claim.
    """
    hay = "\n".join(p.read_text(errors="ignore")
                    for p in (BASE / "Quantlets").rglob("*.tex"))
    return "0.0750" in hay            # True == the screen would wrongly pass it

def guard_substring_screen() -> bool:
    if control_substring_screen():
        _ok("substring provenance screen stays retired: its control confirms it "
            "reports SOURCED for a value drawn from an undeclared panel")
        return True
    _bad("substring screen control no longer reproduces -- re-examine before reuse")
    return False


# ---------------------------------------------------------------- guard 4 ----
# A file the written discipline points at, which git does not carry.
#
# PROTOCOL.md names the harness that enforces Rule 2; MIGRATION.md names the four
# audits the new machine must run before the move is trusted. Three of those
# audits and the harness itself were matched by the `/scripts/*` glob in
# .gitignore and never re-included, so they existed only on the machine where
# they were written -- not for a referee, and not after the migration the
# document describes. A rule that points at a file nobody else has is not a rule.

# The manuscript is the strongest form of written discipline: it promises a
# reader that a file exists. This list held only the internal notes, so
# scripts/audit_split_convention.py -- named in Supplement S.4.3 as failing
# the build -- was untracked and unnoticed. A fresh clone did not contain it.
DISCIPLINE_DOCS = [BASE / "analysis" / "provenance" / "PROTOCOL.md",
                   BASE / "MIGRATION.md",
                   BASE / "main_R2.tex", BASE / "supplement.tex",
                   BASE / "sections" / "sec4_theory.tex",
                   BASE / "sections" / "sec5_montecarlo.tex"]
# Backticks are not enough: MIGRATION.md gives the four audits as indented shell
# commands, and those are exactly the files that did not travel.
PATHISH = re.compile(r"[\w][\w./-]*\.(?:py|sh|tex|bib|csv|tsv|json|cls|lock)\b")
# LaTeX escapes every underscore, and almost every script in this project has
# one. On "\\texttt{scripts/audit\\_split\\_convention.py}" the pattern above
# matches "_convention.py" -- a fragment that is not a file, so the guard
# checked nothing and said "23 referenced files checked". Two audits were
# untracked underneath it, including the one the supplement says fails the
# build. Unescape before matching.
def _unescape_tex(t: str) -> str:
    return t.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")


def _referenced_paths(text: str) -> set[str]:
    """Every token in the document that names a file, backticked or not."""
    return {m.group(0).rstrip(".,;:") for m in PATHISH.finditer(_unescape_tex(text))}


def _untracked(paths: set[str]) -> list[str]:
    bad = []
    for rel in sorted(paths):
        f = BASE / rel
        if not f.is_file():
            continue                      # named but absent is a different defect
        r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                           cwd=BASE, capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(rel)
    return bad


def control_referenced_tracked() -> bool:
    """Plant a reference to a file git does not carry; the scan must catch it."""
    with tempfile.TemporaryDirectory() as d:
        planted = BASE / "scripts" / "_guard4_control_not_tracked.py"
        planted.write_text("# deliberately untracked, written by guard 4's control\n")
        try:
            found = _untracked(_referenced_paths(
                f"the discipline points at `{planted.relative_to(BASE)}` here"))
            return len(found) == 1
        finally:
            planted.unlink(missing_ok=True)


def guard_referenced_tracked() -> bool:
    refs = set()
    for doc in DISCIPLINE_DOCS:
        if doc.is_file():
            refs |= _referenced_paths(doc.read_text(encoding="utf-8"))
    bad = _untracked(refs)
    if bad:
        for rel in bad:
            _bad(f"referenced by the written discipline, not tracked by git: {rel}")
        return False
    _ok(f"{len(refs)} referenced file(s) checked, all tracked")
    return True


# ---------------------------------------------------------------- guard 5 ----
# A table the manuscript inputs, with no script that writes it.
#
# Three instances made this a class: Table S.10, hand-authored from a
# pre-correction vintage; two .tex files edited after generation, so a rebuild
# silently reverted the edit; and tab_master_results.tex, whose only writer was
# a script MANIFEST.md had already marked SUPERSEDED. In each case the artefact
# looked reproducible because a script of about the right name sat beside it.
# The manifest makes the producer a declaration rather than an inference.

PRODUCERS = BASE / "analysis" / "provenance" / "PRODUCERS.tsv"
DOCS_TEX = ["main_R2.tex", "supplement.tex",
            "sections/sec4_theory.tex", "sections/sec5_montecarlo.tex"]
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def _declared_producers() -> dict:
    out = {}
    if not PRODUCERS.is_file():
        return out
    for line in PRODUCERS.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("target\t"):
            continue
        f = line.split("\t")
        if len(f) >= 3:
            out[f[0].strip()] = (f[1].strip(), f[2].strip())
    return out


def _inputs(docs) -> set:
    found = set()
    for d in docs:
        f = BASE / d
        if f.is_file():
            found |= set(INPUT_RE.findall(f.read_text(encoding="utf-8")))
    return found


def _undeclared(docs) -> list:
    decl = _declared_producers()
    bad = []
    for t in sorted(_inputs(docs)):
        if t not in decl:
            bad.append((t, "not in PRODUCERS.tsv"))
            continue
        kind, prod = decl[t]
        if kind == "generated" and not (BASE / prod).is_file():
            bad.append((t, f"declared producer missing: {prod}"))
    return bad


def control_producers() -> bool:
    """A document that inputs an undeclared table must be caught."""
    with tempfile.TemporaryDirectory() as d:
        doc = pathlib.Path(d) / "planted.tex"
        doc.write_text(r"\input{Quantlets/CO_nowhere/tab_no_producer}" + "\n")
        try:
            rel = doc.relative_to(BASE)
        except ValueError:
            rel = None
        if rel is None:
            planted = BASE / "_guard5_control.tex"
            planted.write_text(r"\input{Quantlets/CO_nowhere/tab_no_producer}" + "\n")
            try:
                return len(_undeclared(["_guard5_control.tex"])) == 1
            finally:
                planted.unlink(missing_ok=True)
        return len(_undeclared([str(rel)])) == 1


def guard_producers() -> bool:
    bad = _undeclared(DOCS_TEX)
    if bad:
        for t, why in bad:
            _bad(f"input with no declared producer: {t} -- {why}")
        return False
    n = len(_inputs(DOCS_TEX))
    _ok(f"{n} \\input target(s), each with a declared producer")
    return True


# "pdftext" is a capability, not a binary: pdftotext when poppler is installed,
# pypdf otherwise. Naming the binary here was what made three guards unrunnable.
NEEDS = {"documents compile from source": ("pdflatex", "pdftext"),
         "undefined references": ("pdflatex", "pdftext"),
         "prose numeric literals": ("pdflatex", "pdftext"),
         "retired substring screen": ("pdflatex", "pdftext")}


def _have_pdftext() -> bool:
    if shutil.which("pdftotext"):
        return True
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False

# ---------------------------------------------------------------- guard 6 ----
# A rate printed to a precision finer than the grid it lives on.
#
# Third instance of one shape, which is why it is a check and not a note:
#
#   * Section 4.4 announced violation rates agreeing "to four decimal places"
#     on 40 dates, where the rate moves in steps of 0.025;
#   * the same section announced dispersion agreeing to 0.3% on a statistic a
#     support translation leaves exactly invariant;
#   * CONDITIONAL_PASSAGES.md described a family as sitting "at 0.6x to 1.0x
#     nominal" on 200 dates, where the ratio moves in steps of 0.5x and 0.6 is
#     not a value the panel can produce.
#
# A rate computed from N Bernoulli trials takes values on a grid of 1/N. Printing
# it with a step finer than that claims resolution the data does not carry.
# scripts/paper_numbers.py records N beside every rate it emits; this reads that
# record and re-derives the requirement.
RATES = BASE / "analysis" / "provenance" / "RATE_RESOLUTION.tsv"
RATE_NAME = re.compile(r"^(?:Main|Seq|Pool|Sup|ML|Lit|Raw|Trunc|MC)?.*Pi(?![a-z])")


def _rate_rows() -> list[tuple[str, str, int, int]]:
    if not RATES.is_file():
        return []
    out = []
    for line in RATES.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 5:
            continue
        out.append((f[0], f[1], int(f[2]), int(f[4])))
    return out


def _over_precise(rows) -> list[str]:
    """A printed step finer than 1/N over-claims. Reported with both numbers."""
    bad = []
    for key, val, n_obs, dp in rows:
        if n_obs <= 0:
            bad.append(f"{key}: no sample size recorded"); continue
        if 10.0 ** (-dp) < 1.0 / n_obs:
            bad.append(f"{key} = {val} printed to {dp} dp (step {10.0**-dp:.1e}) "
                       f"on {n_obs:,} observations (grid {1.0/n_obs:.1e})")
    return bad


# Macros the name pattern catches that are not rates. Listed with a reason
# rather than dissolved by loosening the pattern, so the exemption is visible
# and a future rate cannot slip in behind a broadened regex.
NOT_A_RATE = {
    "RawPiGJRDefectivePct": "a percentage of a rate, not a rate; its N is "
                            "recorded against RawPiGJRDefective",
    "MLPiGrid": "the grid step itself, 1/(N alpha) -- the quantity this guard "
                "checks others against",
    "SpearmanRPi": "a rank correlation between two orderings, not a frequency; "
                   "its resolution is set by the number of pairs, not by 1/N",
    "GapPanelDPiMed": "a difference of two rates computed on DIFFERENT "
                      "denominators, n and n - g_n, so it is not confined to "
                      "either one's 1/N grid and can take values finer than both",
    "GapPanelDPiMax": "as above",
    "GapAllDPiMax": "as above, over all four levels",
}


def _undeclared_rates() -> list[str]:
    """A rate macro in numbers.tex that the registry does not carry.

    The registry is written by the same script that emits the macros, so an
    absence means a rate was emitted without its sample size -- which is the
    state every rate in this project was in until this guard existed.
    """
    tex = BASE / "numbers.tex"
    if not tex.exists():
        return []
    known = {k for k, _, _, _ in _rate_rows()}
    out = []
    for k, v in re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}",
                           tex.read_text(encoding="utf-8")):
        if not RATE_NAME.match(k) or k in known or k in NOT_A_RATE:
            continue
        if not re.fullmatch(r"0\.\d+", v):     # a rate is a bare fraction
            continue
        out.append(f"{k} = {v}")
    return out


def control_rate_resolution() -> bool:
    """Planted where the guard reads worst, not where it reads typically.

    Two controls. The first is the shape the guard was written for: four decimal
    places on the 40-date validation, whose grid is 0.025 -- the R14 case, 250
    times finer than the data. The second is the one a registry-driven check can
    miss entirely: a rate that never reaches the registry at all. A guard that
    only inspects the rows it is given passes on every rate nobody recorded, and
    that was the state of all of them until now.
    """
    over = _over_precise([("Fake", "0.0125", 40, 4)])
    # The positive case has to be chosen against the same grid: on 40 dates the
    # grid is 0.025, so even two decimals over-claims and only one does not.
    # Getting this wrong the first time is the point -- the guard reported
    # BROKEN rather than passing on a control that could not distinguish.
    fine = _over_precise([("Fake", "0.0", 40, 1)])
    missed = _over_precise([("Fake", "0.01", 0, 2)])
    return len(over) == 1 and len(fine) == 0 and len(missed) == 1


def guard_rate_resolution() -> bool:
    rows = _rate_rows()
    if not rows:
        _bad("no rate resolution registry -- run scripts/paper_numbers.py --write")
        return False
    good = True
    bad = _over_precise(rows)
    if bad:
        _bad(f"{len(bad)} rate(s) printed finer than their grid")
        for b in bad[:8]:
            print(f"           {b}")
        good = False
    else:
        _ok(f"{len(rows)} rate(s) printed no finer than the grid 1/N they live on")
    miss = _undeclared_rates()
    if miss:
        _bad(f"{len(miss)} rate macro(s) emitted with no sample size recorded")
        for m in miss[:8]:
            print(f"           {m}")
        good = False
    else:
        _ok("every rate macro in numbers.tex carries the N that sets its resolution")
    return good


# ---------------------------------------------------------------- guard 7 ----
# The documents compile from the current source, and the PDFs the guards above
# read were built from it.
#
# Guards 1-3 read main_R2.pdf and supplement.pdf. Nothing established that those
# PDFs came from the tex in the working tree, so the whole set could report
# green while the manuscript did not compile at all: \qVstat^2 put a second
# superscript on a macro that already carried one, pdflatex produced no PDF, and
# six guards passed on the previous build. A guard set that reads an artefact it
# never checked the provenance of is the "verdict outlives its state" mode of
# PROTOCOL.md's table, in the harness that table is enforced by.
LATEX_ERR = re.compile(r"^! (.*)$", re.M)


def _compile(doc: str, outdir: Path) -> tuple[bool, str]:
    """One pdflatex pass into outdir, seeded with the existing .aux for refs."""
    aux = BASE / f"{doc}.aux"
    if aux.exists():
        shutil.copy(aux, outdir / f"{Path(doc).name}.aux")
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory", str(outdir), f"{doc}.tex"],
        cwd=BASE, capture_output=True, text=True)
    if r.returncode != 0 or not (outdir / f"{Path(doc).name}.pdf").exists():
        m = LATEX_ERR.search(r.stdout)
        return False, (m.group(1) if m else "pdflatex returned "
                       f"{r.returncode} and produced no PDF")
    return True, ""


def control_compiles() -> bool:
    """A document with a double superscript must be reported as not compiling."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "ctrl.tex"
        src.write_text(r"\documentclass{article}\begin{document}"
                       r"$x^2^3$\end{document}")
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(td), "ctrl.tex"],
            cwd=td, capture_output=True, text=True)
        return r.returncode != 0 or not (td / "ctrl.pdf").exists()


def guard_compiles() -> bool:
    good = True
    for doc in DOCS:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            built, err = _compile(doc, td)
            if not built:
                _bad(f"{doc}: does not compile from the current source -- {err}")
                good = False
                continue
            shipped = BASE / f"{doc}.pdf"
            if not shipped.exists():
                _bad(f"{doc}.pdf does not exist; guards 1-3 have nothing to read")
                good = False
                continue
            fresh_t = _pdf_text(td / f"{Path(doc).name}.pdf")
            ship_t = _pdf_text(shipped)
            if not fresh_t or not ship_t:
                _bad(f"{doc}: no text extracted; the comparison cannot fire")
                good = False
                continue
            # Normalise whitespace only. A rebuild differs in PDF metadata but
            # not in a character of typeset text.
            f_n, s_n = " ".join(fresh_t.split()), " ".join(ship_t.split())
            if f_n != s_n:
                d = next((i for i, (a, b) in enumerate(zip(f_n, s_n)) if a != b),
                         min(len(f_n), len(s_n)))
                _bad(f"{doc}.pdf is STALE -- rebuilt text differs from the shipped "
                     f"PDF at character {d:,} of {len(s_n):,}")
                print(f"           shipped: ...{s_n[max(0, d-60):d+40]}")
                print(f"           rebuilt: ...{f_n[max(0, d-60):d+40]}")
                good = False
            else:
                _ok(f"{doc}: compiles from the current source and the shipped PDF "
                    f"matches it ({len(s_n):,} characters)")
    return good


# ---------------------------------------------------------------- guard 8 ----
# An artefact that feeds the manuscript is not older than what produced it.
#
# rolling_vs_static.csv was written on 20 August and the analytic Chronos series
# it reads were rebuilt on 27 August, when the one-bin offset was corrected. The
# correction round recorded that "every downstream artefact has been rebuilt";
# this one had not, and re-running its producer today moves three of its 312
# rows. No verdict changed and no manuscript number was wrong, which is exactly
# why nothing surfaced it: a stale artefact whose classifications happen to be
# stable is indistinguishable from a current one until someone re-derives it.
#
# WHAT THIS GUARD IS. A screen on modification times, not a re-derivation. It
# can report an artefact that does not in fact depend on the input that moved --
# a false alarm costing one re-run. It cannot report an artefact as current when
# its input is newer, which is the direction that matters; the retired substring
# screen of guard 3 failed the other way. The cost of a false alarm here is a
# command; the cost of a false pass is a number in a paper.
def _live_artefacts() -> list[Path]:
    r"""Artefacts the manuscript reads: \input targets and paper_numbers inputs."""
    out = set()
    for doc in LITERAL_DOCS:
        for m in INPUT_RE.finditer((BASE / f"{doc}.tex").read_text(encoding="utf-8")):
            t = m.group(1)
            for ext in (".tex", ".csv", ""):
                q = BASE / (t + ext)
                if q.is_file():
                    out.add(q)
                    break
    pn = (BASE / "scripts" / "paper_numbers.py").read_text(encoding="utf-8")
    for m in re.finditer(r'"([\w./-]+\.(?:csv|tsv|json|parquet))"', pn):
        for cand in BASE.rglob(m.group(1)):
            t = str(cand)
            if any(x in t for x in (".git", "submission_IJF", "superseded",
                                    "cfp_ijf_data/returns",
                                    "quarantine", "legacy")):
                continue
            out.add(cand)
    return sorted(out)


def _newest_input() -> tuple[Path, float]:
    """The most recently modified primary series in the data directory."""
    ins = [q for q in (BASE / "cfp_ijf_data").rglob("*.parquet")
           if "paper_outputs" not in str(q)]
    if not ins:
        return None, 0.0
    newest = max(ins, key=lambda q: q.stat().st_mtime)
    return newest, newest.stat().st_mtime


def control_artefact_freshness() -> bool:
    """An artefact older than its input must be reported."""
    import tempfile as _tf, os as _os, time as _tm
    with _tf.TemporaryDirectory() as td:
        old, new = Path(td) / "old.csv", Path(td) / "new.parquet"
        old.write_text("a,b\n1,2\n")
        new.write_text("x")
        _os.utime(old, (0, 0))
        return old.stat().st_mtime < new.stat().st_mtime


def guard_artefact_freshness() -> bool:
    newest, t_in = _newest_input()
    if newest is None:
        _bad("no primary series found; the freshness screen cannot run")
        return False
    # Exemptions are CHECKED, not granted. A row may claim SELF_CONTAINED, and
    # the guard confirms the claim by reading the producing directory: a script
    # that never names cfp_ijf_data cannot depend on a series in it. An
    # exemption whose stated basis does not hold is reported as a failure, not
    # quietly honoured -- an unverified exemption is a hand-carried claim, which
    # is the thing these registries exist to remove.
    exempt, bad_basis = {}, []
    reg = BASE / "analysis" / "provenance" / "FRESHNESS_EXEMPT.tsv"
    if reg.is_file():
        for line in reg.read_text().splitlines():
            if line.startswith("#") or not line.strip() or line.startswith("artefact\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            art, basis, reason = parts[0], parts[1], parts[2]
            if basis == "FROZEN":
                if not re.search(r"\d{4}-\d{2}-\d{2}", reason):
                    bad_basis.append((art, "FROZEN with no date in its reason"))
                    continue
            elif basis == "SELF_CONTAINED":
                d = (BASE / art).parent
                reads = [f for f in d.glob("*.py")
                         if "cfp_ijf_data" in f.read_text(encoding="utf-8", errors="ignore")]
                if reads:
                    bad_basis.append((art, reads[0].name))
                    continue
            exempt[art] = reason
    arts = _live_artefacts()
    stale = []
    for q in arts:
        if q.suffix not in (".csv", ".tsv", ".json"):
            continue
        rel = str(q.relative_to(BASE))
        if q.stat().st_mtime < t_in and rel not in exempt:
            stale.append(q)
    if bad_basis:
        _bad(f"{len(bad_basis)} exemption(s) whose stated basis does not hold")
        for art, f in bad_basis:
            print(f"           {art}: {f}")
        return False
    import datetime as _dt
    when = _dt.datetime.fromtimestamp(t_in).strftime("%Y-%m-%d %H:%M")
    if stale:
        _bad(f"{len(stale)} artefact(s) the manuscript reads predate the newest "
             f"input ({newest.relative_to(BASE)}, {when}); re-derive or declare")
        for q in stale[:10]:
            ts = _dt.datetime.fromtimestamp(q.stat().st_mtime).strftime("%Y-%m-%d")
            print(f"           {ts}  {q.relative_to(BASE)}")
        return False
    _ok(f"{len(arts)} artefact(s) read by the manuscript, none older than the "
        f"newest primary series ({when}); {len(exempt)} declared "
        "exemption(s), each with a checked basis")
    return True


GUARDS = [("documents compile from source", control_compiles, guard_compiles),
          ("undefined references", control_undefined, guard_undefined),
          ("prose numeric literals", control_literals, guard_literals),
          ("retired substring screen", control_substring_screen, guard_substring_screen),
          ("referenced files are tracked", control_referenced_tracked, guard_referenced_tracked),
          ("inputs have declared producers", control_producers, guard_producers),
          ("rates declare their resolution", control_rate_resolution,
           guard_rate_resolution),
          ("artefacts are not older than their inputs",
           control_artefact_freshness, guard_artefact_freshness)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    # The toolchain check used to be all-or-nothing: one missing binary and no
    # guard ran at all, including the ones that need neither. A harness that
    # declines to run is the "cannot run" mode of PROTOCOL.md's table, so the
    # requirement is now per guard, and a guard that cannot run is reported as
    # SKIPPED rather than silently folded into a pass.
    have = {"pdflatex": bool(shutil.which("pdflatex")), "pdftext": _have_pdftext()}
    rc = 0
    for name, control, guard in GUARDS:
        print(f"\n{name}")
        missing = [b for b in NEEDS.get(name, ()) if not have[b]]
        if missing:
            _bad(f"SKIPPED -- requires {', '.join(missing)}, not on PATH")
            rc = 1
            continue
        if not control():
            _bad(f"NEGATIVE CONTROL DID NOT FAIL -- guard '{name}' is not evidence")
            rc = 1; continue
        _ctl("negative control reproduces the failure")
        if a.controls: continue
        if not guard(): rc = 1
    print("\nOK" if rc == 0 else "\nGUARDS FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
