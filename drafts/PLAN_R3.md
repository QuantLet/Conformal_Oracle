# R3: one document, forty pages

R2 is 66 body pages and 22,125 prose words, with a 73-page supplement carrying
another 14,577. R3 is a single document of roughly forty pages, which is about
11,000 words of prose plus the floats that have to travel with it.

That is a selection, not a compression. Roughly seven words in ten leave the
submission. They do not leave the work: the replication repository carries all of
it, and the paper points there.

## The rule that decides each cut

A passage stays if a referee needs it to judge the claims. It goes if it is
evidence *for* the authors that the work was done properly. The distinction is
sharp in this project because so much of the supplement is the second kind --
provenance, harness design, defect census, convention registries. That material
earns its place in `analysis/provenance/`, not in a journal article.

Three consequences, stated in advance so the cutting is not arbitrary:

1. **Proofs go to the repository.** Theorem 4.5 is now attributed as a one-sided
   specialization of Oliveira et al. (2024, Theorem 4); a specialization of a
   published result does not need eight pages of proof in a finance journal. The
   statement, the assumptions and a proof sketch stay.
2. **Every robustness exercise becomes one sentence with a pointer**, unless a
   referee's judgement of a headline claim turns on it.
3. **The verification apparatus leaves the paper entirely**, except for the
   single sentence in the data-availability section that says the numbers are
   emitted from artefacts and the build fails on a mismatch.

## Budget

| section | R2 | R3 | disposition |
|---|---:|---:|---|
| Introduction | 2,274 | 1,200 | tighten; the contribution list shortens because the theorem is now attributed |
| Related literature | 1,243 | 800 | keep; it now carries the 2026 one-sided conformal VaR work |
| Methodology | 3,481 | 2,000 | keep the estimator and the one-sided argument; Remarks 3.1--3.4 reduce to two |
| Coverage under dependence | 3,227 | 1,400 | assumptions, the inherited result with attribution, the GARCH corollary, the measured cost. Rolling and drift bounds become a paragraph |
| Monte Carlo | 2,072 | 500 | one table, one paragraph |
| The panel and the correction | 1,127 | 900 | keep |
| What the diagnostics cannot diagnose | 1,919 | 1,600 | keep: this is the identification result and it survives the priority check |
| What recalibration restores | 1,071 | 800 | keep |
| What it costs, and when to apply it | 1,974 | 1,300 | keep the rule; the window sweep becomes a pointer |
| Limitations | 1,939 | 500 | method limits only |
| Conclusion | 890 | 500 | |
| Data and code | 533 | 400 | gains the pointer sentences the cuts create |
| **body total** | **21,750** | **11,900** | |

Supplement: dropped as a document. Sections S.1--S.18 are already reproducible
from the repository; the ones a referee is most likely to ask for -- the proofs,
the validation gate, the truncation experiment -- get a named file path in the
text rather than a section number.

## What must be watched while cutting

- Every `\ref` to a supplement section becomes a repository path. `audit_supplement_targets.py`
  will fail loudly until each one is converted, which is the intended behaviour.
- Guard 4 reads the manuscript for referenced files, so each new path must be a
  tracked file or the build fails.
- `numbers.tex` is unchanged: cutting prose must not change a single figure, and
  `paper_numbers.py --check` staying green is the evidence for that.
