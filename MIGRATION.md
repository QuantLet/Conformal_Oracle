# Moving this project to another machine

Verified on the source machine, 27 August 2026. Sizes are measured, not estimated.

## What git carries, and what it does not

`git` holds **1,024 tracked files, 6.6 MiB packed** — the manuscript, the
supplement, `sections/`, `scripts/`, `Quantlets/`, `analysis/`, `python/`,
`pipeline/`, and 26 metadata files under `cfp_ijf_data/`. Everything below is
**outside** git and has to travel separately or be rebuilt.

| item | size | tracked | how to restore |
|---|---|---|---|
| `cfp_ijf_data/` (forecast parquets, returns) | **133 MB** | 26 of ~330 files | **copy it.** Not reproducible without GPU inference. |
| `figures/` | 108 MB | 2 of ~40 | `/figures/` is in `.gitignore`. Regenerable, but see the warning below. |
| `.venv_forecasts/` | 1.2 GB | no | rebuild from `requirements-lock.txt` |
| `~/.cache/huggingface` | 12 GB | no | re-downloads on first use; needed only to re-run model inference |
| `cfp_ijf_data.zip` | 105 MB | no | a snapshot of `cfp_ijf_data/`; copy either, not both |

**The figures warning.** `.gitignore` line 94 excludes `/figures/`, so
`figures/fig_mc_convergence.{pdf,png}` — which Section 5 of the manuscript
`\includegraphics`es — is not in the repository. A bare clone will not build
`main_R2.tex`. It is regenerable with

    .venv_forecasts/bin/python analysis/k2_sim/make_figure.py

but the same is not true of every figure in that directory, and the ones without
a producer are the ones to copy.

## The environment does not match `requirements.txt`

`requirements.txt` pins nine packages. Five of the nine pins do not match what is
installed, and the file omits everything needed for model inference:

| package | `requirements.txt` | installed |
|---|---|---|
| pandas | 2.3.3 | **3.0.1** |
| numpy | 2.3.5 | **2.4.3** |
| scipy | 1.16.3 | **1.17.1** |
| matplotlib | 3.10.8 | **3.11.1** |
| pyarrow | 21.0.0 | **23.0.1** |
| lightgbm | 4.6.0 | **4.7.0** |
| torch, chronos-forecasting, transformers, accelerate, einops, huggingface-hub | absent | 2.10.0, 2.2.2, 4.57.6, 1.13.0, 0.8.2, 0.36.2 |

55 installed packages are undeclared. **Every number in the manuscript was
computed under the installed set, not the declared one.** Pinning to
`requirements.txt` on the new machine would rebuild a different environment from
the one the results came from.

`requirements-lock.txt` is a `pip freeze` of the live environment, added for this
reason. Use it to rebuild; keep `requirements.txt` as the loose declaration it is,
or replace it.

## Steps

    # on the new machine
    git clone <remote-or-bundle> "2026 CFP LLM VaR" && cd "2026 CFP LLM VaR"
    rsync -a  <old>/cfp_ijf_data/  cfp_ijf_data/
    rsync -a  <old>/figures/       figures/
    python3.13 -m venv .venv_forecasts
    .venv_forecasts/bin/python -m pip install -r requirements-lock.txt

If there is no git remote, move the history with a bundle rather than by copying
`.git/` (162 MB):

    git bundle create /tmp/cfp.bundle --all      # on the old machine
    git clone /tmp/cfp.bundle "2026 CFP LLM VaR" # on the new one

## Verify the move before trusting it

Run all four audits and both builds. Every one of these passes on the source
machine as of this commit:

    .venv_forecasts/bin/python scripts/paper_numbers.py --check
    .venv_forecasts/bin/python scripts/audit_structural_claims.py
    .venv_forecasts/bin/python scripts/audit_qv_convention.py
    .venv_forecasts/bin/python scripts/audit_supplement_targets.py
    pdflatex main_R2.tex && bibtex main_R2 && pdflatex main_R2.tex && pdflatex main_R2.tex
    pdflatex supplement.tex && bibtex supplement && pdflatex supplement.tex

`paper_numbers.py --check` is the one that matters: it recomputes every asserted
figure from the artefacts and fails if any has drifted. If it passes on the new
machine, `cfp_ijf_data/` arrived intact.

## What needs the GPU, and what does not

Only re-running model inference needs one. The Chronos verification in
`analysis/k1_verify/` ran on CPU here at roughly ten seconds per 1,000 draws;
regenerating both analytic panels (121,923 dates per checkpoint) is the job that
wants the faster machine, and it is the outstanding item — the analytic series
carry the one-bin offset of R14 and have not yet been rebuilt with the corrected
token-to-bin map.
