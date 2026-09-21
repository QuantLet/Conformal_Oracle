# R8 reproducibility status

The public replay command regenerates the paper's displayed results from the
included intermediate numerical inputs. The original 12 September deposit
provided code and final displays but had misplaced source paths and omitted
required inputs. Those packaging faults are repaired in
`R8-2026-09-12-repro1`; the original tag is not moved.

| Layer | Current public package |
|---|---|
| Integrity of code, intermediate inputs and reference outputs | Checked against `REPLAY_MANIFEST.json` |
| Intermediate results to tables, macros and figures | All eight producers executed; 61 exact output comparisons |
| Core statistical utilities and exact mathematical witness | Existing tests and deterministic certificate executed |
| Protection against false passes | Missing-output, changed-output and changed-input negative controls |
| Financial forecasts from raw market histories | Outside this display replay; no new fit is claimed |
| Native model draws and full simulation histories | Separate research archive; not rerun by this command |
| Manuscript compilation | Separate LaTeX submission/source package |

Run the commands in [README.md](README.md). A successful result says precisely
`Regeneration from archived intermediate results; no model fitting`. The JSON
receipt records each command, exact output hashes and the interpreter. A file's
presence or a historical `passed` receipt alone is not treated as regeneration.

The measured validation records are under `validation/`. The first successful
run used the existing recorded scientific runtime. A pip-only virtual environment
reproduced all 34 LaTeX outputs, but differed in 19 figure files because the
Matplotlib wheel bundled another FreeType version. That failure is preserved;
the checker was not weakened. The explicit conda build specification fixes the
renderer as well as package versions. The final public check uses a new clone
and a freshly created environment. The records, rather than this description
alone, establish which tests have completed on which platform.

The old R2 reproducibility ledger is retained in
`history/REPRODUCIBILITY_R2.md`. Its findings concern that historical archive and
are not a verdict on the current R8 outputs. The old tail-closure table is also
kept in `history/unused_exports/`; neither R8 document uses it.
