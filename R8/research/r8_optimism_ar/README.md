# Archived-path AR optimism control

Read PROTOCOL.md and results/optimism_ar/RESULTS.md. The Normal-margin AR0/.8
comparison isolates nonzero target-hit covariance, using six cells and only
previously stored paths. It is a retrospective control, not financial validation.

The committed protocol is e96ee479bb9455d94aeb90a95c3cc86f45d264df in the
local protocol_repository. Original path/results hashes are checked before
the initial --bind operation; lock.json then binds actual source and input
bytes, sizes and mtimes. No original source or result is overwritten.

From the project root, verify completed artifacts:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_optimism_ar/finish.py --verify
```

Recompute numerical outputs in a new empty location:

```sh
/private/tmp/irfa-r8-conda-clean/bin/python research/r8_optimism_ar/run.py --output /private/tmp/irfa-ar-optimism-new
```

Compare its five CSVs to results/optimism_ar/primary. All must match exactly
under the recorded runtime. Do not rerun --bind over the completed lock.
The independent verifier does not import run.py; it reconstructs all losses,
uses an angular change of variables for the hit-covariance integral, and
recomputes simultaneous uncertainty separately.

The prior v1/v2/v3 verifiers are called by the final conservation check.
Relocation requires explicit verified path rebinding; no clean-environment
installation is claimed. The result admits no new financial-panel calculation
and changes no manuscript equation, citation, source or PDF.
