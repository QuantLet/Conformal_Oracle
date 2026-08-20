# CAViaR: independent re-estimation, and the window question

Second implementation: Engle--Manganelli recursions written directly, Powell optimiser, different starting values. Paths are compared, not parameters -- the objective is flat in directions that barely move the fitted quantile path.

| model | assets | Kupiec pass, original window | Kupiec pass, common window | mean π̂ orig | mean π̂ common | mean n orig | mean n common |
|---|---|---|---|---|---|---|---|
| CAViaR-AS | 24 | **15/24** | **15/24** | 0.0111 | 0.0110 | 1678 | 1524 |
| CAViaR-SAV | 24 | **14/24** | **16/24** | 0.0114 | 0.0113 | 1678 | 1524 |

Original run, for comparison:

| model | Kupiec pass | mean π̂ |
|---|---|---|
| CAViaR-AS | 15/24 | 0.0110 |
| CAViaR-SAV | 14/24 | 0.0114 |
| GAS-t | 0/24 | 0.0333 |

