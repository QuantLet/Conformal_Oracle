# Do comparison papers report sampling configuration?

Survey of the four papers this manuscript compares against. Full text retrieved
from arXiv HTML at the **current** version (an earlier pass pinned `v1` and was
discarded — Goel is at v7, Schmitt at v3, and the v1 titles differ).

Search terms: `num_samples`, `top_k` / `top-k` / `top k`, `top_p` / `top-p`,
`nucleus`, `temperature`, `number of samples`, `sample paths`, `random seed`,
`seed`.

## Result

| Paper | arXiv | sample-based forecasters used | sampling parameters reported |
|---|---|---|---|
| **Rahimikia et al.**, *Re(Visiting) Time Series Foundation Models in Finance* | 2511.18578 | **Chronos** (255 mentions), TimesFM (201), Moirai (31), Lag-Llama (13) | **none** |
| **Goel et al.**, *Time-Series Foundation AI Model for Value-at-Risk Forecasting* | 2410.11773 | TimesFM (48), TimeGPT (4), Lag-Llama (1) | **none** |
| Schmitt, *Taming Tail Risk in Financial Markets* | 2602.03903 | none | n/a |
| Zhong, *Proxy-Reliance Control in Conformal Recalibration* | 2603.22569 | none | n/a |

The two apparent hits in Rahimikia are false positives, checked in context:
"the top $k$ columns of $\mathbf{V}$" (principal components) and "the approximate
number of samples … used during model pre-training" (corpus size). Neither is a
generation parameter.

## Scope, stated rather than implied

Only **two** of the four papers are exposed to this question. Schmitt and Zhong
use no time-series foundation models at all — both are conformal-recalibration
papers evaluated on classical forecasters — so reporting sampling configuration
would be meaningless for them. Counting them as omissions would inflate the
finding, and they are recorded as not applicable.

Within the two that are exposed, the distinction matters further:

- **Rahimikia et al. is the clean case.** It is centrally about TSFMs in
  finance, uses Chronos more than any other model, and Chronos is
  token-categorical with a `top_k` default that truncates the sampled support to
  50 of 4094 bins. No sampling parameter is reported anywhere in 448,000
  characters of text and appendices.
- **Goel et al. is weaker.** Its primary forecaster is TimesFM, which exposes a
  quantile head rather than a sampler, so `top_k` has no direct analogue. Only
  its Lag-Llama comparison is sampled, and that is a single mention.

## What can be claimed

That a widely-used default silently truncates the predictive support is
demonstrated here with a dose–response curve
(`analysis/chronos_sampling/RESULTS.md`) and an exact-arithmetic alternative
(`analytic_quantiles.py`). That the leading finance-TSFM benchmarking paper
reports no sampling configuration is a fact about one paper, verifiable by
anyone from its arXiv HTML.

The general claim — that the field does not report these parameters — is **not
established by n = 2**. It would need a systematic survey with a stated sampling
frame (say, every finance paper using a sample-based TSFM in a named set of
venues over a stated window). That is a defensible extension and is not what was
done here.

## Reproduce

    python analysis/litsurvey/fetch.py       # writes raw_current.json
