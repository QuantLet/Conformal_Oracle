## synthetic_returns.csv

2000 daily log-returns generated from a GARCH(1,1) process:

- DGP: `sigma2(t) = omega + alpha * r(t-1)^2 + beta * sigma2(t-1)`
- Parameters: `omega = 1e-6`, `alpha = 0.05`, `beta = 0.90`
- Innovations: standard Normal
- Seed: `numpy.random.default_rng(2026)`
- Dates: business days starting 2018-01-02

The unconditional variance is `omega / (1 - alpha - beta) = 2e-5`,
giving an unconditional daily vol of ~0.45%.
