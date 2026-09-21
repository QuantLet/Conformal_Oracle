# Quality checks for the local 0.4.0 candidate

Run from the package root:

```sh
python -m ruff check --no-cache .
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -ra
python -m mypy --cache-dir /tmp/conformal-oracle-mypy src/conformal_oracle
```

The 2026-09-07 type-check environment used Python 3.13.9, mypy 1.17.1,
NumPy 2.3.5, pandas 2.3.3 and SciPy 1.16.3. The project mypy target remains
Python 3.10. In an isolated development environment, add these matching
annotation packages (they are not runtime dependencies):

```sh
python -m pip install 'numpy==2.3.5' 'pandas-stubs==2.3.3.260113' \
  'scipy-stubs==1.16.3.0' 'scikit-learn-stubs==0.0.3'
```

With that environment, the final full mypy run reports 25 diagnostics in 11
files, all `import-not-found` or `import-untyped`. Missing optional packages
are LightGBM, rpy2, Chronos, TimesFM, GluonTS, uni2ts, huggingface-hub and
Lag-Llama. Statsmodels is installed but has no usable typing metadata here.
The name `statsmodels-stubs` had no distribution in the configured registry.

This is a failing global check, not a complete type-safety certificate.
Unavailable dependency interfaces still require validation in their supported
integration environments. Missing imports are not ignored; no exclusions,
`ignore_missing_imports` settings or new `type: ignore` comments were added.
Do not download model weights or run forecasts merely to run static checks.

The forwarded, mode-dependent keyword dictionaries use `Any` at dispatch
boundaries. Parameter names and types on the concrete static/rolling workers
remain explicit; complete static validation of forwarded options is not
provided. Concrete optional model types are imported only under
`TYPE_CHECKING`, preserving lazy runtime dependency loading.

This pass preserves arithmetic and stored manuscript results. The observable
error-path change is a `RuntimeError` when isotonic recalibration is applied
before fitting; previously this failed with an attribute error. GARCH uses
the canonical `GARCH` spelling accepted by arch's case-insensitive dispatcher.
NumPy scalar results are made explicit as Python floats at scalar return
boundaries. Numerical pandas inputs use `to_numpy()` where arrays are required.
