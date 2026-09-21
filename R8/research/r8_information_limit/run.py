"""Run the complete prespecified analytic grid and write immutable outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
from engine import finite, costs_by_k, minimum_length, poisson_limit, error

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/r8_information_limit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    dest = OUT/('replay' if args.replay else 'run')
    assert not dest.exists(), 'Preserve existing runs'
    before = json.loads((OUT/'before.json').read_text())
    assert sha(ROOT/'research/r8_information_limit/PROTOCOL.md') == before['protocol_sha256']
    assert all(sha(ROOT/p) == h for p,h in before['canonical'].items())
    dest.mkdir()
    rows = []; lengths = []; limits = []; convergence = []
    for alpha in [.01, .05]:
        for epsilon in [.1, .2]:
            costs = costs_by_k(1000, alpha, epsilon)
            for retention in [0., .25, .5, .75]:
                for n in [125, 250, 500, 1000]:
                    rows.append(finite(n, alpha, epsilon, retention, costs))
                for target in [.1, .05]:
                    lengths.append(minimum_length(alpha, epsilon, retention, target))
    for tau in [1.25, 2.5, 5., 10.]:
        for epsilon in [.1, .2]:
            for retention in [0., .25, .5, .75]:
                limit = poisson_limit(tau, epsilon, retention)
                limits.append(limit)
                for alpha in [.01, .001, .0001]:
                    n = round(tau/alpha)
                    assert abs(n*alpha-tau) < 1e-12
                    value = error(n, alpha, epsilon, retention)
                    convergence.append(dict(n=n, alpha=alpha, tau=tau, epsilon=epsilon,
                                            retention=retention, error=value,
                                            poisson_error=limit['poisson_error'],
                                            absolute_difference=abs(value-limit['poisson_error'])))
    frames = {'finite.csv':rows, 'lengths.csv':lengths, 'poisson.csv':limits, 'convergence.csv':convergence}
    for name, data in frames.items():
        pd.DataFrame(data).to_csv(dest/name, index=False, float_format='%.17g')
    d = pd.DataFrame(rows)
    assert len(d) == 64 and len(lengths) == 32 and len(limits) == 32
    assert np.max(abs(d.best_average_error-d.cdf_error)) < 2e-12
    assert np.max(abs(d.scalar_bayes_regret-d.direct_scalar_risk)) < 1e-15
    assert np.max(abs(d.affinity-d.direct_affinity)) < 2e-12
    assert (d.affinity_error_lower <= d.best_average_error+1e-12).all()
    assert (d.testing_scalar_lower <= d.scalar_bayes_regret+1e-15).all()
    inputs = ['research/r8_information_limit/PROTOCOL.md', 'research/r8_information_limit/engine.py',
              'research/r8_information_limit/run.py', 'artifacts/r8_information_limit/before.json']
    result = dict(status='passed', deterministic=True, new_paths=0, new_base_forecasts=0,
                  configurations=64, discrimination_lengths=32, poisson_limits=32,
                  convergence_cells=len(convergence), canonical_unchanged=True,
                  inputs={p:sha(ROOT/p) for p in inputs},
                  outputs={p:sha(dest/p) for p in frames},
                  environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                                   pandas=pd.__version__, scipy=scipy.__version__))
    (dest/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
