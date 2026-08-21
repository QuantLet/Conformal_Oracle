#!/usr/bin/env python3
"""Numerical check of Proposition E.3 (online quantile tracking).

The proposition is deterministic, so a counterexample would be a single sequence
violating the bound. This runs the update on adversarial and stochastic score
paths and checks (i) the iterates stay inside the interval Step 1 claims, and
(ii) the time-average miscoverage is bounded by (2R + eta)/(eta T).

Not a proof. It is the check that would have caught the error in the version this
replaces, which asserted a rate without an explicit constant.
"""
import numpy as np

ALPHA, R = 0.01, 3.0


def run(scores, eta, q1=0.0):
    q = q1
    qs, hits = [q], []
    for s in scores:
        hits.append(1.0 if s > q else 0.0)
        q = q + eta * (hits[-1] - ALPHA)
        qs.append(q)
    return np.array(qs), np.array(hits)


def check(name, scores, eta):
    qs, hits = run(scores, eta)
    T = len(scores)
    lo, hi = -R - eta * ALPHA, R + eta * (1 - ALPHA)
    in_range = bool(np.all((qs >= lo - 1e-12) & (qs <= hi + 1e-12)))
    emp = abs(hits.mean() - ALPHA)
    bound = (2 * R + eta) / (eta * T)
    exact = abs(qs[-1] - qs[0]) / (eta * T)
    print(f"{name:26} T={T:6d}  |emp-a|={emp:.6f}  telescoped={exact:.6f}  "
          f"bound={bound:.6f}  in-range={in_range}  holds={emp <= bound + 1e-12}")
    assert in_range, f"{name}: iterate left the interval"
    assert abs(emp - exact) < 1e-10, f"{name}: telescoping identity failed"
    assert emp <= bound + 1e-12, f"{name}: bound violated"


rng = np.random.default_rng(0)
for T in (250, 1000, 5000):
    eta = R / np.sqrt(T)
    check("iid normal", np.clip(rng.normal(0, 1, T), -R, R), eta)
    check("heavy tail, clipped", np.clip(rng.standard_t(3, T), -R, R), eta)
    check("regime shift", np.clip(np.r_[rng.normal(0, .3, T // 2),
                                        rng.normal(1.5, 1.2, T - T // 2)], -R, R), eta)
    # adversarial: always just above the current iterate would need feedback;
    # the worst fixed sequence for the bound is one that drives q to a boundary
    check("all at +R", np.full(T, R), eta)
    check("all at -R", np.full(T, -R), eta)
    check("alternating extremes", np.where(np.arange(T) % 2 == 0, R, -R), eta)
print("\nAll paths satisfy Step 1 and the bound; telescoping identity exact.")
