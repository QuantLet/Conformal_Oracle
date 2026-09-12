"""Independent checks of the shape--cost simulation; never runs production.

The reference algebra, loss integrals, quantile minimisers and Markov
occupancies below do not import the production engine. Output replay checks
are appended after the producer's schema is available.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/r8_shape_cost/validation"
SIM = ROOT / "artifacts/r8_shape_cost/simulation"
METHODS = ("Raw", "Shift-ERM", "Vol-ERM", "Vol-UERM", "Vol-CP", "Shift-CP")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explicit_pinball(residual, alpha):
    residual = np.asarray(residual)
    return alpha * np.maximum(residual, 0) + (1 - alpha) * np.maximum(-residual, 0)


def innovation_distribution(kind):
    if kind == "normal":
        return stats.norm()
    if kind == "t5":
        return stats.t(df=5, scale=np.sqrt(3 / 5))
    raise ValueError(kind)


def reference_loss(kind, threshold, sigma, alpha=.01):
    """Analytic loss, written independently from the production engine."""
    threshold, sigma = np.broadcast_arrays(threshold, sigma)
    if kind == "normal":
        x = threshold / sigma
        return sigma * stats.norm.pdf(x) + threshold * (stats.norm.cdf(x) - alpha)
    scale = sigma * np.sqrt(3 / 5)
    x = threshold / scale
    return scale * (5 + x * x) / 4 * stats.t.pdf(x, 5) + threshold * (stats.t.cdf(x, 5) - alpha)


def quadrature_loss(kind, threshold, sigma, alpha):
    distribution = innovation_distribution(kind)
    cut = threshold / sigma
    left = integrate.quad(
        lambda x: (1 - alpha) * (threshold - sigma * x) * distribution.pdf(x),
        -np.inf, cut, epsabs=2e-12, epsrel=2e-11,
    )[0]
    right = integrate.quad(
        lambda x: alpha * (sigma * x - threshold) * distribution.pdf(x),
        cut, np.inf, epsabs=2e-12, epsrel=2e-11,
    )[0]
    return left + right


def rational_leftmost(scores, weights, alpha=Fraction(1, 100)):
    """Enumerate exact objectives at every knot, not a cumulative-rank solver."""
    scores = [Fraction(str(x)) for x in scores]
    weights = [Fraction(str(x)) for x in weights]
    candidates = sorted(set(scores))
    values = []
    for c in candidates:
        cost = sum(w * ((1 - alpha) * max(s - c, 0) + alpha * max(c - s, 0))
                   for s, w in zip(scores, weights))
        values.append(cost)
    return candidates[values.index(min(values))]


def validate_subgradient(scores, weights, quantile, alpha=Fraction(1, 100)):
    scores = [Fraction(str(x)) for x in scores]
    weights = [Fraction(str(x)) for x in weights]
    quantile = Fraction(str(quantile))
    total = sum(weights)
    left = sum(w for s, w in zip(scores, weights) if s < quantile)
    right = sum(w for s, w in zip(scores, weights) if s <= quantile)
    target = (1 - alpha) * total
    # Strict lower inequality distinguishes the leftmost member of a flat
    # minimiser interval from its equally optimal right endpoint.
    return quantile in scores and left < target <= right


def transition_occupancy(last_state, horizon):
    """Direct repeated transition matrices, with the first future day j=1."""
    transition = np.array([[.95, .05], [.05, .95]])
    current = np.eye(2)[last_state]
    occupancy = np.zeros(2)
    for _ in range(horizon):
        current = current @ transition
        occupancy += current
    return occupancy / horizon


def reference_occupancy(last_state, horizon):
    correlation = .9
    average_power = correlation * (-np.expm1(horizon * np.log(correlation))) / ((1 - correlation) * horizon)
    stationary = np.array([.5, .5])
    return stationary + average_power * (np.eye(2)[last_state] - stationary)


def check_mathematics():
    loss_cases = 0
    max_error = 0.
    for kind in ("normal", "t5"):
        for alpha in (.01, .05, .5):
            for sigma in (.5, 1., 2.):
                for ratio in (-4., -2., 0., 1.5):
                    q = sigma * ratio
                    expected = float(reference_loss(kind, q, sigma, alpha))
                    actual = quadrature_loss(kind, q, sigma, alpha)
                    max_error = max(max_error, abs(actual - expected))
                    np.testing.assert_allclose(expected, actual, atol=4e-11, rtol=4e-11)
                    loss_cases += 1

    fixtures = [
        ([-3, -1, 0, 2, 4], [1, 1, 1, 1, 1]),
        ([-3, -1, 0, 2, 4], [1, 2, 1, 2, 1]),
        ([-2, -2, 0, 1, 1, 5], [2, 1, 2, 1, 2, 1]),
        ([0, 1, 2, 3], [1, 1, 1, 1]),
        ([0] * 99 + [1], [1] * 100),
        ([0] * 98 + [1, 2], [2] * 100),
    ]
    quantile_cases = 0
    for scores, weights in fixtures:
        for alpha in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 4), Fraction(1, 2)):
            q = rational_leftmost(scores, weights, alpha)
            assert validate_subgradient(scores, weights, q, alpha)
            quantile_cases += 1
    assert not validate_subgradient([0, 1, 2, 3], [1] * 4, 2, Fraction(1, 2))
    assert not validate_subgradient([0] * 99 + [1], [1] * 100, 1)

    occupancy_cases = 0
    for horizon in (1, 2, 10, 107, 428, 1714):
        for last in (0, 1):
            np.testing.assert_allclose(reference_occupancy(last, horizon), transition_occupancy(last, horizon), atol=4e-14, rtol=4e-14)
            occupancy_cases += 1
    wrong_first_day = np.array([1., 0.])
    assert not np.allclose(wrong_first_day, transition_occupancy(0, 1))
    wrong_correlation = .5 + .5 * .95
    assert not np.isclose(wrong_correlation, transition_occupancy(0, 1)[0])

    states = [Fraction(1), Fraction(2)]
    a = sum(states) / 2
    b = (sum(x * x for x in states) / 2) / a
    c = 1 / (sum(1 / x for x in states) / 2)
    assert (a, b, c) == (Fraction(3, 2), Fraction(5, 3), Fraction(4, 3))
    assert (b - c) / (a - c) == 2
    assert b > a > c
    alpha = .01
    crossings = {}
    for kind in ("normal", "t5"):
        distribution = innovation_distribution(kind)
        g = float(distribution.pdf(distribution.ppf(alpha)))
        hstar = np.sqrt(2 * alpha * (1 - alpha)) / g
        values = []
        for multiplier in (.5, 1., 2.):
            h = multiplier * hstar
            delta = (alpha * (1 - alpha) * float(b - c) / g - g * h * h * float(a - c)) / 2000
            values.append(delta)
        assert values[0] > 0 and abs(values[1]) < 1e-16 and values[2] < 0
        crossings[kind] = {"density": g, "h_star": float(hstar), "leading_differences_n1000": values}
    return dict(status="passed", analytic_loss_quadratures=loss_cases,
                maximum_loss_quadrature_error=max_error,
                exact_quantile_objective_fixtures=quantile_cases,
                direct_matrix_occupancy_cases=occupancy_cases,
                exact_moments={"A": str(a), "B": str(b), "C": str(c)},
                crossings=crossings,
                negative_controls={"right_endpoint_flat_minimiser": "detected",
                                   "wrong_99_percent_rank": "detected",
                                   "future_day_zero_inclusion": "detected",
                                   "stay_probability_as_eigenvalue": "detected"})


def check_engine():
    path = ROOT / "research/r8_shape_cost/engine.py"
    spec = importlib.util.spec_from_file_location("shape_engine_under_test", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    fixtures = [
        ([-3., -1., 0., 2., 4.], [1, 2, 1, 2, 1]),
        ([0.] * 99 + [1.], [1] * 100),
        ([0.] * 98 + [1., 2.], [2] * 100),
        ([-2.] * 98 + [0., 0., 1., 3.], [1, 2] * 51),
    ]
    for scores, weights in fixtures:
        assert engine.erm_quantile(scores) == float(rational_leftmost(scores, [1] * len(scores)))
        assert engine.erm_quantile(scores, weights) == float(rational_leftmost(scores, weights))
        if len(scores) >= 100:
            rank = -(-(99 * (len(scores) + 1)) // 100)
            assert engine.cp_quantile(scores) == sorted(scores)[rank - 1]
    for kind in ("normal", "t5"):
        q = np.array([-5., -2., 0., 1., 3.])
        np.testing.assert_allclose(engine.expected_loss(kind, q), reference_loss(kind, q, 1.), atol=2e-14, rtol=2e-14)
    for horizon in (1, 2, 107, 428, 1714):
        for state in (0, 1):
            np.testing.assert_allclose(engine.occupancy(state, horizon), transition_occupancy(state, horizon)[1], atol=4e-14, rtol=4e-14)
    return dict(status="passed", engine_sha256=sha(path), quantile_fixtures=len(fixtures),
                independent_loss_vectors=2, direct_occupancy_comparisons=10)


def assert_close(left, right, label, atol=2e-12, rtol=2e-12):
    np.testing.assert_allclose(left, right, atol=atol, rtol=rtol, err_msg=label)


def recorded_hashes(directory, receipt, key="outputs"):
    for name, digest in receipt[key].items():
        assert sha(directory / name) == digest, ("Recorded hash mismatch", name)


def check_cells(cells):
    expected = [(kind, n, factor) for kind in ("normal", "t5")
                for n in (250, 1000, 4000) for factor in (.5, 2.)]
    assert [(c["kind"], c["n"], c["h_factor"]) for c in cells] == expected
    for c in cells:
        kind, n, factor = c["kind"], c["n"], c["h_factor"]
        dist = innovation_distribution(kind)
        z = dist.ppf(.01)
        g = dist.pdf(z)
        hstar = np.sqrt(.0198) / g
        h = factor * hstar
        d = h / np.sqrt(n)
        assert c["horizon"] == (3 * n) // 7
        assert c["primary"] == (kind == "normal" and n == 1000)
        assert c["cell"] == f"{kind}_n{n}_h{factor:g}"
        for key, value in (("z_alpha", z), ("g", g), ("h_star", hstar), ("h", h), ("d", d)):
            assert_close(c[key], value, key)
        coefficient = c["constant_population_target"]
        derivative = .5 * (dist.cdf(z + d - coefficient) + dist.cdf(z + d - coefficient / 2)) - .01
        assert abs(derivative) < 2e-12
        delta = (.0099 * float(Fraction(1, 3)) / g - g * h * h * float(Fraction(1, 6))) / (2 * n)
        assert_close(c["predicted_delta"], delta, "leading loss difference")


def check_histories_and_coefficients(frame, cells):
    with np.load(SIM / "histories.npz") as saved:
        states = saved["states"]
        uniforms = saved["uniforms"]
    assert states.shape == uniforms.shape == (5000, 4000)
    assert states.dtype == np.uint8 and uniforms.dtype == np.float64
    assert np.isin(states, [0, 1]).all()
    assert ((uniforms > 0) & (uniforms < 1)).all()
    names = [c["cell"] for c in cells]
    coefficients = {
        m: frame[frame.method == m].pivot(index="replication", columns="cell", values="coefficient")
        .reindex(index=np.arange(5000), columns=names).to_numpy()
        for m in METHODS
    }
    assert np.all(coefficients["Raw"] == 0)
    knot_checks = 0
    seed_fingerprints = set()
    for rep in range(5000):
        # Reproduce only the saved input history. No new study outcome is generated.
        scale_rng = np.random.default_rng(np.random.SeedSequence([20260911, 3101, rep]))
        start = int(scale_rng.integers(0, 2))
        flip_uniforms = scale_rng.random(3999)
        reconstructed = np.empty(4000, dtype=np.uint8)
        reconstructed[0] = start
        # Cumulative addition modulo two is independent of the engine's XOR implementation.
        reconstructed[1:] = (start + np.cumsum(flip_uniforms >= .95)) % 2
        np.testing.assert_array_equal(states[rep], reconstructed)
        innovation_rng = np.random.default_rng(np.random.SeedSequence([20260911, 3102, rep]))
        expected_uniforms = innovation_rng.random(4000)
        expected_uniforms = np.maximum(np.nextafter(0., 1.), np.minimum(expected_uniforms, np.nextafter(1., 0.)))
        np.testing.assert_array_equal(uniforms[rep], expected_uniforms)
        seed_fingerprints.add(hashlib.sha256(uniforms[rep].tobytes()).hexdigest())
        innovations = {kind: innovation_distribution(kind).ppf(uniforms[rep]) for kind in ("normal", "t5")}
        for col, cell in enumerate(cells):
            n, kind = cell["n"], cell["kind"]
            scales = states[rep, :n].astype(np.int64) + 1
            standard = cell["z_alpha"] - innovations[kind][:n] + cell["d"]
            scores = scales * standard
            for method in METHODS[1:]:
                candidate = coefficients[method][rep, col]
                values = scores if method.startswith("Shift-") else standard
                weights = scales if method == "Vol-ERM" else np.ones(n, dtype=np.int64)
                left = int(weights[values < candidate].sum())
                right = int(weights[values <= candidate].sum())
                assert np.any(values == candidate), (rep, cell["cell"], method, "not a data knot")
                if method.endswith("CP"):
                    rank = -(-(99 * (n + 1)) // 100)
                    assert left < rank <= right, (rep, cell["cell"], method, left, right, rank)
                else:
                    target = 99 * int(weights.sum())
                    assert 100 * left < target <= 100 * right, (rep, cell["cell"], method, left, right, target)
                knot_checks += 1
        if (rep + 1) % 1000 == 0:
            print("Independent input/rank checks", rep + 1, "/ 5000", flush=True)
    assert len(seed_fingerprints) == 5000
    return states, dict(saved_histories_reproduced=5000, distinct_innovation_histories=5000,
                        coefficient_knot_and_leftmost_subgradient_checks=knot_checks,
                        raw_coefficients_checked=5000 * 12)


def check_conditional_outputs(frame, cells, states):
    checks = 0
    for info in cells:
        cell = frame[frame.cell == info["cell"]]
        n, horizon, kind = info["n"], info["horizon"], info["kind"]
        dist = innovation_distribution(kind)
        for method in METHODS:
            current = cell[cell.method == method].sort_values("replication")
            assert current.replication.tolist() == list(range(5000))
            last = states[:, n - 1]
            np.testing.assert_array_equal(current.last_state.to_numpy(), last)
            # Direct transition-matrix averages, not the producer's occupancy formula.
            high = np.array([transition_occupancy(s, horizon)[1] for s in (0, 1)])[last]
            coefficient = current.coefficient.to_numpy()
            scale = np.array([1., 2.])
            raw = scale[None, :] * (info["z_alpha"] + info["d"])
            correction = coefficient[:, None] * (scale[None, :] if method.startswith("Vol-") else 1.)
            forecasts = raw - correction
            losses = reference_loss(kind, forecasts, scale[None, :])
            hits = dist.cdf(forecasts / scale[None, :])
            probabilities = np.column_stack([1 - high, high])
            expected = (losses * probabilities).sum(axis=1)
            optimal = reference_loss(kind, scale * info["z_alpha"], scale)
            excess = expected - (probabilities * optimal).sum(axis=1)
            values = {
                "forecast_state1": forecasts[:, 0], "forecast_state2": forecasts[:, 1],
                "low_state_loss": losses[:, 0], "high_state_loss": losses[:, 1],
                "low_state_violation": hits[:, 0], "high_state_violation": hits[:, 1],
                "loss": expected, "marginal_loss": losses.mean(axis=1),
                "violation": (hits * probabilities).sum(axis=1), "marginal_violation": hits.mean(axis=1),
                "excess_vs_conditional_oracle": excess, "mean_high_probability": high,
            }
            for name, value in values.items():
                assert_close(current[name].to_numpy(), value, (info["cell"], method, name))
                checks += len(value)
            assert np.all(excess >= -2e-12)
    return dict(independent_conditional_metric_values=checks,
                coefficients_kept_fixed_over_contiguous_horizon=True,
                evaluation_uses_future_days_1_through_H=True)


def check_aggregation(frame, cells, findings):
    names = [c["cell"] for c in cells]
    matrices = {m: frame[frame.method == m].pivot(index="replication", columns="cell", values="loss")
                .reindex(index=np.arange(5000), columns=names).to_numpy() for m in METHODS}
    raw_reference = matrices["Shift-ERM"]
    primary = matrices["Vol-ERM"] - raw_reference
    sensitivity = np.concatenate([matrices[m] - raw_reference for m in ("Vol-UERM", "Vol-CP", "Shift-CP")], axis=1)
    with np.load(SIM / "bootstrap_indices.npz") as saved:
        indices = saved["indices"]
    assert indices.shape == (9999, 5000) and indices.dtype == np.int32
    rng = np.random.default_rng(2026091131)
    np.testing.assert_array_equal(indices, rng.integers(0, 5000, size=(9999, 5000), dtype=np.int32))
    contrasts = pd.read_csv(SIM / "contrasts.csv")
    assert len(contrasts) == 48
    family_receipts = {}
    for family, values, methods in (("primary", primary, ("Vol-ERM",)),
                                     ("sensitivity", sensitivity, ("Vol-UERM", "Vol-CP", "Shift-CP"))):
        with np.load(SIM / f"{family}_bootstrap.npz") as saved:
            recorded_draws = saved["means"]
        assert recorded_draws.shape == (9999, values.shape[1])
        # Frequency weighting reconstructs every history-bootstrap mean without
        # production's values[indices] advanced-indexing implementation.
        replay = np.empty_like(recorded_draws)
        for start in range(0, 9999, 50):
            block = indices[start:start + 50]
            weights = np.array([np.bincount(row, minlength=5000) for row in block], dtype=float)
            replay[start:start + len(block)] = weights @ values / 5000
        assert_close(recorded_draws, replay, family + " bootstrap", atol=3e-13, rtol=3e-12)
        mean = values.sum(axis=0) / 5000
        centered = values - mean
        se = np.sqrt((centered * centered).sum(axis=0) / (4999 * 5000))
        maximum = np.abs((replay - mean) / se).max(axis=1)
        # The higher .95 quantile is an explicit sorted order statistic.
        order = int(np.ceil(.95 * (9999 - 1)))
        critical = float(np.sort(maximum)[order])
        assert_close(findings["families"][family]["critical"], critical, family + " critical", atol=2e-10)
        assert findings["families"][family]["contrasts"] == values.shape[1]
        for pos, method in enumerate(methods):
            for col, info in enumerate(cells):
                j = pos * 12 + col
                row = contrasts[(contrasts.family == family) & (contrasts.method == method) & (contrasts.cell == info["cell"])]
                assert len(row) == 1
                row = row.iloc[0]
                assert row.reference == "Shift-ERM"
                prediction = info["predicted_delta"] if family == "primary" else (
                    0. if method == "Shift-CP" else float(Fraction(1, 6)) *
                    (.0099 / info["g"] - info["g"] * info["h"] ** 2) / (2 * info["n"]))
                for key, value in (("mean_delta", mean[j]), ("MCSE", se[j]), ("critical", critical),
                                   ("simultaneous_lower", mean[j] - critical * se[j]),
                                   ("simultaneous_upper", mean[j] + critical * se[j]),
                                   ("leading_prediction", prediction), ("discrepancy", mean[j] - prediction),
                                   ("n_scaled_delta", info["n"] * mean[j]),
                                   ("n_scaled_prediction", info["n"] * prediction)):
                    assert_close(row[key], value, (family, method, info["cell"], key), atol=2e-10)
        family_receipts[family] = dict(contrasts=values.shape[1], whole_history_draws=9999,
                                       independently_replayed_means=int(replay.size), critical=critical)
    selected = contrasts[(contrasts.family == "primary") & contrasts.primary]
    assert set(selected.cell) == {"normal_n1000_h0.5", "normal_n1000_h2"}
    low = selected[selected.h_factor == .5].iloc[0]
    high = selected[selected.h_factor == 2.].iloc[0]
    supported = low.simultaneous_lower > 0 and high.simultaneous_upper < 0
    adverse = low.simultaneous_upper < 0 or high.simultaneous_lower > 0
    verdict = "supported" if supported else ("contradicted_at_primary_finite_n" if adverse else "unresolved")
    assert findings["primary_sign_crossing"] == verdict
    assert findings["bootstrap_draws"] == 9999 and findings["independent_histories"] == 5000
    assert not findings["quantitative_equivalence_claimed"] and not findings["financial_generalisation_claimed"]
    assert findings["all_primary_cells_observed_sign_agrees"] == bool(np.all(np.sign(primary.mean(axis=0)) == np.sign([c["predicted_delta"] for c in cells])))
    metrics = ["loss", "marginal_loss", "excess_vs_conditional_oracle", "violation", "coefficient",
               "low_state_loss", "high_state_loss", "low_state_violation", "high_state_violation"]
    summary = pd.read_csv(SIM / "method_summary.csv")
    assert len(summary) == 72
    for _, row in summary.iterrows():
        subset = frame[(frame.cell == row.cell) & (frame.method == row.method)]
        for metric in metrics:
            x = subset[metric].to_numpy()
            mean = x.sum() / 5000
            sd = np.sqrt(((x - mean) ** 2).sum() / 4999)
            assert_close(row[metric + "_mean"], mean, (row.cell, row.method, metric, "mean"))
            assert_close(row[metric + "_std"], sd, (row.cell, row.method, metric, "sd"))
    return dict(families=family_receipts, primary_sign_crossing=verdict,
                primary_cells=selected[["cell", "mean_delta", "simultaneous_lower", "simultaneous_upper"]].to_dict("records"),
                method_summary_statistics_checked=72 * len(metrics) * 2,
                inference_unit="Whole independent calibration history, paired across all cells and methods")


def check_outputs():
    execution = json.loads((SIM / "execution.json").read_text())
    findings = json.loads((SIM / "findings.json").read_text())
    lock = json.loads((SIM / "lock.json").read_text())
    for name, digest in lock["files"].items():
        assert sha(ROOT / name) == digest, ("Changed producer/protocol", name)
    assert execution["lock_sha256"] == sha(SIM / "lock.json")
    recorded_hashes(SIM, execution)
    recorded_hashes(SIM, findings)
    assert execution["status"] == findings["status"] == "complete"
    assert (execution["histories"], execution["cells"], execution["rows"]) == (5000, 12, 360000)
    assert tuple(execution["methods"]) == METHODS
    cells = json.loads((SIM / "cells.json").read_text())
    check_cells(cells)
    frame = pd.read_parquet(SIM / "replications.parquet")
    assert len(frame) == 360000
    assert not frame.duplicated(["cell", "replication", "method"]).any()
    assert set(frame.method) == set(METHODS)
    assert np.isfinite(frame.select_dtypes("number")).all().all()
    assert (frame.groupby(["cell", "method"]).size() == 5000).all()
    for c in cells:
        data = frame[frame.cell == c["cell"]]
        for name, value in c.items():
            assert (data[name] == value).all(), (c["cell"], "inconsistent cell metadata", name)
    states, history_checks = check_histories_and_coefficients(frame, cells)
    conditional = check_conditional_outputs(frame, cells, states)
    aggregate = check_aggregation(frame, cells, findings)
    return dict(status="passed", execution_receipt_sha256=sha(SIM / "execution.json"),
                findings_receipt_sha256=sha(SIM / "findings.json"), protocol_lock_sha256=sha(SIM / "lock.json"),
                production_outputs_verified=len(execution["outputs"]) + len(findings["outputs"]),
                stored_rows=360000, **history_checks, **conditional, **aggregate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = check_mathematics()
    if (ROOT / "research/r8_shape_cost/engine.py").exists():
        receipt["production_engine_fixtures"] = check_engine()
    receipt["validator_sha256"] = sha(Path(__file__))
    if not args.deterministic_only:
        receipt["output_replay"] = check_outputs()
    destination = "deterministic.json" if args.deterministic_only else "simulation_validation.json"
    (OUT / destination).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
