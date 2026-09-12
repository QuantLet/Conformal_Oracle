"""Deterministic checks for a proposed risk bridge; no new draws or fitting.

This is development material. It reads the validated R8 simulation outputs,
checks identities by quadrature, and writes only to its own output directory.
It does not extend the validated manuscript's claims.
"""
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/r8_frontier"
SAVED = ROOT / "artifacts/review_20260909/complexity_mc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(n, alpha, conformal):
    value = Decimal(n + int(conformal)) * (1 - Decimal(str(alpha)))
    k = int(value.to_integral_value(rounding=ROUND_CEILING))
    assert 1 <= k <= n
    return k


def normal_check_expectation(mean, sd, alpha):
    """E rho_alpha(X), X normal with the given mean and standard deviation."""
    z = mean / sd
    return sd * stats.norm.pdf(z) + mean * (stats.norm.cdf(z) - (1-alpha))


def return_risk(kind, q, sigma, alpha):
    """Analytic E rho_alpha(Y-q), independently evaluated from saved inputs."""
    if kind == "normal":
        z = q / sigma
        return sigma * (stats.norm.pdf(z) + z * (stats.norm.cdf(z)-alpha))
    scale = sigma * np.sqrt(3/5)
    z = q / scale
    return scale * ((5+z*z)/4 * stats.t.pdf(z, 5)
                    + z*(stats.t.cdf(z, 5)-alpha))


def identity_checks():
    errors = []
    for alpha in (.01, .05, .5):
        p = 1-alpha
        z = stats.norm.ppf(p)
        for optimal_shift in (-.4, 0., .4):
            # S = optimal_shift + N(0,1) - z, so F_S(optimal_shift)=p.
            for c in (-1., -.2, 0., .3, 1.):
                integral = integrate.quad(
                    lambda u: stats.norm.cdf(u-optimal_shift+z)-p,
                    0., c, epsabs=1e-13, epsrel=1e-12)[0]
                direct = (normal_check_expectation(c-optimal_shift+z, 1., alpha)
                          - normal_check_expectation(-optimal_shift+z, 1., alpha))
                errors.append(abs(integral-direct))
    assert max(errors) < 1e-12
    return {"cases": len(errors), "max_absolute_error": max(errors)}


def finite_uniform_checks():
    rows = []
    max_error = 0.
    for alpha in (.01, .05):
        p = 1-alpha
        for n in (125, 250, 500, 1000):
            for conformal in (False, True):
                k = rank(n, alpha, conformal)
                b = k/(n+1)-p
                variance = k*(n+1-k)/((n+1)**2*(n+2))
                # U_(k) has a known beta law; no simulated order statistics.
                law = stats.beta(k, n+1-k)
                for target in (-.005, 0., .005):
                    assert -(1-p) <= target <= p
                    for lam in (0., .5, 1.):
                        exact = .5*(lam*lam*variance
                                     + (lam*(target+b)-target)**2-target**2)
                        numeric = integrate.quad(
                            lambda u: .5*((lam*(target+u-p)-target)**2
                                          - target**2)*law.pdf(u),
                            0., 1., points=[.9, .98, .995],
                            epsabs=1e-12, epsrel=1e-10)[0]
                        max_error = max(max_error, abs(exact-numeric))
                    rows.append({"alpha": alpha, "n": n,
                                 "method": "Shift-CP" if conformal else "Shift-ERM",
                                 "optimal_shift": target, "rank": k,
                                 "rank_bias": b, "rank_variance": variance,
                                 "full_shift_risk_difference": .5*(variance+b*b-target*target),
                                 "break_even_abs_shift": np.sqrt(variance+b*b)})
    assert max_error < 1e-11
    pd.DataFrame(rows).to_csv(OUT/"uniform_exact.csv", index=False)
    return {"quadrature_cases": len(rows)*3, "max_absolute_error": max_error}


def normal_order_statistic_checks():
    """Exact beta-order-statistic quadrature versus the local risk expansion."""
    rows = []
    nodes, weights = special.roots_legendre(2048)
    uniform_nodes = (nodes+1)/2
    weights = weights/2
    low_nodes, low_weights = special.roots_legendre(1024)
    low_nodes, low_weights = (low_nodes+1)/2, low_weights/2
    for alpha in (.01, .05):
        p = 1-alpha
        z = stats.norm.ppf(p)
        density = stats.norm.pdf(z)
        omega = alpha*(1-alpha)
        tau = np.sqrt(omega)/density
        for n in (125, 250, 500, 1000, 4000, 16000, 64000):
            for conformal in (False, True):
                k = rank(n, alpha, conformal)
                order = stats.norm.ppf(stats.beta.ppf(uniform_nodes, k, n+1-k))-z
                lower_order = stats.norm.ppf(stats.beta.ppf(low_nodes, k, n+1-k))-z
                for signal in (-2., 0., 2.):
                    h = signal*tau
                    target = h/np.sqrt(n)
                    for lam in (.5, 1.):
                        estimate = lam*(target+order)
                        risk = normal_check_expectation(estimate-target+z, 1., alpha)
                        raw = normal_check_expectation(-target+z, 1., alpha)
                        exact = n * np.dot(weights, risk-raw)
                        lower = n * np.dot(low_weights,
                            normal_check_expectation(lam*(target+lower_order)-target+z, 1., alpha)-raw)
                        approximation = (lam*lam*omega
                            - (2*lam-lam*lam)*density*density*h*h)/(2*density)
                        rows.append({"alpha": alpha, "n": n,
                                     "method": "Shift-CP" if conformal else "Shift-ERM",
                                     "signal_in_standard_errors": signal, "lambda": lam,
                                     "n_times_risk_difference": exact,
                                     "local_limit": approximation,
                                     "quadrature_resolution_difference": abs(exact-lower),
                                     "absolute_approximation_error": abs(exact-approximation)})
    frame = pd.DataFrame(rows)
    assert np.isfinite(frame.select_dtypes(include=np.number).to_numpy()).all()
    high = frame[frame.n == 64000]
    # Numerical check in a known iid normal family, not a proof for dependent data.
    relative = high.absolute_approximation_error / high.local_limit.abs()
    assert relative.max() < .06
    assert frame.quadrature_resolution_difference.max() < 2e-4
    frame.to_csv(OUT/"normal_order_statistic.csv", index=False)
    return {"cases": len(frame), "max_relative_error_at_n64000": relative.max(),
            "max_quadrature_resolution_difference": frame.quadrature_resolution_difference.max()}


def exact_frontier():
    """Normal location family: solve the finite-n loss break-even points."""
    data = pd.read_csv(OUT/"normal_order_statistic.csv")
    data = data[(data.signal_in_standard_errors == 0) & (data["lambda"] == 1)]
    rows = []
    count_identity_error = 0.
    for record in data.itertuples(index=False):
        alpha, n = record.alpha, record.n
        p, z = 1-alpha, stats.norm.ppf(1-alpha)
        k = rank(n, alpha, record.method == "Shift-CP")
        cost = record.n_times_risk_difference/n
        # Independently evaluate the exact count-law/Fubini identity on the
        # unshifted normal score scale. Integration in u avoids 1/f at endpoints.
        lower = integrate.quad(
            lambda u: (p-stats.norm.cdf(u+z))
            * special.betainc(k, n+1-k, stats.norm.cdf(u+z)),
            -12.-z, 0., epsabs=1e-11, epsrel=1e-9)[0]
        upper = integrate.quad(
            lambda u: (stats.norm.cdf(u+z)-p)
            * special.betaincc(k, n+1-k, stats.norm.cdf(u+z)),
            0., 12.-z, epsabs=1e-11, epsrel=1e-9)[0]
        count_identity_error = max(count_identity_error, abs(cost-lower-upper))
        minimum = normal_check_expectation(z, 1., alpha)
        def improvement(c):
            return normal_check_expectation(z-c, 1., alpha)-minimum
        left = optimize.brentq(lambda c: improvement(c)-cost, -20., 0., xtol=1e-12)
        right = optimize.brentq(lambda c: improvement(c)-cost, 0., 20., xtol=1e-12)
        rows.append({"alpha": alpha, "n": n, "method": record.method, "rank": k,
                     "estimation_cost": cost, "lower_shift_boundary": left,
                     "upper_shift_boundary": right,
                     "lower_raw_violation_rate": stats.norm.sf(z-left),
                     "upper_raw_violation_rate": stats.norm.sf(z-right),
                     "local_violation_half_width": np.sqrt(alpha*(1-alpha)/n)})
    assert count_identity_error < 5e-9
    frame = pd.DataFrame(rows)
    assert (frame.lower_raw_violation_rate < frame.alpha).all()
    assert (frame.upper_raw_violation_rate > frame.alpha).all()
    frame.to_csv(OUT/"normal_exact_break_even.csv", index=False)
    return {"cases": len(frame), "max_count_identity_error": count_identity_error}


def saved_simulation_checks():
    rows, bindings = [], {}
    max_replay_error = 0.
    scalar_estimates = 0
    optimal_shift = .25*np.sqrt(1e-5/(1-.10-.85))
    for kind in ("normal", "t5"):
        for alpha in (.01, .05):
            z = stats.norm.ppf(alpha) if kind == "normal" else np.sqrt(3/5)*stats.t.ppf(alpha, 5)
            innovation_density = (stats.norm.pdf(z) if kind == "normal"
                                  else stats.t.pdf(z/np.sqrt(3/5), 5)/np.sqrt(3/5))
            for n in (125, 250, 500, 1000):
                folder = SAVED/f"{kind}_constant_{n}_{alpha:g}"
                completion = json.loads((folder/"complete.json").read_text())
                for filename in ("parameters.jsonl", "prediction_moments.parquet", "replications.csv"):
                    path = folder/filename
                    digest = sha(path)
                    assert digest == completion["outputs"][filename]
                    bindings[str(path.relative_to(ROOT))] = digest
                params = [json.loads(line) for line in (folder/"parameters.jsonl").read_text().splitlines()]
                moments = pd.read_parquet(folder/"prediction_moments.parquet")
                states = moments[moments.method == "Raw"].sort_values("state")
                sigma = states.sigma.to_numpy()
                assert len(sigma) == 1024 and len(params) == 500
                oracle = sigma*z
                assert np.max(np.abs(oracle-states.oracle_quantile.to_numpy())) < 1e-14
                raw = oracle+optimal_shift
                raw_risk = return_risk(kind, raw, sigma, alpha).mean()
                curvature = innovation_density*np.mean(1/sigma)
                saved = pd.read_csv(folder/"replications.csv")
                for method, key in (("Shift-CP", "shift_cp"), ("Shift-ERM", "shift_erm")):
                    shifts = np.array([record["parameters"][key] for record in params])
                    scalar_estimates += len(shifts)
                    recomputed = return_risk(kind, raw[None, :]-shifts[:, None],
                                             sigma[None, :], alpha).mean(axis=1)
                    original = saved[saved.method == method].sort_values("replication").expected_QS.to_numpy()
                    error = float(np.max(np.abs(recomputed-original)))
                    max_replay_error = max(max_replay_error, error)
                    assert error < 2e-14
                    mse = np.mean((shifts-optimal_shift)**2)
                    variance = np.var(shifts)
                    bias_squared = (shifts.mean()-optimal_shift)**2
                    assert abs(mse-variance-bias_squared) < 1e-16
                    exact_difference = float(np.mean(recomputed)-raw_risk)
                    quadratic = .5*curvature*(mse-optimal_shift**2)
                    rows.append({"innovation": kind, "n_cal": n, "alpha": alpha,
                                 "method": method, "test_state_curvature": curvature,
                                 "shift_bias_squared": bias_squared, "shift_variance": variance,
                                 "shift_MSE": mse,
                                 "exact_QS_difference_x1e4": 1e4*exact_difference,
                                 "quadratic_QS_difference_x1e4": 1e4*quadratic,
                                 "direction_matches": bool((exact_difference < 0) == (quadratic < 0))})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"saved_mc_risk_decomposition.csv", index=False)
    return {"configurations": len(rows)//2, "scalar_estimates_reused": scalar_estimates,
            "independent_calibration_histories_in_original_design": 1000,
            "max_saved_risk_replay_error": max_replay_error,
            "quadratic_direction_matches": int(frame.direction_matches.sum()),
            "quadratic_comparisons": len(frame), "inputs": bindings}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = [ROOT/"source/main_R8.tex", ROOT/"source/supplement_R8.tex",
                 ROOT/"Manuscript_R8.pdf", ROOT/"source/main_R8.pdf",
                 ROOT/"source/supplement_R8.pdf", ROOT/"docs/CONFERENCE_PROGRAM_ABSTRACT.md"]
    canonical += sorted((ROOT/"source/sections_r8").glob("*.tex"))
    protected = {str(path.relative_to(ROOT)): sha(path) for path in canonical}
    report = {"producer_sha256": sha(__file__), "random_draws_generated": 0,
              "models_fitted": 0, "manuscript_changed": False,
              "identity": identity_checks(), "uniform": finite_uniform_checks(),
              "normal_order_statistic": normal_order_statistic_checks(),
              "finite_normal_frontier": exact_frontier(),
              "saved_mc": saved_simulation_checks()}
    assert all(sha(ROOT/path) == digest for path, digest in protected.items())
    report["unchanged_canonical_files"] = protected
    report["outputs"] = {path.name: sha(path) for path in sorted(OUT.glob("*.csv"))}
    (OUT/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    compact = {key: value for key, value in report.items()
               if key not in ("unchanged_canonical_files", "outputs")}
    compact["saved_mc"] = {key: value for key, value in report["saved_mc"].items() if key != "inputs"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
