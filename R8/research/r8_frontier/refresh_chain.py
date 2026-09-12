"""Exact Markov tail counts plus deterministic risk quadrature; no simulation."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from check_risk_bridge import normal_check_expectation, rank

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"artifacts/r8_frontier"


def count_cdf(v, n, k, retention):
    """P(at least k scores <= threshold), with marginal probability v.

    Retention reuses the previous uniform score; otherwise an independent
    uniform score is drawn by the mathematical law. No paths are sampled.
    Track the number of above-threshold scores, discarding paths that can
    no longer meet the count cutoff. Axis 0 indexes quadrature thresholds.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    max_above = n-k
    below = np.zeros((len(v), max_above+1))
    above = np.zeros_like(below)
    below[:, 0] = v
    if max_above >= 1:
        above[:, 1] = 1-v
    p00 = (retention+(1-retention)*v)[:, None]
    p01 = ((1-retention)*(1-v))[:, None]
    p10 = ((1-retention)*v)[:, None]
    p11 = (retention+(1-retention)*(1-v))[:, None]
    for _ in range(1, n):
        next_below = below*p00+above*p10
        next_above = np.zeros_like(above)
        next_above[:, 1:] = below[:, :-1]*p01+above[:, :-1]*p11
        below, above = next_below, next_above
    probability = below.sum(axis=1)+above.sum(axis=1)
    assert probability.min() >= -1e-12 and probability.max() <= 1+1e-12
    return np.clip(probability, 0., 1.)


def enumeration(v, n, k, retention):
    total = 0.
    for path in itertools.product((0, 1), repeat=n):
        if sum(path) < k:
            continue
        probability = v if path[0] else 1-v
        for previous, current in zip(path, path[1:]):
            stationary = v if current else 1-v
            probability *= retention*(previous == current)+(1-retention)*stationary
        total += probability
    return total


def validate_count_law():
    errors = {"iid_beta": 0., "maximum_rank": 0., "enumeration": 0.}
    grid = np.array([.001, .01, .1, .5, .9, .99, .999])
    for n in (6, 125, 250):
        for k in (1, n//2, n):
            result = count_cdf(grid, n, k, 0.)
            errors["iid_beta"] = max(errors["iid_beta"],
                float(np.max(np.abs(result-special.betainc(k, n+1-k, grid)))))
        for retention in (0., .25, .5, .75):
            result = count_cdf(grid, n, n, retention)
            exact = grid*(retention+(1-retention)*grid)**(n-1)
            errors["maximum_rank"] = max(errors["maximum_rank"], float(np.max(np.abs(result-exact))))
    for retention in (0., .25, .5, .75):
        for k in range(1, 7):
            result = count_cdf(grid, 6, k, retention)
            exact = np.array([enumeration(v, 6, k, retention) for v in grid])
            errors["enumeration"] = max(errors["enumeration"], float(np.max(np.abs(result-exact))))
    assert max(errors.values()) < 2e-12
    return errors


def estimation_cost(n, alpha, k, retention, points):
    nodes, weights = special.roots_legendre(points)
    z, p = stats.norm.ppf(1-alpha), 1-alpha
    cost = 0.
    for low, high, lower_half in ((-12.-z, 0., True), (0., 12.-z, False)):
        u = (nodes+1)*(high-low)/2+low
        v = stats.norm.cdf(u+z)
        h = count_cdf(v, n, k, retention)
        integrand = (p-v)*h if lower_half else (v-p)*(1-h)
        cost += float(np.dot(weights, integrand)*(high-low)/2)
    return cost


def make_plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "irfa-r8-frontier-20260909",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.5))
    colours = {0.: "#007F9E", .25: "#6841D9", .5: "#E03F7D", .75: "#E47700"}
    for ax, alpha in zip(axes, (.01, .05)):
        for retention, colour in colours.items():
            g = data[(data.alpha == alpha)&(data.retention == retention)
                     &(data.method == "Shift-CP")].sort_values("n")
            ax.plot(g.n, 100*g.upper_raw_violation_rate, color=colour, marker="o", lw=2)
            ax.plot(g.n, 100*g.lower_raw_violation_rate, color=colour, marker="o", lw=2)
        ax.axhline(100*alpha, color="#5C6470", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_xticks([125, 250, 500, 1000], ["125", "250", "500", "1,000"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(bottom=0)
        ax.set_title(f"Nominal tail: {100*alpha:g}%", loc="left", weight="bold")
        ax.set_xlabel("Calibration observations")
        ax.set_ylabel("Population raw violation rate (%)")
        ax.grid(axis="y", alpha=.16)
    handles = [Line2D([0], [0], color=c, marker="o", lw=2,
                     label=f"Hit persistence {r:g}" + (" (iid)" if r == 0 else ""))
               for r, c in colours.items()]
    fig.suptitle("Tail dependence changes when recalibration pays", fontsize=15,
                 x=.08, ha="left", y=.99, weight="bold")
    fig.text(.08, .925, "Mathematical benchmark: Normal score marginals · conformal rank · independent evaluation", fontsize=9.5)
    fig.text(.08, .875, "Between the two boundaries of each colour, correction worsens expected loss; outside them, it improves it.", fontsize=9)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=4, frameon=False)
    fig.text(.5, .005, "Deterministic Markov-count calculation; points are evaluated sample sizes. These are not financial deployment cutoffs.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.08, right=.985, top=.78, bottom=.22, wspace=.25)
    fig.savefig(OUT/"dependence_break_even.png", dpi=180, transparent=True, bbox_inches="tight")
    fig.savefig(OUT/"dependence_break_even.svg", transparent=True, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = validate_count_law()
    rows = []
    for alpha in (.01, .05):
        z = stats.norm.ppf(1-alpha)
        minimum = normal_check_expectation(z, 1., alpha)
        for n in (125, 250, 500, 1000):
            for retention in (0., .25, .5, .75):
                for conformal in (False, True):
                    k = rank(n, alpha, conformal)
                    cost = estimation_cost(n, alpha, k, retention, 512)
                    lower_cost = estimation_cost(n, alpha, k, retention, 256)
                    def gain(c):
                        return normal_check_expectation(z-c, 1., alpha)-minimum-cost
                    left = optimize.brentq(gain, -20., 0., xtol=1e-12)
                    right = optimize.brentq(gain, 0., 20., xtol=1e-12)
                    rows.append({"alpha": alpha, "n": n, "retention": retention,
                                 "method": "Shift-CP" if conformal else "Shift-ERM", "rank": k,
                                 "hit_variance_factor": (1+retention)/(1-retention),
                                 "estimation_cost": cost,
                                 "quadrature_resolution_difference": abs(cost-lower_cost),
                                 "lower_shift_boundary": left, "upper_shift_boundary": right,
                                 "lower_raw_violation_rate": stats.norm.sf(z-left),
                                 "upper_raw_violation_rate": stats.norm.sf(z-right)})
    frame = pd.DataFrame(rows)
    assert frame.quadrature_resolution_difference.max() < 1e-9
    iid = pd.read_csv(OUT/"normal_exact_break_even.csv")
    paired = frame[frame.retention == 0].merge(iid, on=["alpha", "n", "method"], suffixes=("_chain", "_beta"))
    iid_error = float(np.max(np.abs(paired.estimation_cost_chain-paired.estimation_cost_beta)))
    assert len(paired) == 16 and iid_error < 5e-9
    frame.to_csv(OUT/"refresh_chain.csv", index=False)
    make_plot(frame)
    # P(any score lies beyond x) <= n P(single score lies beyond x).
    # For the two omitted risk tails this bounds the integral by
    # 2n * E[(N(0,1)-12)_+], using the Gaussian positive-part identity.
    truncation_bound = 2*1000*(stats.norm.pdf(12)-12*stats.norm.sf(12))
    paths = [Path(__file__), Path(__file__).with_name("check_risk_bridge.py"),
             Path(__file__).with_name("REFRESH_CHAIN_PROTOCOL.md"),
             OUT/"normal_exact_break_even.csv"]
    report = {"random_draws_generated": 0, "models_fitted": 0, "configurations": len(frame),
              "count_checks_max_errors": checks, "max_iid_risk_difference": iid_error,
              "max_quadrature_resolution_difference": frame.quadrature_resolution_difference.max(),
              "gaussian_truncation_upper_bound": truncation_bound,
              "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [OUT/"refresh_chain.csv", OUT/"dependence_break_even.png", OUT/"dependence_break_even.svg"]}}
    (OUT/"refresh_chain_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
