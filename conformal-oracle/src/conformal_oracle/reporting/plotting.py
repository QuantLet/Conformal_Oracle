"""Matplotlib helpers for audit result visualisation."""

from __future__ import annotations

import matplotlib.pyplot as plt

from conformal_oracle.audit.single_rolling import RollingAuditResult


def plot_rolling_diagnostic(
    result: RollingAuditResult,
    figsize: tuple[float, float] = (12, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """Three-panel diagnostic plot for a rolling audit result.

    Panel 1: qV_roll(t) with zero line
    Panel 2: Replacement ratio R(t) with threshold line at 1.0
    Panel 3: Drift diagnostic (TV distance)
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(result.q_v_roll.index, result.q_v_roll.values, linewidth=0.8)
    axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.5)
    axes[0].set_ylabel("$\\hat{q}_V(t)$")
    axes[0].set_title("Rolling conformal correction")

    axes[1].plot(
        result.replacement_ratio.index,
        result.replacement_ratio.values,
        linewidth=0.8,
        color="C1",
    )
    axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.5)
    axes[1].set_ylabel("$R(t)$")
    axes[1].set_title("Replacement ratio")

    axes[2].plot(
        result.drift_diagnostic.index,
        result.drift_diagnostic.values,
        linewidth=0.8,
        color="C2",
    )
    axes[2].set_ylabel("TV distance")
    axes[2].set_title("Drift diagnostic $\\hat{\\delta}_w(t)$")
    axes[2].set_xlabel("Date")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
