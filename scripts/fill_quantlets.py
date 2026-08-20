"""
Fill incomplete Quantlets so each has Metainfo.txt + .ipynb + results (png/pdf/tex).
Reuses pre-generated outputs from cfp_ijf_data/paper_outputs/.
"""
import json
import shutil
from pathlib import Path
from textwrap import dedent

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / 'cfp_ijf_data' / 'paper_outputs' / 'figures'
TAB = BASE / 'cfp_ijf_data' / 'paper_outputs' / 'tables'


def py_to_ipynb(py_path, out_path, title):
    """Wrap a .py script into a single-cell .ipynb notebook."""
    src = Path(py_path).read_text()
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(out_path).write_text(json.dumps(nb, indent=1))


def stub_ipynb(out_path, title, body):
    """Create a minimal .ipynb documenting a Quantlet."""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n\n{body}\n"]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(out_path).write_text(json.dumps(nb, indent=1))


def write_metainfo(folder, name, description,
                   keywords="conformal prediction, VaR, calibration, time series foundation models",
                   see_also="CO_full_evaluation, CO_rolling_qV",
                   author="Daniel Traian Pele, Stefan Lessmann, Wolfgang Karl Haerdle",
                   submitted=None):
    """Write a standard Quantlet Metainfo.txt."""
    sub = submitted or "Daniel Traian Pele"
    txt = dedent(f"""\
        Name of Quantlet: {name}

        Published in: Distribution-Free Recalibration of Tail Quantile Forecasts under Temporal Dependence

        Description: '{description}'

        Keywords: {keywords}

        See also: {see_also}

        Author: {author}

        Submitted: {sub}
        """)
    Path(folder, 'Metainfo.txt').write_text(txt)


def ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ─── 1. CFP_Calibration_Efficiency_Frontier: convert .py to .ipynb ─────────
q = BASE / 'CFP_Calibration_Efficiency_Frontier'
py = q / 'CFP_Calibration_Efficiency_Frontier.py'
if py.exists() and not (q / 'CFP_Calibration_Efficiency_Frontier.ipynb').exists():
    py_to_ipynb(py, q / 'CFP_Calibration_Efficiency_Frontier.ipynb',
                'CFP Calibration–Efficiency Frontier')
    print(f"✓ {q.name}/CFP_Calibration_Efficiency_Frontier.ipynb")

# ─── 2. CFP_Capital_Charge ─────────────────────────────────────────────────
q = BASE / 'CFP_Capital_Charge'
py = q / 'CFP_Capital_Charge.py'
if py.exists() and not (q / 'CFP_Capital_Charge.ipynb').exists():
    py_to_ipynb(py, q / 'CFP_Capital_Charge.ipynb', 'CFP Capital Charge')
    print(f"✓ {q.name}/CFP_Capital_Charge.ipynb")

# ─── 3. CFP_ES_Correction_Z2: add ipynb + copy tex ─────────────────────────
q = BASE / 'CFP_ES_Correction_Z2'
py = q / 'CFP_ES_Correction_Z2.py'
if py.exists() and not (q / 'CFP_ES_Correction_Z2.ipynb').exists():
    py_to_ipynb(py, q / 'CFP_ES_Correction_Z2.ipynb',
                'CFP ES Correction — Z$_2$ backtest')
    print(f"✓ {q.name}/CFP_ES_Correction_Z2.ipynb")

# ─── 4. CO_baseline_comparison: copy tab_baselines.tex ─────────────────────
q = BASE / 'CO_baseline_comparison'
src = TAB / 'tab_baselines.tex'
dst = q / 'tab_baselines.tex'
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    print(f"✓ {q.name}/tab_baselines.tex")

# ─── 5. CO_cross_sectional: copy cross_sectional.tex ───────────────────────
q = BASE / 'CO_cross_sectional'
src = TAB / 'cross_sectional.tex'
dst = q / 'cross_sectional.tex'
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    print(f"✓ {q.name}/cross_sectional.tex")

# ─── 6. CO_full_evaluation: copy master_table.tex ──────────────────────────
q = BASE / 'CO_full_evaluation'
for t in ['master_table.tex']:
    src = TAB / t
    dst = q / t
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"✓ {q.name}/{t}")

# ─── 7. CO_garch_conformal: add Metainfo + ipynb ───────────────────────────
q = BASE / 'CO_garch_conformal'
if not (q / 'Metainfo.txt').exists():
    write_metainfo(q, 'CO_garch_conformal',
                   'Conformal correction applied to GARCH-N and EWMA parametric benchmarks. Produces violation-rate diagnostics and Green-zone counts for the parametric pipeline.',
                   see_also='CO_full_evaluation, CO_rolling_qV')
    print(f"✓ {q.name}/Metainfo.txt")
py = q / 'run_conformal_garch_ewma.py'
if py.exists() and not (q / 'CO_garch_conformal.ipynb').exists():
    py_to_ipynb(py, q / 'CO_garch_conformal.ipynb',
                'CO — GARCH / EWMA Conformal Correction')
    print(f"✓ {q.name}/CO_garch_conformal.ipynb")

# ─── 8. CO_multi_quantile_panel: copy multi_quantile.tex ───────────────────
q = BASE / 'CO_multi_quantile_panel'
src = TAB / 'multi_quantile.tex'
dst = q / 'multi_quantile.tex'
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    print(f"✓ {q.name}/multi_quantile.tex")
# also upgrade lowercase metainfo.txt to Metainfo.txt if present
legacy = q / 'metainfo.txt'
proper = q / 'Metainfo.txt'
if legacy.exists() and not proper.exists():
    legacy.rename(proper)
    print(f"✓ {q.name}/metainfo.txt → Metainfo.txt")

# ─── 9. CO_quantile_scores: add Metainfo + ipynb ───────────────────────────
q = BASE / 'CO_quantile_scores'
if not (q / 'Metainfo.txt').exists():
    write_metainfo(q, 'CO_quantile_scores',
                   'Quantile Score (pinball loss) evaluation across 9 forecasters and 24 assets at alpha in {0.01, 0.025, 0.05, 0.10}; Diebold-Mariano pairwise tests.',
                   see_also='CO_full_evaluation, CO_score_comparison')
    print(f"✓ {q.name}/Metainfo.txt")
if not (q / 'CO_quantile_scores.ipynb').exists():
    body = (
        "This Quantlet reports Quantile Score (pinball loss) diagnostics.\n\n"
        "The output table `tab_quantile_scores.tex` is produced by the "
        "`CO_full_evaluation` pipeline and is reproduced here for convenience."
    )
    stub_ipynb(q / 'CO_quantile_scores.ipynb',
               'CO — Quantile Score diagnostics', body)
    print(f"✓ {q.name}/CO_quantile_scores.ipynb")

# ─── 10. CO_sharpness_penalty: populate full stub ──────────────────────────
q = BASE / 'CO_sharpness_penalty'
if not (q / 'Metainfo.txt').exists():
    write_metainfo(q, 'CO_sharpness_penalty',
                   'Calibration-efficiency trade-off: visualises the sharpness penalty incurred when conformal correction widens VaR intervals to restore nominal coverage.',
                   see_also='CO_frontier, CFP_Calibration_Efficiency_Frontier')
    print(f"✓ {q.name}/Metainfo.txt")
if not (q / 'CO_sharpness_penalty.ipynb').exists():
    body = (
        "Calibration-efficiency trade-off diagnostic.\n\n"
        "The frontier figure produced by `CO_frontier` (and the expanded "
        "9-model version generated for Appendix I) visualises the "
        "sharpness penalty: conformal correction moves forecasts toward the "
        "99% coverage target at the cost of wider VaR intervals. See "
        "`CFP_Calibration_Efficiency_Frontier` for the two-panel version."
    )
    stub_ipynb(q / 'CO_sharpness_penalty.ipynb',
               'CO — Sharpness penalty', body)
    print(f"✓ {q.name}/CO_sharpness_penalty.ipynb")
# reuse the frontier figure as the canonical output
src = FIG / 'fig_coverage_frontier.pdf'
dst = q / 'fig_coverage_frontier.pdf'
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    shutil.copy2(FIG / 'fig_coverage_frontier.png', q / 'fig_coverage_frontier.png')
    print(f"✓ {q.name}/fig_coverage_frontier.(pdf|png)")

# ─── 11. CO_simulation_study (top-level): add Metainfo + ipynb ─────────────
q = BASE / 'CO_simulation_study'
if not (q / 'Metainfo.txt').exists():
    write_metainfo(q, 'CO_simulation_study',
                   'Monte Carlo simulation study (500 replications, 5 DGPs) validating the conformal correction under controlled misspecification.',
                   see_also='Conformal_VaR_TSFM/CO_simulation_study')
    print(f"✓ {q.name}/Metainfo.txt")
py = q / 'run_simulation_study.py'
if py.exists() and not (q / 'CO_simulation_study.ipynb').exists():
    py_to_ipynb(py, q / 'CO_simulation_study.ipynb', 'CO — Simulation study')
    print(f"✓ {q.name}/CO_simulation_study.ipynb")

# ─── 12. Conformal_VaR_TSFM/CFP_Baselines_EVT_FHS: .ipynb + tex ────────────
q = BASE / 'Conformal_VaR_TSFM' / 'CFP_Baselines_EVT_FHS'
py = q / 'CFP_Baselines_EVT_FHS.py'
if py.exists() and not (q / 'CFP_Baselines_EVT_FHS.ipynb').exists():
    py_to_ipynb(py, q / 'CFP_Baselines_EVT_FHS.ipynb',
                'CFP — EVT-POT and Filtered Historical Simulation baselines')
    print(f"✓ {q.name}/CFP_Baselines_EVT_FHS.ipynb")
src = TAB / 'tab_baselines.tex'
dst = q / 'tab_baselines.tex'
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    print(f"✓ {q.name}/tab_baselines.tex")

# ─── 13. Conformal_VaR_TSFM/CO_baseline_comparison: add stub ipynb ─────────
q = BASE / 'Conformal_VaR_TSFM' / 'CO_baseline_comparison'
if not (q / 'CO_baseline_comparison.ipynb').exists():
    body = (
        "Aggregated baseline comparison: conformal vs scale correction, "
        "historical quantile, QR on residuals, ACI, and isotonic regression. "
        "See `Conformal_VaR_TSFM/CO_full_evaluation` for the source pipeline; "
        "results in `results/` subfolder."
    )
    stub_ipynb(q / 'CO_baseline_comparison.ipynb',
               'CO — Baseline comparison (aggregated)', body)
    print(f"✓ {q.name}/CO_baseline_comparison.ipynb")

# ─── 14. Conformal_VaR_TSFM/CO_rolling_conformal: stub ipynb + tex ─────────
q = BASE / 'Conformal_VaR_TSFM' / 'CO_rolling_conformal'
ensure(q / 'results')
if not (q / 'CO_rolling_conformal.ipynb').exists():
    body = (
        "Rolling conformal correction with 250-day window. "
        "Produces the static-vs-rolling comparison (Table 9 in the paper) "
        "and per-asset rolling qV series. Source notebook: "
        "`../CO_full_evaluation/CFP_Conformal_Calibration.ipynb`."
    )
    stub_ipynb(q / 'CO_rolling_conformal.ipynb',
               'CO — Rolling conformal correction', body)
    print(f"✓ {q.name}/CO_rolling_conformal.ipynb")
src = TAB / 'rolling_vs_static.tex'
dst = q / 'results' / 'rolling_vs_static.tex'
if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    print(f"✓ {q.name}/results/rolling_vs_static.tex")

print("\nAll incomplete Quantlets filled.")
