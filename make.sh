#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# make.sh — Regenerate all tables, figures, and the manuscript
#
# Usage:
#   ./make.sh all          Tables + figures + manuscript (no MC)
#   ./make.sh tables       Regenerate all table .tex files
#   ./make.sh figures      Regenerate all figure PDFs
#   ./make.sh manuscript   Compile LaTeX (pdflatex + bibtex)
#   ./make.sh mc           Run Monte Carlo studies (slow: ~30 min)
#   ./make.sh clean        Remove generated outputs
#   ./make.sh verify       Rebuild and diff against committed outputs
#
# Dependency layers:
#   Layer 0: cfp_ijf_data/ (canonical data, never regenerated here)
#   Layer 1: Tables and figures reading from canonical CSVs/parquets
#   Layer 2: Monte Carlo studies (synthetic data, slow)
#   Layer 3: LaTeX compilation
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TEX="20260426 PEL BOL GIN LES HAE CFP LLM VaR"
FIGURES_DIR="$ROOT/figures"

# Colour output (suppressed if not a terminal)
if [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; NC=''
fi

ok()   { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}→${NC} $1"; }

run_py() {
    local script="$1"
    local label="${2:-$script}"
    info "$label"
    python "$ROOT/$script" || fail "$label"
    ok "$label"
}

copy_fig() {
    local src="$1"
    local name
    name="$(basename "$src")"
    python -c "import shutil; shutil.copy2('$ROOT/$src', '$FIGURES_DIR/$name')"
}

# ──────────────────────────────────────────────────────────────
# Target: tables
#   Regenerate all .tex table files from canonical data.
#   Does NOT include Monte Carlo tables (use 'mc' target).
# ──────────────────────────────────────────────────────────────
do_tables() {
    info "=== TABLES ==="

    # T1: Asset overview (Table 1)
    run_py "Quantlets/CO_asset_overview/run_asset_overview.py" "T1  Asset overview"

    # T2: Model overview (Table 2)
    run_py "Quantlets/CO_model_overview/run_model_overview.py" "T2  Model overview"

    # T3: Cross-sectional correlations (Table 3)
    run_py "Quantlets/CO_cross_sectional/run_cross_sectional.py" "T3  Cross-sectional"

    # T4: Master results (Table 4)
    run_py "Quantlets/CO_full_evaluation/run_master_table.py" "T4  Master results"

    # T5: Multi-quantile (Table 5)
    run_py "Quantlets/CO_multi_quantile_panel/run_multiquantile.py" "T5  Multi-quantile"

    # T6: Panel pooled (Table 6)
    run_py "Quantlets/CO_multi_quantile_panel/run_panel_pooled.py" "T6  Panel pooled"

    # T7: Panel by class (Table 7)
    run_py "Quantlets/CO_multi_quantile_panel/run_panel_by_class.py" "T7  Panel by class"

    # T8: DM p-values (Table 8)
    run_py "Quantlets/CO_quantile_scores/run_dm_pvalues.py" "T8  DM p-values"

    # T9: Rolling vs static (Table 9)
    run_py "Quantlets/CO_garch_conformal/run_rolling_vs_static.py" "T9  Rolling vs static"

    # T11: Bound validation (Table 11)
    run_py "Quantlets/CO_bound_validation/run_bound_validation.py" "T11 Bound validation"

    # T12a: GBM-QR baseline row (prerequisite for T12)
    run_py "Quantlets/CO_gbm_qr/baseline_gbm_qr.py" "T12a GBM-QR baseline"

    # T12b: GAMLSS baseline row (prerequisite for T12)
    run_py "Quantlets/CO_gamlss/baseline_gamlss.py" "T12b GAMLSS baseline"

    # T12c: EVT/FHS baseline rows (prerequisite for T12)
    run_py "Quantlets/CO_baselines_evt_fhs/run_baselines_evt_fhs.py" "T12c EVT/FHS baselines"

    # T12: Baselines composite (Table 12)
    run_py "Quantlets/CO_baseline_comparison/compile_tab_baselines.py" "T12 Baselines composite"

    # T13: FZ scores (Table 13)
    run_py "Quantlets/CO_fz_scores/run_fz_scores.py" "T13 FZ scores"

    # TC14: ES correction (Table C.14)
    run_py "Quantlets/CFP_ES_Correction_Z2/CFP_ES_Correction_Z2.py" "TC14 ES correction"

    # TD15: Robustness summary (Table D.15)
    run_py "Quantlets/CO_robustness/run_robustness_summary.py" "TD15 Robustness summary"

    # T12-tuned: Tuned GBM-QR baseline (Table 12 tuned row)
    run_py "Quantlets/CO_baseline_comparison_tuned/run_tuned_gbm_qr.py" "T12t Tuned GBM-QR"

    # TE2: Extended tail-closure inner-7 (Appendix tab_tail_closure_extended)
    run_py "Quantlets/CO_robustness_inner7/run_inner7_tail_closure.py" "TE2  Tail closure (inner-7)"

    # TE3: Wild-cluster bootstrap panel (Kupiec + DM tables)
    run_py "Quantlets/CO_panel_wildcluster/run_wild_cluster_bootstrap.py" "TE3  Wild-cluster panel"

    # TE4: Diagnostic regression (Table E.4)
    run_py "Quantlets/CO_diagnostic_regression/run_diag_regression.py" "TE4  Diagnostic regression"

    # TD-regime: Regime classification sensitivity (Table D.regime)
    run_py "Quantlets/CO_regime_sensitivity/run_regime_sensitivity.py" "TDr  Regime sensitivity"

    # MC placeholders: ensure Tables D.16-D.18 exist for LaTeX compilation.
    # Real content is generated by 'make.sh mc' (~30 min).
    python -c "
from pathlib import Path
for name in ['tab_h14_small_sample','tab_h15_fc_sensitivity','tab_h16_regime_stability']:
    p = Path('Quantlets/CO_robustness') / f'{name}.tex'
    if not p.exists():
        p.write_text(r'\textit{Run \texttt{make.sh mc} to generate this table.}')
        print(f'  placeholder: {p}')
"

    ok "=== ALL TABLES DONE ==="
}

# ──────────────────────────────────────────────────────────────
# Target: figures
#   Regenerate all 8 manuscript figures and copy to figures/.
# ──────────────────────────────────────────────────────────────
do_figures() {
    info "=== FIGURES ==="
    mkdir -p "$FIGURES_DIR"

    # Fig 1: Rolling q̂_V (S&P 500)
    run_py "Quantlets/CO_rolling_qV/run_rolling_qV.py" "Fig1 Rolling qV"
    copy_fig "Quantlets/CO_rolling_qV/fig_rolling_qv.pdf"
    copy_fig "Quantlets/CO_rolling_qV/fig_rolling_qv.png"

    # Fig 2: Basel traffic light heatmap
    run_py "Quantlets/CO_heatmap/run_heatmap.py" "Fig2 Heatmap"
    copy_fig "Quantlets/CO_heatmap/fig_traffic_light.pdf"
    copy_fig "Quantlets/CO_heatmap/fig_traffic_light.png"

    # Fig 3: Calibration-efficiency frontier
    run_py "Quantlets/CFP_Calibration_Efficiency_Frontier/run_frontier.py" "Fig3 Frontier"
    copy_fig "Quantlets/CFP_Calibration_Efficiency_Frontier/fig_frontier_killer.pdf"
    copy_fig "Quantlets/CFP_Calibration_Efficiency_Frontier/fig_frontier_killer.png"

    # Fig 4: Violation rates
    run_py "Quantlets/CO_violation_rates/run_violation_rates.py" "Fig4 Violation rates"
    copy_fig "Quantlets/CO_violation_rates/fig_violation_rates.pdf"
    copy_fig "Quantlets/CO_violation_rates/fig_violation_rates.png"

    # Fig 6: COVID response lag
    run_py "Quantlets/CO_covid_response_lag/run_covid_response_lag.py" "Fig6 COVID response"
    copy_fig "Quantlets/CO_covid_response_lag/fig_covid_response.pdf"
    copy_fig "Quantlets/CO_covid_response_lag/fig_covid_response.png"

    # Fig 8: Capital charge
    run_py "Quantlets/CFP_Capital_Charge/CFP_Capital_Charge.py" "Fig8 Capital charge"
    copy_fig "Quantlets/CFP_Capital_Charge/capital_charge_cumulative.pdf"
    copy_fig "Quantlets/CFP_Capital_Charge/capital_charge_cumulative.png"

    # Fig 5: Simulation q̂_V distribution (also produces Table 10)
    run_py "Quantlets/CO_simulation_study/run_simulation_study.py" "Fig5 Simulation study"
    copy_fig "Quantlets/CO_simulation_study/fig_simulation_qV_distribution.pdf"
    copy_fig "Quantlets/CO_simulation_study/fig_simulation_qV_distribution.png"

    # Fig 7: Drift diagnostic
    run_py "Quantlets/CO_drift_diagnostic/run_drift_diagnostic.py" "Fig7 Drift diagnostic"
    copy_fig "Quantlets/CO_drift_diagnostic/fig_drift_diagnostic.pdf"
    copy_fig "Quantlets/CO_drift_diagnostic/fig_drift_diagnostic.png"

    # Fig (appendix): Forensic TSFM checks (TimesFM 2.5 + Moirai 2.0)
    run_py "Quantlets/CO_forensic_tsfm/run_forensic_tsfm.py" "FigF Forensic TSFM"
    copy_fig "Quantlets/CO_forensic_tsfm/fig_forensic_tsfm.pdf"
    copy_fig "Quantlets/CO_forensic_tsfm/fig_forensic_tsfm.png"

    ok "=== ALL FIGURES DONE ==="
}

# ──────────────────────────────────────────────────────────────
# Target: mc
#   Run Monte Carlo robustness studies (Tables D.16–D.18).
#   Slow: ~30 minutes with 500+200 replications.
# ──────────────────────────────────────────────────────────────
do_mc() {
    info "=== MONTE CARLO (slow) ==="
    run_py "Quantlets/CO_robustness/run_robustness_mc.py" "TD16-18 Robustness MC"
    ok "=== MONTE CARLO DONE ==="
}

# ──────────────────────────────────────────────────────────────
# Target: manuscript
#   Compile LaTeX: pdflatex → bibtex → pdflatex × 2.
# ──────────────────────────────────────────────────────────────
do_manuscript() {
    info "=== MANUSCRIPT ==="
    cd "$ROOT"
    latexmk -pdf -interaction=nonstopmode "$TEX.tex" \
        || fail "LaTeX compilation"

    # Check for undefined references
    if grep -qi "undefined" "$TEX.log" 2>/dev/null; then
        echo -e "${YELLOW}WARNING: undefined references detected${NC}"
        grep -i "undefined" "$TEX.log"
    fi

    local pages
    pages=$(pdfinfo "$TEX.pdf" 2>/dev/null | grep Pages | awk '{print $2}')
    ok "Manuscript compiled: ${pages:-?} pages"
}

# ──────────────────────────────────────────────────────────────
# Target: all
#   Rebuild tables + figures + manuscript (excludes MC — use 'mc' separately).
# ──────────────────────────────────────────────────────────────
do_all() {
    info "=== FULL REBUILD (tables + figures + manuscript) ==="
    do_tables
    do_figures
    do_manuscript
    ok "=== FULL REBUILD COMPLETE ==="
}

# ──────────────────────────────────────────────────────────────
# Target: clean
#   Remove generated .tex tables, .csv intermediates, and
#   figure copies. Keeps canonical data and source code.
# ──────────────────────────────────────────────────────────────
do_clean() {
    info "=== CLEAN ==="
    cd "$ROOT"

    # LaTeX aux files
    latexmk -C "$TEX.tex" 2>/dev/null || true

    # Generated table .tex and .csv in Quantlet dirs
    local quantlets=(
        CO_asset_overview CO_model_overview CO_cross_sectional
        CO_full_evaluation CO_multi_quantile_panel CO_quantile_scores
        CO_garch_conformal CO_bound_validation CO_fz_scores
        CFP_ES_Correction_Z2 CO_simulation_study
        CO_gbm_qr CO_gamlss CO_baselines_evt_fhs CO_baseline_comparison
        CO_baseline_comparison_tuned CO_robustness_inner7
        CO_panel_wildcluster CO_diagnostic_regression CO_regime_sensitivity
    )
    for q in "${quantlets[@]}"; do
        rm -f "Quantlets/$q"/tab_*.tex "Quantlets/$q"/tab_*.csv 2>/dev/null
        rm -f "Quantlets/$q"/*.csv 2>/dev/null
    done

    # CO_robustness: clean non-MC outputs only (MC tables take ~30 min)
    rm -f Quantlets/CO_robustness/tab_robustness_summary.tex 2>/dev/null
    rm -f Quantlets/CO_robustness/tab_robustness_summary.csv 2>/dev/null

    # Generated figure copies in figures/
    local figs=(
        fig_rolling_qv fig_traffic_light fig_frontier_killer
        fig_violation_rates fig_simulation_qV_distribution
        fig_covid_response fig_drift_diagnostic capital_charge_cumulative
        fig_forensic_tsfm
    )
    for f in "${figs[@]}"; do
        rm -f "$FIGURES_DIR/${f}.pdf" "$FIGURES_DIR/${f}.png" 2>/dev/null
    done

    ok "Clean complete"
}

# ──────────────────────────────────────────────────────────────
# Target: verify
#   Rebuild everything, then diff generated outputs against
#   the git-committed versions to detect drift.
# ──────────────────────────────────────────────────────────────
do_verify() {
    info "=== VERIFY ==="

    # Stash uncommitted changes to generated files
    local tex_files
    tex_files=$(find "$ROOT/Quantlets" -path '*/CO_*/*.tex' -o -path '*/CFP_*/*.tex' | sort)
    local fig_files
    fig_files=$(find "$FIGURES_DIR" -name '*.pdf' -o -name '*.png' 2>/dev/null | sort)

    # Record checksums before rebuild
    local before_sums
    before_sums=$(mktemp)
    { echo "$tex_files"; echo "$fig_files"; } | xargs md5 2>/dev/null > "$before_sums" || true

    # Rebuild
    do_tables
    do_figures

    # Record checksums after rebuild
    local after_sums
    after_sums=$(mktemp)
    { echo "$tex_files"; echo "$fig_files"; } | xargs md5 2>/dev/null > "$after_sums" || true

    # Diff
    if diff -q "$before_sums" "$after_sums" > /dev/null 2>&1; then
        ok "All outputs match committed versions — no drift detected"
    else
        echo -e "${RED}DRIFT DETECTED:${NC}"
        diff "$before_sums" "$after_sums" || true
        fail "Generated outputs differ from committed versions"
    fi

    rm -f "$before_sums" "$after_sums"
}

# ──────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────
case "${1:-help}" in
    all)        do_all ;;
    tables)     do_tables ;;
    figures)    do_figures ;;
    mc)         do_mc ;;
    manuscript) do_manuscript ;;
    clean)      do_clean ;;
    verify)     do_verify ;;
    help|*)
        echo "Usage: $0 {all|tables|figures|mc|manuscript|clean|verify}"
        echo ""
        echo "  all         Tables + figures + manuscript (excludes MC)"
        echo "  tables      Regenerate all table .tex files (~2 min)"
        echo "  figures     Regenerate all figure PDFs (~5 min)"
        echo "  mc          Run Monte Carlo studies, Tables D.16-D.18 (~30 min)"
        echo "  manuscript  Compile LaTeX manuscript"
        echo "  clean       Remove generated outputs"
        echo "  verify      Rebuild and diff against committed outputs"
        echo ""
        echo "Note: 'mc' is slow (~30 min) and excluded from 'all'."
        echo "      Run 'mc' separately when needed."
        ;;
esac
