"""
Monte Carlo Simulation Study + Pinball Loss + Calibration Plot
==============================================================
Part 1: 3 DGPs x 500 MC replications (well-specified, moderate, severe misspecification)
Part 2: Pinball (quantile) loss for simulation DGPs
Part 3: Calibration plot using empirical LLM data (GPT-3.5, GPT-4, GPT-4o on S&P 500)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/danielpele/Documents/CFP LLM VaR"
ASSETS_DIR = os.path.join(BASE_DIR, "Assets data")

# ================================================================
# PART 1: MONTE CARLO SIMULATION STUDY
# ================================================================

def simulate_garch_data(T, omega, alpha_garch, beta, df_innov=None, seed=None):
    """
    Simulate GARCH(1,1) returns.
    df_innov=None -> Normal innovations; df_innov=v -> Student-t(v) innovations.
    Returns: (returns, sigma2) arrays of length T.
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(T + 1)
    returns = np.zeros(T)
    sigma2[0] = omega / (1 - alpha_garch - beta)  # unconditional variance

    for t in range(T):
        if df_innov is None:
            z = rng.standard_normal()
        else:
            # Scale t-distribution to have unit variance
            z = rng.standard_t(df_innov) / np.sqrt(df_innov / (df_innov - 2))
        returns[t] = np.sqrt(sigma2[t]) * z
        sigma2[t + 1] = omega + alpha_garch * returns[t] ** 2 + beta * sigma2[t]

    return returns, sigma2[:T]


def simulate_sv_data(T, mu_sv=-10, phi_sv=0.98, sigma_eta=0.15, seed=None):
    """
    Simulate Stochastic Volatility returns.
    h_t = mu*(1-phi) + phi * h_{t-1} + eta_t, eta_t ~ N(0, sigma_eta^2)
    where h_t = log(sigma_t^2), so E[h_t] = mu (unconditional mean).
    r_t = sigma_t * epsilon_t, epsilon_t ~ N(0,1)
    With mu=-10: sigma ~ exp(-5) ~ 0.0067 (realistic daily vol).
    Returns: (returns, sigma2) arrays of length T.
    """
    rng = np.random.default_rng(seed)
    log_sigma2 = np.zeros(T)
    returns = np.zeros(T)

    # Stationary initialization: unconditional mean = mu, variance = sigma_eta^2 / (1 - phi^2)
    uncond_var = sigma_eta ** 2 / (1 - phi_sv ** 2)
    log_sigma2[0] = mu_sv + np.sqrt(uncond_var) * rng.standard_normal()

    for t in range(T):
        sigma_t = np.exp(log_sigma2[t] / 2.0)
        returns[t] = sigma_t * rng.standard_normal()
        if t < T - 1:
            log_sigma2[t + 1] = mu_sv * (1 - phi_sv) + phi_sv * log_sigma2[t] + sigma_eta * rng.standard_normal()

    sigma2 = np.exp(log_sigma2)
    return returns, sigma2


def simulate_regime_switching_data(T, p11=0.98, p22=0.95,
                                    mu1=0.0003, sigma1=0.01,
                                    mu2=-0.001, sigma2_rs=0.04, seed=None):
    """
    Simulate Markov Regime-Switching returns.
    s_t in {1, 2}: P(1->1)=p11, P(2->2)=p22
    Regime 1: low vol, Regime 2: high vol.
    Returns: (returns, sigma2) arrays of length T.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    states = np.zeros(T, dtype=int)

    # Stationary distribution: pi_1 = (1 - p22) / (2 - p11 - p22)
    pi1 = (1 - p22) / (2 - p11 - p22)
    states[0] = 1 if rng.random() < pi1 else 2

    for t in range(T):
        if states[t] == 1:
            returns[t] = mu1 + sigma1 * rng.standard_normal()
            sigma2[t] = sigma1 ** 2
        else:
            returns[t] = mu2 + sigma2_rs * rng.standard_normal()
            sigma2[t] = sigma2_rs ** 2
        if t < T - 1:
            if states[t] == 1:
                states[t + 1] = 1 if rng.random() < p11 else 2
            else:
                states[t + 1] = 2 if rng.random() < p22 else 1

    return returns, sigma2


def garch_normal_forecast_from_data(returns, alpha_var=0.01):
    """
    Fit GARCH(1,1)-Normal on data by estimating parameters from sample variance,
    then compute VaR forecasts. Used for DGPs where true GARCH params are unknown.
    Uses fixed reasonable GARCH params: omega from sample var, alpha=0.05, beta=0.90.
    """
    T = len(returns)
    # Use fixed GARCH parameters (misspecified for SV/RS)
    alpha_garch = 0.05
    beta = 0.90
    sample_var = np.var(returns)
    omega = sample_var * (1 - alpha_garch - beta)
    if omega <= 0:
        omega = 1e-8

    sigma2 = np.zeros(T + 1)
    sigma2[0] = sample_var

    var_forecast = np.zeros(T)
    es_forecast = np.zeros(T)

    z_alpha = stats.norm.ppf(alpha_var)  # negative
    phi_z = stats.norm.pdf(z_alpha)

    for t in range(T):
        sigma_t = np.sqrt(max(sigma2[t], 1e-12))
        var_forecast[t] = -sigma_t * z_alpha
        es_forecast[t] = sigma_t * phi_z / alpha_var
        sigma2[t + 1] = omega + alpha_garch * returns[t] ** 2 + beta * sigma2[t]

    return var_forecast, es_forecast


def garch_normal_forecast(returns, omega, alpha_garch, beta, alpha_var=0.01):
    """
    Compute GARCH-Normal VaR and ES forecasts using the GARCH recursion on data.
    VaR_t and ES_t are computed using sigma_t from GARCH recursion.
    Returns: (var_forecast, es_forecast) as positive numbers (losses).
    """
    T = len(returns)
    sigma2 = np.zeros(T + 1)
    sigma2[0] = omega / (1 - alpha_garch - beta)

    var_forecast = np.zeros(T)
    es_forecast = np.zeros(T)

    z_alpha = stats.norm.ppf(alpha_var)  # negative
    phi_z = stats.norm.pdf(z_alpha)

    for t in range(T):
        sigma_t = np.sqrt(sigma2[t])
        # VaR_t = -sigma_t * z_alpha (positive, since z_alpha < 0)
        var_forecast[t] = -sigma_t * z_alpha
        # ES_t = sigma_t * phi(z_alpha) / alpha (positive)
        es_forecast[t] = sigma_t * phi_z / alpha_var
        # Update sigma2 for next step
        sigma2[t + 1] = omega + alpha_garch * returns[t] ** 2 + beta * sigma2[t]

    return var_forecast, es_forecast


def conformal_correction(returns, var_forecast, alpha_var=0.01, f_cal=0.70):
    """
    Apply one-sided conformal correction identical to the LLM pipeline.
    s_t^V = q_lo_t - r_t where q_lo_t = -VaR_t
    q_hat_V = quantile of calibration scores at level ceil((n+1)*(1-alpha))/n
    corrected VaR = -(q_lo - q_hat_V) = VaR + q_hat_V
    """
    T = len(returns)
    n_cal = int(T * f_cal)

    # q_lo = -VaR (negative quantile)
    q_lo = -var_forecast

    # Calibration scores: s_t = q_lo_t - r_t
    s_V = q_lo[:n_cal] - returns[:n_cal]

    # Quantile of calibration scores
    n = len(s_V)
    q_level = np.ceil((n + 1) * (1 - alpha_var)) / n
    q_level = min(q_level, 1.0)
    q_hat_V = np.quantile(s_V, q_level)

    # Corrected VaR on test set
    corrected_var = -(q_lo - q_hat_V)  # = var_forecast + q_hat_V

    # Raw violations on test set
    test_returns = returns[n_cal:]
    test_raw_var = var_forecast[n_cal:]
    test_corr_var = corrected_var[n_cal:]

    raw_violations = test_returns < -test_raw_var
    corr_violations = test_returns < -test_corr_var

    n_test = T - n_cal
    raw_pi = raw_violations.sum() / n_test
    corr_pi = corr_violations.sum() / n_test

    # Traffic light
    def traffic_light(pi):
        annual_exc = pi * 250
        if annual_exc <= 4:
            return 'G'
        elif annual_exc <= 9:
            return 'Y'
        else:
            return 'R'

    raw_TL = traffic_light(raw_pi)
    corr_TL = traffic_light(corr_pi)

    return {
        'q_hat_V': q_hat_V,
        'raw_pi': raw_pi,
        'corr_pi': corr_pi,
        'raw_TL': raw_TL,
        'corr_TL': corr_TL,
        'n_cal': n_cal,
        'n_test': n_test,
        'test_raw_var': test_raw_var,
        'test_corr_var': test_corr_var,
        'test_returns': test_returns,
    }


def pinball_loss(returns, var_forecast, alpha=0.01):
    """
    Quantile score (pinball loss) for VaR at level alpha.
    S(q, y) = (alpha - I(y < -q)) * (y - (-q))
            = (alpha - I(y < -q)) * (y + q)
    where q = VaR (positive) and y = return.
    Lower is better.
    """
    q = var_forecast  # positive
    y = returns
    indicator = (y < -q).astype(float)
    loss = (indicator - alpha) * (y + q)
    return np.mean(loss)


def run_simulation_study():
    print("=" * 70)
    print("PART 1: Monte Carlo Simulation Study")
    print("=" * 70)

    omega = 1e-6
    alpha_garch = 0.05
    beta = 0.90
    T = 1000
    N_MC = 500
    alpha_var = 0.01
    f_cal = 0.70

    dgps = [
        {'name': 'Normal', 'df': None, 'label': 'Well-specified', 'type': 'garch'},
        {'name': 'Student-t(5)', 'df': 5, 'label': 'Moderate misspec.', 'type': 'garch'},
        {'name': 'Student-t(3)', 'df': 3, 'label': 'Severe misspec.', 'type': 'garch'},
        {'name': 'Stoch. Vol.', 'df': None, 'label': 'Structural misspec.', 'type': 'sv'},
        {'name': 'Regime Switch.', 'df': None, 'label': 'Regime switching', 'type': 'rs'},
    ]

    all_results = []
    pinball_results = {d['name']: {'raw': [], 'corr': []} for d in dgps}

    for dgp_idx, dgp in enumerate(dgps):
        print(f"\nDGP {dgp_idx + 1}: {dgp['name']} ({dgp['label']})")
        for rep in range(N_MC):
            if (rep + 1) % 100 == 0:
                print(f"  Replication {rep + 1}/{N_MC}")

            seed = dgp_idx * 10000 + rep

            if dgp['type'] == 'garch':
                returns, sigma2_true = simulate_garch_data(
                    T, omega, alpha_garch, beta, df_innov=dgp['df'], seed=seed
                )
                # GARCH-Normal forecast (known params for DGPs 1-3)
                var_forecast, es_forecast = garch_normal_forecast(
                    returns, omega, alpha_garch, beta, alpha_var=alpha_var
                )
            elif dgp['type'] == 'sv':
                returns, sigma2_true = simulate_sv_data(
                    T, mu_sv=-10, phi_sv=0.98, sigma_eta=0.15, seed=seed
                )
                # GARCH-Normal forecast (misspecified — ignores SV structure)
                var_forecast, es_forecast = garch_normal_forecast_from_data(
                    returns, alpha_var=alpha_var
                )
            elif dgp['type'] == 'rs':
                returns, sigma2_true = simulate_regime_switching_data(
                    T, p11=0.98, p22=0.95,
                    mu1=0.0003, sigma1=0.01, mu2=-0.001, sigma2_rs=0.04, seed=seed
                )
                # GARCH-Normal forecast (misspecified — ignores regimes)
                var_forecast, es_forecast = garch_normal_forecast_from_data(
                    returns, alpha_var=alpha_var
                )

            # Conformal correction
            res = conformal_correction(returns, var_forecast, alpha_var=alpha_var, f_cal=f_cal)

            all_results.append({
                'dgp': dgp_idx + 1,
                'dgp_name': dgp['name'],
                'rep': rep + 1,
                'raw_pi': res['raw_pi'],
                'q_hat_V': res['q_hat_V'],
                'corr_pi': res['corr_pi'],
                'raw_TL': res['raw_TL'],
                'corr_TL': res['corr_TL'],
            })

            # Pinball loss on test set
            n_cal = res['n_cal']
            raw_pb = pinball_loss(res['test_returns'], res['test_raw_var'], alpha=alpha_var)
            corr_pb = pinball_loss(res['test_returns'], res['test_corr_var'], alpha=alpha_var)
            pinball_results[dgp['name']]['raw'].append(raw_pb)
            pinball_results[dgp['name']]['corr'].append(corr_pb)

    # Save results CSV
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(BASE_DIR, "simulation_study_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SIMULATION STUDY SUMMARY")
    print("=" * 70)

    # Description map for each DGP
    description_map = {
        'Normal': 'Well-specified',
        'Student-t(5)': 'Moderate misspec.',
        'Student-t(3)': 'Severe misspec.',
        'Stoch. Vol.': 'Structural misspec.',
        'Regime Switch.': 'Regime switching',
    }

    summary_rows = []
    for dgp_idx, dgp in enumerate(dgps):
        mask = df_results['dgp'] == dgp_idx + 1
        sub = df_results[mask]
        mean_qV = sub['q_hat_V'].mean()
        sd_qV = sub['q_hat_V'].std()
        mean_raw_pi = sub['raw_pi'].mean()
        mean_corr_pi = sub['corr_pi'].mean()
        green_pct = (sub['corr_TL'] == 'G').mean() * 100
        raw_green_pct = (sub['raw_TL'] == 'G').mean() * 100

        summary_rows.append({
            'DGP': dgp_idx + 1,
            'Innovations': dgp['name'],
            'Description': description_map[dgp['name']],
            'Mean_qV': mean_qV,
            'SD_qV': sd_qV,
            'Raw_Coverage': mean_raw_pi,
            'Corr_Coverage': mean_corr_pi,
            'Raw_Green_pct': raw_green_pct,
            'Corr_Green_pct': green_pct,
        })

        print(f"DGP {dgp_idx + 1} ({dgp['name']}):")
        print(f"  Mean q_hat_V = {mean_qV:.6f}, SD = {sd_qV:.6f}")
        print(f"  Raw pi = {mean_raw_pi:.4f}, Corr pi = {mean_corr_pi:.4f}")
        print(f"  Raw Green% = {raw_green_pct:.1f}%, Corr Green% = {green_pct:.1f}%")

    # ---------- LaTeX table: simulation study ----------
    tex_path = os.path.join(BASE_DIR, "tab_simulation_study.tex")
    with open(tex_path, 'w') as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo Simulation Study: Conformal Correction under Misspecification}
\label{tab:simulation_study}
\begin{tabular}{cllcccccc}
\toprule
DGP & Innovations & Description & Mean $\hat{q}_V$ & SD($\hat{q}_V$) & Raw $\hat{\pi}$ & Corr $\hat{\pi}$ & Raw Green\% & Corr Green\% \\
\midrule
""")
        for row in summary_rows:
            f.write(f"{row['DGP']} & {row['Innovations']} & {row['Description']} & "
                    f"{row['Mean_qV']:.4f} & {row['SD_qV']:.4f} & "
                    f"{row['Raw_Coverage']:.4f} & {row['Corr_Coverage']:.4f} & "
                    f"{row['Raw_Green_pct']:.1f} & {row['Corr_Green_pct']:.1f} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} 500 Monte Carlo replications, $T=1000$. DGPs 1--3: GARCH(1,1) with $\omega=10^{-6}$, $\alpha=0.05$, $\beta=0.90$.
DGP 4: Stochastic Volatility ($\mu=-10$, $\phi=0.98$, $\sigma_\eta=0.15$).
DGP 5: Markov Regime Switching ($p_{11}=0.98$, $p_{22}=0.95$).
Forecaster always uses GARCH(1,1)-Normal. $\alpha_{\text{VaR}}=0.01$, calibration fraction $f=0.70$.
Traffic Light: Green $\leq 4$ annual violations per 250 days.
\end{tablenotes}
\end{table}
""")
    print(f"Saved: {tex_path}")

    # ---------- Figure: q_V distribution ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    data_boxes = []
    box_labels = [
        'Normal\n(well-spec.)',
        't(5)\n(moderate)',
        't(3)\n(severe)',
        'SV\n(struct.)',
        'Regime\n(switch)',
    ]
    for dgp_idx, dgp in enumerate(dgps):
        mask = df_results['dgp'] == dgp_idx + 1
        data_boxes.append(df_results.loc[mask, 'q_hat_V'].values)

    bp = ax.boxplot(data_boxes, labels=box_labels, patch_artist=True,
                    widths=0.5, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))

    colors = ['#2196F3', '#FF9800', '#F44336', '#9C27B0', '#795548']  # blue, orange, red, purple, brown
    color_labels = ['DGP 1: Normal', 'DGP 2: Student-t(5)', 'DGP 3: Student-t(3)',
                    'DGP 4: Stoch. Vol.', 'DGP 5: Regime Switch.']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Create legend handles
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, alpha=0.6, label=l) for c, l in zip(colors, color_labels)]
    legend_handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1, alpha=0.7, label='$\\hat{q}_V = 0$'))

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('Conformal correction $\\hat{q}_V$', fontsize=12)
    ax.set_xlabel('Data Generating Process', fontsize=12)
    ax.set_title('Distribution of $\\hat{q}_V$ across 500 MC replications', fontsize=13)
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=9, framealpha=0.0)
    ax.grid(False)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fpath = os.path.join(BASE_DIR, f"fig_simulation_qV_distribution.{ext}")
        fig.savefig(fpath, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fig_simulation_qV_distribution.pdf/.png")

    return df_results, pinball_results, summary_rows


# ================================================================
# PART 2: PINBALL LOSS (QUANTILE SCORE)
# ================================================================

def run_pinball_analysis(pinball_results):
    print("\n" + "=" * 70)
    print("PART 2: Pinball Loss (Quantile Score)")
    print("=" * 70)

    dgp_names = ['Normal', 'Student-t(5)', 'Student-t(3)', 'Stoch. Vol.', 'Regime Switch.']
    dgp_labels = ['Well-specified', 'Moderate misspec.', 'Severe misspec.', 'Structural misspec.', 'Regime switching']
    true_dfs = [r'$\infty$', '5', '3', 'SV', 'RS']

    tex_path = os.path.join(BASE_DIR, "tab_quantile_scores.tex")
    with open(tex_path, 'w') as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Quantile Score (Pinball Loss) for Simulation DGPs at $\alpha=0.01$}
\label{tab:quantile_scores}
\begin{tabular}{clcccc}
\toprule
DGP & Innovations & True $\nu$ & Raw Pinball & Corr Pinball & Improvement \\
\midrule
""")
        for i, name in enumerate(dgp_names):
            raw_mean = np.mean(pinball_results[name]['raw'])
            corr_mean = np.mean(pinball_results[name]['corr'])
            raw_sd = np.std(pinball_results[name]['raw'])
            corr_sd = np.std(pinball_results[name]['corr'])
            if raw_mean != 0:
                improvement = (raw_mean - corr_mean) / abs(raw_mean) * 100
            else:
                improvement = 0.0

            print(f"DGP {i+1} ({name}):")
            print(f"  Raw pinball  = {raw_mean:.6f} (SD={raw_sd:.6f})")
            print(f"  Corr pinball = {corr_mean:.6f} (SD={corr_sd:.6f})")
            print(f"  Improvement  = {improvement:.1f}%")

            f.write(f"{i+1} & {name} & {true_dfs[i]} & "
                    f"{raw_mean:.6f} ({raw_sd:.6f}) & "
                    f"{corr_mean:.6f} ({corr_sd:.6f}) & "
                    f"{improvement:+.1f}\\% \\\\\n")

        f.write(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Pinball loss $S_\alpha(q, y) = (\mathbf{1}_{y < -q} - \alpha)(y + q)$ averaged over test set.
500 MC replications, standard deviations in parentheses. Improvement = (Raw $-$ Corr) / $|$Raw$|$.
\end{tablenotes}
\end{table}
""")
    print(f"Saved: {tex_path}")


# ================================================================
# PART 3: CALIBRATION PLOT
# ================================================================

def find_csv(model_dir, asset, w):
    """Find CSV file matching asset and window in a model directory."""
    files = os.listdir(model_dir)
    for f in files:
        if not f.endswith('.csv'):
            continue
        if f"_w={w}.csv" in f and (
            f"_{asset}_" in f or f"_{asset.lower()}_" in f or
            f"_{asset.upper()}_" in f
        ):
            return os.path.join(model_dir, f)
    # Fallback
    for f in files:
        if not f.endswith('.csv'):
            continue
        if f"w={w}" in f and asset.lower() in f.lower():
            return os.path.join(model_dir, f)
    return None


def load_simulation_data(model_dir, asset, w):
    """Load simulation CSV: returns (dates, samples) where samples is (T, 1024)."""
    fpath = find_csv(model_dir, asset, w)
    if fpath is None:
        return None, None
    df = pd.read_csv(fpath, parse_dates=["Date"])
    dates = df["Date"].values
    samples = df.iloc[:, 1:].values
    return dates, samples


def load_realized_returns(asset):
    """Load realized log-returns from asset Excel file."""
    asset_file_map = {
        "CRIX": "CRIX.xlsx", "SP500": "SP500.xlsx",
        "SPGTCLTR": "SPGTCLTR.xlsx", "stoxx": "stoxx.xlsx",
        "cact": "cact.xlsx", "gdaxi": "gdaxi.xlsx",
        "cbu": "cbu.xlsx", "ftse": "ftse.xlsx", "djci": "djci.xlsx",
    }
    fpath = os.path.join(ASSETS_DIR, asset_file_map[asset])
    df = pd.read_excel(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df[["Date", "log_return"]].dropna().reset_index(drop=True)


def run_calibration_plot():
    print("\n" + "=" * 70)
    print("PART 3: Calibration Plot")
    print("=" * 70)

    alpha_levels = [0.005, 0.01, 0.025, 0.05, 0.10]
    alpha_var_base = 0.01
    f_cal = 0.70

    models = {
        'GPT-3.5': os.path.join(BASE_DIR, 'gpt3.5_simulations'),
        'GPT-4': os.path.join(BASE_DIR, 'gpt4_simulations'),
        'GPT-4o': os.path.join(BASE_DIR, 'gpt_4o_simulations'),
    }

    asset = 'SP500'
    w = 30

    # Load realized returns
    realized_df = load_realized_returns(asset)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Perfect calibration line
    ax.plot([0, 0.12], [0, 0.12], 'k:', linewidth=1.5, label='Perfect calibration', alpha=0.7)

    model_styles = {
        'GPT-3.5': {'color': '#2196F3', 'raw_ls': '--', 'corr_ls': '-', 'marker': 's'},
        'GPT-4': {'color': '#4CAF50', 'raw_ls': '--', 'corr_ls': '-', 'marker': 'o'},
        'GPT-4o': {'color': '#FF9800', 'raw_ls': '--', 'corr_ls': '-', 'marker': '^'},
    }

    for model_name, model_dir in models.items():
        print(f"\nProcessing {model_name}, {asset}, w={w}...")

        dates, samples = load_simulation_data(model_dir, asset, w)
        if dates is None:
            print(f"  WARNING: Could not load data for {model_name}")
            continue

        # Align dates with realized returns
        sim_dates = pd.to_datetime(dates)
        merged = pd.DataFrame({'Date': sim_dates})
        merged = merged.merge(realized_df, on='Date', how='inner')

        if len(merged) == 0:
            print(f"  WARNING: No date overlap for {model_name}")
            continue

        # Get indices of matching dates in simulation data
        sim_date_set = {pd.Timestamp(d): i for i, d in enumerate(sim_dates)}
        aligned_indices = []
        aligned_returns = []
        for _, row in merged.iterrows():
            ts = pd.Timestamp(row['Date'])
            if ts in sim_date_set:
                aligned_indices.append(sim_date_set[ts])
                aligned_returns.append(row['log_return'])

        aligned_indices = np.array(aligned_indices)
        aligned_returns = np.array(aligned_returns)
        aligned_samples = samples[aligned_indices]

        T = len(aligned_returns)
        n_cal = int(T * f_cal)
        n_test = T - n_cal

        print(f"  T={T}, n_cal={n_cal}, n_test={n_test}")

        raw_pis = []
        corr_pis = []

        for alpha_level in alpha_levels:
            # Compute conformal correction AT THIS alpha level
            q_lo_cal = np.quantile(aligned_samples[:n_cal], alpha_level, axis=1)
            s_V_cal = q_lo_cal - aligned_returns[:n_cal]
            n_sv = len(s_V_cal)
            q_level_a = np.ceil((n_sv + 1) * (1 - alpha_level)) / n_sv
            q_level_a = min(q_level_a, 1.0)
            q_hat_V_a = np.quantile(s_V_cal, q_level_a)

            # Raw quantile at this alpha level from LLM samples (test set)
            q_lo = np.quantile(aligned_samples[n_cal:], alpha_level, axis=1)
            raw_var = -q_lo  # positive

            # Raw violations
            raw_viol = aligned_returns[n_cal:] < q_lo
            raw_pi = raw_viol.mean()
            raw_pis.append(raw_pi)

            # Apply conformal correction at this alpha level
            corr_q_lo = q_lo - q_hat_V_a
            corr_var = -corr_q_lo

            # Corrected violations
            corr_viol = aligned_returns[n_cal:] < corr_q_lo
            corr_pi = corr_viol.mean()
            corr_pis.append(corr_pi)

            print(f"  alpha={alpha_level:.3f}: raw_pi={raw_pi:.4f}, corr_pi={corr_pi:.4f}, q_hat_V={q_hat_V_a:.6f}")

        style = model_styles[model_name]
        ax.plot(alpha_levels, raw_pis, linestyle=style['raw_ls'], color=style['color'],
                marker=style['marker'], markersize=6, linewidth=1.5,
                label=f'{model_name} (raw)', alpha=0.8)
        ax.plot(alpha_levels, corr_pis, linestyle=style['corr_ls'], color=style['color'],
                marker=style['marker'], markersize=7, linewidth=2,
                label=f'{model_name} (corrected)', alpha=0.9)

    ax.set_xlabel('Nominal level $\\alpha$', fontsize=12)
    ax.set_ylabel('Empirical violation rate $\\hat{\\pi}$', fontsize=12)
    ax.set_title('Calibration Plot: S\\&P 500, $w=30$', fontsize=13)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.005, 0.115)
    ax.set_ylim(-0.005, max(0.15, ax.get_ylim()[1]))
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fpath = os.path.join(BASE_DIR, f"fig_calibration_plot.{ext}")
        fig.savefig(fpath, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: fig_calibration_plot.pdf/.png")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("COMPREHENSIVE SIMULATION STUDY")
    print("=" * 70)

    # Part 1: Monte Carlo
    df_results, pinball_results, summary_rows = run_simulation_study()

    # Part 2: Pinball Loss
    run_pinball_analysis(pinball_results)

    # Part 3: Calibration Plot
    run_calibration_plot()

    print("\n" + "=" * 70)
    print("ALL DONE. Output files:")
    print(f"  {os.path.join(BASE_DIR, 'simulation_study_results.csv')}")
    print(f"  {os.path.join(BASE_DIR, 'tab_simulation_study.tex')}")
    print(f"  {os.path.join(BASE_DIR, 'tab_quantile_scores.tex')}")
    print(f"  {os.path.join(BASE_DIR, 'fig_simulation_qV_distribution.pdf')}")
    print(f"  {os.path.join(BASE_DIR, 'fig_simulation_qV_distribution.png')}")
    print(f"  {os.path.join(BASE_DIR, 'fig_calibration_plot.pdf')}")
    print(f"  {os.path.join(BASE_DIR, 'fig_calibration_plot.png')}")
    print("=" * 70)
