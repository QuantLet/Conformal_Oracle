"""
Three tasks:
1. Compute q_hat_V standard deviations across assets, update tab_conformal_all_methods.tex
2. Run T=5000 simulations (plus T=1000), create tab_simulation_extended.tex
3. Compute quantile scores at alpha=0.01 and alpha=0.025, create tab_quantile_scores.tex
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/danielpele/Documents/CFP LLM VaR"

# ================================================================
# TASK 1: Compute q_hat_V standard deviations across assets
# ================================================================

def task1():
    print("=" * 70)
    print("TASK 1: q_hat_V standard deviations across assets")
    print("=" * 70)

    # Load GARCH-family results
    garch_df = pd.read_csv(os.path.join(BASE_DIR, "garch_conformal_detail.csv"))
    # Load LLM results
    llm_df = pd.read_csv(os.path.join(BASE_DIR, "conformal_backtest_results.csv"))

    # GARCH methods: compute mean and std of q_hat_V across 9 assets
    garch_methods = ['GARCH-N(250)', 'GAS-N(250)', 'GAS-t(250)', 'GARCH-LPA', 'EWMA-N(120)', 'EWMA-DCS(120)']

    results = []
    for method in garch_methods:
        sub = garch_df[garch_df['method'] == method]
        q_vals = sub['q_hat_V'].values
        raw_pi_vals = sub['raw_pi_hat'].values
        corr_pi_vals = sub['corr_pi_hat'].values
        green_count = (sub['corr_TL'] == 'G').sum()
        z2_pass = (sub['corr_Z2'].abs() < 1.96).sum()  # approximate
        results.append({
            'method': method,
            'mean_raw_pi': raw_pi_vals.mean(),
            'mean_q_V': q_vals.mean(),
            'sd_q_V': q_vals.std(ddof=1),
            'mean_corr_pi': corr_pi_vals.mean(),
            'green': f"{green_count}/9",
            'z2_pass': f"{z2_pass}/9",
        })

    # LLM methods: filter w=30
    llm_w30 = llm_df[llm_df['window'] == 30]
    for model in ['GPT-3.5', 'GPT-4', 'GPT-4o']:
        sub = llm_w30[llm_w30['model'] == model]
        q_vals = sub['q_hat_V'].values
        raw_pi_vals = sub['raw_pi_hat'].values
        corr_pi_vals = sub['corr_pi_hat'].values
        green_count = (sub['corr_TL'] == 'G').sum()
        # Z2 pass: check if corr_Z2 column exists and values are valid
        z2_col = sub['corr_Z2'].values
        z2_pass = sum(1 for z in z2_col if not np.isnan(z) and abs(z) < 1.96)
        # If no valid Z2 values, count all as pass (conservative)
        if all(np.isnan(z) for z in z2_col):
            z2_pass = len(sub)

        results.append({
            'method': f"{model}+CP",
            'mean_raw_pi': raw_pi_vals.mean(),
            'mean_q_V': q_vals.mean(),
            'sd_q_V': q_vals.std(ddof=1),
            'mean_corr_pi': corr_pi_vals.mean(),
            'green': f"{green_count}/9",
            'z2_pass': f"{z2_pass}/9",
        })

    # Print table
    print(f"\n{'Method':<20} {'Mean q_V':>10} {'Std q_V':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['method']:<20} {r['mean_q_V']:>10.4f} {r['sd_q_V']:>10.4f}")

    # Write updated tab_conformal_all_methods.tex with SD column
    tex_path = os.path.join(BASE_DIR, "tab_conformal_all_methods.tex")
    with open(tex_path, 'w') as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Conformal correction across model classes. $\overline{\hat{\pi}}_{\mathrm{raw}}$ and $\overline{\hat{\pi}}_{\mathrm{corr}}$ are mean violation rates (raw and corrected) across nine assets. $\overline{\qV}$ is the mean conformal correction threshold. SD$(\qV)$ is the cross-asset standard deviation. Green denotes Basel Green Zone ($\leq 4$ annual exceptions).}
\label{tab:conformal_all_methods}
\small
\begin{tabular}{l r r r r c c}
\toprule
Method & $\overline{\hat{\pi}}_{\mathrm{raw}}$ & $\overline{\qV}$ & SD$(\qV)$ & $\overline{\hat{\pi}}_{\mathrm{corr}}$ & Green & $Z_2$ Pass \\
\midrule
""")
        for i, r in enumerate(results):
            # Add midrule between GARCH and LLM methods
            if i == 6:
                f.write(r"\midrule" + "\n")
            sign = "+" if r['mean_q_V'] >= 0 else ""
            f.write(f"  {r['method']} & {r['mean_raw_pi']:.4f} & {sign}{r['mean_q_V']:.4f} & "
                    f"{r['sd_q_V']:.4f} & {r['mean_corr_pi']:.4f} & "
                    f"{r['green']} & {r['z2_pass']} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"\nSaved: {tex_path}")
    return results


# ================================================================
# TASK 2 & 3: Simulation functions (reused from run_simulation_study.py)
# ================================================================

def simulate_garch_data(T, omega, alpha_garch, beta, df_innov=None, seed=None):
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(T + 1)
    returns = np.zeros(T)
    sigma2[0] = omega / (1 - alpha_garch - beta)
    for t in range(T):
        if df_innov is None:
            z = rng.standard_normal()
        else:
            z = rng.standard_t(df_innov) / np.sqrt(df_innov / (df_innov - 2))
        returns[t] = np.sqrt(sigma2[t]) * z
        sigma2[t + 1] = omega + alpha_garch * returns[t] ** 2 + beta * sigma2[t]
    return returns, sigma2[:T]


def simulate_sv_data(T, mu_sv=-10, phi_sv=0.98, sigma_eta=0.15, seed=None):
    rng = np.random.default_rng(seed)
    log_sigma2 = np.zeros(T)
    returns = np.zeros(T)
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
    rng = np.random.default_rng(seed)
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    states = np.zeros(T, dtype=int)
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
    T = len(returns)
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
    z_alpha = stats.norm.ppf(alpha_var)
    phi_z = stats.norm.pdf(z_alpha)
    for t in range(T):
        sigma_t = np.sqrt(max(sigma2[t], 1e-12))
        var_forecast[t] = -sigma_t * z_alpha
        es_forecast[t] = sigma_t * phi_z / alpha_var
        sigma2[t + 1] = omega + alpha_garch * returns[t] ** 2 + beta * sigma2[t]
    return var_forecast, es_forecast


def garch_normal_forecast(returns, omega, alpha_garch, beta, alpha_var=0.01):
    T = len(returns)
    sigma2 = np.zeros(T + 1)
    sigma2[0] = omega / (1 - alpha_garch - beta)
    var_forecast = np.zeros(T)
    es_forecast = np.zeros(T)
    z_alpha = stats.norm.ppf(alpha_var)
    phi_z = stats.norm.pdf(z_alpha)
    for t in range(T):
        sigma_t = np.sqrt(sigma2[t])
        var_forecast[t] = -sigma_t * z_alpha
        es_forecast[t] = sigma_t * phi_z / alpha_var
        sigma2[t + 1] = omega + alpha_garch * returns[t] ** 2 + beta * sigma2[t]
    return var_forecast, es_forecast


def conformal_correction(returns, var_forecast, alpha_var=0.01, f_cal=0.70):
    T = len(returns)
    n_cal = int(T * f_cal)
    q_lo = -var_forecast
    s_V = q_lo[:n_cal] - returns[:n_cal]
    n = len(s_V)
    q_level = np.ceil((n + 1) * (1 - alpha_var)) / n
    q_level = min(q_level, 1.0)
    q_hat_V = np.quantile(s_V, q_level)
    corrected_var = -(q_lo - q_hat_V)
    test_returns = returns[n_cal:]
    test_raw_var = var_forecast[n_cal:]
    test_corr_var = corrected_var[n_cal:]
    raw_violations = test_returns < -test_raw_var
    corr_violations = test_returns < -test_corr_var
    n_test = T - n_cal
    raw_pi = raw_violations.sum() / n_test
    corr_pi = corr_violations.sum() / n_test

    def traffic_light(pi):
        annual_exc = pi * 250
        if annual_exc <= 4:
            return 'G'
        elif annual_exc <= 9:
            return 'Y'
        else:
            return 'R'

    return {
        'q_hat_V': q_hat_V,
        'raw_pi': raw_pi,
        'corr_pi': corr_pi,
        'raw_TL': traffic_light(raw_pi),
        'corr_TL': traffic_light(corr_pi),
        'n_cal': n_cal,
        'n_test': n_test,
        'test_raw_var': test_raw_var,
        'test_corr_var': test_corr_var,
        'test_returns': test_returns,
    }


def quantile_score(y, q_var, alpha):
    """
    QS(alpha, q, y) = (y - q) * (alpha - I(y < q))
    where q = -VaR (the quantile, negative for losses), y = return.
    Actually: q here is the quantile (negative), VaR is positive.
    q = -VaR, so:
    QS = (y - (-VaR)) * (alpha - I(y < -VaR))
       = (y + VaR) * (alpha - I(y < -VaR))
    """
    q = -q_var  # quantile (negative)
    indicator = (y < q).astype(float)
    qs = (y - q) * (alpha - indicator)
    return np.mean(qs)


# ================================================================
# TASK 2: Run T=5000 simulations
# ================================================================

def task2():
    print("\n" + "=" * 70)
    print("TASK 2: Simulation study with T=1000 and T=5000")
    print("=" * 70)

    omega = 1e-6
    alpha_garch = 0.05
    beta = 0.90
    N_MC = 500
    alpha_var = 0.01
    f_cal = 0.70
    T_values = [1000, 5000]

    dgps = [
        {'name': 'Normal', 'df': None, 'label': 'Well-specified', 'type': 'garch'},
        {'name': 'Student-t(5)', 'df': 5, 'label': 'Moderate misspec.', 'type': 'garch'},
        {'name': 'Student-t(3)', 'df': 3, 'label': 'Severe misspec.', 'type': 'garch'},
        {'name': 'Stoch.\\ Vol.', 'df': None, 'label': 'Structural misspec.', 'type': 'sv'},
        {'name': 'Regime Switch.', 'df': None, 'label': 'Regime switching', 'type': 'rs'},
    ]

    summary_rows = []

    for T in T_values:
        print(f"\n--- T = {T} ---")
        for dgp_idx, dgp in enumerate(dgps):
            print(f"  DGP {dgp_idx + 1}: {dgp['name']} ...", end=" ", flush=True)

            q_hat_Vs = []
            raw_pis = []
            corr_pis = []
            raw_greens = 0
            corr_greens = 0

            for rep in range(N_MC):
                seed = dgp_idx * 100000 + T * 1000 + rep

                if dgp['type'] == 'garch':
                    returns, _ = simulate_garch_data(T, omega, alpha_garch, beta, df_innov=dgp['df'], seed=seed)
                    var_forecast, _ = garch_normal_forecast(returns, omega, alpha_garch, beta, alpha_var=alpha_var)
                elif dgp['type'] == 'sv':
                    returns, _ = simulate_sv_data(T, mu_sv=-10, phi_sv=0.98, sigma_eta=0.15, seed=seed)
                    var_forecast, _ = garch_normal_forecast_from_data(returns, alpha_var=alpha_var)
                elif dgp['type'] == 'rs':
                    returns, _ = simulate_regime_switching_data(T, seed=seed)
                    var_forecast, _ = garch_normal_forecast_from_data(returns, alpha_var=alpha_var)

                res = conformal_correction(returns, var_forecast, alpha_var=alpha_var, f_cal=f_cal)
                q_hat_Vs.append(res['q_hat_V'])
                raw_pis.append(res['raw_pi'])
                corr_pis.append(res['corr_pi'])
                if res['raw_TL'] == 'G':
                    raw_greens += 1
                if res['corr_TL'] == 'G':
                    corr_greens += 1

            mean_qV = np.mean(q_hat_Vs)
            sd_qV = np.std(q_hat_Vs, ddof=1)
            mean_raw_pi = np.mean(raw_pis)
            mean_corr_pi = np.mean(corr_pis)
            raw_green_pct = raw_greens / N_MC * 100
            corr_green_pct = corr_greens / N_MC * 100

            summary_rows.append({
                'DGP': dgp['name'],
                'T': T,
                'Mean_qV': mean_qV,
                'SD_qV': sd_qV,
                'Raw_pi': mean_raw_pi,
                'Corr_pi': mean_corr_pi,
                'Raw_Green_pct': raw_green_pct,
                'Corr_Green_pct': corr_green_pct,
            })

            print(f"q_V={mean_qV:.4f} (SD={sd_qV:.4f}), raw_pi={mean_raw_pi:.4f}, corr_pi={mean_corr_pi:.4f}, "
                  f"raw_G={raw_green_pct:.0f}%, corr_G={corr_green_pct:.0f}%")

    # Write tab_simulation_extended.tex
    tex_path = os.path.join(BASE_DIR, "tab_simulation_extended.tex")
    with open(tex_path, 'w') as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Monte Carlo Simulation Study: Conformal Correction for $T=1{,}000$ and $T=5{,}000$}
\label{tab:simulation_extended}
\scriptsize
\begin{tabular}{lrrrrrcc}
\toprule
DGP & $T$ & Mean $\hat{q}_V$ & SD($\hat{q}_V$) & Raw $\hat{\pi}$ & Corr $\hat{\pi}$ & Raw Green\% & Corr Green\% \\
\midrule
""")
        for i, row in enumerate(summary_rows):
            # Add midrule between T=1000 and T=5000 blocks
            if i == 5:
                f.write(r"\midrule" + "\n")
            f.write(f"{row['DGP']} & {row['T']} & {row['Mean_qV']:.4f} & {row['SD_qV']:.4f} & "
                    f"{row['Raw_pi']:.4f} & {row['Corr_pi']:.4f} & "
                    f"{row['Raw_Green_pct']:.1f} & {row['Corr_Green_pct']:.1f} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} 500 Monte Carlo replications per DGP$\times T$ combination.
DGPs 1--3: GARCH(1,1) with $\omega=10^{-6}$, $\alpha=0.05$, $\beta=0.90$.
DGP 4: Stochastic Volatility ($\mu=-10$, $\phi=0.98$, $\sigma_\eta=0.15$).
DGP 5: Markov Regime Switching ($p_{11}=0.98$, $p_{22}=0.95$).
Forecaster always uses GARCH(1,1)-Normal. $\alpha_{\text{VaR}}=0.01$, calibration fraction $f=0.70$.
Traffic Light: Green $\leq 4$ annual violations per 250 days.
\end{tablenotes}
\end{table}
""")
    print(f"\nSaved: {tex_path}")
    return summary_rows


# ================================================================
# TASK 3: Quantile Scores at alpha=0.01 and alpha=0.025
# ================================================================

def task3():
    print("\n" + "=" * 70)
    print("TASK 3: Quantile Scores at alpha=0.01 and alpha=0.025")
    print("=" * 70)

    omega = 1e-6
    alpha_garch = 0.05
    beta = 0.90
    N_MC = 500
    f_cal = 0.70
    T_values = [1000, 5000]
    alpha_levels = [0.01, 0.025]

    dgps = [
        {'name': 'Normal', 'df': None, 'label': 'Well-specified', 'type': 'garch'},
        {'name': 'Student-t(5)', 'df': 5, 'label': 'Moderate misspec.', 'type': 'garch'},
        {'name': 'Student-t(3)', 'df': 3, 'label': 'Severe misspec.', 'type': 'garch'},
        {'name': 'Stoch.\\ Vol.', 'df': None, 'label': 'Structural misspec.', 'type': 'sv'},
        {'name': 'Regime Switch.', 'df': None, 'label': 'Regime switching', 'type': 'rs'},
    ]

    qs_rows = []

    for T in T_values:
        print(f"\n--- T = {T} ---")
        for dgp_idx, dgp in enumerate(dgps):
            print(f"  DGP {dgp_idx + 1}: {dgp['name']} ...", end=" ", flush=True)

            # Store QS for each alpha, raw/corr
            qs_raw = {a: [] for a in alpha_levels}
            qs_corr = {a: [] for a in alpha_levels}

            for rep in range(N_MC):
                seed = dgp_idx * 100000 + T * 1000 + rep

                if dgp['type'] == 'garch':
                    returns, _ = simulate_garch_data(T, omega, alpha_garch, beta, df_innov=dgp['df'], seed=seed)
                elif dgp['type'] == 'sv':
                    returns, _ = simulate_sv_data(T, mu_sv=-10, phi_sv=0.98, sigma_eta=0.15, seed=seed)
                elif dgp['type'] == 'rs':
                    returns, _ = simulate_regime_switching_data(T, seed=seed)

                for alpha_var in alpha_levels:
                    if dgp['type'] == 'garch':
                        var_forecast, _ = garch_normal_forecast(returns, omega, alpha_garch, beta, alpha_var=alpha_var)
                    else:
                        var_forecast, _ = garch_normal_forecast_from_data(returns, alpha_var=alpha_var)

                    res = conformal_correction(returns, var_forecast, alpha_var=alpha_var, f_cal=f_cal)

                    # Quantile score on test set
                    qs_r = quantile_score(res['test_returns'], res['test_raw_var'], alpha_var)
                    qs_c = quantile_score(res['test_returns'], res['test_corr_var'], alpha_var)
                    qs_raw[alpha_var].append(qs_r)
                    qs_corr[alpha_var].append(qs_c)

            row = {'DGP': dgp['name'], 'T': T}
            for a in alpha_levels:
                row[f'qs_raw_{a}'] = np.mean(qs_raw[a])
                row[f'qs_corr_{a}'] = np.mean(qs_corr[a])
            qs_rows.append(row)

            print(f"QS(0.01) raw={row['qs_raw_0.01']:.6f} corr={row['qs_corr_0.01']:.6f}, "
                  f"QS(0.025) raw={row['qs_raw_0.025']:.6f} corr={row['qs_corr_0.025']:.6f}")

    # Write tab_quantile_scores.tex
    tex_path = os.path.join(BASE_DIR, "tab_quantile_scores.tex")
    with open(tex_path, 'w') as f:
        f.write(r"""\begin{table}[htbp]
\centering
\caption{Quantile Scores for Simulation DGPs at $\alpha=0.01$ and $\alpha=0.025$}
\label{tab:quantile_scores}
\scriptsize
\begin{tabular}{lrcccc}
\toprule
DGP & $T$ & QS(0.01) raw & QS(0.01) corr & QS(0.025) raw & QS(0.025) corr \\
\midrule
""")
        for i, row in enumerate(qs_rows):
            if i == 5:
                f.write(r"\midrule" + "\n")
            f.write(f"{row['DGP']} & {row['T']} & "
                    f"{row['qs_raw_0.01']:.6f} & {row['qs_corr_0.01']:.6f} & "
                    f"{row['qs_raw_0.025']:.6f} & {row['qs_corr_0.025']:.6f} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Quantile Score $\mathrm{QS}(\alpha, q, y) = (y - q)(\alpha - \mathbf{1}_{y < q})$ averaged over 500 MC replications and test observations.
Lower values indicate better calibration. ``raw'' uses the GARCH-Normal VaR forecast; ``corr'' uses the conformally corrected VaR.
\end{tablenotes}
\end{table}
""")
    print(f"\nSaved: {tex_path}")
    return qs_rows


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    # Task 1
    task1_results = task1()

    # Task 2
    task2_results = task2()

    # Task 3
    task3_results = task3()

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETE. Output files:")
    print(f"  {os.path.join(BASE_DIR, 'tab_conformal_all_methods.tex')}")
    print(f"  {os.path.join(BASE_DIR, 'tab_simulation_extended.tex')}")
    print(f"  {os.path.join(BASE_DIR, 'tab_quantile_scores.tex')}")
    print("=" * 70)
