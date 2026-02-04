"""
Module for computing statistical summaries and generating diagnostic plots from Monte Carlo simulations
and UKF tracking results.

This module provides functions to analyze DataFrames from Monte Carlo runs and to visualize the performance
of SRUKF estimates over time. It leverages Pandas for data manipulation, SciPy for statistics, and Matplotlib for plotting.

The `compute_statistical_figures` function extends Pandas' describe() with additional metrics like skew, kurtosis,
median, and MAD. The `generate_performance_plots` function creates multiple diagnostic figures,
including state/parameter estimates with uncertainty bands, error convergence, and model fit comparisons,
saving them as PNG files in a configured directory.

Notes / high-level behaviour
----------------------------
- Assumes input DataFrames have specific columns for estimates and errors; computes transposed summaries with extra stats.
- Plotting function handles trimming of histories to match time spans, computes propagated sensitivities using
  `compute_sens_and_std` from utils, and applies tight y-limits for better visualization.
- Uncertainty bands use a configurable sigma multiplier (default 1.96 for ~95% CI).
- Plots are generated in subplots for parameters/states/sensitivity, plus separate figures for error norms and capacitance fits.

Logging and output
------------------
- No explicit logging; outputs are side effects (saved PNG files) and returned DataFrames.
- Plots include labels, legends, grids, and titles for clarity.
- Empty axes in subplots are turned off; layouts are tightened for compactness.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os
from typing import List
from model import solve_ODE, compute_static_sensitivity
from utils import compute_sens_and_std

OUTPUT_DIR = "monte_carlo_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "diagnostic_plots")

def compute_statistical_figures(df):
    """
    Compute descriptive statistics for Monte Carlo results stored in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least these columns:
        'est_stiff', 'est_elec', 'est_Q', 'est_offset', 'est_sensitivity',
        'err_stiff_norm', 'err_elec_norm', 'err_Q_norm', 'err_offset_norm',
        'err_sensitivity_per'.

    Returns
    -------
    pandas.DataFrame
        Transposed summary produced by pandas .describe() augmented with 'skew',
        'kurtosis', 'median', and 'mad' (mean absolute deviation from the mean).
    """
    stats_cols = [
        'est_stiff', 'est_elec', 'est_Q', 'est_offset', 'est_sensitivity',
        'err_stiff_norm', 'err_elec_norm', 'err_Q_norm', 'err_offset_norm',
        'err_sensitivity_per'
    ]
    stats = df[stats_cols].describe().T
    stats['skew'] = df[stats_cols].apply(skew, nan_policy='omit')
    stats['kurtosis'] = df[stats_cols].apply(kurtosis, nan_policy='omit')
    stats['median'] = df[stats_cols].median()
    stats['mad'] = (df[stats_cols] - df[stats_cols].mean()).abs().mean()
    return stats

def generate_performance_plots(history: List[np.ndarray], covariance_history: List[np.ndarray],
                               true_params: np.ndarray,
                               true_disp: np.ndarray, true_vel: np.ndarray,
                               true_measurements: np.ndarray, noisy_data: np.ndarray,
                               t_span: np.ndarray,
                               min_voltage: float,
                               sigma_multiplier: float = 1.96,
                               save_prefix: str = ""
                               ) -> None:
    """
    Generate diagnostic performance plots for a UKF/Monte Carlo run and save PNG files.

    Parameters
    ----------
    history : list or np.ndarray, shape (T, n_states)
        Time history of estimated states/parameters.
    covariance_history : list of np.ndarray or np.ndarray, shape (T, n_states, n_states)
        Time history of covariance matrices.
    true_params : np.ndarray, shape (4,)
        Ground-truth parameters [stiff, elec, Q, offset].
    true_disp : np.ndarray, shape (>= T,)
        Ground-truth displacement time series.
    true_vel : np.ndarray, shape (>= T,)
        Ground-truth velocity time series.
    true_measurements : np.ndarray, shape (>= T,)
        Ground-truth ΔC time series.
    noisy_data : np.ndarray, shape (>= T,)
        Noisy measurements (ΔC).
    t_span : np.ndarray, shape (>= T,)
        Time vector for plotting.
    min_voltage : float
        Voltage used to compute model output when plotting final fit.
    sigma_multiplier : float, optional
        Multiplier for ±σ bands (default 1.96).
    save_prefix : str, optional
        String prefix appended to saved filenames.

    Returns
    -------
    None
        Side effect: writes PNG files into the configured plots directory.
    """
    history_array = np.array(history)
    n_hist, n_states = history_array.shape
    cov_history_trimmed = covariance_history[:n_hist]
    sigma_array = np.array([np.sqrt(np.diag(P)) for P in cov_history_trimmed])
    min_length = min(n_hist, len(t_span))
    history_array = history_array[:min_length]
    sigma_array = sigma_array[:min_length]
    t_span_trimmed = t_span[:min_length]
    param_and_state_names = ['Over Stiffness', 'Over Electrode', 'Q', 'Offset',
                             'Position', 'Velocity']
    n_plots = n_states + 1
    ncols = 2
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    true_values = [true_params[0], true_params[1], true_params[2], true_params[3],
                   true_disp[:min_length], true_vel[:min_length]]
    true_model_state = np.zeros(5)
    true_model_state[0:2] = true_params[0:2]
    true_model_state[2] = 0.
    true_model_state[3:5] = true_params[2:4]
    true_sens_val = compute_static_sensitivity(true_model_state)
    def tight_ylim(est, sigma, true_val=None, margin=0.08):
        upper = est + sigma_multiplier * sigma
        lower = est - sigma_multiplier * sigma
        all_vals = np.concatenate([est, upper, lower])
        if true_val is not None:
            if np.isscalar(true_val):
                all_vals = np.append(all_vals, true_val)
            else:
                all_vals = np.concatenate([all_vals, true_val])
        ymin, ymax = all_vals.min(), all_vals.max()
        if ymax == ymin:
            delta = 0.1 if abs(ymax) < 1e-8 else abs(ymax) * 0.1
            return ymin - delta, ymax + delta
        delta = (ymax - ymin) * margin
        return ymin - delta, ymax + delta
    for i in range(n_plots):
        ax = axes[i]
        if i < n_states:
            est = history_array[:, i]
            sigma_i = sigma_array[:, i] if i < sigma_array.shape[1] else np.zeros(min_length)
            half_band = sigma_multiplier * sigma_i
            upper = est + half_band
            lower = est - half_band
            ax.plot(t_span_trimmed, est, '-', linewidth=2, label='UKF Estimate', color='tab:blue')
            ax.fill_between(t_span_trimmed, lower, upper, alpha=0.25, color='tab:blue',
                            label=f'±{sigma_multiplier}σ')
            if i < 4:
                true_val_scalar = true_values[i]
                ax.axhline(y=true_val_scalar, color='r', linestyle='--', linewidth=2, label='True Value')
                true_for_ylim = true_val_scalar
            else:
                true_arr = np.asarray(true_values[i])
                ax.plot(t_span_trimmed, true_arr, 'r--', linewidth=2, label='True Value')
                true_for_ylim = true_arr
            ax.set_ylim(tight_ylim(est, sigma_i, true_for_ylim))
            label = param_and_state_names[i] if i < len(param_and_state_names) else f'State {i}'
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(label)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{label} Estimation with Uncertainty Bounds')
        else:
            params_hist = history_array[:, :4]
            cov_hist = np.stack([P[:4, :4] for P in cov_history_trimmed])
            est_sens_series, std_sens = compute_sens_and_std(params_hist, cov_hist)
            half_band = sigma_multiplier * std_sens
            upper = est_sens_series + half_band
            lower = est_sens_series - half_band
            ax.plot(t_span_trimmed, est_sens_series, '-', linewidth=2, color='tab:blue', label='UKF Estimate')
            ax.fill_between(t_span_trimmed, lower, upper, alpha=0.25, color='tab:blue',
                            label=f'±{sigma_multiplier}σ')
            ax.axhline(y=true_sens_val, color='r', linestyle='--', linewidth=2, label='True Value')
            ax.set_ylim(tight_ylim(est_sens_series, std_sens, true_sens_val))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('|ΔC| (F)')
            ax.set_title('Sensitivity Estimation with Uncertainty Bounds')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{save_prefix}_params_states.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    param_errors = np.linalg.norm(history_array[:, :4] - true_params, axis=1)
    sigma_norms = np.array([np.sqrt(np.trace(P[:4, :4])) for P in cov_history_trimmed])
    upper_err = param_errors + sigma_multiplier * sigma_norms
    lower_err = np.maximum(0.0, param_errors - sigma_multiplier * sigma_norms)
    plt.figure(figsize=(10, 6))
    plt.plot(t_span_trimmed, param_errors, 'r-', linewidth=2.5, label='Parameter Error (norm)')
    plt.fill_between(t_span_trimmed, lower_err, upper_err, alpha=0.25, color='red',
                     label=f'±{sigma_multiplier}σ on norm')
    plt.ylim(tight_ylim(param_errors, sigma_norms, margin=0.1))
    plt.xlabel('Time (s)')
    plt.ylabel('Parameter Estimation Error (Euclidean norm)')
    plt.title('UKF Parameter Error Convergence with Uncertainty')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"{save_prefix}_error_convergence.png"), dpi=200, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 6))
    final_estimate = history_array[-1]
    model_state = np.array([final_estimate[0], final_estimate[1], 0.0, final_estimate[2], final_estimate[3]])
    estimated_dC = solve_ODE(model_state, min_voltage)[:min_length]
    all_cap_data = np.concatenate([true_measurements[:min_length], estimated_dC, noisy_data[:min_length]])
    margin = (all_cap_data.max() - all_cap_data.min()) * 0.08 if (
                                                                             all_cap_data.max() - all_cap_data.min()) > 0 else 0.1
    ylim_cap = (all_cap_data.min() - margin, all_cap_data.max() + margin)
    plt.plot(t_span_trimmed, true_measurements[:min_length], 'g-', linewidth=2.5, label='True ΔC', alpha=0.8)
    plt.plot(t_span_trimmed, estimated_dC, 'b--', linewidth=2.5, label='UKF Estimated ΔC')
    plt.plot(t_span_trimmed, noisy_data[:min_length], 'gray', linewidth=1, alpha=0.5, label='Noisy Measurements')
    plt.ylim(ylim_cap)
    plt.xlabel('Time (s)')
    plt.ylabel('Capacitance Difference ΔC (F)')
    plt.title('UKF Model Fit: True vs Estimated Capacitance')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"{save_prefix}_model_fit.png"), dpi=200, bbox_inches='tight')
    plt.close()