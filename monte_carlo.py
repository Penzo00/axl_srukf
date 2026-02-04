"""
Monte Carlo validation driver for SR-UKF MEMS parameter estimation.

This module orchestrates a full Monte Carlo (MC) validation and diagnostic pipeline
for the SR-UKF based parameter tracker. It provides utilities for sampling true
parameters, generating noisy measurement data by integrating a physical model,
running per-run UKF tracking (possibly iteratively until convergence), collecting
results, producing summary statistics, saving generated datasets, and creating
diagnostic plots for distributional and per-run performance.

High-level workflow
-------------------
1. Sample `N_MC_RUNS` true parameter vectors from truncated normal distributions
   (see PARAM_DIST).
2. For each run:
     a. Generate the true ΔC time series by integrating the MEMS model.
     b. Add Gaussian measurement noise (NOISE_STD).
     c. Run the ParameterTracker (SR-UKF) to estimate parameters from noisy data.
     d. Optionally iterate the tracking (reinitialize with updated priors) until
        convergence or maximum iterations is reached.
     e. Record final estimates, uncertainties, errors, and diagnostics.
3. Collect results from all runs in a DataFrame, save CSV and .npz data, and
   compute statistical summaries and diagnostic plots.

Important constants & directories
--------------------------------
- N_MC_RUNS : int
    Number of Monte Carlo trials.
- OUTPUT_DIR, PLOTS_DIR, SAMPLES_DIR : str / directories
    Output locations for CSV, sample archives, and diagnostic plots. Created on import.
- PARAM_DIST : dict
    Per-parameter sampling specifications (mean, std, lower/upper truncation bounds).
- INITIAL_GUESS : np.ndarray
    UKF initial state guess used to start tracking.
- NOISE_STD : float
    Standard deviation of measurement noise added to simulated ΔC.

Parallel execution
------------------
- `main_monte_carlo()` uses `multiprocessing.Pool` to run MC trials in parallel using
  `worker()` -> `run_single_mc_simulation()`. The number of workers defaults to `os.cpu_count()`.
- Be aware that heavy numerical code (Numba functions, ODE integration) will be invoked
  in child processes; ensure each worker can import modules and compile Numba JIT code
  independently (first-call compile overhead per process).

I/O & reproducibility
----------------------
- Results, generated samples, and plots are saved under OUTPUT_DIR. The generated samples
  directory includes a timestamp-based subfolder for reproducibility.
- `np.random.seed(42)` at module import sets a deterministic global seed for sampling in
  single-process contexts; each MC run also reseeds using the run index for reproducible
  noise per run.

Usage
-----
Call `main_monte_carlo()` to run the full pipeline. For scripted/CI usage you may want to:
- Reduce `N_MC_RUNS` for quick checks.
- Set `exploration_config` hyperparameters to control convergence or sampling behavior.
"""
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from time import time
import multiprocessing

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.signal import chirp
from scipy.stats import truncnorm
from tqdm import tqdm
from config import UKFConfig, logger, total_duration, _step, dur_chirp, f_end, f_start
from utils import compute_min_A, compute_stats, find_peaks_1d, find_valleys_1d, sens_jacobian
from model import solve_ODE_disp, solve_ODE, compute_static_sensitivity
from tracker import ParameterTracker
from analysis import compute_statistical_figures, generate_performance_plots

N_MC_RUNS = 1_000
OUTPUT_DIR = "monte_carlo_results"
CSV_FILENAME_PREFIX = "mc_parameter_estimation_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "diagnostic_plots")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "generated_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

PARAM_DIST = {
    'over_stiffness': {'mean': 0.35, 'std': 0.1, 'low': 0.15, 'high': 0.55},
    'over_electrode': {'mean': 0.35, 'std': 0.1, 'low': 0.15, 'high': 0.55},
    'Q': {'mean': 0.50, 'std': 0.05, 'low': 0.40, 'high': 0.60},
    'offset': {'mean': 0.00, 'std': 0.1, 'low': -0.2, 'high': 0.20},
}

INITIAL_GUESS = np.array([0.35, 0.35, 0.5, 0.0, 0.0, 0.0])
NOISE_STD = 4.8 * np.sqrt(0.3) * 0.001 * 10 ** 1.5

exploration_config = UKFConfig()

def save_generated_samples(true_params_all: np.ndarray, all_noisy_data: np.ndarray,
                           all_true_dC: np.ndarray, timestamp: str) -> str:
    """
    Save generated Monte Carlo samples and dataset metadata.

    This helper writes:
      - a compressed NumPy .npz archive containing `true_params`, `noisy_data`, `true_dC`,
        `noise_std`, and `initial_guess`
      - a human-readable `dataset_info.txt` summarising the dataset and sampling config

    Parameters
    ----------
    true_params_all : np.ndarray, shape (N_MC_RUNS, 4)
        True parameter vectors used for each Monte Carlo trial.
    all_noisy_data : np.ndarray, shape (N_MC_RUNS, T)
        Noisy measurement time series for each run.
    all_true_dC : np.ndarray, shape (N_MC_RUNS, T)
        True (noise-free) ΔC time series for each run.
    timestamp : str
        Timestamp string used to create a unique samples directory.

    Returns
    -------
    str
        Path to the directory where generated samples and metadata were saved.

    Side effects
    ------------
    - Creates `SAMPLES_DIR/samples_{timestamp}` and writes:
        * generated_samples.npz
        * dataset_info.txt
    - Prints summary to standard output.

    Notes
    -----
    - The .npz archive contains all arrays as provided; for large N_MC_RUNS this file
      can become large—consider subsampling or storing per-run files if disk space is constrained.
    """

    samples_dir = os.path.join(SAMPLES_DIR, f"samples_{timestamp}")
    os.makedirs(samples_dir, exist_ok=True)
    # Save to .npz file
    npz_path = os.path.join(samples_dir, "generated_samples.npz")
    np.savez(
        npz_path,
        true_params=true_params_all, # Shape: (N_MC_RUNS, 4)
        noisy_data=all_noisy_data, # Shape: (N_MC_RUNS, data_length)
        true_dC=all_true_dC, # Shape: (N_MC_RUNS, data_length)
        noise_std=NOISE_STD,
        initial_guess=INITIAL_GUESS
    )
    # Save dataset info as txt
    info_path = os.path.join(samples_dir, "dataset_info.txt")
    with open(info_path, 'w') as f:
        f.write("=== MONTE CARLO DATASET INFO ===\n\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write(f"Number of Monte Carlo runs: {N_MC_RUNS}\n")
        f.write(f"True parameters shape: {true_params_all.shape}\n")
        f.write(f"Noisy data shape: {all_noisy_data.shape}\n")
        f.write(f"True dC shape: {all_true_dC.shape}\n")
        f.write(f"Noise standard deviation: {NOISE_STD}\n")
        f.write(f"Data length per run: {all_noisy_data.shape[1]}\n\n")
        f.write("Parameter distributions:\n")
        for key, dist in PARAM_DIST.items():
            f.write(f" {key}: mean={dist['mean']}, std={dist['std']}, "
                    f"range=[{dist['low']}, {dist['high']}]\n")
        f.write("\nTo load the data:\n")
        f.write(" data = np.load('generated_samples.npz')\n")
        f.write(" true_params = data['true_params']\n")
        f.write(" noisy_data = data['noisy_data']\n")
        f.write(" true_dC = data['true_dC']\n")
    print(f"Generated samples saved to: {samples_dir}")
    print(f" - generated_samples.npz")
    print(f" - metadata.json")
    print(f" - functions.txt")
    print(f" - dataset_info.txt")
    return samples_dir

def sample_true_parameters(run_seed: int) -> np.ndarray:
    """
    Sample a single true parameter vector from truncated normal distributions.

    Each of the four parameters (over_stiffness, over_electrode, Q, offset) is sampled
    independently from a truncated normal distribution defined in PARAM_DIST.

    Parameters
    ----------
    run_seed : int
        Seed used to initialize NumPy's RNG for reproducible sampling for the given run.

    Returns
    -------
    np.ndarray, shape (4,)
        Sampled true parameter vector: [over_stiffness, over_electrode, Q, offset].

    Notes
    -----
    - The function uses `scipy.stats.truncnorm.rvs` to sample within the provided bounds.
    - Seeding is performed inside the function to ensure each run is reproducible when
      called in parallel with different seeds.
    """

    np.random.seed(run_seed)
    params = np.zeros(4)
    for idx, key in enumerate(['over_stiffness', 'over_electrode', 'Q', 'offset']):
        dist = PARAM_DIST[key]
        a, b = (dist['low'] - dist['mean']) / dist['std'], (dist['high'] - dist['mean']) / dist['std']
        params[idx] = truncnorm.rvs(a, b, loc=dist['mean'], scale=dist['std'])
    return params

def run_single_mc_simulation(run_id: int, true_params: np.ndarray) -> dict:
    """
    Execute one Monte Carlo trial: simulate data, run UKF tracking (iteratively), and collect diagnostics.

    Workflow inside this function
    -----------------------------
    1. Build a 5D model vector from `true_params` and compute a static sensitivity.
    2. Compute the minimal safe drive amplitude via `compute_min_A(...)`.
    3. Simulate the system using `solve_ODE(...)` with the chosen amplitude and extract chirp portion.
    4. Add Gaussian noise (NOISE_STD) to obtain `noisy_data`.
    5. Instantiate `ParameterTracker` and run `tracker.run_tracking()` to estimate parameters.
       - The code supports iterative re-initialization of the tracker while holding some states fixed
         and checks convergence across parameter estimates between iterations.
    6. Compute final parameter errors (normalized by the defined parameter ranges),
       sensitivity error, uncertainties, and gather run-level diagnostics (runtime, convergence).

    Parameters
    ----------
    run_id : int
        Integer identifier for the run. Used to seed the RNG and name outputs.
    true_params : np.ndarray, shape (4,)
        True parameters for this run: [over_stiffness, over_electrode, Q, offset].

    Returns
    -------
    dict
        Dictionary containing scalar and array results for the run. Keys include:
        - 'run_id', 'true_stiff', 'true_elec', 'true_Q', 'true_offset', 'true_sensitivity'
        - 'est_stiff', 'est_elec', 'est_Q', 'est_offset', 'est_sensitivity'
        - 'std_stiff', 'std_elec', 'std_Q', 'std_offset', 'std_sensitivity'
        - 'err_stiff_norm', 'err_elec_norm', 'err_Q_norm', 'err_offset_norm', 'err_sensitivity_per'
        - 'total_param_error_norm', 'failed', 'noisy_data', 'true_dC', 'runtime', 'min_voltage'
        - times for detected peaks/valleys: 'third_peak_time', 'fourth_peak_time', 'third_valley_time', 'fourth_valley_time'
        - 'history', 'cov_history', 'iterations_used', 'converged'

    Exceptions & robustness
    -----------------------
    - If `tracker.run_tracking()` raises a `LinAlgError` flagged as "Irreversible Cholesky failure",
      the function captures it, marks the iteration as failed, and returns the best available estimate.
    - The function uses truncated sampling and floors where appropriate to avoid NaNs; still,
      callers should validate outputs before downstream aggregation.

    Performance & parallelism
    -------------------------
    - This function is intentionally self-contained and seeds RNG locally so it can safely run
      inside a multiprocessing worker. However, heavy initialization (Numba compilation, ODE integration)
      may happen per worker—expect compile-time overhead on first use inside each process.
    """

    start_run = time()  # Start measuring time for this MC run
    np.random.seed(run_id)  # For reproducible noise generation
    model_5d = np.zeros(5)
    model_5d[0:2] = true_params[0:2]
    model_5d[2] = 0.0
    model_5d[3:5] = true_params[2:4]
    true_sens = compute_static_sensitivity(model_5d)
    min_A = compute_min_A(model_5d)
    # Generate true data
    true_dC = solve_ODE(model_5d, min_A)  # Assuming solve_ODE is defined elsewhere
    t_eval = np.arange(0.0, total_duration + _step, _step)
    idx_chirp_full = t_eval <= dur_chirp
    dC_chirp_full = true_dC[idx_chirp_full]
    peaks = find_peaks_1d(dC_chirp_full)
    valleys = find_valleys_1d(dC_chirp_full)
    t_eval_chirp = t_eval[idx_chirp_full]
    third_peak_time = t_eval_chirp[peaks[2]] if len(peaks) >= 3 else np.nan
    fourth_peak_time = t_eval_chirp[peaks[3]] if len(peaks) >= 3 else np.nan
    third_valley_time = t_eval_chirp[valleys[2]] if len(valleys) >= 3 else np.nan
    fourth_valley_time = t_eval_chirp[valleys[3]] if len(valleys) >= 3 else np.nan
    noisy_data = true_dC + np.random.normal(0, NOISE_STD, len(true_dC))
    # UKF tracking parameters
    initial_cov_diag = exploration_config.initial_covariance
    initial_cov = np.diag(initial_cov_diag)
    process_noise = np.array(exploration_config.process_noise)
    measurement_noise = NOISE_STD ** 2 * 100

    # Convergence parameters
    CONVERGENCE_THRESHOLD = 0.0001  # 0.01% relative change
    MAX_ITERATIONS = exploration_config.num_iter
    min_iterations = 2  # Minimum iterations before checking convergence

    failed = False
    # Initial state and covariance for first iteration
    current_state = INITIAL_GUESS.copy()
    current_cov = initial_cov.copy()
    final_mean = None
    final_cov = None
    all_estimates = None
    smoothed_covs = None

    # Store previous iteration's results for convergence check
    prev_estimate = None
    prev_params = None

    # Track iterations
    iter_num = 0
    converged = False

    while iter_num < MAX_ITERATIONS and (iter_num < min_iterations or not converged):
        tracker = ParameterTracker(exploration_config)
        tracker.initialize(current_state, current_cov, process_noise, measurement_noise)
        iter_failed = False
        try:
            final_estimate, all_estimates, _, _, smoothed_covs = tracker.run_tracking(noisy_data, min_A)
        except np.linalg.LinAlgError as e:
            if "Irreversible Cholesky failure" in str(e):
                iter_failed = True
                final_estimate = tracker.ukf.x.copy()
                smoothed_covs = None
            else:
                raise

        # Check convergence after minimum iterations
        if iter_num >= min_iterations - 1 and prev_estimate is not None:
            # Get parameter estimates from current and previous iteration
            current_params = final_estimate[:4]

            # Calculate relative changes for each parameter
            rel_changes = np.abs((current_params - prev_params) / prev_params)
            # Handle near-zero parameters
            for i in range(4):
                if abs(prev_params[i]) < 1e-12:
                    rel_changes[i] = abs(current_params[i] - prev_params[i])

            # Check if all parameters have converged
            if np.all(rel_changes < CONVERGENCE_THRESHOLD):
                converged = True
                if tracker.show:
                    logger.info(
                        f"Run {run_id}: Convergence reached at iteration {iter_num + 1}. "
                        f"Relative changes: stiff={rel_changes[0]:.4%}, "
                        f"elec={rel_changes[1]:.4%}, Q={rel_changes[2]:.4%}, "
                        f"offset={rel_changes[3]:.4%}"
                    )

        # Store current results for next convergence check
        prev_estimate = final_estimate.copy()
        prev_params = prev_estimate[:4]

        final_mean = final_estimate
        final_cov = smoothed_covs[-1] if smoothed_covs is not None else tracker.ukf.get_covariance()
        failed = failed or iter_failed

        # Prepare for next iteration if not converged and not at max iterations
        if iter_num < MAX_ITERATIONS - 1 and not converged:
            # Prepare for next iteration: reset dynamic states and their covariances
            current_state = final_mean.copy()
            current_state[4:] = 0.0
            current_cov = final_cov.copy()
            current_cov[4:, 4:] = np.diag([initial_cov_diag[i] for i in range(4, 6)])
            current_cov[4:, :4] = 0.0
            current_cov[:4, 4:] = 0.0

        iter_num += 1

    # Log convergence status
    if converged:
        logger.info(f"Run {run_id}: Converged after {iter_num} iterations (max={MAX_ITERATIONS})")
    else:
        logger.info(f"Run {run_id}: Did not converge after {iter_num} iterations")

    est_params = final_mean[:4]
    est_std = np.sqrt(np.diag(final_cov[:4, :4]))
    est_model_5d = np.zeros(5)
    est_model_5d[0:2] = est_params[0:2]
    est_model_5d[2] = 0.0
    est_model_5d[3:5] = est_params[2:4]
    est_sens = compute_static_sensitivity(est_model_5d)
    J_sens = sens_jacobian(est_params)
    var_sens = J_sens @ final_cov[:4, :4] @ J_sens
    std_sens = np.sqrt(var_sens)
    # Normalization ranges
    range_stiff = PARAM_DIST['over_stiffness']['high'] - PARAM_DIST['over_stiffness']['low']
    range_elec = PARAM_DIST['over_electrode']['high'] - PARAM_DIST['over_electrode']['low']
    range_Q = PARAM_DIST['Q']['high'] - PARAM_DIST['Q']['low']
    range_offset = PARAM_DIST['offset']['high'] - PARAM_DIST['offset']['low']
    # Errors
    err_stiff = abs(est_params[0] - true_params[0]) / range_stiff
    err_elec = abs(est_params[1] - true_params[1]) / range_elec
    err_Q = abs(est_params[2] - true_params[2]) / range_Q
    err_offset = abs(est_params[3] - true_params[3]) / range_offset
    err_sens = abs(est_sens - true_sens) / abs(true_sens) if abs(true_sens) > 1e-8 else np.nan
    total_param_error = err_stiff + err_elec + err_Q + err_offset
    runtime = time() - start_run  # Total time for this MC run

    return {
        'run_id': run_id,
        'true_stiff': true_params[0],
        'true_elec': true_params[1],
        'true_Q': true_params[2],
        'true_offset': true_params[3],
        'true_sensitivity': true_sens,
        'est_stiff': est_params[0],
        'est_elec': est_params[1],
        'est_Q': est_params[2],
        'est_offset': est_params[3],
        'est_sensitivity': est_sens,
        'std_stiff': est_std[0],
        'std_elec': est_std[1],
        'std_Q': est_std[2],
        'std_offset': est_std[3],
        'std_sensitivity': std_sens,
        'err_stiff_norm': err_stiff * 100,
        'err_elec_norm': err_elec * 100,
        'err_Q_norm': err_Q * 100,
        'err_offset_norm': err_offset * 100,
        'err_sensitivity_per': err_sens * 100 if not np.isnan(err_sens) else np.nan,
        'total_param_error_norm': total_param_error * 100,
        'failed': failed,
        'noisy_data': noisy_data,  # Store for saving
        'true_dC': true_dC,  # Store for saving
        'runtime': runtime,  # Add runtime
        'min_voltage': min_A,  # Add min_voltage
        'third_peak_time': third_peak_time,
        'fourth_peak_time': fourth_peak_time,
        'third_valley_time': third_valley_time,
        'fourth_valley_time': fourth_valley_time,
        'history': all_estimates,
        'cov_history': smoothed_covs,
        'iterations_used': iter_num,  # Add number of iterations actually used
        'converged': converged  # Add convergence status
    }

def worker(args: Tuple[int, np.ndarray]) -> dict:
    """
    Worker wrapper to enable multiprocessing.Pool mapping.

    Parameters
    ----------
    args : tuple
        Tuple of (run_id, params) accepted by `run_single_mc_simulation`.

    Returns
    -------
    dict
        The result dictionary returned by `run_single_mc_simulation(run_id, params)`.

    Notes
    -----
    - A trivial adaptor used so Pool.imap / starmap style calls can be used uniformly.
    """

    i, params = args
    return run_single_mc_simulation(i, params)

def main_monte_carlo() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the entire Monte Carlo validation campaign end-to-end and save results.

    High-level actions performed
    ----------------------------
    1. Sample `N_MC_RUNS` true parameters.
    2. Persist histograms of sampled parameter distributions.
    3. Execute MC runs in parallel using multiprocessing.
    4. Collect per-run results and assemble a DataFrame, saving it to CSV.
    5. Save generated noisy samples (.npz) and textual metadata.
    6. Compute a statistical summary of the results and save to text.
    7. Produce a variety of diagnostic plots (error distributions, peak/valley timing, sensitivity comparisons).
    8. For best/worst sensitivity error cases, generate performance plots using `generate_performance_plots`.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - df: DataFrame containing per-run results (one row per MC run).
        - stats: DataFrame or textual summary containing computed statistics (same as what is written
                 to the `statistical_summary_{timestamp}.txt` file).

    Side effects
    ------------
    - Writes CSV, .npz sample archives, multiple PNG diagnostic plots, and a textual statistics summary
      to the OUTPUT_DIR and PLOTS_DIR directories.
    - Prints progress and status messages to stdout and logs via `logger`.

    Performance considerations
    --------------------------
    - The function spawns a process pool of size `os.cpu_count()`. If each worker must compile
      Numba code, the first-run overhead per process may be substantial. For reproducible CI runs,
      consider running once to compile (warm-up) or running with fewer processes.
    - Storing all histories and covariance traces in memory for very large N_MC_RUNS can be memory-heavy.
      Consider saving per-run artifacts to disk and reducing in-memory aggregation if needed.

    Notes on outputs & postprocessing
    ---------------------------------
    - The saved CSV stores numeric diagnostics and some arrays; large arrays (noisy_data, true_dC)
      are excluded from the CSV and saved to the .npz archive instead.
    - The function computes time-based statistics using `compute_stats` (a Numba-accelerated helper)
      and writes both numeric reports and plots for human analysis.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"{CSV_FILENAME_PREFIX}_{timestamp}.csv")
    print(f"Starting Monte Carlo validation with {N_MC_RUNS} runs...")
    print(f"Results will be saved to: {csv_path}")
    # Sample true parameters for all runs
    print("Sampling true parameters for all runs...")
    true_params_all = np.array([sample_true_parameters(i) for i in range(N_MC_RUNS)])
    # Generate plots of sampled parameter distributions
    print("Generating plots of sampled parameter distributions...")
    param_names = ['over_stiffness', 'over_electrode', 'Q', 'offset']
    for idx, name in enumerate(param_names):
        plt.figure(figsize=(12, 8))
        sns.histplot(true_params_all[:, idx], kde=True, bins=50)
        plt.title(f'Distribution of Sampled {name}')
        plt.xlabel(name)
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, f"sampled_{name}_distribution_{timestamp}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()
    print(f"Sampled parameter distribution plots saved in: {PLOTS_DIR}")
    print("Running Monte Carlo simulations in parallel...")
    start_parallel = time()

    args = [(i, true_params_all[i]) for i in range(N_MC_RUNS)]
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(worker, args), total=N_MC_RUNS, desc="Monte Carlo Runs", unit="run"))

    parallel_time = time() - start_parallel
    mean_parallel = parallel_time / N_MC_RUNS
    # Extract results and noisy data
    all_noisy_data = []
    all_true_dC = []
    all_histories = []
    all_cov_histories = []
    for result in results:
        all_noisy_data.append(result.pop('noisy_data'))
        all_true_dC.append(result.pop('true_dC'))
        all_histories.append(result.pop('history'))
        all_cov_histories.append(result.pop('cov_history'))
    all_noisy_data = np.array(all_noisy_data)
    all_true_dC = np.array(all_true_dC)
    # Save generated samples
    print("Saving generated samples...")
    save_generated_samples(true_params_all, all_noisy_data, all_true_dC, timestamp)
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")
    # Save failed cases if any
    df_failed = df[df['failed']]
    if not df_failed.empty:
        failed_path = os.path.join(OUTPUT_DIR, f"failed_parameter_cases_{timestamp}.csv")
        df_failed.to_csv(failed_path, index=False)
        print(f"Failed parameter cases saved to: {failed_path}")
    # Compute and save statistical summary
    print("Computing statistical summary...")
    stats = compute_statistical_figures(df)
    stats_columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis', 'median', 'mad']
    # Compute time statistics
    time_arr = df['runtime'].values
    time_stats_arr = compute_stats(time_arr)
    time_stats = pd.DataFrame([time_stats_arr], columns=stats_columns)
    # Compute min_voltage statistics
    min_voltage_arr = df['min_voltage'].values
    min_voltage_stats_arr = compute_stats(min_voltage_arr)
    min_voltage_stats = pd.DataFrame([min_voltage_stats_arr], columns=stats_columns)
    # Compute third_peak_time statistics
    third_peak_time_arr = df['third_peak_time'].values
    third_peak_time_stats_arr = compute_stats(third_peak_time_arr)
    third_peak_time_stats = pd.DataFrame([third_peak_time_stats_arr], columns=stats_columns)
    # Compute fourth_peak_time statistics
    fourth_peak_time_arr = df['fourth_peak_time'].values
    fourth_peak_time_stats_arr = compute_stats(fourth_peak_time_arr)
    fourth_peak_time_stats = pd.DataFrame([fourth_peak_time_stats_arr], columns=stats_columns)
    # Compute third_valley_time statistics
    third_valley_time_arr = df['third_valley_time'].values
    third_valley_time_stats_arr = compute_stats(third_valley_time_arr)
    third_valley_time_stats = pd.DataFrame([third_valley_time_stats_arr], columns=stats_columns)
    # Compute fourth_valley_time statistics
    fourth_valley_time_arr = df['fourth_valley_time'].values
    fourth_valley_time_stats_arr = compute_stats(fourth_valley_time_arr)
    fourth_valley_time_stats = pd.DataFrame([fourth_valley_time_stats_arr], columns=stats_columns)
    stats_path = os.path.join(OUTPUT_DIR, f"statistical_summary_{timestamp}.txt")
    with open(stats_path, "w") as f:
        f.write(f"Monte Carlo UKF Validation - {N_MC_RUNS} runs\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=== STATISTICAL FIGURES ===\n\n")
        f.write(stats.to_string())
        f.write("\n\n=== ADDITIONAL STATISTICS ===\n\n")
        f.write(f"Success rate: {(len(df) - len(df_failed)) / len(df) * 100:.2f}%\n")
        f.write(f"Failed runs: {len(df_failed)}\n")
        f.write(f"\nError statistics (median ± mad):\n")
        for col in ['err_stiff_norm', 'err_elec_norm', 'err_Q_norm', 'err_offset_norm', 'err_sensitivity_per']:
            col_arr = df[col].values
            col_stats = compute_stats(col_arr)
            median_val = col_stats[5]
            mad_val = col_stats[11]
            f.write(f" {col}: {median_val:.4f} ± {mad_val:.4f} %\n")
        f.write("\n\n=== TIME STATISTICS (PER RUN) ===\n\n")
        f.write(time_stats.to_string())
        f.write("\n\n=== MIN VOLTAGE STATISTICS (PER RUN) ===\n\n")
        f.write(min_voltage_stats.to_string())
        f.write("\n\n=== THIRD PEAK TIME STATISTICS (PER RUN) ===\n\n")
        f.write(third_peak_time_stats.to_string())
        f.write("\n\n=== FOURTH PEAK TIME STATISTICS (PER RUN) ===\n\n")
        f.write(fourth_peak_time_stats.to_string())
        f.write("\n\n=== THIRD VALLEY TIME STATISTICS (PER RUN) ===\n\n")
        f.write(third_valley_time_stats.to_string())
        f.write("\n\n=== FOURTH VALLEY TIME STATISTICS (PER RUN) ===\n\n")
        f.write(fourth_valley_time_stats.to_string())
        f.write(f"\n\nMean runtime per run (parallel): {mean_parallel:.4f} seconds\n")
        # f.write(f"Mean runtime per run (serial estimate from subset): {mean_serial:.4f} seconds\n")
    print(f"Statistical summary saved to: {stats_path}")
    # Save best/worst cases
    df_valid = df.dropna(subset=['err_sensitivity_per'])
    best_5 = df_valid.nsmallest(5, 'total_param_error_norm')
    worst_5 = df_valid.nlargest(5, 'total_param_error_norm')
    extremes_path = os.path.join(OUTPUT_DIR, f"best_worst_cases_{timestamp}.csv")
    pd.concat([best_5, worst_5]).to_csv(extremes_path, index=False)
    print(f"Best/worst 5 cases saved to: {extremes_path}")
    # Save tracking plots for best and worst sensitivity error cases
    print("Saving tracking plots for best and worst cases...")
    best_idx = df_valid['err_sensitivity_per'].idxmin()
    worst_idx = df_valid['err_sensitivity_per'].idxmax()
    for case, idx in [('best', best_idx), ('worst', worst_idx)]:
        history = all_histories[idx]
        covariance_history = all_cov_histories[idx]
        true_params = true_params_all[idx]
        noisy_data = all_noisy_data[idx]
        true_measurements = all_true_dC[idx]
        min_A = df.loc[idx, 'min_voltage']
        model_5d = np.zeros(5)
        model_5d[0:2] = true_params[0:2]
        model_5d[2] = 0.0
        model_5d[3:5] = true_params[2:4]
        true_disp, true_vel = solve_ODE_disp(model_5d, min_A)
        t_span = np.arange(0.0, len(true_measurements) * _step, _step)
        generate_performance_plots(history, covariance_history, true_params,
                                   true_disp, true_vel,
                                   true_measurements, noisy_data, t_span,
                                   min_A, save_prefix=f"{case}_{timestamp}")
        # Plot noisy responses
        plt.figure(figsize=(10, 6))
        plt.plot(t_span, noisy_data, 'gray', label='Noisy ΔC')
        plt.xlabel('Time (s)')
        plt.ylabel('ΔC (F)')
        plt.title('Noisy Responses (ΔC vs Time)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, f"{case}_{timestamp}_noisy_deltaC.png"), dpi=200, bbox_inches='tight')
        plt.close()

        # Plot voltage stimuli history
        def single_block_voltage(t, min_voltage):
            t = np.asarray(t)
            V = np.zeros_like(t)
            mask_chirp = (0 <= t) & (t < dur_chirp)
            V[mask_chirp] = min_voltage * chirp(t[mask_chirp], f0=f_start, f1=f_end, t1=dur_chirp, phi=-90)
            mask_const = (dur_chirp <= t) & (t < dur_chirp)
            V[mask_const] = -min_voltage
            # zero elsewhere within block; outside caller should handle
            return V

        plt.figure(figsize=(10, 6))
        V = single_block_voltage(t_span, min_voltage=min_A)
        plt.plot(t_span, V, 'b-', label='Right Electrode Voltage (V_right)')
        plt.plot(t_span, -V, 'r--', label='Left Electrode Voltage (V_left)')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage Stimuli History (V vs Time)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, f"{case}_{timestamp}_voltage_history.png"), dpi=200, bbox_inches='tight')
        plt.close()
    # Generate diagnostic plots
    print("Generating diagnostic plots...")
    # Sensitivity error distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['err_sensitivity_per'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Sensitivity Relative Error (%)')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"sensitivity_error_distribution_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # Min voltage distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['min_voltage'], kde=True, bins=50)
    plt.title('Distribution of Minimum Voltage')
    plt.xlabel('Minimum Voltage')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"min_voltage_distribution_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # Third peak time distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['third_peak_time'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Third Peak Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"third_peak_time_distribution_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # Fourth peak time distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['fourth_peak_time'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Fourth Peak Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"fourth_peak_time_distribution_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # Third valley time distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['third_valley_time'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Third Valley Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"third_valley_time_distribution_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # Fourth valley time distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['fourth_valley_time'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Fourth Valley Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"fourth_valley_time_distribution_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # Error boxplots
    plt.figure(figsize=(14, 10))
    error_cols = ['err_stiff_norm', 'err_elec_norm', 'err_Q_norm', 'err_offset_norm', 'err_sensitivity_per']
    df[error_cols].boxplot()
    plt.title('Normalized Parameter & Sensitivity Errors Across All Runs')
    plt.ylabel('Normalized Error (%)')
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"error_boxplots_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # True vs Estimated Sensitivity scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df['true_sensitivity'], df['est_sensitivity'], alpha=0.6)
    min_s = min(df['true_sensitivity'].min(), df['est_sensitivity'].min())
    max_s = max(df['true_sensitivity'].max(), df['est_sensitivity'].max())
    plt.plot([min_s, max_s], [min_s, max_s], 'r--', lw=2, label='Perfect estimation')
    plt.xlabel('True Sensitivity')
    plt.ylabel('Estimated Sensitivity')
    plt.title('True vs Estimated Sensitivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, f"sensitivity_true_vs_est_{timestamp}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"All diagnostic plots saved in: {PLOTS_DIR}")
    print("\nMonte Carlo validation completed successfully!")
    return df, stats