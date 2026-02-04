"""
Module for parameter tracking using a Square-Root Unscented Kalman Filter (SR-UKF).

This module provides the `ParameterTracker` class which wraps an SR-UKF (from `srukf.SRUKF`)
to estimate corrected MEMS parameters online from a stream of capacitance measurements.
It integrates model propagation, measurement prediction (via `Capacitance_left` /
`Capacitance_right`), sensitivity computation, convergence monitoring, and basic logging.

Notes / high-level behaviour
----------------------------
- The tracker maintains an SR-UKF instance (`self.ukf`) that holds state x, covariance history,
  and UKF internals. The UKF state dimension is 6 (by default in SRUKF instantiation here).
- Measurements are scalar capacitance-difference values derived from the model's capacitance
  functions; these are wrapped in a small measurement function which is compiled with `numba.njit`
  inside `run_tracking` (see warnings below).
- At each measurement step the tracker:
    1. forms a process function that either is identity (first iteration) or calls `propagate_state`
       to advance the model by dt.
    2. calls `ukf.predict(process_func, meas_func)` to predict sigma points / measurement statistics.
    3. computes (optionally) a true measurement Jacobian `H_k_true` via `compute_meas_gradient`.
    4. calls `ukf.update(...)` to perform the measurement update.
    5. computes the scalar sensitivity `sens = compute_static_sensitivity(model_state)` and
       its propagated uncertainty `std_s` using `sens_jacobian` and the UKF covariance.
    6. appends estimates and covariance history and checks a convergence criterion based on
       relative changes in sensitivity and its std over a sliding window.

Logging and output
------------------
- `self.show` controls periodic logging of the current estimate and ±1.96·σ diagonal uncertainties.
- `self.running_time` is set to the wall-clock time (seconds) spent in `run_tracking` (from entry to return).
- `run_tracking` returns five values:
    (current_estimate, estimates, filtered_estimates, filtered_covs, filtered_covs)
  Note: the last two returned values are currently identical (both `filtered_covs`); this is
  preserved for backwards compatibility with existing calling code but may be redundant.

Example usage
-------------
>>> tracker = ParameterTracker(config=my_config)
>>> tracker.initialize(initial_guess=np.array([0.35,0.35,0.5,0.,0.,0.]),
...                    initial_uncertainty=np.eye(6)*1e-2,
...                    process_noise=np.eye(6)*1e-4,
...                    measurement_noise=1e-3)
>>> final_est, estimates, filtered_estimates, filtered_covs, _ = tracker.run_tracking(meas_array, min_voltage=1.2)
"""
import numpy as np
import time
from typing import List, Tuple, Optional

from model import propagate_state, meas_func
from srukf import SRUKF
from config import UKFConfig, logger, _step
from model import compute_meas_gradient, compute_static_sensitivity
from utils import sens_jacobian


class ParameterTracker:
    """
    MEMS parameter estimation tracker using an SR-UKF.

    Attributes
    ----------
    ukf : srukf.SRUKF
        Instance of the square-root UKF holding the filter state, covariance and history.
        It is instantiated with `state_dim=6` by default.
    measurement_data : Optional[np.ndarray]
        Most recent measurement array passed to `run_tracking`.
    dt : float
        Time step used for propagation (taken from `config._step`).
    running_time : float
        Last measured wall-clock runtime (seconds) for `run_tracking`.
    show : bool
        When True, periodically logs progress and diagnostic info to `logger`.

    Methods
    -------
    initialize(initial_guess, initial_uncertainty, process_noise=None, measurement_noise=None)
        Initialize UKF state, covariance and optionally noise parameters.
    run_tracking(measurement_data, min_voltage) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        Run the tracking loop for the provided measurements and return estimates and covariances.
    """
    def __init__(self, config: UKFConfig = None) -> None:
        """
        Create a ParameterTracker.

        Parameters
        ----------
        config : UKFConfig, optional
            Configuration object passed to the SRUKF constructor. If `None`, SRUKF uses its default config.
        """
        self.ukf = SRUKF(state_dim=6, config=config)
        self.measurement_data = None
        self.dt = _step # Assume _step is defined globally
        self.running_time = 0.0 # Initialize running time
        self.show = True

    def initialize(self, initial_guess: np.ndarray, initial_uncertainty: np.ndarray,
                   process_noise: Optional[np.ndarray] = None,
                   measurement_noise: Optional[float] = None) -> None:
        """
        Initialize the UKF internal state and optionally set process/measurement noise.

        Parameters
        ----------
        initial_guess : np.ndarray, shape (6,)
            Initial state vector for the UKF. Expected layout:
            [est_stiff, est_elec, est_Q, est_offset, position, velocity]
            (matching the SRUKF state convention).
        initial_uncertainty : np.ndarray, shape (6, 6)
            Initial covariance (or square-root equivalent depending on SRUKF API).
        process_noise : Optional[np.ndarray], shape (6, 6)
            Additive process noise covariance to set inside the UKF. If None, leave existing value.
        measurement_noise : Optional[float]
            Measurement noise variance (scalar) to set in the UKF. If None, leave existing value.

        Notes
        -----
        - This function delegates to `self.ukf.initialize(...)` and to `ukf.set_process_noise`
          / `ukf.set_measurement_noise` if noise arguments are provided.
        """
        self.ukf.initialize(initial_guess, initial_uncertainty)
        if process_noise is not None:
            self.ukf.set_process_noise(process_noise)
        if measurement_noise is not None:
            self.ukf.set_measurement_noise(measurement_noise)

    def run_tracking(self, measurement_data: np.ndarray, min_voltage: float) -> Tuple[
        np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Run the UKF tracking loop over `measurement_data`.

        At each time step the UKF predict and update are executed. The update optionally
        uses a true measurement Jacobian computed by `compute_meas_gradient`. After each
        update the code computes a scalar sensitivity value and its propagated uncertainty
        (std) using `compute_static_sensitivity`, `sens_jacobian` and the UKF covariance.

        Parameters
        ----------
        measurement_data : np.ndarray, shape (T,)
            Sequence of scalar capacitance-difference measurements (ΔC) to be processed
            sequentially by the filter. Length T is the number of iterations performed.
        min_voltage : float
            Voltage argument passed to `propagate_state` when advancing the model.
            Also used when computing the final estimated model output for diagnostics.

        Returns
        -------
        current_estimate : np.ndarray, shape (6,)
            The final UKF state estimate (most recent).
        estimates : list[np.ndarray]
            List of state estimates at each iteration (each a 1D np.ndarray of length 6).
        filtered_estimates : list[np.ndarray]
            Alias for `estimates` (preserved for API compatibility).
        filtered_covs : list[np.ndarray]
            Filtered covariance history (sliced from `self.ukf.covariance_history`).
        filtered_covs_dup : list[np.ndarray]
            Duplicate of `filtered_covs` (note: currently identical on return; consider refactoring).

        Behavioural details
        ------------------
        - A local `meas_func(state)` is defined and decorated with `@njit`. It computes a scalar
          measurement as 10*(Capacitance_right(pos, over2_m, offset_m) - Capacitance_left(...)).
          The function guards against non-finite outputs by substituting 0.0.
        - The process function used in `ukf.predict` is `lambda state: state` for the first
          iteration and `lambda state: propagate_state(state, current_time, self.dt, min_voltage=min_voltage)`
          afterwards.
        - Convergence is checked every iteration once at least `CONVERGENCE_WINDOW` samples
          have been collected. Convergence criterion compares relative changes of sensitivity
          and its std over the window to `CONVERGENCE_THRESHOLD`. If both are below the threshold,
          the loop is terminated early.
        - Periodic logging prints the current estimate and ±1.96·σ diagonal uncertainties; the
          logging cadence is approximately `iteration % (num_measurements // 3) == 0`.

        Example
        -------
        >>> tracker = ParameterTracker(config=my_config)
        >>> tracker.initialize(initial_guess, initial_uncertainty)
        >>> final_est, estimates, filt_est, filt_covs, _ = tracker.run_tracking(measurements, min_voltage=1.2)
        """
        start_time = time.time() # Start measuring tracking time
        self.measurement_data = measurement_data
        estimates = []
        self.ukf.sigma_points_pred = None
        num_measurements = len(measurement_data)
        current_time = 0.0
        # Convergence monitoring parameters
        CONVERGENCE_WINDOW = 100
        CONVERGENCE_THRESHOLD = 0.0001  # 0.01%
        sensitivity_history = []
        std_sensitivity_history = []
        current_estimate = [0.35, .35, .5, 0., 0., 0.]
        for iteration in range(num_measurements):
            if iteration == 0:
                process_func = lambda state: state
            else:
                process_func = lambda state: propagate_state(state, current_time, self.dt, min_voltage=min_voltage)
            P_xy, P_yy, z_pred, S_zz_tilde = self.ukf.predict(process_func, meas_func)
            # Compute true Jacobian if config.use_true_jacobian is True
            H_k_true = compute_meas_gradient(self.ukf.x)
            # Pass the true Jacobian to update
            current_estimate = self.ukf.update(
                float(measurement_data[iteration]),
                P_xy, P_yy, z_pred, S_zz_tilde,
                H_k_true=H_k_true
            )
            estimates.append(current_estimate.copy())
            params = current_estimate[:4]
            model_state = np.zeros(5)
            model_state[0:2] = params[0:2]
            model_state[2] = 0.
            model_state[3:5] = params[2:4]
            sens = compute_static_sensitivity(model_state)
            J = sens_jacobian(params)
            cov = self.ukf.get_covariance()[:4, :4]
            var_s = np.dot(J, np.dot(cov, J))
            std_s = np.sqrt(var_s)
            sensitivity_history.append(sens)
            std_sensitivity_history.append(std_s)
            if iteration >= CONVERGENCE_WINDOW:
                # Get the values from CONVERGENCE_WINDOW iterations ago
                sens_old = sensitivity_history[-CONVERGENCE_WINDOW]
                std_old = std_sensitivity_history[-CONVERGENCE_WINDOW]

                # Calculate relative changes
                sens_change = abs((sens - sens_old) / sens_old) if abs(sens_old) > 1e-12 else abs(sens - sens_old)
                std_change = abs((std_s - std_old) / std_old) if abs(std_old) > 1e-12 else abs(std_s - std_old)

                # Check if both sensitivity and its uncertainty have converged
                if sens_change < CONVERGENCE_THRESHOLD and std_change < CONVERGENCE_THRESHOLD:
                    if self.show:
                        logger.info(
                            f"Convergence reached at iteration {iteration}: "
                            f"sensitivity change={sens_change:.6%}, "
                            f"std change={std_change:.6%}"
                        )
                    break
            if iteration > 0:
                current_time += self.dt
            if self.show:
                cov_full = self.ukf.get_covariance()
                sigma = 1.96 * np.sqrt(np.diag(cov_full))
                vals = np.round(current_estimate, 4)
                if iteration % (num_measurements // 3) == 0:
                    state_str = "[" + ", ".join(f"{v:.4f}±{s:.4f}" for v, s in zip(vals, sigma)) + "]"
                    logger.info(
                        f"Iteration {iteration:3d}/{num_measurements}: t={current_time:.4f}, innovation={self.ukf.innovation_norm:.4f}, sens={sens:.4f}±{std_s:.4f}, state={state_str}")
        filtered_estimates = estimates
        filtered_covs = self.ukf.covariance_history[1:]
        self.running_time = time.time() - start_time
        return current_estimate, estimates, filtered_estimates, filtered_covs, filtered_covs