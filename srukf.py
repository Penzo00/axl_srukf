"""
Square-Root Unscented Kalman Filter (SR-UKF) implementation with additional
mechanisms described in the code comments (GMCC, Strong Tracking Fading (STF),
constrained gain projection, and adaptive noise estimation).

This module provides the `SRUKF` class — a compact SR-UKF tailored for a
6-dimensional state + scalar measurement (meas_dim == 1) and intended for
parameter/state estimation problems (e.g., MEMS parameter tracking).

Design notes
------------
- Square-root formulation: the filter stores and propagates the lower-triangular
  square-root factor `S` such that `P = S @ S.T`. This improves numerical stability
  and is used for QR-based covariance reconstitution.
- Measurement dimension is fixed to 1 in this implementation. Many internal
  routines assume scalar measurement outputs.
- The code contains three advanced features referenced in the comments:
    * GMCC: Generalized Maximum Correntropy Criterion based iterative update.
    * STF: Strong Tracking Fading (adaptive multiplication of predicted covariance).
    * Constrained gain: projection of gain/estimate onto linear inequality constraints.
- The implementation uses several helper functions imported from `utils`:
  - `generate_sigma_points`, `compute_ut_weights`, `_regularize_covariance`,
    `compute_constrained_gain`, `_compute_weighted_deviations`, `_compute_ggd_weight`.
  Ensure these are present and tested.

Caveats & recommendations
-------------------------
- `min_eig_floor` (from config) is used to avoid division by (near) zero and to
  regularize covariance inversions.
- The GMCC iterative update uses matrix inversions and per-iteration weight updates.
  If `B_k` or other matrices are ill-conditioned, numerical regularization occurs.
- The filter returns `self.x` after update. In case of a failed Cholesky on covariance
  update, the implementation returns the previous state and does not raise.
"""
import numpy as np
from typing import Callable, Optional, List, Tuple

from numpy.linalg import cholesky, qr, inv, LinAlgError

from config import UKFConfig, logger, min_eig_floor
from utils import (generate_sigma_points, compute_ut_weights, _regularize_covariance,
                    compute_constrained_gain, _compute_weighted_deviations, _compute_ggd_weight)

class SRUKF:
    """
        Square-Root Unscented Kalman Filter with GMCC, STF, constrained-gain and adaptive noise.

        Parameters
        ----------
        state_dim : int
            Dimensionality of the filter state vector (n). This implementation expects n=6 in the
            calling code but supports any n for which helper utilities are configured.
        config : UKFConfig, optional
            Configuration object providing algorithm hyperparameters (weights, noise, flags).
            If None, a default UKFConfig() is constructed.

        Attributes
        ----------
        state_dim : int
            State dimensionality.
        meas_dim : int
            Measurement dimensionality (fixed to 1 in this class).
        config : UKFConfig
            Configuration object holding hyperparameters.
        num_sigma : int
            Number of sigma points used (state_dim + 2 for this custom sigma scheme).
        alpha, beta : float
            Unscented transform parameters (read from config).
        weights_m, weights_c : np.ndarray
            Unscented transform mean and covariance weights (length num_sigma).
        x : np.ndarray, shape (state_dim,)
            Current state estimate.
        S : np.ndarray, shape (state_dim, state_dim)
            Lower-triangular square-root factor of state covariance.
        Q, S_q : np.ndarray
            Process-noise covariance and its square-root factor.
        R, S_r : float
            Measurement noise variance and its sqrt (scalar).
        V_k : float
            Residual covariance used by the STF (scalar).
        estimates_history, covariance_history, predicted_states, predicted_covs, cross_history, residual_history : lists
            Histories for debugging/analysis.
        constraint_A, constraint_b : Optional[np.ndarray]
            Linear inequality constraint matrices (A, b) representing A x <= b.
        innovation_norm : float
            Magnitude of last innovation.
        failed_updates : int
            Number of failed update attempts (numerical issues).

        Notes
        -----
        - The sigma-point generation and weight scheme are provided externally via util functions.
        - The filter uses a QR decomposition to reconstruct square-root covariance from a
          weighted deviations matrix augmented with process noise square-root.
        - Many algorithmic choices (GMCC iteration limits, STF toggles, adaptive flags) are
          controlled via `config`.
        """
    def __init__(self, state_dim: int, config: UKFConfig = None) -> None:
        self.state_dim = state_dim
        self.meas_dim = 1  # Fixed to 1 as per note
        self.config = config or UKFConfig()

        self.num_sigma = self.state_dim + 2
        self.beta = self.config.beta
        self.alpha = self.config.alpha

        self.weights_m, self.weights_c = compute_ut_weights()

        # State and covariance
        self.x: np.ndarray = np.zeros(state_dim)
        self.S: np.ndarray = np.eye(state_dim)

        # Noise matrices
        self.Q: np.ndarray = np.diag(self.config.process_noise)
        self.S_q: np.ndarray = cholesky(self.Q)
        self.R: float = 1.0
        self.S_r: float = 1.0

        # STF variables
        self.V_k: float = 0  # Residual covariance, scalar

        # History
        self.step = 0
        self.estimates_history: List[np.ndarray] = []
        self.covariance_history: List[np.ndarray] = []
        self.predicted_states: List[np.ndarray] = []
        self.predicted_covs: List[np.ndarray] = []
        self.cross_history: List[np.ndarray] = []
        self.residual_history: List[float] = []

        # Constraints (Paper 2)
        self.constraint_A = None
        self.constraint_b = None
        self.set_constraints_from_bounds([(0, 1.4), (0, 1.75), (0.01, 10), (-1.2, 1.2), (-1.2, 1.2), (-1e4, 1e4)])

        # Initialize other variables
        self.innovation_norm: float = 0.0
        self.failed_updates: int = 0

    def set_constraints_from_bounds(self, bounds):
        """
                Build linear inequality constraint matrices from per-state bounds.

                Parameters
                ----------
                bounds : iterable of (lb, ub)
                    Sequence of (lower_bound, upper_bound) tuples for each state dimension. Use
                    `np.inf`/`-np.inf` for unbounded sides.

                Effects
                -------
                Sets `self.constraint_A` and `self.constraint_b` such that A x <= b encodes the
                provided bounds. If no finite bounds are provided, the attributes are set to None.

                Notes
                -----
                - For each finite lower bound lb on state i, a row e_i^T x <= lb is added.
                - For each finite upper bound ub on state i, a row -e_i^T x <= -ub is added
                  (so that e_i^T x >= ub becomes -e_i^T x <= -ub).
                """
        A_list = []
        b_list = []
        for i, (lb, ub) in enumerate(bounds):
            if np.isfinite(lb):
                row = np.zeros(self.state_dim)
                row[i] = 1.0
                A_list.append(row)
                b_list.append(lb)
            if np.isfinite(ub):
                row = np.zeros(self.state_dim)
                row[i] = -1.0
                A_list.append(row)
                b_list.append(-ub)
        self.constraint_A = np.array(A_list) if A_list else None
        self.constraint_b = np.array(b_list) if b_list else None

    def initialize(self, initial_state: np.ndarray, initial_covariance: np.ndarray) -> None:
        """
        Initialize filter state and internal square-root covariance.

        Parameters
        ----------
        initial_state : np.ndarray, shape (state_dim,)
            Initial state estimate vector.
        initial_covariance : np.ndarray, shape (state_dim, state_dim)
            Initial covariance matrix (symmetric). It will be regularized and
            cholesky-decomposed to initialize `self.S`.

        Effects
        -------
        - Sets `self.x` to the provided initial state (copy).
        - Sets `self.S` to `cholesky(_regularize_covariance(initial_covariance))`.
        - Resets histories and counters (step, predicted lists, etc.) and pushes the
          initial state & covariance into the history lists.

        Notes
        -----
        - `_regularize_covariance` ensures positive definiteness (floors small eigenvalues).
        - The method copies arrays to avoid in-place aliasing side-effects.
        """
        self.x = initial_state.copy().astype(np.float64)
        self.S = cholesky(_regularize_covariance(initial_covariance))

        self.estimates_history = [self.x.copy()]
        self.covariance_history = [self.S @ self.S.T]
        self.step = 0
        self.predicted_states = []
        self.predicted_covs = []
        self.cross_history = []
        self.step = 0

    @staticmethod
    def _propagate_sigma_points(sigma_points: np.ndarray,
                                func: Callable, output_dim: int) -> np.ndarray:
        """
        Propagate sigma points through a function `func`.

        Parameters
        ----------
        sigma_points : np.ndarray, shape (state_dim, num_sigma)
            Columns are sigma points to be propagated.
        func : Callable[[np.ndarray], np.ndarray]
            Function mapping a single state-like vector to an output vector (length = output_dim).
        output_dim : int
            Dimensionality of the function's output.

        Returns
        -------
        np.ndarray
            If `output_dim == 1`, returns a 1D array of length `num_sigma` (flattened).
            Otherwise, returns an array of shape (output_dim, num_sigma) containing
            propagated sigma points as columns.

        Notes
        -----
        - This routine is written as a simple loop to avoid depending on vectorized
          semantics of `func`. If `func` supports batch evaluation, performance may be
          improved by a batch call.
        """
        num_sigma = sigma_points.shape[1]
        sigma_points_pred = np.zeros((output_dim, num_sigma))
        for i in range(num_sigma):
            sigma_points_pred[:, i] = func(sigma_points[:, i])

        if output_dim == 1:
            return sigma_points_pred.flatten()
        return sigma_points_pred

    def predict(self, process_func: Callable[[np.ndarray], np.ndarray],
                measurement_func: Callable[[np.ndarray], np.ndarray]) -> Tuple:
        """
        Prediction step (square-root formulation).

        Steps performed
        ----------------
        1. Generate sigma points from current state `self.x` and square-root `self.S`.
        2. Propagate sigma points through the process model (process_func).
        3. Compute predicted state mean using UT weights.
        4. Compute weighted deviations and assemble a weighted-deviations matrix W.
        5. Augment W with process-noise square-root `S_q` and perform QR to compute new `S`.
        6. Predict measurement sigma values via measurement_func and compute predicted
           measurement mean `z_pred`, scalar measurement covariance `P_yy`, and
           cross-covariance `P_xy`.

        Parameters
        ----------
        process_func : Callable[[np.ndarray], np.ndarray]
            Function that maps a state vector to the predicted state (same shape as state).
            For stationary models, this can be the identity function.
        measurement_func : Callable[[np.ndarray], np.ndarray]
            Function that maps a state vector to measurement(s). Expected to return a scalar
            (1D or 0D) because meas_dim == 1 in this implementation.

        Returns
        -------
        Tuple containing
        - P_xy : np.ndarray, shape (state_dim, 1)
            Cross-covariance between state and measurement.
        - P_yy : float
            Predicted measurement variance (scalar).
        - z_pred : float
            Predicted measurement mean.
        - S_zz_tilde : float
            Square-root of the measurement-innovation intermediate (before adding R).

        Notes
        -----
        - The function uses `_compute_weighted_deviations` to produce a weighted deviation
          matrix `W` using covariance weights, then augments `W` with `S_q` and performs QR.
        - For scalar measurement, the implementation reduces measurement sigma output to
          1D vectors and computes scalar variances using weighted sums.
        - The returned `P_xy` shape is `(state_dim, 1)` for convenient multiplication in the update step.
        """

        # Generate sigma points
        sigma_points = generate_sigma_points(self.x, self.S)

        # Propagate sigma points through process model
        sigma_points_pred = self._propagate_sigma_points(sigma_points, process_func, self.state_dim)

        # Compute predicted state mean
        x_pred = sigma_points_pred @ self.weights_m

        # Compute square-root of predicted covariance using QR decomposition
        # Form weighted deviations matrix
        W = _compute_weighted_deviations(sigma_points_pred, x_pred, self.weights_c)

        # Add process noise for predicted cov
        W_aug = np.hstack([W, self.S_q])

        # QR decomposition to get square-root covariance
        _, R = qr(W_aug.T)
        self.S = R.T  # Lower triangular square root

        # Update state
        self.x = x_pred.copy()

        # Measurement prediction
        Z_sigma = self._propagate_sigma_points(sigma_points_pred, measurement_func, self.meas_dim)
        z_pred = np.dot(Z_sigma, self.weights_m)

        # Compute measurement covariance
        dz = Z_sigma - z_pred

        P_zz_reg = np.sum(self.weights_c * dz**2)

        S_zz_tilde = np.sqrt(P_zz_reg)

        P_yy = P_zz_reg + self.R

        # Compute cross-covariance P_xy
        dx = sigma_points_pred - x_pred[:, np.newaxis]
        P_xy = np.dot(dx, self.weights_c * dz)[:, np.newaxis]

        return P_xy, P_yy, z_pred, S_zz_tilde

    def update(self, measurement: float, P_xy: np.ndarray, P_yy: float,
               z_pred: float,
               S_zz_tilde: float,
               H_k_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update step: apply measurement to correct the state estimate.

        This method implements a multi-mode update:
          - optionally uses a provided true measurement Jacobian `H_k_true`;
          - optionally performs Strong Tracking Fading (STF) to scale predicted covariance;
          - optionally performs a GMCC iterative robust update (if enabled in config);
          - otherwise performs the standard Kalman update for scalar measurement;
          - optionally projects gain/estimate to satisfy linear inequality constraints;
          - updates covariance via Joseph form and stores square-root via Cholesky;
          - optionally performs adaptive process/measurement noise updates.

        Parameters
        ----------
        measurement : float
            Observed scalar measurement at the current time step.
        P_xy : np.ndarray, shape (state_dim, 1)
            Cross-covariance between state and measurement from predict().
        P_yy : float
            Predicted measurement variance from predict().
        z_pred : float
            Predicted measurement mean from predict().
        S_zz_tilde : float
            Square-root of the measurement-innovation intermediate (before adding R).
        H_k_true : Optional[np.ndarray], shape (1, state_dim) or (state_dim,)
            If provided and `config.use_true_jacobian` is True, this Jacobian is used
            directly instead of the UT-based approximation.

        Returns
        -------
        np.ndarray, shape (state_dim,)
            Updated state estimate (self.x is updated in-place and returned).

        Notes and algorithmic details
        -----------------------------
        - Innovation `v = measurement - z_pred` and its absolute magnitude are stored.
        - Residual history is maintained for adaptive noise estimation and STF computations.
        - If `config.use_true_jacobian` and `H_k_true` provided, `H_k` is set to that Jacobian.
          Otherwise, an approximate Jacobian is computed as `inv(predicted_P) @ P_xy`.
        - STF: if enabled, computes scalar quantities and a fading factor phi_k to scale predicted_S.
        - GMCC (if enabled): performs an iterative reweighted update using Generalized Gaussian
          Density weights. The algorithm constructs an augmented linear system (B_k) and iteratively
          updates an estimate `x_hat` until relative change falls below `config.mcc_eps` or maximum
          iterations `config.mcc_max_iter` is reached. Internal weights use `_compute_ggd_weight`.
        - Standard update path computes Kalman gain `K = P_xy / P_yy` for scalar measurement.
        - Constrained gain: if enabled and constraints exist, calls `compute_constrained_gain`
          to project gain & estimate into constraint-consistent space.
        - Covariance update uses Joseph form to improve numerical stability:
              P_updated = (I - eL H_k) P_pred (I - eL H_k)^T + (eL * R) eL^T
          where `eL` is the effective gain vector (shape (state_dim,)).
        - The new square-root `S` is computed by Cholesky of regularized `P_updated`. If
          Cholesky fails, the filter returns the current state without updating the covariance.

        Adaptive noise estimation
        -------------------------
        - If `config.enable_adaptive` True, process and measurement noise are updated using
          the residual history and forgetting factor `b_k` per the referenced equations.
        - Process noise diagonal `S_q` and measurement sqrt `S_r` are updated and squared to
          produce `Q` and `R` respectively.

        Robustness & numerical caveats
        ------------------------------
        - Matrix inversions are guarded by `_regularize_covariance` where helpful.
        - When computing scalar denominators, `min_eig_floor` is used to avoid division by zero.
        - GMCC's iterative update can be sensitive to initial conditions; ensure `predicted_S`
          and `S_r` are reasonable before enabling it.
        - Constrained gain projection may raise; exceptions are caught and a warning logged.

        Returns
        -------
        np.ndarray
            The updated state `self.x` (also stored in history).

        Raises
        ------
        None (errors are typically handled internally and warnings logged).
        """
        self.step += 1

        # Store predicted state
        predicted_x = self.x.copy()
        predicted_S = self.S.copy()
        predicted_P = predicted_S @ predicted_S.T

        # Innovation
        v = measurement - z_pred
        self.innovation_norm = abs(v)

        # Store residual
        self.residual_history.append(v)
        if len(self.residual_history) > self.config.window_size:
            self.residual_history.pop(0)

        # Update residual covariance V_k (Eq. 48 from Paper 1)
        self.V_k = v ** 2 if self.step == 1 else (self.config.epsilon * self.V_k + v ** 2) / (1 + self.config.epsilon)

        # ============ MEASUREMENT JACOBIAN SELECTION ============
        if self.config.use_true_jacobian and H_k_true is not None:
            H_k = H_k_true.flatten()
        else:
            # Use the approximation
            inv_P = inv(_regularize_covariance(predicted_P))
            H_k = (inv_P @ P_xy).flatten()

        # ============ STRONG TRACKING FADING (STF) ============
        if self.config.enable_fading:
            # L_k = H_k (state,)
            Q_L = H_k @ self.Q @ H_k  # scalar

            N_k = self.V_k - self.config.psi * self.R - Q_L
            M_k = S_zz_tilde ** 2 - Q_L

            # Compute traces (scalars)
            trace_N = N_k
            trace_M = M_k

            # Fading factor
            phi_k = max(1.0, float(trace_N / trace_M)) if trace_M > 0 else 1.0

            # Apply fading to state covariance
            predicted_S *= np.sqrt(phi_k)
            predicted_P = predicted_S @ predicted_S.T

        # ============ GMCC UPDATE ============
        if self.config.enable_mcc:
            # Linear regression model
            I_n = np.eye(self.state_dim)

            # Build B_k matrix
            zeros_state1 = np.zeros((self.state_dim, 1))
            zeros_1state = np.zeros((1, self.state_dim))
            B_k = np.block([
                [predicted_S, zeros_state1],
                [zeros_1state, np.array([[self.S_r]])]
            ])

            try:
                B_k_inv = inv(B_k)
            except LinAlgError:
                # Regularize B_k
                B_k_inv = inv(_regularize_covariance(B_k))

            # Build J_k and W_k
            term = v + np.dot(H_k, predicted_x)  # scalar
            J_k = B_k_inv @ np.concatenate([predicted_x, np.array([term])])

            H_k_row = H_k[None, :]  # (1, state)
            W_k_stack = np.vstack([I_n, H_k_row])
            W_k = B_k_inv @ W_k_stack

            # Initialize estimate
            x_hat = predicted_x.copy()
            K = np.zeros_like(x_hat)
            # Iterative GMCC update
            for iter_count in range(self.config.mcc_max_iter):
                # Compute error
                e_k = J_k - W_k @ x_hat

                # Compute GGD weights
                C_x_inv = np.ones(self.state_dim)
                for i in range(self.state_dim):
                    C_x_inv[i] /= np.maximum(_compute_ggd_weight(float(e_k[i])), min_eig_floor)

                C_z_inv = 1. / np.maximum(_compute_ggd_weight(float(e_k[self.state_dim])), min_eig_floor)

                # Update covariances with GMCC weights
                P_k_k1 = predicted_S @ np.diag(C_x_inv) @ predicted_S.T
                R_k = (self.S_r ** 2) * C_z_inv

                # Compute Kalman gain with updated covariances
                S = np.maximum(np.dot(H_k, P_k_k1 @ H_k) + R_k, min_eig_floor)  # scalar
                K = (P_k_k1 @ H_k) / S  # (state,)

                # Update state estimate
                x_new = predicted_x + K * v

                # Check convergence
                rel_change = np.linalg.norm(x_new - x_hat) / (np.linalg.norm(x_hat) if np.linalg.norm(x_hat) > 0 else 1)

                x_hat = x_new.copy()

                if rel_change <= self.config.mcc_eps:
                    break

            # Final update
            ex = x_hat  # (state,)
            eL = K  # (state,)

        else:
            # Standard Kalman update
            K = P_xy / P_yy
            ex = predicted_x + K.flatten() * v
            eL = K.flatten()

        # ============ CONSTRAINED GAIN ============
        if self.config.enable_constraints and self.constraint_A is not None and self.constraint_b is not None:
            try:
                constrained_gain, constrained_estimate = compute_constrained_gain(
                    eL[:, None], ex, v, P_yy, self.constraint_A, self.constraint_b
                )
                ex = constrained_estimate
                eL = constrained_gain.flatten()

            except Exception as e:
                logger.warning(f"Constrained gain failed: {e}")
                # Continue with unconstrained estimate

        # Update covariance using Joseph form
        I = np.eye(self.state_dim)
        IKH = I - np.outer(eL, H_k)
        P_updated = IKH @ predicted_P @ IKH.T + (eL[:, None] * self.R) @ eL[:, None].T

        # Update square root
        try:
            self.S = cholesky(_regularize_covariance(P_updated))
        except LinAlgError:
            self.S = cholesky(_regularize_covariance(cholesky(P_updated @ P_updated.T)))

        # ============ ADAPTIVE NOISE ESTIMATION ============
        if self.config.enable_adaptive:
            # Compute d_k
            b_k = self.config.adaptive_forgetting_b
            d_k = (1 - b_k) / np.maximum((1 - b_k ** (self.step + 1)), min_eig_floor)

            # Compute F_k (innovation covariance)

            F_k = sum(res ** 2 for res in self.residual_history) / len(self.residual_history) if len(self.residual_history) > 0 else v * v

            # Update process noise
            eL_flat = eL
            temp = np.outer(eL_flat, eL_flat) * F_k  # (state, state)
            sqrt_temp = np.sqrt(np.maximum(np.diag(temp), 0))
            old_s_q_diag = np.diag(self.S_q)
            new_s_q_diag = (1 - d_k) * old_s_q_diag + d_k * sqrt_temp + np.sqrt(np.diag(self.config.Q_min))
            self.S_q = np.diag(new_s_q_diag)
            self.Q = self.S_q @ self.S_q.T

            # Update measurement noise
            sqrt_f = np.sqrt(max(F_k, 0))
            old_s_r = self.S_r
            new_s_r = (1 - d_k) * old_s_r + d_k * sqrt_f + np.sqrt(self.config.R_min)
            self.S_r = new_s_r
            self.R = self.S_r ** 2

        # Update state
        self.x = ex

        # Store history
        self.estimates_history.append(self.x.copy())
        self.covariance_history.append(self.S @ self.S.T)

        return self.x

    def get_covariance(self) -> np.ndarray:
        """
        Return the full covariance matrix reconstructed from the square-root factor.

        Returns
        -------
        np.ndarray, shape (state_dim, state_dim)
            Current state covariance `P = S @ S.T`.
        """
        return self.S @ self.S.T

    def get_noise_matrices(self):
        """
        Return a snapshot of current noise matrices and covariances.

        Returns
        -------
        dict
            Dictionary with keys:
              - 'process_noise': Q (np.ndarray)
              - 'measurement_noise': R (float)
              - 'initial_covariance': initial covariance stored at initialization (or None)
              - 'final_covariance': most recent covariance from history (or None)
        """
        return {
            'process_noise': self.Q.copy(),
            'measurement_noise': self.R,
            'initial_covariance': self.covariance_history[0] if self.covariance_history else None,
            'final_covariance': self.covariance_history[-1] if self.covariance_history else None
        }

    def set_process_noise(self, Q: np.ndarray) -> None:
        """
        Set process noise covariance (and update square-root).

        Parameters
        ----------
        Q : np.ndarray or 1D array-like
            If a 1D array is provided, it is treated as the diagonal of Q.
            If a 2D array is provided it must be a square process covariance matrix.

        Effects
        -------
        Updates `self.Q` and `self.S_q` accordingly.
        """
        self.Q = np.diag(Q) if Q.ndim == 1 else Q
        self.S_q = cholesky(self.Q)

    def set_measurement_noise(self, R: float) -> None:
        """
        Set measurement noise variance.

        Parameters
        ----------
        R : float
            Measurement noise variance (scalar). Must be non-negative.

        Effects
        -------
        Updates `self.R` and `self.S_r` in-place.
        """
        self.R = float(R)
        self.S_r = np.sqrt(self.R)