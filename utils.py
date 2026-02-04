"""
Utilities module providing Numba-accelerated functions for signal processing, statistical computations,
and support for SRUKF and ODE-based modeling.
This module contains a collection of @njit-decorated functions optimized for performance in a Numba environment.
These functions support tasks such as weighted deviation calculations, sigma point generation for unscented transforms,
covariance regularization, peak/valley detection in signals, safety checks for chirp responses,
bisection searches for minimal amplitudes, finite-difference Jacobians, percentile calculations,
descriptive statistics, and sensitivity propagation.
The functions are designed to integrate with ODE solvers (via `solve_ODE` from `model`) and
configuration parameters (from `config`).

Notes / high-level behaviour
----------------------------
- Many functions are compiled with Numba's @njit for speed and to enable parallel execution
  where applicable (e.g., using prange).
- Covariance handling emphasizes numerical stability: `_regularize_covariance` ensures SPD matrices via eigenvalue flooring.
- Signal processing functions like `find_peaks_1d` and `find_valleys_1d` use strict inequalities to detect local extrema,
  ignoring plateaus and edges.
- The chirp safety check (`compute_chirp_range_safe`) incorporates explosion detection and
  strict peak/valley count validation to handle solver instabilities.
- Statistical functions (`percentile`, `compute_stats`) handle NaNs and provide a fixed set of descriptors,
  including skew and kurtosis.
- Sensitivity-related functions (`sens_jacobian`, `compute_sens_and_std`) use finite differences and quadratic forms
  for uncertainty propagation.

Logging and output
------------------
- No built-in logging; functions are pure and return arrays or scalars directly.
- Exceptions in non-critical paths (e.g., inversion failures in `compute_constrained_gain`) fallback to unconstrained values silently.
"""
from typing import Tuple

from numba import njit, prange
import numpy as np
from numpy.linalg import cholesky, eigh, inv

from config import q, min_eig_floor, dur_chirp, total_duration, _step
from model import solve_ODE, compute_static_sensitivity

@njit
def _compute_weighted_deviations(points: np.ndarray, mean: np.ndarray,
                                 weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted deviations of columns in `points` from `mean`.

    For each column i of `points`, compute:
        W[:, i] = (points[:, i] - mean) * sqrt(abs(weights[i]))

    Parameters
    ----------
    points : np.ndarray, shape (n, m)
        Columns are sample points (state vectors).
    mean : np.ndarray, shape (n,)
        Mean vector to subtract from each column of `points`.
    weights : np.ndarray, shape (m,)
        Scalar weights for each column. Absolute value is used before square root.

    Returns
    -------
    W : np.ndarray, shape (n, m)
        Weighted deviations.

    Notes
    -----
    Negative weights are tolerated by using `abs(weights)`, but
    from our definition of covariance, abs() could be omitted.
    """

    deviations = points - mean[:, np.newaxis]
    sqrt_weights = np.sqrt(np.abs(weights))
    W = deviations * sqrt_weights[np.newaxis, :]

    return W

@njit
def _compute_ggd_weight(e: float) -> float:
    """
    Simple exponential weight used for a "generalized Gaussian density" style weight.

    Parameters
    ----------
    e : float
        Input residual or error.

    Returns
    -------
    float
        Weight computed as 0.25 * exp(-abs(0.5 * e)).

    Notes
    -----
    This is not the general expression, but the one simplifying Theta function.
    """
    return 0.25 * np.exp(-np.abs(0.5 * e))

@njit
def generate_sigma_points(x: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Generate sigma points via a square-root formulation.

    A coefficient matrix `sigma_c` of shape (n, n+2) is built using the global
    `q` vector; the sigma points are computed as:
        sigma_points = x[:, None] + S @ sigma_c

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        State mean vector.
    S : np.ndarray, shape (n, n)
        Square-root factor of state covariance (P = S @ S.T).

    Returns
    -------
    np.ndarray, shape (n, n+2)
        Sigma points as columns.

    Notes
    -----
    - The construction of sigma_c depends on `q` imported from the config module.
      q is computed from our suggested alpha.
    - This is a S3F sigma-point scheme.
    """
    n = x.shape[0]
    sigma_c = np.zeros((n, n + 2))
    for row in range(n):
        jt = row + 1
        qt = q[row]
        for col in range(1, jt + 1):
            sigma_c[row, col] = -qt / jt
        sigma_c[row, jt + 1] = qt
    sigma_points = x[:, np.newaxis] + S @ sigma_c
    return sigma_points

@njit
def compute_ut_weights(state_dim: int = 6, alpha: float = 0.518638, beta: float = 2.0) -> tuple:
    """
    Compute unscented-transform weights for a custom sigma-point set.

    Parameters
    ----------
    state_dim : int
        Dimension of the state (n).
    alpha : float
        UT scaling parameter.
    beta : float
        UT distribution-knowledge parameter (2 is optimal for Gaussian).

    Returns
    -------
    tuple
        (weights_m, weights_c) two 1D arrays of length (state_dim + 2) containing mean and covariance weights.
    """
    n = state_dim
    Wi = 1 / (alpha ** 2 * (n + 1))
    W0_m = 1 - 1 / alpha ** 2
    W0_c = W0_m + (1 - alpha ** 2 + beta)
    weights_m = np.zeros(n + 2)
    weights_m[0] = W0_m
    for i in range(1, n + 2):
        weights_m[i] = Wi
    weights_c = np.zeros(n + 2)
    weights_c[0] = W0_c
    for i in range(1, n + 2):
        weights_c[i] = Wi
    return weights_m, weights_c

@njit
def _regularize_covariance(P: np.ndarray) -> np.ndarray:
    """
    Ensure covariance matrix `P` is symmetric positive definite (SPD).

    The function first enforces symmetry (0.5*(P + P.T)), then attempts a
    Cholesky factorization. If Cholesky fails, it computes an eigen-decomposition,
    floors eigenvalues to `min_eig_floor`, and reconstructs a SPD matrix.

    Parameters
    ----------
    P : np.ndarray, shape (n, n)
        Symmetric (or near-symmetric) covariance matrix.

    Returns
    -------
    np.ndarray, shape (n, n)
        Regularized symmetric positive definite covariance matrix.

    Notes
    -----
    - This routine uses `cholesky` and `eigh` from numpy.linalg. Under Numba's
      nopython mode these may not be available; if `@njit` compilation fails,
      move this function out of JIT compilation or replace with a Numba-friendly approach.
    """
    P = 0.5 * (P + P.T)  # Ensure symmetry

    try:
        # Try Cholesky first
        _ = cholesky(P)
        return P
    except:
        # Regularize by adding to diagonal
        eigvals, eigvecs = eigh(P)

        # Ensure all eigenvalues are positive
        eigvals = np.maximum(eigvals, min_eig_floor)

        # Reconstruct matrix
        P_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        P_reg = 0.5 * (P_reg + P_reg.T)  # Ensure symmetry

        return P_reg

@njit
def compute_constrained_gain(unconstrained_gain: np.ndarray,
                             unconstrained_estimate: np.ndarray,
                             innovation: float,
                             innovation_cov: float,
                             A: np.ndarray, b: np.ndarray):
    """
    Project an unconstrained estimate onto active linear inequality constraints
    A x <= b and compute a constrained gain.

    The method:
    1. Detects active constraints where A @ estimate - b <= 1e-12.
    2. Builds a projection of the estimate onto the equality space of active constraints:
       x_c = x - A_a.T @ (A_a A_a.T)^{-1} (A_a x - b_a).
    3. Applies a gain correction proportional to the projection scaled by a scalar
       derived from innovation and innovation covariance.

    Parameters
    ----------
    unconstrained_gain : np.ndarray
        The original gain (vector or matrix) before constraint correction.
    unconstrained_estimate : np.ndarray, shape (n,)
        The state estimate to be projected.
    innovation : float
        Measurement innovation (scalar).
    innovation_cov : float
        Measurement innovation covariance (scalar).
    A : np.ndarray, shape (m, n)
        Constraint matrix for inequalities A x <= b.
    b : np.ndarray, shape (m,)
        Constraint RHS.

    Returns
    -------
    (constrained_gain, constrained_estimate)
        Tuple containing corrected gain and the projected estimate.

    Notes
    -----
    - If (A_a A_a.T) is singular or its inversion fails the function returns the unconstrained inputs.
    """
    # Check active constraints
    constr_vals = A @ unconstrained_estimate - b
    active_mask = constr_vals <= 1e-12

    if not np.any(active_mask):
        return unconstrained_gain, unconstrained_estimate

    # Form active constraint matrices
    A_a = A[active_mask, :]
    b_a = b[active_mask]
    n_active = A_a.shape[0]

    if n_active == 0:
        return unconstrained_gain, unconstrained_estimate

    try:
        # Compute (A_a A_a^T)^{-1}
        AA_t = A_a @ A_a.T
        AA_t_inv = inv(AA_t)

        # Compute innovation covariance inverse
        Sigma_y_inv = 1.0 / innovation_cov if innovation_cov > min_eig_floor else 1.0 / min_eig_floor

        # Compute scalar term
        r_Sigma_inv_r = innovation ** 2 * Sigma_y_inv
        scalar_term = 0.0 if abs(r_Sigma_inv_r) < min_eig_floor else 1.0 / r_Sigma_inv_r

        # Compute correction term
        correction_term = A_a @ unconstrained_estimate - b_a

        # Compute projection
        projection = A_a.T @ AA_t_inv @ correction_term

        # Compute constrained estimate (Eq. 37 from Paper 2)
        constrained_estimate = unconstrained_estimate - projection

        # Compute constrained gain (Eq. 36 from Paper 2)
        gain_scalar = scalar_term * (innovation * Sigma_y_inv)
        gain_correction = (projection * gain_scalar)[:, None]
        constrained_gain = unconstrained_gain - gain_correction

        return constrained_gain, constrained_estimate

    except:
        # If inversion fails, return unconstrained
        return unconstrained_gain, unconstrained_estimate

@njit
def find_peaks_1d(signal: np.ndarray) -> np.ndarray:
    """
    Find indices of local maxima in a 1D signal using strict inequalities.

    A point i (1 <= i <= n-2) is considered a peak if signal[i] > signal[i-1] and signal[i] > signal[i+1].

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1D signal array.

    Returns
    -------
    np.ndarray, shape (k,)
        Indices of detected peaks. Edges are never peaks; plateaus are not peaks.
    """
    n = signal.shape[0]
    # boolean mask for peaks
    mask = np.zeros(n, dtype=np.bool_)
    count = 0
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            mask[i] = True
            count += 1
    # allocate result
    res = np.empty(count, dtype=np.int64)
    j = 0
    for i in range(n):
        if mask[i]:
            res[j] = i
            j += 1
    return res

@njit
def find_valleys_1d(signal: np.ndarray) -> np.ndarray:
    """
    Find indices of local minima in a 1D signal using strict inequalities.

    A point i (1 <= i <= n-2) is considered a valley if signal[i] < signal[i-1] and signal[i] < signal[i+1].

    This is implemented by finding peaks in the negated signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1D signal array.

    Returns
    -------
    np.ndarray, shape (k,)
        Indices of detected valleys. Edges are never valleys; plateaus are not valleys.
    """
    return find_peaks_1d(-signal)

@njit
def compute_chirp_range_safe(uncertainty: np.ndarray, A: float,
                             safety_val: float = 10., max_voltage: float = 20.,
                             max_abs_threshold: float = 150.) -> bool:
    """
    Determine whether a chirp response is 'unsafe' for amplitude A by checking
    the difference between the 3rd and 4th peaks/valleys in the chirp interval.

    Modified to handle exploded solutions: if max(|dC_chirp_full|) exceeds a threshold,
    treat as unsafe (return True).

    Parameters
    ----------
    uncertainty : np.ndarray
        Parameter uncertainty vector passed to the ODE solver.
    A : float
        Trial amplitude for the chirp.
    safety_val : float, optional
        Threshold for absolute difference between successive peaks/valleys to be considered unsafe.
    max_voltage : float, optional
        Hard voltage cap; if A > max_voltage the function returns True immediately.
    max_abs_threshold : float, optional
        Threshold for max absolute value in dC_chirp_full to detect exploded/unphysical solutions.

    Returns
    -------
    bool
        True if chirp is considered unsafe (i.e., difference > safety_val for peaks or valleys,
        or exploded, or A > max_voltage); False otherwise.
    """
    if A > max_voltage:
        return True
    dC = solve_ODE(uncertainty, A)
    t_eval = np.arange(0.0, total_duration + _step, _step)
    idx_chirp_full = t_eval <= dur_chirp
    dC_chirp_full = dC[idx_chirp_full]

    # Add explosion check before peak/valley detection
    if np.max(np.abs(dC_chirp_full)) > max_abs_threshold:
        return True  # Exploded/unphysical: treat as unsafe

    peaks = find_peaks_1d(dC_chirp_full)
    valleys = find_valleys_1d(dC_chirp_full)

    # Add strict check for expected number (exactly 4 peaks/valleys)
    if peaks.shape[0] != 4 or valleys.shape[0] != 4:
        return False  # Invalid if not exactly 4: treat as safe (not yet unstable)

    # Since exactly 4, indices 2 and 3 are safe
    third_max_val = dC_chirp_full[peaks[2]]
    fourth_max_val = dC_chirp_full[peaks[3]]
    third_min_val = dC_chirp_full[valleys[2]]
    fourth_min_val = dC_chirp_full[valleys[3]]

    return (np.abs(fourth_max_val - third_max_val) > safety_val and fourth_max_val > third_max_val and fourth_min_val > third_min_val) or \
        (np.abs(fourth_min_val - third_min_val) > safety_val and fourth_min_val < third_min_val and fourth_max_val < third_max_val)

@njit
def compute_min_A(uncertainty: np.ndarray, A_start: float = 0.8,
                         tol: float = 1e-3, max_A: float = 20.0) -> float:
    """
    Find the minimal amplitude A such that compute_chirp_range_safe(uncertainty, A) is True.

    Uses exponential bracketing followed by bisection on the monotonic function
    f(A) = compute_chirp_range_safe(uncertainty, A).

    Parameters
    ----------
    uncertainty : np.ndarray
        Parameter uncertainty vector passed to the ODE solver.
    A_start : float, optional
        Starting amplitude to test.
    tol : float, optional
        Tolerance for bisection convergence.
    max_A : float, optional
        Maximum amplitude to consider; fallback if no True found.

    Returns
    -------
    float
        Minimal amplitude A (up to tol) within [A_start, max_A] for which the safety function returns True.

    Notes
    -----
    - Each evaluation calls the ODE solver; we should consider caching evaluations to reduce cost.
    """
    A_low = A_start
    if compute_chirp_range_safe(uncertainty, A_low):
        return A_low

    A_high = A_low * 2.0
    while (not compute_chirp_range_safe(uncertainty, A_high)) and (A_high < max_A):
        A_low = A_high
        A_high *= 2.0

    if A_high >= max_A:
        return max_A

    while (A_high - A_low) > tol:
        mid = 0.5 * (A_low + A_high)
        if compute_chirp_range_safe(uncertainty, mid):
            A_high = mid
        else:
            A_low = mid
    return A_high

@njit
def sens_jacobian(p: np.ndarray) -> np.ndarray:
    """
    Compute finite-difference Jacobian (size 4) of a sensitivity mapping for parameter vector p
    using a central-difference scheme.

    The function builds a 5-element parameter vector m from p and calls compute_static_sensitivity(m).
    The Jacobian is computed using central differences:
        J[j] = (f(p + dp e_j) - f(p - dp e_j)) / (2 * dp)

    Parameters
    ----------
    p : np.ndarray, shape (4,)
        Parameter vector.

    Returns
    -------
    np.ndarray, shape (4,)
        Approximate Jacobian of sensitivity with respect to p.

    Notes
    -----
    - Uses a relative step size by default; step = eps * abs(p[j]) (or eps if near zero).
    - Central differences are used for higher accuracy than forward differences.
    """
    eps = 1e-9
    J = np.zeros(4)
    for j in prange(4):
        # relative step (avoid zero step)
        dp = eps if abs(p[j]) < 1e-8 else eps * abs(p[j])

        # positive perturbation
        p_plus = p.copy()
        p_plus[j] += dp
        m_plus = np.zeros(5)
        m_plus[0] = p_plus[0]
        m_plus[1] = p_plus[1]
        m_plus[2] = 0.0
        m_plus[3] = p_plus[2]
        m_plus[4] = p_plus[3]
        f_plus = compute_static_sensitivity(m_plus)

        # negative perturbation
        p_minus = p.copy()
        p_minus[j] -= dp
        m_minus = np.zeros(5)
        m_minus[0] = p_minus[0]
        m_minus[1] = p_minus[1]
        m_minus[2] = 0.0
        m_minus[3] = p_minus[2]
        m_minus[4] = p_minus[3]
        f_minus = compute_static_sensitivity(m_minus)

        # central difference
        J[j] = (f_plus - f_minus) / (2.0 * dp)

    return J

@njit
def percentile(arr: np.ndarray, p: float) -> float:
    """
    Compute the p-th percentile of `arr` using linear interpolation.

    Parameters
    ----------
    arr : np.ndarray
        1D data array.
    p : float
        Percentile to compute (0-100).

    Returns
    -------
    float
        Percentile value or np.nan if input array is empty.
    """
    if len(arr) == 0:
        return np.nan
    sorted_arr = np.sort(arr)
    k = (len(arr) - 1) * (p / 100.0)
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return sorted_arr[int(k)]
    d0 = sorted_arr[int(f)] * (c - k)
    d1 = sorted_arr[int(c)] * (k - f)
    return d0 + d1

@njit
def compute_stats(arr: np.ndarray) -> np.ndarray:
    """
    Compute a set of descriptive statistics from `arr` (NaNs removed).

    Returned array layout:
      [count, mean, std, min, q25, q50, q75, max, skew, kurtosis_excess, q50, mad]

    Parameters
    ----------
    arr : np.ndarray
        1D data array possibly containing NaNs.

    Returns
    -------
    np.ndarray, shape (12,)
        Summary statistics. Note q50 (median) appears twice in the returned array.

    Notes
    -----
    - `std` uses numpy's default (population std). Use ddof=1 to compute sample std if required.
    - `mad` is mean absolute deviation from the mean.
    - Consider returning a structured dtype or dict for clarity.
    """
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.full(12, np.nan)
    mean = np.mean(arr)
    var = np.var(arr)
    std = np.sqrt(var)
    minv = np.min(arr)
    maxv = np.max(arr)
    q25 = percentile(arr, 25)
    q50 = percentile(arr, 50)
    q75 = percentile(arr, 75)
    dev = arr - mean
    m3 = np.mean(dev ** 3)
    m4 = np.mean(dev ** 4)
    skew_val = m3 / (std ** 3 + 1e-16)
    kurt_val = m4 / (std ** 4 + 1e-16) - 3
    mad = np.mean(np.abs(dev))
    count = float(n)
    return np.array([count, mean, std, minv, q25, q50, q75, maxv, skew_val, kurt_val, q50, mad])

@njit
def compute_sens_and_std(params_hist: np.ndarray, cov_hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each parameter vector in `params_hist`, compute static sensitivity and the
    propagated standard deviation using the finite-difference Jacobian.

    Parameters
    ----------
    params_hist : np.ndarray, shape (N, 4)
        History of parameter vectors.
    cov_hist : np.ndarray, shape (N, 4, 4)
        History of parameter covariance matrices.

    Returns
    -------
    sens : np.ndarray, shape (N,)
        Sensitivity values for each parameter vector.
    stds : np.ndarray, shape (N,)
        Standard deviations of the sensitivities computed as sqrt(J @ cov @ J.T).

    Notes
    -----
    - Each step calls compute_static_sensitivity and sens_jacobian; expensive for large N.
    """
    n = params_hist.shape[0]
    sens = np.zeros(n)
    stds = np.zeros(n)
    for k in prange(n):
        p = params_hist[k]
        m = np.zeros(5)
        m[0:2] = p[0:2]
        m[3:5] = p[2:4]
        sens[k] = compute_static_sensitivity(m)
        J = sens_jacobian(p)
        cov_p = cov_hist[k]
        var_s = np.dot(J, np.dot(cov_p, J))
        stds[k] = np.sqrt(var_s)
    return sens, stds