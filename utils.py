"""
srukf_utils — Numba-accelerated utilities for SRUKF, ODE-model probing, and robust signal processing
===============================================================================================

Summary
-------
This module provides a compact, high-performance collection of numerical utilities intended for use
inside a square-root unscented Kalman filter (SRUKF) workflow and for running/analysing ODE-based
chirp experiments. All core routines that benefit from speed are decorated with ``@njit`` so they
can be compiled by Numba and run in nopython mode.

Key capabilities
----------------
- Linear algebra helpers for sigma-point generation and weighted deviations.
- Covariance regularization to enforce symmetric positive definiteness (SPD).
- Constrained-gain projection to handle linear inequality constraints A x <= b.
- Robust, Numba-friendly peak/valley detection (smoothing + prominence + distance suppression).
- Cycle-aware pairing of extrema for reliable 3rd→4th peak/valley comparisons in noisy traces.
- Chirp safety test that preserves strict directional logic while handling exploded ODE solutions.
- Utility statistics (percentiles, compute_stats) and sensitivity propagation via finite differences.
- Minimal bisection/bracketing routine (compute_min_A) to find the smallest amplitude that triggers
  the safety predicate.

Intended usage
--------------
This module is designed to be imported into a larger model/estimation package. Typical usage
patterns include:
- Calling ``generate_sigma_points`` and ``compute_ut_weights`` from SRUKF update/prediction steps.
- Computing robust peaks/valleys from ODE-derived traces (``dC``) to decide whether a chirp is
  "unsafe" using ``compute_chirp_range_safe``.
- Running ``compute_min_A`` as a wrapper that finds the minimal amplitude that violates safety.
- Estimating static sensitivities and propagated uncertainties via ``compute_sens_and_std``.

Module-level requirements and expected globals
----------------------------------------------
The module expects the following objects to be available in the importing module's namespace (they
are imported here from ``config`` and ``model`` in the current codebase):

- ``_step`` : float
    Integration / sampling timestep (seconds). Used to convert time-based distances to samples.
- ``total_duration`` : float
    Duration used to build the solver's evaluation times.
- ``dur_chirp`` : float
    Chirp interval duration (seconds); used to slice the ODE response.
- ``q`` : np.ndarray
    Coefficients used to construct the sigma-point coefficient matrix.
- ``min_eig_floor`` : float
    Minimal eigenvalue floor used for covariance regularization and numerical safe-guards.
- ``solve_ODE(uncertainty, A)`` : callable
    Function that runs the ODE model and returns a 1D array ``dC`` sampled at ``_step`` intervals.
- ``compute_static_sensitivity(m)`` : callable
    Function returning the static sensitivity (scalar) for a 5-element parameter vector ``m``.

Important behavior and design choices
------------------------------------
- Numba-first: Many helpers are ``@njit``-decorated and written in a style that avoids Python objects
  (lists, comprehensions, dynamic resizing) to ensure compilation. If Numba compilation fails for a
  particular host environment, move a given function out of Numba or provide fallbacks.
- Robust extrema detection:
  - The naive 3-sample test ``x[i] > x[i-1] and x[i] > x[i+1]`` is replaced by a pipeline:
    smoothing (moving average), candidate detection, local-prominence calculation, minimum-distance
    suppression (greedy preference for larger peaks). This dramatically reduces spurious extrema
    from Gaussian noise while retaining true cycle peaks.
  - Valleys are found by applying the same pipeline to ``-signal`` ensuring symmetry of behavior.
  - Cycle pairing: to compare the "3rd" and "4th" extrema reliably, peaks are paired using valleys as
    cycle anchors (dominant peak between valley intervals). This prevents misalignment when noise
    creates extra small extrema near cycle boundaries.
- Safety boolean logic preserved: the chirp safety predicate uses your original directional rule:
  mark unsafe only when both peaks and valleys move in the same direction AND one of the changes
  (peak or valley difference) exceeds ``safety_val``. The module does **not** change that logic —
  it only makes detection of the extrema themselves more robust.
- Explosion detection: if the ODE response's maximum absolute value exceeds ``max_abs_threshold``,
  the solution is considered exploded/unphysical and the test returns True (unsafe).
- Conservative failure modes: when there are insufficient robust extrema to make a decision, the
  module returns ``False`` (safe). This choice favors avoiding false positives when the signal is
  ambiguous; tune this behavior if your application requires otherwise.

Tunable parameters and where to change them
------------------------------------------
Internal parameters (defaults are conservative and can be adjusted to match sampling rate and SNR):
- smoothing window (samples) inside ``find_peaks_1d`` — default: 7
- minimum peak separation (seconds converted to samples using ``_step``) — default: 0.2 s
- prominence threshold multiplier — default: 2.5 * estimated noise sigma (estimated from the first
  5% of the signal)
- explosion threshold ``max_abs_threshold`` in ``compute_chirp_range_safe`` — default: 150.

Examples
--------
Minimal conceptual examples (pseudocode):

    # 1) find peaks on a noisy ODE response
    dC = solve_ODE(uncertainty, A)
    t_eval = np.arange(0.0, total_duration + _step, _step)
    dC_chirp = dC[t_eval <= dur_chirp]
    peaks = find_peaks_1d(dC_chirp)
    valleys = find_valleys_1d(dC_chirp)

    # 2) compute whether the chirp is unsafe
    unsafe = compute_chirp_range_safe(uncertainty, A)

    # 3) find minimal unsafe amplitude
    A_crit = compute_min_A(uncertainty, A_start=0.8)
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
def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a simple moving-average of a 1D signal with naive edge handling.

    The average is computed using a sliding window of length ``window`` centered
    at every sample. Near the edges the window is truncated so that the average
    uses only available samples (no padding).

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal samples.
    window : int
        Window length to use for the moving average. If ``window`` is even, the
        function treats half = window // 2 and uses an asymmetric window
        (consistent with integer division). ``window`` should be >= 1.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``signal`` containing the smoothed samples.

    Notes
    -----
    - This implementation is intentionally simple and NumPy/Numba friendly.
    - It is **not** optimized for very large windows or high-performance
      convolution; its purpose is to remove high-frequency noise before peak
      detection in a Numba-compatible way.
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=signal.dtype)
    half = window // 2
    for i in range(n):
        left = i - half
        right = i + half
        if left < 0:
            left = 0
        if right >= n:
            right = n - 1
        s = 0.0
        cnt = 0
        for j in range(left, right + 1):
            s += signal[j]
            cnt += 1
        out[i] = s / cnt
    return out


@njit
def _std_from_slice(arr: np.ndarray) -> float:
    """
    Compute the population standard deviation of a 1D slice.

    This uses the population formula (division by N), which is consistent with
    a noise sigma estimate for prominence thresholds.

    Parameters
    ----------
    arr : np.ndarray
        1D numeric array. If ``arr`` is empty, returns 0.0.

    Returns
    -------
    float
        Population standard deviation of ``arr``.
    """
    n = arr.shape[0]
    if n == 0:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += arr[i]
    mean /= n
    var = 0.0
    for i in range(n):
        diff = arr[i] - mean
        var += diff * diff
    var /= n
    return np.sqrt(var)


@njit
def _suppress_peaks_greedy(peaks: np.ndarray, values: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Greedy suppression of peaks that are closer than ``min_dist`` samples.

    The algorithm:
      * Sort candidate peaks by descending value (largest peaks first).
      * Accept a candidate if it is farther than ``min_dist`` from any already
        accepted peak.
      * Return the accepted peaks sorted in ascending (time) order.

    Parameters
    ----------
    peaks : np.ndarray
        1D integer array of candidate peak indices (sample indices).
    values : np.ndarray
        1D array of same length as ``peaks`` with corresponding peak values.
    min_dist : int
        Minimum allowed distance (in samples) between kept peaks.

    Returns
    -------
    np.ndarray
        Array of accepted peak indices, sorted ascending. May be empty.

    Notes
    -----
    - This is a simple greedy strategy that prefers larger peaks. It is easy
      to reason about and is robust for noisy signals where small local maxima
      appear near large true peaks.
    """
    k = peaks.shape[0]
    if k == 0:
        return np.empty(0, dtype=np.int64)

    # Build index order by values descending (selection sort to stay Numba-friendly)
    order = np.empty(k, dtype=np.int64)
    used = np.zeros(k, dtype=np.bool_)
    for i in range(k):
        max_j = -1
        max_val = -1e300
        for j in range(k):
            if not used[j] and values[j] > max_val:
                max_val = values[j]
                max_j = j
        used[max_j] = True
        order[i] = max_j

    # Greedily accept peaks if not within min_dist of any accepted
    accepted = np.empty(k, dtype=np.int64)
    accepted_count = 0
    for idx_i in range(k):
        j = order[idx_i]
        pj = peaks[j]
        too_close = False
        for a in range(accepted_count):
            if abs(int(accepted[a]) - int(pj)) <= min_dist:
                too_close = True
                break
        if not too_close:
            accepted[accepted_count] = pj
            accepted_count += 1

    # Shrink and return in ascending order
    res = np.empty(accepted_count, dtype=np.int64)
    for i in range(accepted_count):
        res[i] = accepted[i]

    # Simple in-place sort (ascending)
    for a in range(res.shape[0] - 1):
        for b in range(a + 1, res.shape[0]):
            if res[b] < res[a]:
                tmp = res[a]; res[a] = res[b]; res[b] = tmp
    return res


@njit
def _find_candidate_peaks(sm: np.ndarray) -> np.ndarray:
    """
    Find candidate peaks on a smoothed signal using a 3-point strict test.

    A candidate at index i satisfies sm[i] > sm[i-1] and sm[i] > sm[i+1].

    Parameters
    ----------
    sm : np.ndarray
        Smoothed 1D signal.

    Returns
    -------
    np.ndarray
        Indices of candidate peaks (unsuppressed / unscreened). May be empty.
    """
    n = sm.shape[0]
    cand = np.empty(n, dtype=np.int64)
    cnt = 0
    for i in range(1, n - 1):
        if sm[i] > sm[i - 1] and sm[i] > sm[i + 1]:
            cand[cnt] = i
            cnt += 1
    if cnt == 0:
        return np.empty(0, dtype=np.int64)
    res = np.empty(cnt, dtype=np.int64)
    for i in range(cnt):
        res[i] = cand[i]
    return res


@njit
def _compute_prominences(sm: np.ndarray, cand: np.ndarray, search_range: int):
    """
    Compute a simple local prominence for each candidate peak.

    Prominence is computed as: peak_value - max(local_left_min, local_right_min),
    where local minima are searched within ``search_range`` samples from the peak.

    Parameters
    ----------
    sm : np.ndarray
        Smoothed 1D signal.
    cand : np.ndarray
        Indices of candidate peaks.
    search_range : int
        Half-window (in samples) to search for left/right minima.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple of (prominences, peak_values) as arrays aligned to ``cand``.
    """
    k = cand.shape[0]
    prominences = np.empty(k, dtype=sm.dtype)
    vals = np.empty(k, dtype=sm.dtype)
    n = sm.shape[0]
    for idx in range(k):
        i = cand[idx]
        # left min
        left_min = sm[i]
        left_bound = i - search_range
        if left_bound < 0:
            left_bound = 0
        j = i - 1
        while j >= left_bound:
            if sm[j] < left_min:
                left_min = sm[j]
            j -= 1
        # right min
        right_min = sm[i]
        right_bound = i + search_range
        if right_bound > n - 1:
            right_bound = n - 1
        j = i + 1
        while j <= right_bound:
            if sm[j] < right_min:
                right_min = sm[j]
            j += 1
        # prominence
        larger_min = left_min if left_min > right_min else right_min
        prominences[idx] = sm[i] - larger_min
        vals[idx] = sm[i]
    return prominences, vals


@njit
def _select_peaks(sm: np.ndarray, cand: np.ndarray,
                  prominences: np.ndarray, vals: np.ndarray,
                  min_prominence: float, min_distance: int) -> np.ndarray:
    """
    Filter candidate peaks by minimum prominence and apply distance suppression.

    Parameters
    ----------
    sm : np.ndarray
        Smoothed signal (only used for type/consistency; values also passed).
    cand : np.ndarray
        Candidate peak indices.
    prominences : np.ndarray
        Prominences aligned with ``cand``.
    vals : np.ndarray
        Peak values aligned with ``cand``.
    min_prominence : float
        Minimum required prominence (absolute units) for a peak to be kept.
    min_distance : int
        Minimum number of samples between kept peaks.

    Returns
    -------
    np.ndarray
        Final kept peak indices (ascending). May be empty.
    """
    sel_cnt = 0
    tmp_peaks = np.empty(cand.shape[0], dtype=np.int64)
    tmp_vals = np.empty(cand.shape[0], dtype=sm.dtype)
    for i in range(cand.shape[0]):
        if prominences[i] >= min_prominence:
            tmp_peaks[sel_cnt] = cand[i]
            tmp_vals[sel_cnt] = vals[i]
            sel_cnt += 1
    if sel_cnt == 0:
        return np.empty(0, dtype=np.int64)
    peaks_arr = np.empty(sel_cnt, dtype=np.int64)
    vals_arr = np.empty(sel_cnt, dtype=sm.dtype)
    for i in range(sel_cnt):
        peaks_arr[i] = tmp_peaks[i]
        vals_arr[i] = tmp_vals[i]
    kept = _suppress_peaks_greedy(peaks_arr, vals_arr, min_distance)
    return kept


# -----------------------
# Public functions (names preserved)
# -----------------------

@njit
def find_peaks_1d(signal: np.ndarray) -> np.ndarray:
    """
    Robust peak finder for noisy 1D signals (Numba-compatible).

    This replaces the naive 3-sample strict test with a robust pipeline:
      1. Small moving-average smoothing to reduce high-frequency Gaussian noise.
      2. Candidate peak detection using a 3-point strict test on the smoothed trace.
      3. Local prominence computation for every candidate (searching a small
         neighborhood for left/right minima).
      4. Filter candidates by a minimum prominence threshold derived from an
         estimate of the local noise sigma.
      5. Greedy suppression of peaks that are closer than a configured minimum
         distance (prefer larger peaks).
      6. Return final peak indices sorted in ascending (time) order.

    The intent is to (a) reduce spurious tiny maxima created by noise, (b)
    preserve true cycle peaks, and (c) be fully Numba-friendly (no Python-only
    calls).

    Parameters
    ----------
    signal : np.ndarray
        1D signal array containing samples (float). The function expects
        reasonably sampled cycles and that a short initial segment can be used
        to estimate noise variance.

    Returns
    -------
    np.ndarray
        1D integer array of indices of detected peaks (ascending). May be empty.

    Notes
    -----
    - Tunable internal parameters (safe defaults chosen):
        * smoothing window: 7 samples
        * minimum distance: ~0.2 s converted to samples using global ``_step``
          (falls back to a small fraction of the signal length if ``_step``
          is unavailable)
        * prominence threshold: 2.5 * estimated noise sigma from the first
          5% of samples (population std)
    - If you find real peaks being removed increase the smoothing window or
      decrease ``prominence_sigma_mult`` inside the function (2.0-3.0 typical).
    - The function is deterministic and suitable for inclusion in an njit-ed
      pipeline.
    """
    n = signal.shape[0]
    if n < 3:
        return np.empty(0, dtype=np.int64)

    # Parameters (tweak if needed)
    smooth_window = 7  # odd, small smoothing to remove high-frequency noise
    # min distance between peaks (in samples); try to derive from global _step
    try:
        min_distance_samples = int(0.2 / _step)
    except Exception:
        min_distance_samples = max(1, n // 50)
    if min_distance_samples < 1:
        min_distance_samples = 1

    # estimate noise sigma from a short pre-chirp segment (first 5% of samples, at least 2)
    pre_n = n // 20
    if pre_n < 2:
        pre_n = 2
    noise_sigma = _std_from_slice(signal[:pre_n])
    prominence_sigma_mult = 2.5
    min_prominence = prominence_sigma_mult * noise_sigma

    # smooth -> candidates -> prominences -> selection
    sm = _moving_average(signal, smooth_window)
    cand = _find_candidate_peaks(sm)
    if cand.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    search_range = max(1, min_distance_samples // 2)
    prominences, vals = _compute_prominences(sm, cand, search_range)
    peaks = _select_peaks(sm, cand, prominences, vals, min_prominence, min_distance_samples)
    return peaks


@njit
def find_valleys_1d(signal: np.ndarray) -> np.ndarray:
    """
    Robust valley finder implemented as peaks on the negated signal.

    This preserves the original function name/signature but delegates to the
    robust peak-finding pipeline applied to ``-signal`` so that minima are
    detected with identical logic and robustness.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.

    Returns
    -------
    np.ndarray
        Indices of detected valleys (ascending). May be empty.
    """
    return find_peaks_1d(-signal)


@njit
def _pair_peaks_valleys_by_last_cycles(signal: np.ndarray, peaks: np.ndarray, valleys: np.ndarray):
    """
    Pair peaks and valleys robustly to extract the '3rd' and '4th' extrema.

    Strategy:
      - Use the last two valleys as cycle anchors: v3 = valleys[-2], v4 = valleys[-1].
      - Find the dominant (largest) peak between the previous valley and v3 -> p3.
      - Find the dominant peak between v3 and v4 -> p4.
      - Fallback: if either p3 or p4 cannot be found, use the last two peaks.

    Returns
    -------
    tuple (int, int, int, int)
        (p3_idx, p4_idx, v3_idx, v4_idx) or (-1, -1, -1, -1) on failure.

    Notes
    -----
    - This pairing approach reduces the chance of mis-aligned comparisons when
      small spurious extrema appear at cycle boundaries.
    """
    vn = valleys.shape[0]
    pn = peaks.shape[0]
    if vn < 2:
        return -1, -1, -1, -1
    v4_idx = valleys[vn - 1]
    v3_idx = valleys[vn - 2]

    if vn >= 3:
        prev_v = valleys[vn - 3]
    else:
        prev_v = 0

    # find dominant peak in (prev_v, v3)
    p3_idx = -1
    p3_val = -1e300
    for i in range(pn):
        p = peaks[i]
        if prev_v < p < v3_idx:
            if signal[p] > p3_val:
                p3_val = signal[p]
                p3_idx = p

    # find dominant peak in (v3, v4)
    p4_idx = -1
    p4_val = -1e300
    for i in range(pn):
        p = peaks[i]
        if v3_idx < p < v4_idx:
            if signal[p] > p4_val:
                p4_val = signal[p]
                p4_idx = p

    # fallback to last two peaks if missing
    if p3_idx == -1 or p4_idx == -1:
        if pn >= 2:
            p3_idx = peaks[pn - 2]
            p4_idx = peaks[pn - 1]
        elif pn == 1:
            p3_idx = peaks[0]
            p4_idx = peaks[0]
        else:
            return -1, -1, -1, -1

    return p3_idx, p4_idx, v3_idx, v4_idx


@njit
def compute_chirp_range_safe(uncertainty: np.ndarray, A: float,
                             safety_val: float = 10., max_voltage: float = 20.,
                             max_abs_threshold: float = 150.) -> bool:
    """
    Determine whether a chirp response is 'unsafe' for a given amplitude ``A``.

    The test inspects the behavior of the 3rd and 4th peaks and valleys within
    the chirp interval and applies the original directional safety logic:
      - Unsafe if the peak change from 3 -> 4 exceeds ``safety_val`` AND
        both peaks and valleys **increase** between the two cycles.
      - OR unsafe if the valley change from 3 -> 4 exceeds ``safety_val`` AND
        both valleys and peaks **decrease** between the two cycles.

    This implementation preserves the exact final boolean logic you requested
    but makes detection robust against Gaussian measurement noise by:
      * smoothing the trace,
      * using prominence-based peak/valley selection,
      * enforcing a minimum distance between peaks,
      * pairing peaks and valleys by cycle anchors (valleys) so that the 3rd/4th
        extrema are the intended cycle extrema.

    Parameters
    ----------
    uncertainty : np.ndarray
        Parameter uncertainty vector passed to the ODE solver. The routine
        forwards this into ``solve_ODE`` (must be available in the module).
    A : float
        Trial amplitude for the chirp.
    safety_val : float, optional
        Threshold (absolute units, same units as the signal) that a change
        between the 3rd and 4th extrema must exceed to be considered unsafe.
    max_voltage : float, optional
        Hard cap on amplitude; if ``A > max_voltage`` the function returns True.
    max_abs_threshold : float, optional
        If the maximum absolute value of the chirp response exceeds this value,
        treat the solution as exploded/unphysical and return True.

    Returns
    -------
    bool
        True if the chirp is considered unsafe (exploded, amplitude too high,
        or directional change in extrema meeting the safety criteria);
        False otherwise.

    Raises
    ------
    (No explicit raises — this function is njit-decorated and relies on
    surrounding module globals like ``_step``, ``total_duration``, ``dur_chirp``
    and a callable ``solve_ODE``. Ensure these exist and are numeric/callable.)

    Notes
    -----
    - The function returns ``False`` (safe) when there are insufficient robustly
      detected extrema to make a decision (rather than risking false positives).
    - If you prefer a stricter/looser detection, tune the internal parameters of
      ``find_peaks_1d`` (smoothing window, prominence multiplier, minimum
      distance).
    """
    # immediate hard-limit checks
    if A > max_voltage:
        return True

    # Obtain the full dC trace from the ODE solver (module-level solve_ODE is required)
    dC = solve_ODE(uncertainty, A)

    # t_eval uses module-level total_duration and _step; these must be defined
    t_eval = np.arange(0.0, total_duration + _step, _step)
    idx_chirp_full = t_eval <= dur_chirp
    dC_chirp_full = dC[idx_chirp_full]

    # explosion/unphysical check (early exit)
    if np.max(np.abs(dC_chirp_full)) > max_abs_threshold:
        return True

    # robust peak / valley detection
    peaks = find_peaks_1d(dC_chirp_full)
    valleys = find_valleys_1d(dC_chirp_full)

    # require at least two valleys and two peaks to form meaningful 3rd/4th comparison
    if peaks.shape[0] < 2 or valleys.shape[0] < 2:
        return False

    # Pair peaks and valleys to extract the intended p3/p4 and v3/v4 indices
    p3_idx, p4_idx, v3_idx, v4_idx = _pair_peaks_valleys_by_last_cycles(dC_chirp_full, peaks, valleys)
    if p3_idx == -1 or p4_idx == -1 or v3_idx == -1 or v4_idx == -1:
        return False

    third_max_val = dC_chirp_full[p3_idx]
    fourth_max_val = dC_chirp_full[p4_idx]
    third_min_val = dC_chirp_full[v3_idx]
    fourth_min_val = dC_chirp_full[v4_idx]

    # Original directional safety logic (kept verbatim)
    cond1 = (np.abs(fourth_max_val - third_max_val) > safety_val and
             fourth_max_val > third_max_val and
             fourth_min_val > third_min_val)
    cond2 = (np.abs(fourth_min_val - third_min_val) > safety_val and
             fourth_min_val < third_min_val and
             fourth_max_val < third_max_val)

    return cond1 or cond2


@njit
def compute_min_A(uncertainty: np.ndarray, A_start: float = 0.8,
                  tol: float = 1e-3, max_A: float = 20.0) -> float:
    """
    Find the minimal amplitude A for which the chirp is considered unsafe.

    The routine performs:
      1. An initial check at ``A_start``. If already unsafe, return ``A_start``.
      2. Exponential bracketing: double the amplitude until the safety test
         ``compute_chirp_range_safe`` returns True or ``max_A`` is reached.
      3. Bisection on the interval [A_low, A_high] until the width is <= ``tol``.

    Parameters
    ----------
    uncertainty : np.ndarray
        Parameter uncertainty vector passed through to the ODE solver used by
        ``compute_chirp_range_safe``.
    A_start : float, optional
        Initial amplitude to test. Must be > 0 in practice.
    tol : float, optional
        Absolute tolerance for the final amplitude. The returned amplitude is
        accurate within ``tol`` (the algorithm returns the upper bracket).
    max_A : float, optional
        Maximum amplitude to consider. If the bracket expansion reaches or
        exceeds ``max_A`` without finding an unsafe amplitude, ``max_A`` is
        returned as a conservative fallback.

    Returns
    -------
    float
        Minimal amplitude A (to within ``tol``) in the interval
        [``A_start``, ``max_A``] for which ``compute_chirp_range_safe`` returns True.
        If ``compute_chirp_range_safe`` is already True at ``A_start``, returns
        ``A_start``. If no unsafe amplitude is found up to ``max_A``, returns
        ``max_A``.

    Notes
    -----
    - Each evaluation of ``compute_chirp_range_safe`` typically calls the ODE
      solver; this can be expensive. Consider caching results of
      ``solve_ODE(uncertainty, A)`` externally if many repeated calls are
      expected with the same arguments.
    - The function assumes monotonicity of the boolean safety predicate with
      respect to amplitude (i.e., once unsafe at some A, larger A remain
      unsafe). This is the basis for safe bracketing + bisection.
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