"""
Module-level conventions (used throughout the functions below)

Units and dimensions used by the functions in this module (consistent with the provided table):
- length: micrometers (µm)
- mass: micrograms (µg)
- time: milliseconds (ms)
- frequency: kilohertz (kHz)
- stiffness K: nanonewtons per micrometer (nN/µm)
- damping C_damping: units consistent with (nN·ms/µm) or as used by the code (see notes)
- capacitance: femtofarads (fF)
- speed: µm/ms
- acceleration: µm/ms^2
- vacuum permittivity `eps` in code: fF/µm (as implemented)

Notes
-----
- All functions are decorated with `@njit` (Numba nopython) in your code; ensure any
  objects they reference are Numba-compatible (pure numeric arrays and scalars).
- The code uses a consistent time-step `_step` (ms) and global durations (`dur_chirp`, `total_duration`).
- Be careful with units if you mix SI units—these docstrings follow the units used in the code/table.
"""

from typing import Tuple

import numpy as np
from numba import njit, prange

from config import _step, total_duration, dur_chirp, f_start, f_end, min_eig_floor


@njit
def Generate_K(over1_m: float) -> float:
    """
    Generate_K(over1_m: float) -> float
    ----------------------------------

    Compute an approximate stiffness coefficient K as a function of an "over-etch"
    or geometric correction parameter `over1_m`.

    Parameters
    ----------
    over1_m : float
        Over-etch length for the stiffness geometry, expressed in micrometers (µm).

    Returns
    -------
    float
        Stiffness value `K` in **nanonewtons per micrometer (nN/µm)** as used by the caller.
        (The implementation multiplies an internal estimate by 1000 before returning;
        consult the code if you require a different unit scaling.)

    Notes
    -----
    - The function uses a simple Euler-Bernoulli beam approximation:
        k = 3 * E * I / (n * L_short^3)
      where E, I, n, and L_short are computed from `over1_m`.
    - `E` is set to 160e3 (units chosen to match the rest of the code and return units).
    - `I` is estimated using a beam width `width_beam` with a rectangular-section moment of inertia,
      using the shape parameters embedded in the function. Adjust if you have different geometry.
    - This function is `@njit`-compiled; it returns a scalar float appropriate for use in other Numba functions.
    """

    E = 160e3

    L_long = 212.9 - 2 * over1_m
    L_short = L_long / 2
    width_beam = 2.8 - 2 * over1_m
    I = 30 * width_beam ** 3 / 12
    n = 10
    k = 3 * E * I / (n * L_short ** 3)
    return k * 1000

@njit
def Generate_C_damping(Q: float, K: float, M: float) -> float:
    """
    Generate_C_damping(Q: float, K: float, M: float) -> float
    --------------------------------------------------------

    Compute an equivalent linear viscous damping coefficient from a quality factor Q,
    stiffness K, and mass M.

    Parameters
    ----------
    Q : float
        Mechanical quality factor (dimensionless).
    K : float
        Stiffness (same units returned by `Generate_K`, e.g. nN/µm).
    M : float
        Effective mass (units consistent with the rest of the code — assumed µg).

    Returns
    -------
    float
        Damping coefficient `C_damping` in the code's internal units (consistent with mass, time, and stiffness).
        The function uses ω = sqrt(K / M) and returns C = ω * M / Q.

    Notes
    -----
    - Because the code uses non-SI unit conventions (µm, µg, ms), this formula produces a
      damping coefficient consistent with those units. If you need SI units, convert K and M first.
    - The function is `@njit` and returns a scalar.
    """

    omega = np.sqrt(K / M)
    C_damping = omega * M / Q
    return C_damping

@njit
def _cap_from_gap(d: float, over2_m: float = 0.0) -> float:
    """
    _cap_from_gap(d: float, over2_m: float = 0.0) -> float
    ----------------------------------------------------

    Compute the parallel-plate + fringe capacitance approximation given a gap `d`.

    Parameters
    ----------
    d : float
        Gap between electrodes (µm). The routine floors very small gaps to `min_eig_floor` to avoid singularity.
    over2_m : float, optional
        Geometric over-etch on electrode region (µm). Default is 0.0.

    Returns
    -------
    float
        Capacitance in **femtofarads (fF)** as computed by the model.

    Notes
    -----
    - The function uses an empirical formula containing terms:
        C = eps * L * (w/d + (1/pi) * (1 + log(2*w*pi/d) + log(1 + ...))
      where `eps` is defined in the function as 8.854e-3 (units selected to match fF/µm).
    - The routine ensures `d >= min_eig_floor` to avoid division by zero and returns 0.0 if the result is not finite.
    - This function is Numba-jitted (`@njit`) and must only call other Numba-compatible routines.
    """
    eps = 8.854e-3
    L = 101 - 2 * over2_m
    L_mov = 104.5 - over2_m
    w = 30
    h = 3.5 - 2 * over2_m
    L = np.sqrt(L * L_mov)
    d = np.maximum(d, min_eig_floor)
    C = eps * L * (w / d + 1 / np.pi * (
            1 + np.log(2 * w * np.pi / d) +
            np.log(1 + 2 * h / d + 2 * np.sqrt(h / d + (h / d) ** 2))
    ))
    if not np.isfinite(C):
        C = 0.0
    return C

@njit
def _dC_dd(d: float, over2_m: float, h_eps: float = 1e-9) -> float:
    """
    _dC_dd(d: float, over2_m: float, h_eps: float = 1e-9) -> float
    ------------------------------------------------------------

    Numerical derivative of capacitance with respect to the gap `d` using central difference.

    Parameters
    ----------
    d : float
        Gap (µm).
    over2_m : float
        Electrode over-etch parameter (µm), forwarded to `_cap_from_gap`.
    h_eps : float, optional
        Finite-difference step (µm). Default is 1e-9.

    Returns
    -------
    float
        Approximation to dC/dd (fF per µm).

    Notes
    -----
    - Central difference: (C(d + h) - C(d - h)) / (2*h). The step `h_eps` is small — ensure it is large enough
      not to fall below machine precision in the chosen units.
    - Function is decorated with `@njit`.
    """
    d_plus = d + h_eps
    d_minus = d - h_eps
    C_plus = _cap_from_gap(d_plus, over2_m)
    C_minus = _cap_from_gap(d_minus, over2_m)
    return (C_plus - C_minus) / (2.0 * h_eps)

@njit
def dC_dx_right(x: float, over2_m: float, offset_m: float) -> float:
    """
    dC_dx_right(x: float, over2_m: float, offset_m: float) -> float
    ---------------------------------------------------------------

    Chain-rule derivative dC/dx for the right electrode configuration.

    Parameters
    ----------
    x : float
        Displacement of the moving electrode (µm). Positive x moves the structure toward the right electrode.
    over2_m : float
        Electrode over-etch (µm).
    offset_m : float
        Static offset (µm) shifting the baseline gap.

    Returns
    -------
    float
        dC/dx for the right electrode in units fF per µm (using the chain rule).

    Notes
    -----
    - For the right electrode the gap is computed as d = d0 - x - offset_m; thus dd/dx = -1 and dC/dx = dC/dd * (-1).
    - Uses `_dC_dd` internally. Numba jitted.
    """
    d0 = 1.2 + 2 * over2_m
    d = d0 - x - offset_m
    dCdd = _dC_dd(d, over2_m)
    # chain rule: dC/dx = dC/dd * dd/dx ; dd/dx = -1 for right
    return dCdd * (-1.0)

@njit
def dC_dx_left(x: float, over2_m: float, offset_m: float) -> float:
    """
    dC_dx_left(x: float, over2_m: float, offset_m: float) -> float
    --------------------------------------------------------------

    Chain-rule derivative dC/dx for the left electrode configuration.

    Parameters
    ----------
    x : float
        Displacement of the moving electrode (µm). Positive x moves the structure toward the right electrode.
    over2_m : float
        Electrode over-etch (µm).
    offset_m : float
        Static offset (µm) shifting the baseline gap.

    Returns
    -------
    float
        dC/dx for the left electrode (fF per µm).

    Notes
    -----
    - For the left electrode the gap is computed as d = d0 + x + offset_m; thus dd/dx = +1 and dC/dx = dC/dd * (+1).
    - Uses `_dC_dd` internally. Numba jitted.
    """
    d0 = 1.2 + 2 * over2_m
    d = d0 + x + offset_m
    return _dC_dd(d, over2_m)

@njit
def Capacitance_right(x: float, over2_m: float, offset_m: float) -> float:
    """
    Capacitance_right(x: float, over2_m: float, offset_m: float) -> float
    --------------------------------------------------------------------

    Compute the right-electrode capacitance for a given position and geometry.

    Parameters
    ----------
    x : float
        Displacement (µm).
    over2_m : float
        Over-etch on electrode (µm).
    offset_m : float
        Static offset (µm).

    Returns
    -------
    float
        Capacitance (fF) for the right electrode.

    Notes
    -----
    - Computes gap d = d0 - x - offset_m with d0 = 1.2 + 2 * over2_m, then calls `_cap_from_gap`.
    - Numba jitted.
    """
    d0 = 1.2 + 2 * over2_m
    d = d0 - x - offset_m  # offset_m positive -> nearer right electrode
    return _cap_from_gap(d, over2_m)


@njit
def Capacitance_left(x: float, over2_m: float, offset_m: float) -> float:
    """
    Capacitance_left(x: float, over2_m: float, offset_m: float) -> float
    -------------------------------------------------------------------

    Compute the left-electrode capacitance for a given position and geometry.

    Parameters
    ----------
    x : float
        Displacement (µm).
    over2_m : float
        Over-etch on electrode (µm).
    offset_m : float
        Static offset (µm).

    Returns
    -------
    float
        Capacitance (fF) for the left electrode.

    Notes
    -----
    - Computes gap d = d0 + x + offset_m with d0 = 1.2 + 2 * over2_m, then calls `_cap_from_gap`.
    - Numba jitted.
    """
    d0 = 1.2 + 2 * over2_m
    d = d0 + x + offset_m  # offset_m positive -> further from left electrode
    return _cap_from_gap(d, over2_m)

@njit
def _chirp_voltage(t_local: float, min_voltage: float) -> float:
    """
    _chirp_voltage(t_local: float, min_voltage: float) -> float
    ----------------------------------------------------------

    Return a linear frequency chirp voltage waveform evaluated at local time t_local.

    Parameters
    ----------
    t_local : float
        Local time within the chirp window (ms). Expected domain: [0, dur_chirp).
    min_voltage : float
        Peak voltage amplitude to scale the sine wave (same units as caller's voltage).

    Returns
    -------
    float
        Instantaneous voltage (same units as `min_voltage`) at time `t_local`. If `t_local` is out of
        the chirp interval (> dur_chirp), returns 0.0 (clamped).

    Notes
    -----
    - Frequency sweeps linearly from `f_start` to `f_end` over `dur_chirp` (frequencies in kHz).
    - The phase is integrated: phase(t) = 2π (f_start*t + 0.5*k*t^2) where k = (f_end - f_start)/dur_chirp.
    - The function is `@njit` and expects global `f_start`, `f_end`, `dur_chirp` to be numeric scalars.
    """
    if t_local > dur_chirp:
        return 0.0

    k = (f_end - f_start) / dur_chirp
    phase = 2.0 * np.pi * (f_start * t_local + 0.5 * k * t_local ** 2)
    return min_voltage * np.sin(phase)

@njit
def single_block_voltage(t_local: float, min_voltage: float) -> float:
    """
    single_block_voltage(t_local: float, min_voltage: float) -> float
    ----------------------------------------------------------------

    Return a single-block electrode excitation waveform which is a chirp for t in [0, dur_chirp)
    and zero otherwise.

    Parameters
    ----------
    t_local : float
        Local time (ms).
    min_voltage : float
        Peak amplitude used by the chirp.

    Returns
    -------
    float
        Voltage at time `t_local`. Chirp waveform inside chirp window, otherwise 0.

    Notes
    -----
    - Simple wrapper that calls `_chirp_voltage` when 0 <= t_local < dur_chirp.
    """
    return _chirp_voltage(t_local, min_voltage=min_voltage) if 0.0 <= t_local < dur_chirp else 0.0

@njit
def F_ele(x: float, t: float, over2_m: float, offset_m: float, min_voltage: float) -> float:
    """
    F_ele(x: float, t: float, over2_m: float, offset_m: float, min_voltage: float) -> float
    -------------------------------------------------------------------------------------

    Compute the total electrostatic force on the moving electrode as the sum of left and right contributions.

    Parameters
    ----------
    x : float
        Displacement (µm).
    t : float
        Global time (ms). If t is outside [0, total_duration), the applied voltage is considered zero.
    over2_m : float
        Electrode over-etch (µm).
    offset_m : float
        Static offset (µm).
    min_voltage : float
        Peak voltage amplitude (units consistent with measurement model).

    Returns
    -------
    float
        Total electrostatic force (units consistent with K, M, and the rest of the model). The code multiplies
        each electrode contribution by 0.5 and then by V^2 as implemented; the returned force is a scalar.

    Notes
    -----
    - Internally:
        V = single_block_voltage(t, min_voltage)
        V_right = V, V_left = -V (opposite polarity)
        F = 0.5 * dC_dx_right * V_right^2 + 0.5 * dC_dx_left * V_left^2
    - The function returns 0.0 if the computed force is not finite.
    - Numba jitted.
    """
    if t < 0.0 or t >= total_duration:
        V_right = 0.0
        V_left = 0.0
    else:
        V = single_block_voltage(t, min_voltage=min_voltage)
        V_right = V
        V_left = -V

    F_right_full = 0.5 * dC_dx_right(x, over2_m, offset_m) * (V_right ** 2)
    F_left_full = 0.5 * dC_dx_left(x, over2_m, offset_m) * (V_left ** 2)

    F_total = F_left_full + F_right_full
    if not np.isfinite(F_total):
        F_total = 0.0
    return F_total

@njit
def Generate_ODE(t: float, y: np.ndarray, K: float, M: float, C_damping: float,
                 over2_m: float, offset_m: float, min_voltage: float) -> np.ndarray:
    """
    Generate_ODE(t: float, y: np.ndarray, K: float, M: float, C_damping: float,
                 over2_m: float, offset_m: float, min_voltage: float) -> np.ndarray
    ----------------------------------------------------------------------

    Right-hand side of the second-order MEMS ODE converted to first-order form.

    Parameters
    ----------
    t : float
        Current time (ms).
    y : np.ndarray, shape (2,)
        State vector [displacement (µm), velocity (µm/ms)].
    K : float
        Stiffness (nN/µm or consistent unit).
    M : float
        Mass (µg or consistent unit).
    C_damping : float
        Viscous damping coefficient (code units consistent with M and K).
    over2_m : float
        Electrode over-etch (µm).
    offset_m : float
        Static offset (µm).
    min_voltage : float
        Voltage amplitude used in force calculation.

    Returns
    -------
    np.ndarray, shape (2,)
        Time derivative of state [velocity, acceleration] where acceleration is computed from:
            dx2dt = (F_ele + offset_m*K - C_damping*x2 - K*x1) / M
        Note: the code multiplies `F_ele(...) * 10` in the acceleration, consistent with the
        internal scaling used in this model.

    Notes
    -----
    - Units must be consistent across K, M, C_damping, and F_ele. The function is `@njit`.
    """
    x1 = y[0]
    x2 = y[1]

    dx1dt = x2  # Velocity
    # Acceleration: (F_ele + spring_offset - damping - spring) / mass
    dx2dt = (F_ele(x1, t, over2_m, offset_m, min_voltage=min_voltage) * 10 + offset_m * K -
             C_damping * x2 - K * x1) / M
    return np.array([dx1dt, dx2dt])

@njit
def solve_ODE_disp(uncertainty: np.ndarray, min_voltage: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    solve_ODE_disp(uncertainty: np.ndarray, min_voltage: float) -> Tuple[np.ndarray, np.ndarray]
    --------------------------------------------------------------------------------------------

    Integrate the MEMS ODE over the simulation time and return displacement and velocity history.

    Parameters
    ----------
    uncertainty : np.ndarray, shape (5,)
        Parameter vector of the form:
            [over_stiffness_um, over_electrode_um, over_hole_um, Q, offset_um]
        Units: µm for geometric parameters, Q (dimensionless), offset in µm.
    min_voltage : float
        Voltage amplitude for the excitation waveform.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (disp, vel) arrays of length `n_steps` where n_steps = len(np.arange(0, total_duration + _step, _step)).
        - disp: displacement time series (µm)
        - vel: velocity time series (µm/ms)

    Notes
    -----
    - The function constructs K = 2 * Generate_K(over1_m) and uses a fixed mass M = 5.67501869 (units consistent with code).
    - Damping is computed from Q via Generate_C_damping.
    - The integrator used at each step is `dop853_step`, an explicit 12-stage Runge-Kutta method (DOP853 coefficients embedded).
    - The ODE is advanced with a constant step `dt = _step` for `total_duration`.
    - The returned arrays are Numba-jitted and thus are plain NumPy arrays.
    """
    over_stiffness_um, over_electrode_um, over_hole_um, Q, offset_um = uncertainty

    over1_m = over_stiffness_um
    over2_m = over_electrode_um
    offset_m = offset_um

    K = 2 * Generate_K(over1_m)
    M = 5.67501869
    C_damping = Generate_C_damping(Q, K, M)

    x0 = np.array([offset_m, 0.0])

    dt = _step
    t_eval = np.arange(0.0, total_duration + _step, _step)
    n_steps = len(t_eval)

    disp = np.zeros(n_steps)
    vel = np.zeros(n_steps)

    y = x0.copy()
    t = 0.0

    disp[0] = y[0]
    vel[0] = y[1]

    for i in range(1, n_steps):
        y = dop853_step(Generate_ODE, y, t, dt, K, M, C_damping, over2_m, offset_m, min_voltage=min_voltage)
        t += dt
        disp[i] = y[0]
        vel[i] = y[1]

    return disp, vel

@njit(parallel=True)
def solve_ODE(uncertainty: np.ndarray, min_voltage: float) -> np.ndarray:
    """
    solve_ODE(uncertainty: np.ndarray, min_voltage: float) -> np.ndarray
    -------------------------------------------------------------------

    Compute the capacitance difference time series ΔC(t) by integrating the mechanical ODE
    and evaluating capacitances at each displacement sample.

    Parameters
    ----------
    uncertainty : np.ndarray, shape (5,)
        [over_stiffness_um, over_electrode_um, over_hole_um, Q, offset_um]
    min_voltage : float
        Voltage amplitude for the chirp/block waveform.

    Returns
    -------
    np.ndarray, shape (n_steps,)
        ΔC(t) = 10 * (C_right - C_left) sampled at every time step from 0 to total_duration (inclusive).
        Values are in fF (after the multiplication factor used in the code). Non-finite entries are set to 0.0.

    Notes
    -----
    - This function calls `solve_ODE_disp` to obtain displacement history, then computes per-sample capacitances:
          dC_posi[i] = Capacitance_right(disp[i], over2_m, offset_m)
          dC_neg[i]  = Capacitance_left(disp[i], over2_m, offset_m)
      and returns dC = (dC_posi - dC_neg) * 10 for each sample.
    - Decorated with `@njit(parallel=True)` and uses `prange` to parallelize the per-sample capacitance computation.
    - Ensure `over2_m` and `offset_m` are within the physical ranges expected by the capacitance model.
    """
    over_stiffness_um, over_electrode_um, over_hole_um, Q, offset_um = uncertainty

    over2_m = over_electrode_um
    offset_m = offset_um

    disp, _ = solve_ODE_disp(uncertainty, min_voltage=min_voltage)

    n = len(disp)
    dC_posi = np.zeros(n)
    dC_neg = np.zeros(n)
    for i in prange(n):
        dC_posi[i] = Capacitance_right(disp[i], over2_m, offset_m)
        dC_neg[i] = Capacitance_left(disp[i], over2_m, offset_m)
    dC = (dC_posi - dC_neg) * 10
    dC[~np.isfinite(dC)] = 0.0

    return dC

@njit
def meas_func(state: np.ndarray) -> np.ndarray:
    """
    meas_func(state: np.ndarray) -> np.ndarray
    ------------------------------------------

    Measurement function mapping a full UKF state vector to a scalar measurement ΔC.

    Parameters
    ----------
    state : np.ndarray, shape (>=6,)
        UKF-style state vector where:
          state[1] = over2_m (electrode over-etch, µm)
          state[3] = offset_m (µm)
          state[4] = position (µm)
        (This ordering matches the SRUKF state convention used in your project.)

    Returns
    -------
    np.ndarray, shape (1,)
        Single-element array containing ΔC = 10 * (C_right - C_left) in fF. Non-finite values are clamped to 0.0.

    Notes
    -----
    - This is the measurement mapping used by the filter's prediction/update. It is `@njit`.
    """
    pos = state[4]
    over2_m = state[1]
    offset_m = state[3]
    dC_posi = Capacitance_right(pos, over2_m, offset_m)
    dC_neg = Capacitance_left(pos, over2_m, offset_m)
    dC = (dC_posi - dC_neg) * 10
    if not np.isfinite(dC):
        dC = 0.0
    return np.array([dC])

@njit
def compute_meas_gradient(state: np.ndarray, h_eps: float = 1e-9) -> np.ndarray:
    """
    compute_meas_gradient(state: np.ndarray, h_eps: float = 1e-9) -> np.ndarray
    ----------------------------------------------------------------------------

    Finite-difference approximation of the measurement Jacobian ∂h/∂x for the scalar measurement `meas_func`.

    Parameters
    ----------
    state : np.ndarray, shape (n,)
        State vector at which to evaluate the gradient.
    h_eps : float, optional
        Finite-difference step for central differences (default 1e-9).

    Returns
    -------
    np.ndarray, shape (n,)
        Numerical gradient vector of the scalar measurement with respect to each state component.

    Notes
    -----
    - Uses central difference: (h(x + eps e_i) - h(x - eps e_i)) / (2*eps).
    - The function is `@njit` and therefore must call other Numba-compatible routines (it calls `meas_func`).
    """
    gradient = np.zeros_like(state)
    for i in range(len(state)):
        state_plus = state.copy()
        state_plus[i] += h_eps
        h_plus = meas_func(state_plus)[0]

        state_minus = state.copy()
        state_minus[i] -= h_eps
        h_minus = meas_func(state_minus)[0]

        gradient[i] = (h_plus - h_minus) / (2.0 * h_eps)
    return gradient

@njit
def compute_static_sensitivity(model_state_5d: np.ndarray) -> float:
    """
    compute_static_sensitivity(model_state_5d: np.ndarray) -> float
    ---------------------------------------------------------------

    Compute a static sensitivity metric (ΔC change) for the 5-component model state.

    Parameters
    ----------
    model_state_5d : np.ndarray, shape (5,)
        Parameter / state vector in the form:
          [over1_m, over2_m, over3_m, Q, offset_m]
        Units: µm for geometric params, Q dimensionless, offset µm.

    Returns
    -------
    float
        Scalar sensitivity equal to 10 * (deltaC_final - deltaC_initial), where deltaC = C_right - C_left.
        Units: fF (after the factor 10 in the code).

    Notes
    -----
    - Procedure:
        1. Compute initial ΔC at `offset_m` (C_r_initial - C_l_initial).
        2. Compute static equilibrium deflection x_ss = offset_m + F_ext / K where F_ext = M * g.
        3. Compute final ΔC at x_ss.
        4. Return scaled difference (sensitivity = 10 * (deltaC_final - deltaC_initial)).
    - Uses Generate_K and Capacitance_* functions. M uses the same mass constant used elsewhere in the module.
    - Function is `@njit`.
    """
    over1_m, over2_m, over3_m, Q, offset_m = model_state_5d
    K = 2.0 * Generate_K(over1_m)
    M = 5.67501869
    C_r_initial = Capacitance_right(offset_m, over2_m, offset_m)
    C_l_initial = Capacitance_left(offset_m, over2_m, offset_m)
    deltaC_0 = C_r_initial - C_l_initial
    g = 9.80665
    F_ext = M * g
    x_ss = offset_m + F_ext / K
    C_r_final = Capacitance_right(x_ss, over2_m, offset_m)
    C_l_final = Capacitance_left(x_ss, over2_m, offset_m)
    deltaC_final = C_r_final - C_l_final
    sensitivity = deltaC_final - deltaC_0
    return sensitivity * 10

# DOP853 coefficients
C_DOP853 = np.array([
    0.0, 0.05260015, 0.07890023, 0.11835034, 0.28164966, 0.33333333,
    0.25, 0.30769231, 0.65128205, 0.6, 0.85714286, 1.0
])

A_DOP853 = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.052600152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.019725057, 0.059175171, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0295875855, 0.0, 0.0887627564, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.241365134, 0.0, -0.884549479, 0.924834003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.037037037, 0.0, 0.0, 0.170828609, 0.125467688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.037109375, 0.0, 0.0, 0.170252211, 0.060216539, -0.017578125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0370920001, 0.0, 0.0, 0.170383926, 0.10726203, -0.0153194377, 0.00827378916, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.624110959, 0.0, 0.0, -3.36089263, -0.868219347, 27.5920997, 20.1540676, -43.4898842, 0.0, 0.0, 0.0, 0.0],
    [0.477662536, 0.0, 0.0, -2.48811462, -0.590290827, 21.2300514, 15.2792336, -33.2882110, -0.0203312017, 0.0, 0.0, 0.0],
    [-0.93714243, 0.0, 0.0, 5.18637243, 1.09143735, -8.14978701, -18.5200657, 22.7394871, 2.49360555, -3.04676447, 0.0, 0.0],
    [2.27331015, 0.0, 0.0, -10.5344955, -2.00087206, -17.9589319, 27.9488845, -2.85899828, -8.87285693, 12.3605672, 0.643392746, 0.0]
])

B_DOP853 = np.array([
    0.05429373, 0.0, 0.0, 0.0, 0.0, 4.45031289,
    1.8915179, -5.80120396, 0.31116437, -0.15216095, 0.2013654, 0.04471062
])

@njit
def dop853_step(ode_func, y0: np.ndarray, t0: float, dt: float, K: float, M: float, C_damping: float, over2_m: float,
                offset_m: float, min_voltage: float) -> np.ndarray:
    """
    dop853_step(ode_func, y0: np.ndarray, t0: float, dt: float, K: float, M: float, C_damping: float,
                over2_m: float, offset_m: float, min_voltage: float) -> np.ndarray
    ---------------------------------------------------------------------------------------

    Single fixed-step integration using an explicit Dormand–Prince 8(5,3) (DOP853) coefficient tableau.

    Parameters
    ----------
    ode_func : Callable
        ODE right-hand-side function taking (t, y, K, M, C_damping, over2_m, offset_m, min_voltage) and returning dy/dt.
    y0 : np.ndarray, shape (m,)
        Current state vector.
    t0 : float
        Current time (ms).
    dt : float
        Time step (ms) to advance.
    K, M, C_damping, over2_m, offset_m, min_voltage : floats
        Parameters forwarded to `ode_func`.

    Returns
    -------
    np.ndarray, shape (m,)
        Estimated state at time t0 + dt computed as y0 + dt * sum(B_j * k_j).

    Notes
    -----
    - The function uses hard-coded DOP853 coefficients (C_DOP853, A_DOP853, B_DOP853).
    - This is an explicit fixed-step integrator (no adaptivity implemented here).
    - It is `@njit` and expects `ode_func` to be Numba-compatible (you pass `Generate_ODE`).
    - Accuracy and stability depend on `dt` relative to the dynamic time scales of the system.
    """
    k = np.zeros((12, y0.shape[0]))
    k[0] = ode_func(t0, y0, K, M, C_damping, over2_m, offset_m, min_voltage=min_voltage)
    for i in range(1, 12):
        dy_i = np.zeros_like(y0)
        for j in range(i):
            dy_i += A_DOP853[i, j] * k[j]
        yi = y0 + dt * dy_i
        k[i] = ode_func(t0 + C_DOP853[i] * dt, yi, K, M, C_damping, over2_m, offset_m, min_voltage=min_voltage)
    dy = np.zeros_like(y0)
    for j in range(12):
        dy += B_DOP853[j] * k[j]
    return y0 + dt * dy

@njit
def propagate_state(state: np.ndarray, current_t: float, dt: float, min_voltage: float) -> np.ndarray:
    """
    propagate_state(state: np.ndarray, current_t: float, dt: float, min_voltage: float) -> np.ndarray
    ---------------------------------------------------------------------------------------------------

    Propagate a full filter state forward by one time step using the mechanical ODE integrator.

    Parameters
    ----------
    state : np.ndarray, shape (6,)
        Filter state vector organized as:
          [over_stiffness_um, over_electrode_um, Q, offset_um, pos, vel]
        Units: µm for geometry/pos/offset, Q dimensionless.
    current_t : float
        Current time (ms).
    dt : float
        Time step (ms).
    min_voltage : float
        Waveform amplitude used for forcing.

    Returns
    -------
    np.ndarray, shape (6,)
        New state vector after propagation:
          [over_stiffness_um, over_electrode_um, Q, offset_um, new_pos, new_vel]

    Notes
    -----
    - The function:
        1. extracts mechanical parameters from the state,
        2. computes K = 2 * Generate_K(over1_m) and C_damping from Q,
        3. integrates the mechanical ODE over the interval [current_t, current_t + dt] using dop853_step,
        4. returns the unchanged parameter entries and the updated (pos, vel).
    - Decorated with `@njit` for speed; the integrator and ODE functions must be Numba-compatible.
    - Keep `dt` sufficiently small relative to the highest frequency content in the chirp to ensure stability and accuracy.
    """
    over_stiffness_um = state[0]
    over_electrode_um = state[1]
    Q = state[2]
    offset_um = state[3]
    pos = state[4]
    vel = state[5]

    over1_m = over_stiffness_um
    over2_m = over_electrode_um
    offset_m = offset_um

    K = 2 * Generate_K(over1_m)
    M = 5.67501869
    C_damping = Generate_C_damping(Q, K, M)

    y0 = np.array([pos, vel])

    y_new = dop853_step(Generate_ODE, y0, current_t, dt, K, M, C_damping, over2_m, offset_m, min_voltage=min_voltage)
    new_pos = y_new[0]
    new_vel = y_new[1]


    new_state = np.array([over_stiffness_um, over_electrode_um, Q, offset_um,
                          new_pos, new_vel])
    return new_state