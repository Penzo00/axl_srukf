"""
Configuration and global constants for the SR-UKF MEMS parameter-tracking package.

This module centralizes global parameters, low-level numerical tolerances, time/frequency
constants, logging configuration, and the `UKFConfig` dataclass that bundles algorithmic
hyperparameters and noise defaults.

Intended usage
--------------
- Import this module where the filter, models, and simulation code need consistent
  parameterization and default values:
      from config import UKFConfig, f_start, total_duration, _step, logger

- Override settings by creating a new `UKFConfig(...)` instance or by modifying fields
  when instantiating classes that accept a `UKFConfig`.

Key global constants (units & semantics)
---------------------------------------
- Frequencies:
    * `f_start`, `f_end` : float (kHz)
      Start and end chirp frequencies in kilohertz. Example: f_start = 1.0 (1 kHz).
- Time:
    * `T` : float (ms)
      Period at `f_start` in **milliseconds**. Computed as T = 1.0 / f_start.
    * `_step` : float (ms)
      Base time step used throughout the code (set to T / 1200).
    * `dur_chirp` : float (ms)
      Duration of the frequency chirp block (set to 2.0 * T).
    * `dur_zero` : float (ms)
      Quiet interval duration appended after the chirp (set to 0.75 * T).
    * `block` : float (ms)
      Block duration = dur_chirp + dur_zero.
    * `total_duration` : float (ms)
      Simulation total duration; currently set equal to `block`.
- Numerical tolerances:
    * `min_eig_floor` : float
      Small eigenvalue floor used to stabilize inversions and avoid division-by-zero.
      Set to `np.finfo(float).eps * 1e3` (machine epsilon scaled up).
- Sigma/sr-UKF helper:
    * `q` : np.ndarray, shape (6,)
      Coefficients used by the square-root sigma-point construction. Computed by:
          q[t] = 1.37218717 * sqrt((t + 1) / (t + 2))  for t = 0..5
- RNG:
    * `np.random.seed(42)` — sets the global NumPy RNG seed for reproducible pseudo-random draws.

Logging
-------
- `logger` is a module-level `logging.Logger` configured with a human-readable format and
  default level INFO. Use `logger` for consistent diagnostic messages across modules.

Notes and compatibility
-----------------------
- Units: the codebase frequently uses non-SI units: lengths in micrometers (µm), mass in
  micrograms (µg), time in milliseconds (ms), frequency in kHz, and capacitance in fF.
  Before converting values to SI, verify units carefully when interfacing external data.
- `min_eig_floor` is chosen to be safely larger than machine epsilon but may need tuning
  for extremely ill-conditioned covariance matrices on low-precision hardware.
- The `q` vector length equals 6 because a 6-dimensional state is used by the SR-UKF in
  this project; update it if you change the state dimension.
- This module is safe to import in Numba-compiled contexts (it only defines numeric constants
  and pure-Python dataclasses). However, passing Python objects into nopython-compiled
  functions requires care; prefer passing numerical arrays and scalars.
"""

import logging
import numpy as np
from dataclasses import dataclass, field

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Global parameters
f_start = 1.  # 0.8 kHz -> 800 Hz
f_end = 1.2  # 1.5 kHz -> 1500 Hz
T = 1.0 / f_start  # period at start frequency (ms)
f_drive = f_start  # base drive frequency (kHz)
_step = T / 1200
dur_chirp = 2. * T
dur_zero = .75 * T
block = dur_chirp + dur_zero
total_duration = block
min_eig_floor = np.finfo(float).eps * 1e3
q = np.zeros(6)
for t in range(6):
    q[t] = 1.37218717 * np.sqrt((t + 1) / (t + 2))

@dataclass
class UKFConfig:
    """
    Configuration container for the Square-Root Unscented Kalman Filter and associated
    algorithmic extensions used in this project (GMCC, STF, constrained gains, adaptive noise).

    This dataclass centralizes boolean feature toggles, UT parameters, noise defaults,
    and algorithm hyperparameters so that the filter and tracking code can accept a single,
    documented configuration object.

    Attributes
    ----------
    enable_adaptive : bool, default True
        If True, enable the adaptive noise estimation block that adjusts `Q` and `R` using
        innovation statistics and an exponential forgetting factor.
    enable_mcc : bool, default True
        If True, enable the GMCC (Generalized Maximum Correntropy Criterion) iterative update
        instead of the standard Kalman update. This provides robustness to non-Gaussian measurement errors.
    enable_fading : bool, default True
        If True, enable Strong Tracking Fading (STF) that can scale the predicted covariance
        to improve tracking of abrupt changes.
    enable_constraints : bool, default True
        If True, apply linear inequality constraints (A x <= b) through a constrained-gain projection.
    use_true_jacobian : bool, default True
        If True and the true measurement Jacobian is available, use it instead of the UT-based approximation.

    alpha : float, default 0.518638
        Unscented transform (UT) scaling parameter (affects sigma point spread).
    beta : float, default 2.0
        UT parameter capturing prior knowledge about the distribution (2 is optimal for Gaussian).

    process_noise : tuple[float, ...], default (1e-10, 1e-10, 1e-10, 1e-10, 1e-6, 1e-5)
        Default process-noise diagonal entries for the 6-state model. Interpreted as variances
        and converted to a diagonal matrix where required.
    initial_covariance : tuple[float, ...], default (0.01, 0.01, 0.0025, 0.01, 0.01, 0.01)
        Default variances used to initialize the filter state covariance if no custom covariance is provided.

    adaptive_forgetting_b : float, default 0.98
        Exponential forgetting factor used in adaptive noise estimation (b in equations).
    min_eig_floor : float, default <module min_eig_floor>
        Small value used to floor eigenvalues and denominators to avoid numerical instability.

    mcc_eps : float, default 1e-5
        Convergence tolerance for the GMCC iterative update (relative change threshold).
    mcc_max_iter : int, default 20
        Maximum number of GMCC iterations per update.
    mu : float, default 1.0
        GMCC-related parameter (kept for compatibility with algorithmic formulations).
    gamma_val : float, default 2.0
        GMCC-related parameter (tunable kernel-shape or scaling factor depending on implementation).

    psi : float, default 3.0
        STF tuning parameter used in the formulation of the fading detection equation.
    epsilon : float, default 0.98
        STF smoothing parameter used when updating residual covariance V_k.
    window_size : int, default 10
        Window length (number of recent residuals) used for adaptive statistics such as F_k.

    Q_min : np.ndarray, shape (6, 6), default min_eig_floor * I_6
        Minimum allowable process-noise covariance (used to prevent Q reduction to zero).
    R_min : float, default 1e-4
        Minimum allowable measurement noise variance.

    num_iter : int, default 20
        Default number of tracking iterations (convenience field used by scripts or tests).

    Examples
    --------
    Create default config and override a few fields:

    >>> cfg = UKFConfig()
    >>> cfg.enable_mcc = False
    >>> cfg.process_noise = (1e-8, 1e-8, 1e-8, 1e-8, 1e-5, 1e-4)

    >>> # Instantiate SRUKF with a custom config:
    >>> from srukf import SRUKF
    >>> ukf = SRUKF(state_dim=6, config=cfg)

    Implementation notes
    --------------------
    - `Q_min` is created with a `default_factory` to ensure each UKFConfig instance
      receives its own array. It uses the module-level `min_eig_floor`.
    - Many algorithms in the code expect arrays (e.g. process_noise converted to a diagonal
      matrix). When handing these values into numerical code (Numba or pure NumPy), convert
      sequences to `np.ndarray` with dtype `float64` as appropriate.
    - This dataclass is intentionally lightweight and serializable by default; you may
      extend it with validation methods if stricter checks are desired (e.g., ensure
      positive definiteness of Q_min or non-negative R_min).
    """
    # Algorithm features
    enable_adaptive: bool = True
    enable_mcc: bool = True
    enable_fading: bool = True
    enable_constraints: bool = True
    use_true_jacobian: bool = True

    # UKF parameters
    alpha: float = 0.518638
    beta: float = 2.0

    # Noise parameters
    process_noise: tuple = (1e-10, 1e-10, 1e-10, 1e-10, 1e-6, 1e-5)
    initial_covariance: tuple = (0.01, 0.01, 0.0025, 0.01, 0.01, 0.01)

    # Adaptive parameters
    adaptive_forgetting_b: float = 0.98
    min_eig_floor: float = min_eig_floor

    # GMCC parameters
    mcc_eps: float = 1e-5
    mcc_max_iter: int = 20
    mu: float = 1.0
    gamma_val: float = 2.0

    # STF parameters
    psi: float = 3.0
    epsilon: float = 0.98
    window_size: int = 10

    # Minimum noise values
    Q_min: np.ndarray = field(default_factory=lambda: min_eig_floor * np.eye(6))
    R_min: float = 1e-4

    # Number of tracking iterations
    num_iter: int = 50