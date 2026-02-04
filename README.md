# axl_srukf
**Square‑Root Unscented Kalman Filter (SRUKF) for MEMS X‑Accelerometer Parameter Tracking**

A custom **Square‑Root Unscented Kalman Filter (SRUKF)** implementation designed for **MEMS X‑accelerometer sensitivity calibration**.

The project estimates fabrication uncertainties such as **overetch beam**, **overetch electrodes**, **Q‑factor**, and **mechanical offset** using SRUKF tested on Monte Carlo simulations with truncated Gaussian distributions of the unknown parameters.

The goal is to validate and refine parameter tracking in MEMS devices through robust numerical methods and reproducible workflows, obtaining the best possible results in a fast way.

---
## Overview
The system implements a **6‑dimensional SRUKF** with **scalar capacitance‑difference measurements**.  
It simulates MEMS dynamics under chirp excitation, injects realistic noise, and performs iterative estimation to converge on fabrication parameters.  
Monte Carlo runs provide statistical validation, while utilities ensure numerical stability (square‑root filtering, eigenvalue flooring, constrained gains).

### Key Features
- **Advanced SRUKF mechanisms**: Generalized Maximum Correntropy Criterion iterations, Strong Tracking Fading adaptation, constrained Kalman gains, square‑root covariance propagation.
- **Monte Carlo validation** for robustness and statistical confidence.
- **Diagnostic plots** and extended statistical summaries (mean, std, skew, kurtosis, median, MAD, percentiles).
- **Configurable hyperparameters** for frequencies, durations, noise levels, tolerances.
- **Numba‑accelerated utilities** for speed and reproducibility.
- **Multiprocessing** support for parallel Monte Carlo trials.

---
## Usage
### Run Monte Carlo simulations
```bash
python run_monte_carlo.py
```

This will:
- Run `N_MC_RUNS` Monte Carlo trials
- Measure wall‑clock time
- Print a concise summary (total time, average per‑run time, CPU core count)
- Save a timestamped execution‑time report in `OUTPUT_DIR`
  (e.g., `execution_time_20260204_120000.txt`)

---
## Custom Tracking Example
```python
from tracker import ParameterTracker
from config import UKFConfig
import numpy as np

config = UKFConfig()
tracker = ParameterTracker(config)

tracker.initialize(
    initial_guess=np.array([0.35, 0.35, 0.5, 0., 0., 0.]),
    initial_uncertainty=np.eye(6) * 1e-2,
    process_noise=np.eye(6) * 1e-4,
    measurement_noise=1e-3
)

meas_array = ...  # noisy capacitance‑difference data
final_est, estimates, filtered_estimates, filtered_covs, _ = tracker.run_tracking(
    meas_array,
    min_voltage=1.2
)
```

---
## Python Dependencies (Python ≥ 3.13 recommended)
The following packages are required:
### Core Scientific Stack
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`

### Performance & Parallelism
- `numba`
- `multiprocessing` (standard library)

### Progress & Logging
- `tqdm`
- `logging` (standard library)

### Install command
```bash
pip install numpy scipy pandas matplotlib seaborn numba tqdm
```

---
## Installation
Clone and run directly (no setup.py required):
```bash
git clone https://github.com/Penzo00/axl_srukf
cd axl_srukf
python run_monte_carlo.py
```
The code is offline‑friendly after dependencies are installed.

---
## Outputs
### Console Summary
- Total execution time
- Average time per Monte Carlo run
- Detected CPU core count

### Timing Report
Saved in `OUTPUT_DIR` as:
```
execution_time_<YYYYMMDD_HHMMSS>.txt
```
Contains detailed timing metadata.

### Monte Carlo Results
- CSV DataFrame with final estimates, errors, uncertainties, and diagnostics
- `.npz` archives of raw generated samples (in timestamped subfolder under `SAMPLES_DIR`)

### Diagnostic Plots
Saved in `PLOTS_DIR` as PNG files:
- State/parameter estimates with ±1.96σ uncertainty bands
- Error convergence over time
- Sensitivity propagation
- Model‑fit comparisons and capacitance overlays

### Statistical Summaries
Extended Pandas describe() with:
- skew, kurtosis
- median, MAD
- selected percentiles
- custom metrics

---
## Directory Structure
```
OUTPUT_DIR/     # Timing reports, CSV results
PLOTS_DIR/      # Diagnostic PNG figures
SAMPLES_DIR/    # Timestamped Monte Carlo sample archives (.npz)
```

---
## Modules Overview
- **run_monte_carlo.py** — Entry‑point script: runs Monte Carlo simulations, measures wall‑clock time, prints summary, saves timestamped timing report
- **monte_carlo.py** — Core Monte Carlo driver: parameter sampling, noisy data generation, per‑run SRUKF tracking (with optional iteration to convergence), result collection, CSV/.npz saving
- **tracker.py** — High‑level `ParameterTracker` wrapper around SRUKF for online parameter estimation from capacitance streams
- **srukf.py** — Core square‑root UKF implementation with GMCC, Strong Tracking Fading, constrained gains, and adaptive mechanisms
- **utils.py** — Numba‑accelerated utilities: sigma‑point generation, covariance regularization, robust peak/valley detection, chirp safety checks, sensitivity Jacobians, statistics
- **analysis.py** — Statistical summaries (extended describe) and diagnostic plot generation
- **config.py** — Global constants, units, time/frequency parameters, `UKFConfig` dataclass, logging setup
- **model.py** — MEMS physical model, ODE integration, capacitance functions (left/right), sensitivity computation

---
## Notes
### Units (consistent throughout the codebase)
- length: µm (micrometers)
- mass: µg (micrograms)
- time: ms (milliseconds)
- frequency: kHz (kilohertz)
- capacitance: fF (femtofarads)
- stiffness: nN/µm
- damping: consistent with nN·ms/µm

### Numerical Stability
- Square‑root covariance propagation
- Eigenvalue flooring (`min_eig_floor`)
- Regularization helpers
- Cholesky fallback handling

### Performance
- Heavy numerical sections accelerated with Numba JIT
- Monte Carlo trials parallelized via `multiprocessing.Pool` (defaults to all CPU cores)
- First execution incurs Numba compilation overhead (subsequent runs are faster)

### Reproducibility
- Global NumPy RNG seed = **42**
- Per‑run reseeding using run index for deterministic noise/realizations
- Timestamped output folders

### Limitations
- Mass assumed known/constant
- No proof‑mass overetch modeled
- No external acceleration tracking
- Measurement dimension fixed to scalar ΔC

---
## Contributions
Detailed function‑level behavior is documented in module docstrings.  
Pull requests, bug reports, and feature suggestions are welcome.
