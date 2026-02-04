# axl_srukf  
**Square‑Root Unscented Kalman Filter (SRUKF) for MEMS X‑Accelerometer Parameter Tracking**

A custom **Square‑Root Unscented Kalman Filter (SRUKF)** implementation designed for **MEMS X‑accelerometer sensitivity calibration**.  
The project estimates key fabrication parameters — including **overetch beam**, **overetch electrodes**, **Q‑factor**, and **mechanical offset** — using:

- Monte Carlo simulations  
- Numba‑accelerated signal‑processing utilities  
- Diagnostic and statistical analysis  
- Physical modeling and noise‑aware measurement functions  

The goal is to validate and refine parameter tracking in MEMS devices through robust numerical methods and reproducible workflows.

---

## Overview

The system implements a **6‑dimensional SRUKF** with **scalar capacitance‑difference measurements**.  
It simulates MEMMS dynamics under chirp excitation, injects realistic noise, and performs iterative estimation to converge on fabrication parameters.

Monte Carlo runs provide statistical validation, while utilities ensure numerical stability (square‑root filtering, eigenvalue flooring, constrained gains).

### Key Features

- **Advanced SRUKF mechanisms**: GMCC iterations, STF adaptation, constrained Kalman gains, square‑root covariance propagation.
- **Monte Carlo validation** for robustness and statistical confidence.
- **Diagnostic plots** and extended statistical summaries.
- **Configurable hyperparameters** for frequencies, durations, noise levels, tolerances.
- **Numba‑accelerated utilities** for speed and reproducibility.

---

## Usage

### Run Monte Carlo simulations

```bash
python main.py
```

### Programmatic usage

```python
from main import main
df, stats = main()
```

This will:

- Run `N_MC_RUNS` Monte Carlo trials  
- Measure wall‑clock time  
- Print a summary (total time, per‑run time, CPU cores)  
- Save a timestamped report in `OUTPUT_DIR`  
  (e.g., `execution_time_20250101_153000.txt`)

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

meas_array = ...  # noisy capacitance data
final_est, estimates, filtered_estimates, filtered_covs, _ = tracker.run_tracking(
    meas_array,
    min_voltage=1.2
)
```

---

## Python Dependencies (Python 3.13)

The following packages are required based on all imports in the project:

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

### Install command (if needed)

```bash
pip install numpy scipy pandas matplotlib seaborn numba tqdm
```

---

## Installation

Clone and run directly (no setup.py required):

```bash
git clone <repo-url>
cd axl_srukf
python main.py
```

Assumes the scientific Python stack is already installed (offline‑friendly).

---

## Outputs

### Console Summary
- Total execution time  
- Average per Monte Carlo run  
- CPU core count  

### Timing Report
Saved as:

```
execution_time_<timestamp>.txt
```

### Monte Carlo Results
- CSV DataFrame with estimates, errors, diagnostics  
- `.npz` archives for raw samples  

### Diagnostic Plots
Saved in `PLOTS_DIR`:

- Parameter/state estimates  
- Uncertainty envelopes  
- Error convergence  
- Model‑fit overlays  

### Statistical Summaries
Extended statistics:

- mean, std, skew, kurtosis  
- median, MAD  
- percentiles  
- custom metrics  

---

## Directory Structure

```
OUTPUT_DIR/   # Reports, CSVs
PLOTS_DIR/    # Diagnostic figures
SAMPLES_DIR/  # Monte Carlo sample archives
```

---

## Modules Overview

- **main.py** — Monte Carlo simulations, timing, output generation  
- **utils.py** — Numba‑accelerated utilities (sigma points, covariance regularization, peak detection, chirp checks, Jacobians, statistics)  
- **analysis.py** — Statistical figures + diagnostic plots  
- **config.py** — Constants, UKF hyperparameters, logging setup  
- **model.py** — MEMS dynamics, ODE solvers, capacitance models  
- **monte_carlo.py** — Parameter sampling, noise injection, SRUKF runs, data saving  
- **srukf.py** — Core SRUKF implementation (square‑root, GMCC, STF, constrained gains)  
- **tracker.py** — High‑level parameter‑tracking wrapper  

---

## Notes

### Units
- µm (length)  
- µg (mass)  
- ms (time)  
- kHz (frequency)  
- fF (capacitance)

### Numerical Stability
- Eigenvalue flooring (`min_eig_floor`)  
- Covariance regularization  
- Square‑root propagation  

### Performance
- Numba JIT acceleration  
- Multiprocessing for Monte Carlo  
- First run incurs compilation overhead  

### Reproducibility
- Global RNG seed = **42**  
- Per‑run reseeding  

### Limitations
- Mass assumed constant
- No overetch proof mass
- No acceleration tracking

---

## Contributions

Refer to module docstrings for detailed function behavior.  
Pull requests and issue reports are welcome.
