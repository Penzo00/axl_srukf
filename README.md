# axl_srukf
---
## Square‑Root Unscented Kalman Filter (SRUKF) for MEMS X‑Accelerometer Parameter Tracking
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
meas_array = ... # noisy capacitance‑difference data
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
git clone [https://github.com/Penzo00/axl_srukf](https://github.com/Penzo00/axl_srukf.git)
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
OUTPUT_DIR/ # Timing reports, CSV results
PLOTS_DIR/ # Diagnostic PNG figures
SAMPLES_DIR/ # Timestamped Monte Carlo sample archives (.npz)
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
## Benchmark Results (1000 Monte Carlo Runs)
**Date:** 2026-02-04 20:21:04

### Statistical Figures
```
                      count      mean       std       min       25%       50%       75%       max      skew  kurtosis    median       mad
est_stiff          1000.0  0.346135  0.088447  0.152095  0.279475  0.344934  0.412478  0.547335  0.018130 -0.723386  0.344934  0.073232
est_elec           1000.0  0.349986  0.085814  0.150314  0.288831  0.350918  0.412407  0.550057  0.002270 -0.577543  0.350918  0.070216
est_Q              1000.0  0.499997  0.044481  0.400138  0.466786  0.500963  0.533059  0.599792  0.020773 -0.759926  0.500963  0.036974
est_offset         1000.0 -0.002353  0.087956 -0.197547 -0.066651 -0.000472  0.064330  0.198526 -0.012287 -0.666873 -0.000472  0.072794
est_sensitivity    1000.0  5.065889  1.645247  2.060547  3.868755  4.785103  5.993825 13.215268  0.925159  1.070808  4.785103  1.297251
err_stiff_norm     1000.0  0.020348  0.022392  0.000010  0.006544  0.014506  0.026074  0.223296  3.440448 19.448481  0.014506  0.014641
err_elec_norm      1000.0  0.029039  0.028989  0.000021  0.009624  0.021085  0.039446  0.297259  2.792552 13.937414  0.021085  0.020115
err_Q_norm         1000.0  0.059817  0.057315  0.000031  0.022914  0.045331  0.076575  0.490194  2.607790 10.683526  0.045331  0.039166
err_offset_norm    1000.0  0.006777  0.006195  0.000019  0.002479  0.005306  0.009413  0.060429  2.449260 11.155912  0.005306  0.004412
err_sensitivity_per1000.0  0.010426  0.009839  0.000014  0.003726  0.008071  0.013846  0.069958  2.233295  7.279996  0.008071  0.006916
```
<img width="2312" height="1736" alt="error_boxplots_20260204_200917" src="https://github.com/user-attachments/assets/4d7ca9b1-695a-4d55-9913-e9dfdbacdbda" />
<img width="1683" height="1401" alt="sensitivity_true_vs_est_20260204_200917" src="https://github.com/user-attachments/assets/0b911fd6-1269-4739-b89a-dd3076f5b54e" />
<img width="2017" height="1093" alt="worst_20260204_200917_model_fit" src="https://github.com/user-attachments/assets/aaa0cee4-f7b1-4103-9659-bbfb2385487c" /><img width="2779" height="3182" alt="worst_20260204_200917_params_states" src="https://github.com/user-attachments/assets/19fea595-75c7-4074-ac01-a7f2df9d0cb3" />
<img width="2011" height="1093" alt="best_20260204_200917_model_fit" src="https://github.com/user-attachments/assets/7f478145-470b-42a5-9aed-aec9df1f6e10" /><img width="2779" height="3182" alt="best_20260204_200917_params_states" src="https://github.com/user-attachments/assets/5d5ba963-3d9b-4c84-bab5-b155d84acd87" />

<img width="1993" height="1402" alt="sampled_over_electrode_distribution_20260204_200917" src="https://github.com/user-attachments/assets/7d86b736-1f0b-4f65-8510-5a25fbbbf90c" />
<img width="1993" height="1401" alt="sampled_Q_distribution_20260204_200917" src="https://github.com/user-attachments/assets/fb797190-9574-440b-956a-8e1b96a14056" />
<img width="1993" height="1401" alt="sampled_offset_distribution_20260204_200917" src="https://github.com/user-attachments/assets/31660d66-707c-488d-a058-c601cd7a48d5" />
<img width="1993" height="1402" alt="sampled_over_stiffness_distribution_20260204_200917" src="https://github.com/user-attachments/assets/45c98a8e-2848-4104-bb21-744cbebf468c" />


### Additional Statistics
- **Success rate:** 100.00%
- **Failed runs:** 0

**Error statistics (median ± MAD):**
- err_stiff_norm: 0.0145 ± 0.0146 %
- err_elec_norm: 0.0211 ± 0.0201 %
- err_Q_norm: 0.0453 ± 0.0392 %
- err_offset_norm: 0.0053 ± 0.0044 %
- err_sensitivity_per: 0.0081 ± 0.0069 %

### Timing and Other Per-Run Statistics
- **Mean runtime per run (parallel execution):** 0.6980 seconds
- **Serial per-run time statistics:** mean 16.45 s, std 7.75 s, median 14.94 s

**Hardware:** Intel® Core™ i9-12900F (Alder Lake, Socket LGA 1700, 10 nm, 8 Performance + 8 Efficient cores, 24 threads, 30 MB L3 cache, base TDP 65 W).

---
## Contributions
Detailed function‑level behavior is documented in module docstrings.
Pull requests, bug reports, and feature suggestions are welcome.

---
## AI Usage
This description and the documentation of the codes were created mainly using LLMs. These were also used to speed up the implementation of algorithms from the references and some ideas in iter.

---
## References
```
@article{papakonstantinou2022scaled,
  title={A Scaled Spherical Simplex Filter (S3F) with a decreased n+ 2 sigma points set size and equivalent 2n+ 1 Unscented Kalman Filter (UKF) accuracy},
  author={Papakonstantinou, Konstantinos G and Amir, Mariyam and Warn, Gordon P},
  journal={Mechanical Systems and Signal Processing},
  volume={163},
  pages={107433},
  year={2022},
  publisher={Elsevier}
}
@article{li2022constrained,
  title={Constrained unscented Kalman filter for parameter identification of structural systems},
  author={Li, Dan and Wang, Yang},
  journal={Structural Control and Health Monitoring},
  volume={29},
  number={4},
  pages={e2908},
  year={2022},
  publisher={Wiley Online Library}
}
@inproceedings{gu2009algorithm,
  title={Algorithm of adaptive fading memory UKF in bearings-only target tracking},
  author={Gu, Xiao-dong and Yuan, Zhi-yong and Qiu, Zhi-ming},
  booktitle={2009 2nd International Congress on Image and Signal Processing},
  pages={1--4},
  year={2009},
  organization={IEEE}
}
@article{xie2025state,
  title={State of charge estimation of Li-ion batteries based on strong tracking adaptive square root unscented Kalman filter with generalized maximum correntropy criterion},
  author={Xie, Hao and Lin, Jingli and Huang, Ziyi and Kuang, Rui and Hao, Yuanchao},
  journal={Journal of Energy Storage},
  volume={111},
  pages={115401},
  year={2025},
  publisher={Elsevier}
}
@article{de2017joseph,
  title={Joseph covariance formula adaptation to square-root sigma-point kalman filters},
  author={De Vivo, Francesco and Brandl, Alberto and Battipede, Manuela and Gili, Piero},
  journal={Nonlinear dynamics},
  volume={88},
  number={3},
  pages={1969--1986},
  year={2017},
  publisher={Springer}
}
@inproceedings{van2001square,
  title={The square-root unscented Kalman filter for state and parameter-estimation},
  author={Van Der Merwe, Rudolph and Wan, Eric A},
  booktitle={2001 IEEE international conference on acoustics, speech, and signal processing. Proceedings (Cat. No. 01CH37221)},
  volume={6},
  pages={3461--3464},
  year={2001},
  organization={IEEE}
}
@inproceedings{wan2000unscented,
  title={The unscented Kalman filter for nonlinear estimation},
  author={Wan, Eric A and Van Der Merwe, Rudolph},
  booktitle={Proceedings of the IEEE 2000 adaptive systems for signal processing, communications, and control symposium (Cat. No. 00EX373)},
  pages={153--158},
  year={2000},
  organization={Ieee}
```

