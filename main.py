"""
Run Monte Carlo simulations and record execution time.

This module calls `main_monte_carlo()` from the `monte_carlo` package, measures
wall-clock execution time, computes average time per Monte Carlo run, prints a
concise summary to stdout, and saves a timestamped execution-time report to
`OUTPUT_DIR`.

Usage
-----
Run as a script:
    python run_monte_carlo.py

Or import and call programmatically:
    from run_monte_carlo import main
    df, stats = main()

Dependencies
------------
- monte_carlo.main_monte_carlo: callable returning (pandas.DataFrame, dict)
- monte_carlo.N_MC_RUNS: integer number of Monte Carlo runs
- monte_carlo.OUTPUT_DIR: directory path for saving timing reports

Outputs
-------
- Console output with total and per-run timings and CPU core count.
- A file named `execution_time_<YYYYMMDD_HHMMSS>.txt` saved to OUTPUT_DIR
  containing detailed timing metadata.

Notes
-----
- The script measures wall-clock time using time.time(). For CPU time,
  consider using time.process_time() or resource.getrusage.
"""

import time
import os
from datetime import datetime
from monte_carlo import main_monte_carlo, N_MC_RUNS, OUTPUT_DIR

def main():
    """
    Execute the Monte Carlo workflow and record execution timing.

    Calls `main_monte_carlo()` to run the Monte Carlo simulation(s), times the
    overall execution, computes an average per-run duration, prints a short
    summary, and writes a timestamped execution-time report into `OUTPUT_DIR`.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame returned by `main_monte_carlo()` containing Monte Carlo results.
    stats : dict
        Summary statistics or metadata returned by `main_monte_carlo()`.

    Side effects
    ------------
    - Prints an execution time summary to stdout.
    - Writes a file `execution_time_<YYYYMMDD_HHMMSS>.txt` to `OUTPUT_DIR` with:
        - total execution time (seconds)
        - average time per run (seconds)
        - number of Monte Carlo runs (`N_MC_RUNS`)
        - number of CPU cores available
        - timestamp

    Raises
    ------
    ZeroDivisionError
        If `N_MC_RUNS` is zero (division by zero when computing average per run).
    OSError / FileNotFoundError
        If the `OUTPUT_DIR` is not writable or cannot be created.

    Notes
    -----
    - If `OUTPUT_DIR` may not exist, consider creating it before writing, e.g.:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    - For more robust logging in production, replace `print()` with the `logging`
      module and set appropriate log levels/handlers.

    Example
    -------
    >>> df, stats = main()
    >>> print(stats["some_key"])
    """

    start = time.time()
    df, stats = main_monte_carlo()
    end = time.time()
    total_time = end - start
    avg_time_per_run = total_time / N_MC_RUNS
    print(f"\n=== EXECUTION TIME SUMMARY ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per run: {avg_time_per_run:.2f} seconds")
    print(f"Number of CPU cores used: {os.cpu_count()}")
    # Save execution time info
    time_path = os.path.join(OUTPUT_DIR, f"execution_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(time_path, 'w') as f:
        f.write("=== EXECUTION TIME INFORMATION ===\n\n")
        f.write(f"Total parallel execution time: {total_time:.4f} seconds\n")
        f.write(f"Average time per run: {avg_time_per_run:.4f} seconds\n")
        f.write(f"Number of Monte Carlo runs: {N_MC_RUNS}\n")
        f.write(f"Number of CPU cores available: {os.cpu_count()}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Execution time info saved to: {time_path}")
    return df, stats
if __name__ == "__main__":
    df, stats = main()