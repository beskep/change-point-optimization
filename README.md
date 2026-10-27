# Change-Point Optimization for Building Energy Models

Code for the paper "Quantifying Change-Point Estimation Bias in Building Energy Models:
A Monte Carlo Comparison of Optimization Methods".

The five-parameter change-point model (CPM) relates daily or monthly energy use to
outdoor temperature through a heating change-point ($T_h$) and a cooling change-point ($T_c$).
This repository compares methods for estimating the change-points by Monte Carlo simulation.
Synthetic datasets with known reference parameters are generated from measured weather data and
from CPM parameter distributions derived from 441 office buildings.
Each optimization method is evaluated by its change-point error, energy fit and computational cost.

## Optimization methods

Implemented in [`src/cpm/optimizer.py`](src/cpm/optimizer.py).

| Name in code          | Method                                                               |
| --------------------- | -------------------------------------------------------------------- |
| `Reference`           | Reference model fitted at the reference change-points                |
| `IMT`                 | Two-stage grid search of the ASHRAE Inverse Modeling Toolkit         |
| `GridSearch`          | Single-stage grid search                                             |
| `TRR`                 | Trust-Region Reflective least squares from a random start            |
| `MultiStartTRR`       | TRR from multiple random starts                                      |
| `InformedStartTRR`    | TRR started near the mean of the reference change-point distribution |
| `Muggeo`              | Muggeo's method (`piecewise-regression`)                             |
| `SegmentedRegression` | Exhaustive search over splits of the temperature-sorted data         |

## Repository structure

```text
src/cpm/              Library: synthetic datasets, optimizers, shared utilities
data/                 Inputs for generating the Monte Carlo simulation datasets
scripts/
  s01distribution.py  Parameter distributions of the measured CPMs
  s02optimize.py      Monte Carlo simulation
  s03evaluate/        Aggregation, statistics, tables and figures
conf.toml             Experiment settings (optimizers, repetitions, conditions)
env.example.toml      Template for env.toml (workspace location)
matplotlibrc          Figure style
```

## Configuration

Copy `env.example.toml` to `env.toml` and set `dirs.root` to a workspace directory where outputs are stored.
The directory layout under the workspace is defined in [`src/cpm/dirs.py`](src/cpm/dirs.py).
All scripts read `env.toml`, `conf.toml`, `matplotlibrc` and the files in `data/` from the current directory,
so run them from the repository root.

This project uses [uv](https://docs.astral.sh/uv/); run `uv sync` once before running any script.

Each script is a command-line application.
Use `--help` to list its commands and options, e.g. `uv run scripts/s02optimize.py --help`.

## Input data

The measured electricity use data contain sensitive building-level information and are not included.
The simulation does not use them directly.
It requires only the following inputs.

- Daily outdoor temperature
  - `data/daily-outdoor-temperature.parquet` with columns `station`, `year`, `date` and `temperature` [°C].
  - Obtained from the daily mean air temperature of the
    [Korea Meteorological Administration ASOS stations (2022–2025)](https://data.kma.go.kr/cmmn/main.do).
  - Station-years with fewer than 365 daily observations are excluded,
    leaving 299 station-years (weather datasets) from 96 stations.
- CPM parameter distributions
  - Derived from the measured buildings (see the paper) and provided in the `data/` directory.
  - Change-point distribution: `data/change-points-distribution.json`.
    Mean (`mean`), covariance (`cov`), standard deviation (`std`) and correlation matrix (`r`)
    of ($T_h$, $T_c$) [°C].
  - Baseline energy use distribution: `data/baseline-distribution.json`.
    Parameters of $\log(E_b) \sim (1-\epsilon) N(\mu, \sigma^2) + \epsilon N(\mu, \lambda \sigma^2)$,
    where $E_b$ is the daily baseline energy use [kWh/m²].
  - Statistics of the measured CPMs: `data/measured-cpm-statistics.csv`.
    Count, mean, standard deviation and percentiles of the baseline energy use [kWh/m²],
    slopes [kWh/m²K] and change-points [°C] of the 441 measured buildings.
    The 10% and 50% percentiles of the slopes are used as the reference slopes.

## Workflow

### 1. Parameter distributions (requires the measured data)

`scripts/s01distribution.py` derives the parameter files above from the measured CPMs.
It is included for transparency but cannot be run without the non-public data.

| Command                                      | Output                                                           |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `raw-dist`                                   | `measured-cpm-statistics.csv`, `change-points-distribution.json` |
| `fit-baseline-distribution`, `baseline-dist` | Candidate mixture distributions of daily baseline energy use     |
| `baseline-params`                            | `baseline-distribution.json`                                     |
| `power-analysis`                             | Statistical power of the slope t-tests                           |

### 2. Monte Carlo simulation

```sh
uv run scripts/s02optimize.py optimize
```

Runs every optimizer in `conf.toml` on all synthetic datasets
(daily and monthly resolution × two slope conditions × 20 repetitions per weather dataset)
and writes one result file per case to `<workspace>/03.optimize`.
Existing result files are skipped, so an interrupted run can be resumed.
Random seeds are fixed by weather dataset and repetition, so all methods are evaluated on identical datasets.

### 3. Evaluation

```sh
uv run scripts/s03evaluate/s01aggregate.py preprocess
uv run scripts/s03evaluate/s02energy.py
uv run scripts/s03evaluate/s03change_point.py
uv run scripts/s03evaluate/s04misc.py
```

`s01aggregate.py preprocess` must run first.
`s03change_point.py` and `s04misc.py` without a command run their subcommands in order.
Outputs are written to `<workspace>/04.evaluate`.

| Script / command                                               | Result in the paper                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| `s02energy.py`                                                 | Objective minimization and computational cost tables         |
| `s03change_point.py`                                           | Change-point and heating/cooling energy use error tables     |
| `s04misc.py sse-cp`                                            | Change-point bias versus relative SSE figure                 |
| `s04misc.py bias-plot`                                         | Change-point bias figure                                     |
| `s04misc.py jensen`                                            | Effect of monthly averaging on annual heating/cooling energy |
| `s04misc.py sr-bracket`, `muggeo-failure`, `muggeo-base-slope` | Diagnostics of SR and Muggeo reported in the text            |

## License

- Code and `data/` files other than the temperature data: MIT (see [`LICENSE`](LICENSE))
- `data/daily-outdoor-temperature.parquet`: Korea Meteorological Administration data,
  Korea Open Government License (KOGL) Type 1.

<!-- TODO: add citation (journal, DOI) once published -->
