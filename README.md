# coverage_eCI

This repository is dedicated to the code availability for the paper: **Coverage of Empirical Confidence Intervals for the Ensemble Mean of General Circulation Models in Projecting Climate Change Impacts on Health**.

## Description

In projections of climate change impacts on health, the impacts projected under multiple general circulation models (GCMs) are commonly summarized by their mean, together with the empirical confidence interval (eCI), i.e., the interval between the empirical quantiles of the impacts pooled across the GCMs and the coefficient draws.
The code computes and evaluates the following intervals from the coefficient draws:

- `interval_eci`: the eCI, quantiles of the impacts pooled across the GCMs and the coefficient draws
- `interval_mu`: quantiles of the GCM-ensemble mean computed within each coefficient draw
- `interval_re_t`: the interval for the mean over a population of GCMs, $\hat{\mu} \pm t_{L-1,1-\alpha/2} (\hat{V} + S^2/L)^{1/2}$
- `interval_re_sim`: quantiles of $\mu^{(j)} + (S/\sqrt{L}) T^{(j)}$ with $T^{(j)} \sim t_{L-1}$
- `interval_pi_new`: the prediction interval for the impact under a new GCM

## Requirements

- Python ≥ 3.9.0
- Required libraries: [requirements.txt](requirements.txt)

## License

This repository is licensed under the [MIT License](LICENSE).

## Usage

### Compute the intervals

```python
import numpy as np
from eci_intervals import interval_eci, interval_mu, interval_re_t

# H_draws: array (J, L) of the impacts H_l(theta^(j)) for the coefficient draws j and the GCMs l
# H_hat: array (L,) of the impacts H_l(theta_hat) at the point estimate
mu_hat = H_hat.mean()
eci = interval_eci(H_draws, alpha=0.05)
i_mu = interval_mu(H_draws, alpha=0.05)
i_re = interval_re_t(H_hat, H_draws, alpha=0.05)
```

### Perform the experiments in the paper

For preparation, execute the following commands.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

To perform all experiments, execute the following command.

```sh
bash run_all_experiments.sh
```

To perform each experiment separately, set the configurations at the top of each script and execute the following commands from the `scripts` directory.
Each simulation script accepts `--quick` for a small run to check the code.

1. Exact coverage under the linear Gaussian model (configurations, a distant GCM, differences among the GCM-specific impacts depending on the coefficient, and the super-population mean)

```sh
python3 exact_coverage.py
```

2. Simulation study mimicking projections of heat-related mortality

```sh
python3 simulation_projection.py
```

3. Downloading the data of the reanalysis[^1]

```sh
python3 download_data.py
```

4. Reanalysis of the heat-related mortality projections in London

```sh
python3 reanalysis_london.py
```

5. Creating tables of the paper

```sh
python3 make_tables.py
```

6. Creating figures of the paper

```sh
python3 create_figures.py
```

The results are saved in `results_exact`, `results_simulation`, and `results_reanalysis`, and the tables and figures of the paper are saved in `results_for_paper/tables` and `results_for_paper/images`.
Cells of the tables whose results are not available are filled with `\TBDcell`.

### Code for version 1 of the preprint

The numerical results in version 1 of the preprint on medRxiv (doi: [10.64898/2026.08.01.26359345](https://doi.org/10.64898/2026.08.01.26359345)) were obtained with the code at the tag [`medrxiv-v1`](https://github.com/t-yui/coverage_eCI/tree/medrxiv-v1) of this repository, which is also available in the branch [`medRxiv_v1`](https://github.com/t-yui/coverage_eCI/tree/medRxiv_v1).

### Computation time

With `N_JOBS = 4`, the projection-type simulation with $2{,}000$ repetitions takes about 5 minutes.
The exact computations take about 40 minutes, most of which is the numerical minimization of the coverage when the differences among the GCM-specific impacts depend on the coefficient.
The code was checked with Python 3.9 and 3.13.
