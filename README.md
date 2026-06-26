# coverage_eCI

This repository is dedicated to the data availability for the paper "Aligning Empirical Uncertainty Intervals with the GCM-Ensemble Mean in Projecting Climate Change Impacts on Health."

# How to conduct simulations

The scripts in this repository requires Python 3.9 or later.

## Package Installation

```bash
pip install requirements.txt
```

## Run simulation code

```bash
python simulation_gcm_intervals_estimated_variance.py \
--n-repetitions 10000 \
--sample-size 100 \
--n-draws 10000 \
--seed 1 \
--output simulation_results.csv
```
