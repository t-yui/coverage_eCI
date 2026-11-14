#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact (J -> infinity) coverage computations under the linear Gaussian model:
  - configurations of the GCM-specific impacts in the location-shift model (c_1 = ... = c_L)
  - a configuration with one distant GCM, as a function of its distance
  - the super-population mean and the impact under a new GCM under the normal random-effects model
"""

import os
from pathlib import Path

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from eci_intervals import (
    exact_coverage_eci,
    exact_width_ratio_common,
    exact_coverage_re_t,
    exact_coverage_re_sim,
    exact_coverage_pi_new,
    coverage_eci_superpopulation,
    coverage_mu_superpopulation,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "ALPHA": 0.05,
    # configurations in the location-shift model (standardized impacts)
    "CONFIGURATIONS": {
        "symmetric, delta=0.0": [-0.0, 0.0, 0.0],
        "symmetric, delta=0.5": [-0.5, 0.0, 0.5],
        "symmetric, delta=1.0": [-1.0, 0.0, 1.0],
        "symmetric, delta=2.0": [-2.0, 0.0, 2.0],
        "one distant GCM, L=50": [0.0] * 49 + [100.0],
    },
    # one distant GCM: b = (0, ..., 0, beta)
    "DISTANT_GCM": {
        0.05: {"L_LIST": [10, 40, 41, 50, 100], "BETA_MAX": 300.0},
        0.34: {"L_LIST": [3, 5, 6, 10, 20], "BETA_MAX": 60.0},
    },
    "N_BETA": 601,
    # super-population mean
    "SUPERPOP_L_LIST": [3, 5, 10, 20],
    "RHO_GRID": [0.0] + list(np.logspace(-2, 2, 33)),
    "N_MC_SUPERPOP": 20000,
    "RANDOM_STATE": 20261112,
    "RESULT_DIR": ROOT / "results_exact",
}


def table_configurations(alpha):
    rows = []
    for name, b in CONFIG["CONFIGURATIONS"].items():
        b = np.asarray(b, dtype=float)
        e = b - b.mean()
        rows.append(
            {
                "configuration": name,
                "L": b.size,
                "mu_L": b.mean(),
                "coverage_mu": 1 - alpha,
                "coverage_eci": float(exact_coverage_eci(e, alpha)[0]),
                "width_ratio": float(exact_width_ratio_common(e, alpha)[0]),
            }
        )
    return pd.DataFrame(rows)


def curves_distant_gcm():
    rows = []
    for alpha, spec in CONFIG["DISTANT_GCM"].items():
        betas = np.linspace(0.0, spec["BETA_MAX"], CONFIG["N_BETA"])
        for L in spec["L_LIST"]:
            b = np.zeros((betas.size, L))
            b[:, -1] = betas
            e = b - b.mean(axis=1, keepdims=True)
            cov = exact_coverage_eci(e, alpha)
            rows += [
                {"alpha": alpha, "L": L, "beta": be, "coverage_eci": c}
                for be, c in zip(betas, cov)
            ]
    return pd.DataFrame(rows)


def curves_superpopulation(alpha, rng):
    rows = []
    for L in CONFIG["SUPERPOP_L_LIST"]:
        for rho in CONFIG["RHO_GRID"]:
            omega = (rho / L) / (1 + rho / L)
            omega_new = (1 + 1 / L) * rho / (1 + (1 + 1 / L) * rho)
            cov_eci, se_eci = coverage_eci_superpopulation(
                L, rho, alpha, CONFIG["N_MC_SUPERPOP"], rng
            )
            rows.append(
                {
                    "alpha": alpha,
                    "L": L,
                    "rho": rho,
                    "coverage_eci": cov_eci,
                    "mc_se_eci": se_eci,
                    "coverage_mu": coverage_mu_superpopulation(L, rho, alpha),
                    "coverage_re_t": exact_coverage_re_t(L, omega, alpha),
                    "coverage_re_sim": exact_coverage_re_sim(L, omega, alpha),
                    "coverage_new": exact_coverage_pi_new(L, omega_new, alpha),
                }
            )
        print(f"  L={L} done", flush=True)
    return pd.DataFrame(rows)


def main():
    out = Path(CONFIG["RESULT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
    alpha = CONFIG["ALPHA"]

    print("configurations in the location-shift model", flush=True)
    tab = table_configurations(alpha)
    print(tab.to_string(index=False))
    tab.to_csv(out / "configurations.csv", index=False)

    print("one distant GCM", flush=True)
    curves_distant_gcm().to_csv(out / "distant_gcm.csv", index=False)

    print("super-population mean", flush=True)
    curves_superpopulation(alpha, rng).to_csv(out / "superpopulation.csv", index=False)


if __name__ == "__main__":
    main()
