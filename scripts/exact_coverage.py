#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact (J -> infinity) coverage computations under the linear Gaussian model:
  - configurations of the GCM-specific impacts with a common sensitivity
  - a configuration with one distant GCM, as a function of its distance
  - heterogeneous sensitivities
  - the super-population mean under the normal random-effects model
"""

import os
from pathlib import Path

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from eci_intervals import (
    exact_coverage_eci,
    exact_width_ratio_common,
    exact_coverage_re_t,
    exact_coverage_re_sim,
    coverage_eci_superpopulation,
    coverage_mu_superpopulation,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "ALPHA": 0.05,
    # configurations with a common sensitivity (standardized impacts)
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
    # heterogeneous sensitivities: e = (E1, -eta, ..., -eta)
    "HETERO_L_LIST": [2, 3, 5, 10, 20],
    "HETERO_E1": 0.01,
    "HETERO_ETA_MAX": 8.0,
    "N_ETA": 321,
    # numerical infimum under heterogeneous sensitivities
    "INF_L_LIST": [2, 3, 5, 10, 20, 40],
    "INF_N_START": 20,
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
            rows += [{"alpha": alpha, "L": L, "beta": be, "coverage_eci": c} for be, c in zip(betas, cov)]
    return pd.DataFrame(rows)


def curves_heterogeneous(alpha):
    etas = np.linspace(0.0, CONFIG["HETERO_ETA_MAX"], CONFIG["N_ETA"])
    rows = []
    for L in CONFIG["HETERO_L_LIST"]:
        e = np.tile(-etas[:, None], (1, L))
        e[:, 0] = CONFIG["HETERO_E1"]
        cov = exact_coverage_eci(e, alpha)
        rows += [{"alpha": alpha, "L": L, "eta": et, "coverage_eci": c} for et, c in zip(etas, cov)]
    return pd.DataFrame(rows)


def infimum_heterogeneous(alpha, rng):
    """numerical minimum of the coverage over e with min(e) <= 0 <= max(e)"""

    def objective(x):
        if not (x.min() <= 0.0 <= x.max()):
            return 1.0 + min(abs(x.min()), abs(x.max()))
        return float(exact_coverage_eci(x, alpha)[0])

    rows = []
    for L in CONFIG["INF_L_LIST"]:
        best = 1.0
        for s in range(CONFIG["INF_N_START"]):
            if s % 2 == 0:
                x0 = -rng.uniform(2.0, 20.0, L)
                x0[0] = rng.uniform(0.0, 0.5)
            else:
                x0 = rng.normal(0.0, rng.choice([1.0, 3.0, 10.0]), L)
            res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-10})
            best = min(best, res.fun)
        rows.append({"alpha": alpha, "L": L, "numerical_minimum": best, "one_minus_L_alpha_half": max(0.0, 1 - L * alpha / 2)})
        print(f"  L={L}: minimum={best:.4f}", flush=True)
    return pd.DataFrame(rows)


def curves_superpopulation(alpha, rng):
    rows = []
    for L in CONFIG["SUPERPOP_L_LIST"]:
        for rho in CONFIG["RHO_GRID"]:
            omega = (rho / L) / (1 + rho / L)
            cov_eci, se_eci = coverage_eci_superpopulation(L, rho, alpha, CONFIG["N_MC_SUPERPOP"], rng)
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
                }
            )
        print(f"  L={L} done", flush=True)
    return pd.DataFrame(rows)


def main():
    out = Path(CONFIG["RESULT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
    alpha = CONFIG["ALPHA"]

    print("configurations with a common sensitivity", flush=True)
    tab = table_configurations(alpha)
    print(tab.to_string(index=False))
    tab.to_csv(out / "configurations.csv", index=False)

    print("one distant GCM", flush=True)
    curves_distant_gcm().to_csv(out / "distant_gcm.csv", index=False)

    print("heterogeneous sensitivities", flush=True)
    curves_heterogeneous(alpha).to_csv(out / "heterogeneous.csv", index=False)
    infimum_heterogeneous(alpha, rng).to_csv(out / "heterogeneous_infimum.csv", index=False)

    print("super-population mean", flush=True)
    curves_superpopulation(alpha, rng).to_csv(out / "superpopulation.csv", index=False)


if __name__ == "__main__":
    main()
