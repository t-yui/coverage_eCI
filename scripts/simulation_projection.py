#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation study mimicking projections of heat-related mortality under multiple GCMs.

Fixed ensembles (target: mu_L(theta_0), the mean over the considered GCMs):
  spread          : warming at evenly spaced quantiles of N(WARMING_MEAN, WARMING_SD^2)
  hot_model       : L - 1 GCMs at evenly spaced quantiles of N(WARMING_MEAN, CLUSTER_SD^2)
                    and one GCM with the warming HOT_WARMING
  large_ensemble  : L - 1 members with the warming WARMING_MEAN (differing only in the
                    internal variability) and one GCM with the warming HOT_WARMING
Random ensembles (targets: mu_inf(theta_0), the mean over a population of GCMs, and the
impact under a new GCM): GCMs sampled with replacement from a fixed population of
POOL_SIZE GCMs with the warming drawn from N(WARMING_MEAN, WARMING_SD^2).
"""

import os
from pathlib import Path

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import argparse
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed
from eci_intervals import interval_eci, interval_mu, interval_re_t, interval_re_sim, interval_pi_new
from projection_model import (
    climatology_knots,
    true_coefficients,
    simulate_temperature,
    simulate_epi_data,
    fit_quasi_poisson,
    heat_bin_counts,
    attributable_deaths_per_bin,
    impacts,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "N_YEARS_OBS": 10,
    "N_FUTURE_YEARS": 21,
    "DAILY_DEATHS": [20, 150],
    "ALPHA_LIST": [0.05, 0.10, 0.34],
    "WARMING_MEAN": 2.5,
    "WARMING_SD": 0.8,
    "CLUSTER_SD": 0.4,
    "HOT_WARMING": 6.0,
    "SUMMER_AMPLIFICATION": 0.4,
    "FIXED_DESIGNS": {
        "spread": [3, 5, 10, 20, 50],
        "hot_model": [5, 10, 20, 50],
        "large_ensemble": [5, 10, 20, 50],
    },
    "RANDOM_L_LIST": [3, 5, 10, 20],
    "POOL_SIZE": 1000,
    "N_DRAWS": 1000,
    "N_REPEATS": 2000,
    "N_JOBS": 4,
    "RANDOM_STATE": 2026,
    "RESULT_DIR": ROOT / "results_simulation",
}


# GCM ensembles (fixed across the repetitions)


def fixed_warming(design, L):
    m, sd, hot = CONFIG["WARMING_MEAN"], CONFIG["WARMING_SD"], CONFIG["HOT_WARMING"]
    if design == "spread":
        return m + sd * norm.ppf((np.arange(L) + 0.5) / L)
    if design == "hot_model":
        base = m + CONFIG["CLUSTER_SD"] * norm.ppf((np.arange(L - 1) + 0.5) / (L - 1))
        return np.append(base, hot)
    if design == "large_ensemble":
        return np.append(np.full(L - 1, m), hot)
    raise ValueError(design)


def build_ensembles(seed_seq):
    n_days = 365 * CONFIG["N_FUTURE_YEARS"]
    designs = [(d, L) for d, Ls in CONFIG["FIXED_DESIGNS"].items() for L in Ls]
    children = seed_seq.spawn(len(designs) + 1)
    fixed = {}
    for (design, L), ss in zip(designs, children[:-1]):
        warming = fixed_warming(design, L)
        temp = simulate_temperature(L, n_days, np.random.default_rng(ss), warming, CONFIG["SUMMER_AMPLIFICATION"])
        fixed[(design, L)] = {"warming": warming, "counts": heat_bin_counts(temp)}
    rng_pool = np.random.default_rng(children[-1])
    warming = rng_pool.normal(CONFIG["WARMING_MEAN"], CONFIG["WARMING_SD"], CONFIG["POOL_SIZE"])
    temp = simulate_temperature(CONFIG["POOL_SIZE"], n_days, rng_pool, warming, CONFIG["SUMMER_AMPLIFICATION"])
    pool = {"warming": warming, "counts": heat_bin_counts(temp)}
    return fixed, pool


# one repetition


def run_repetition(rep, seed_seq, setup, n_draws):
    rng = np.random.default_rng(seed_seq)
    knots, theta0 = setup["knots"], setup["theta0"]
    F = CONFIG["N_FUTURE_YEARS"]
    rows = []
    for D in CONFIG["DAILY_DEATHS"]:
        y, X, p = simulate_epi_data(CONFIG["N_YEARS_OBS"], D, theta0, knots, rng)
        theta_hat, cov = fit_quasi_poisson(y, X, p)
        chol = np.linalg.cholesky(cov)
        theta_draws = theta_hat[None, :] + rng.standard_normal((n_draws, p)) @ chol.T
        g_hat = attributable_deaths_per_bin(theta_hat, knots, D)
        g_draws = attributable_deaths_per_bin(theta_draws, knots, D)

        # fixed ensembles
        for (design, L), ens in setup["fixed"].items():
            H_true = setup["H_true_fixed"][(design, L, D)]
            mu_L = H_true.mean()
            H_hat = impacts(g_hat, ens["counts"], F)[0]
            H_draws = impacts(g_draws, ens["counts"], F)
            for alpha in CONFIG["ALPHA_LIST"]:
                e_lo, e_hi = interval_eci(H_draws, alpha)
                m_lo, m_hi = interval_mu(H_draws, alpha)
                key = (rep, D, "fixed", design, L, alpha)
                rows.append(key + ("eci", "mu_L", e_lo, e_hi, mu_L, float(e_lo <= mu_L <= e_hi)))
                rows.append(key + ("mu", "mu_L", m_lo, m_hi, mu_L, float(m_lo <= mu_L <= m_hi)))
                frac = np.mean((e_lo <= H_true) & (H_true <= e_hi))
                rows.append(key + ("eci", "H_s", e_lo, e_hi, np.nan, frac))

        # random ensembles
        pool_true = setup["H_true_pool"][D]
        mu_inf = pool_true.mean()
        for L in CONFIG["RANDOM_L_LIST"]:
            idx = rng.integers(0, CONFIG["POOL_SIZE"], size=L + 1)
            counts = setup["pool"]["counts"][idx[:L]]
            h_new = pool_true[idx[L]]
            H_hat = impacts(g_hat, counts, F)[0]
            H_draws = impacts(g_draws, counts, F)
            t_draws = rng.standard_t(L - 1, n_draws)
            for alpha in CONFIG["ALPHA_LIST"]:
                key = (rep, D, "random", "population", L, alpha)
                intervals = {
                    ("eci", "mu_inf"): interval_eci(H_draws, alpha),
                    ("mu", "mu_inf"): interval_mu(H_draws, alpha),
                    ("re_t", "mu_inf"): interval_re_t(H_hat, H_draws, alpha),
                    ("re_sim", "mu_inf"): interval_re_sim(H_hat, H_draws, alpha, t_draws),
                }
                for (method, target), (lo, hi) in intervals.items():
                    rows.append(key + (method, target, lo, hi, mu_inf, float(lo <= mu_inf <= hi)))
                lo, hi = intervals[("eci", "mu_inf")]
                rows.append(key + ("eci", "H_new", lo, hi, h_new, float(lo <= h_new <= hi)))
                lo, hi = interval_pi_new(H_hat, H_draws, alpha)
                rows.append(key + ("pi_new", "H_new", lo, hi, h_new, float(lo <= h_new <= hi)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small run for checking the code")
    args = parser.parse_args()
    n_rep = 8 if args.quick else CONFIG["N_REPEATS"]
    n_draws = 200 if args.quick else CONFIG["N_DRAWS"]

    out = Path(CONFIG["RESULT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    root_seed = np.random.SeedSequence(CONFIG["RANDOM_STATE"])
    seed_setup, seed_ens, seed_rep = root_seed.spawn(3)

    t0 = time.time()
    knots = climatology_knots(np.random.default_rng(seed_setup))
    theta0 = true_coefficients(knots)
    fixed, pool = build_ensembles(seed_ens)
    H_true_fixed = {}
    for (design, L), ens in fixed.items():
        for D in CONFIG["DAILY_DEATHS"]:
            g0 = attributable_deaths_per_bin(theta0, knots, D)
            H_true_fixed[(design, L, D)] = impacts(g0, ens["counts"], CONFIG["N_FUTURE_YEARS"])[0]
    H_true_pool = {
        D: impacts(attributable_deaths_per_bin(theta0, knots, D), pool["counts"], CONFIG["N_FUTURE_YEARS"])[0]
        for D in CONFIG["DAILY_DEATHS"]
    }
    setup = {
        "knots": knots,
        "theta0": theta0,
        "fixed": fixed,
        "pool": pool,
        "H_true_fixed": H_true_fixed,
        "H_true_pool": H_true_pool,
    }
    print(f"setup done in {time.time() - t0:.1f}s; knots = {np.round(knots, 2)}", flush=True)

    # true GCM-specific impacts
    truth_rows = []
    for (design, L, D), H in H_true_fixed.items():
        for l, h in enumerate(H):
            truth_rows.append(("fixed", design, L, D, l + 1, fixed[(design, L)]["warming"][l], h))
    for D, H in H_true_pool.items():
        for l, h in enumerate(H):
            truth_rows.append(("random", "population", CONFIG["POOL_SIZE"], D, l + 1, pool["warming"][l], h))
    suffix = "_quick" if args.quick else ""
    pd.DataFrame(truth_rows, columns=["ensemble", "design", "L", "daily_deaths", "gcm", "warming", "impact"]).to_csv(
        out / f"projection_truth{suffix}.csv", index=False
    )

    seeds = seed_rep.spawn(n_rep)
    res = Parallel(n_jobs=CONFIG["N_JOBS"], verbose=5)(
        delayed(run_repetition)(r, s, setup, n_draws) for r, s in enumerate(seeds)
    )
    df = pd.DataFrame(
        [row for rows in res for row in rows],
        columns=["rep", "daily_deaths", "ensemble", "design", "L", "alpha", "method", "target", "lower", "upper", "truth", "cover"],
    )
    df.to_csv(out / f"projection_raw{suffix}.csv.gz", index=False, compression="gzip")
    print(f"finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
