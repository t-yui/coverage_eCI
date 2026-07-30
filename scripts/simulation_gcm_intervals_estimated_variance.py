#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from tqdm import tqdm
from logzero import logger


"""
# usage
python simulation_gcm_intervals_estimated_variance.py \
--n-repetitions 100000 \
--sample-size 100 \
--n-draws 10000 \
--seed 1 \
--output simulation_results.csv
"""


def _type7_quantile_from_repeated_base(
    sorted_base: np.ndarray,
    *,
    p: float,
    base_multiplicity: int,
    shifted_multiplicity: int,
    shift: float,
) -> np.ndarray:
    n_rows, n_draws = sorted_base.shape
    del n_rows

    total_size = (base_multiplicity + shifted_multiplicity) * n_draws
    h = (total_size - 1) * p  # zero-based type-7 position
    lower_index = int(np.floor(h))
    upper_index = int(np.ceil(h))
    fraction = float(h - lower_index)

    lower_value = sorted_base[:, lower_index // base_multiplicity]
    upper_value = sorted_base[:, upper_index // base_multiplicity]
    return (1.0 - fraction) * lower_value + fraction * upper_value


def run_simulation(
    *,
    n_repetitions: int,
    sample_size: int,
    n_draws: int,
    seed: int,
    batch_size: int,
    t_draw=False,
) -> pd.DataFrame:

    if n_repetitions <= 0:
        raise ValueError("n_repetitions must be positive.")
    if sample_size <= 1:
        raise ValueError("sample_size must be greater than 1.")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    theta_0 = 0.0
    scale = 1.0
    deltas = (0.0, 0.5, 1.0, 2.0)

    setting_configs: list[tuple[str, str, int, float, float]] = [
        ("symmetric", "Symmetric", 3, delta, 0.0) for delta in deltas
    ]
    setting_configs.append(("asymmetric", "Asymmetric", 50, np.nan, 2.0 * scale))

    master_seed_sequence = np.random.SeedSequence(seed)
    setting_seed_sequences = master_seed_sequence.spawn(len(setting_configs))

    z_975 = float(norm.ppf(0.975))
    limiting_coverage_mu = float(2.0 * t.cdf(z_975, df=sample_size - 1) - 1.0)

    rows: list[dict[str, float | int | str]] = []

    for setting_config, setting_seed_sequence in zip(
        setting_configs,
        setting_seed_sequences,
    ):
        kind, setting_label, L, delta, b_bar = setting_config
        rng = np.random.default_rng(setting_seed_sequence)

        covered_mu = 0
        covered_eci = 0
        sum_width_mu = 0.0
        sum_width_eci = 0.0
        sum_width_ratio = 0.0

        for iteration_start in tqdm(
            range(0, n_repetitions, batch_size),
            desc="Setting: {0}".format(setting_config),
        ):
            current_batch = min(batch_size, n_repetitions - iteration_start)

            observations = rng.normal(
                loc=theta_0,
                scale=np.sqrt(sample_size),
                size=(current_batch, sample_size),
            )
            theta_hat = observations.mean(axis=1)
            observation_variance = observations.var(axis=1, ddof=1)
            sigma_hat_squared = observation_variance / sample_size
            sigma_hat = np.sqrt(sigma_hat_squared)

            # Coefficient sampling
            if t_draw:
                _draws = rng.standard_t(
                    df=sample_size - 1,
                    size=(current_batch, n_draws),
                )
            else:
                _draws = rng.normal(size=(current_batch, n_draws))
            coefficient_draws = theta_hat[:, None] + sigma_hat[:, None] * _draws

            # extract quantiles
            q_mu = np.quantile(
                coefficient_draws,
                q=(0.025, 0.975),
                axis=1,
                method="linear",
            )
            width_mu = q_mu[1] - q_mu[0]

            if kind == "symmetric":
                d = delta * scale

                pooled = np.concatenate(
                    (
                        coefficient_draws - d,
                        coefficient_draws,
                        coefficient_draws + d,
                    ),
                    axis=1,
                )
                q_eci = np.quantile(
                    pooled,
                    q=(0.025, 0.975),
                    axis=1,
                    method="linear",
                )
                width_eci = q_eci[1] - q_eci[0]

                target = theta_0
                mu_covered = (q_mu[0] <= target) & (target <= q_mu[1])
                eci_covered = (q_eci[0] <= target) & (target <= q_eci[1])

            elif kind == "asymmetric":
                shift = 100.0 * scale

                sorted_draws = np.sort(coefficient_draws, axis=1)
                eci_low = _type7_quantile_from_repeated_base(
                    sorted_draws,
                    p=0.025,
                    base_multiplicity=49,
                    shifted_multiplicity=1,
                    shift=shift,
                )
                eci_high = _type7_quantile_from_repeated_base(
                    sorted_draws,
                    p=0.975,
                    base_multiplicity=49,
                    shifted_multiplicity=1,
                    shift=shift,
                )
                width_eci = eci_high - eci_low

                mu_low = q_mu[0] + b_bar
                mu_high = q_mu[1] + b_bar
                target = theta_0 + b_bar

                mu_covered = (mu_low <= target) & (target <= mu_high)
                eci_covered = (eci_low <= target) & (target <= eci_high)

            covered_mu += int(np.count_nonzero(mu_covered))
            covered_eci += int(np.count_nonzero(eci_covered))
            sum_width_mu += float(np.sum(width_mu))
            sum_width_eci += float(np.sum(width_eci))
            sum_width_ratio += float(np.sum(width_eci / width_mu))

        coverage_mu = covered_mu / n_repetitions
        coverage_eci = covered_eci / n_repetitions

        rows.append(
            {
                "setting": setting_label,
                "L": L,
                "delta": delta,
                "b_bar": b_bar,
                "coverage_I_mu": coverage_mu,
                "coverage_I_eCI": coverage_eci,
                "mcse_I_mu": np.sqrt(coverage_mu * (1.0 - coverage_mu) / n_repetitions),
                "mcse_I_eCI": np.sqrt(
                    coverage_eci * (1.0 - coverage_eci) / n_repetitions
                ),
                "mean_width_I_mu": sum_width_mu / n_repetitions,
                "mean_width_I_eCI": sum_width_eci / n_repetitions,
                "mean_within_repetition_width_ratio": (sum_width_ratio / n_repetitions),
                "ratio_of_mean_widths": sum_width_eci / sum_width_mu,
                "limiting_coverage_I_mu_J_infinity": limiting_coverage_mu,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repetitions", type=int, default=100_000)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--n-draws", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation_results.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Drawing from normal")
    results_normal = run_simulation(
        n_repetitions=args.n_repetitions,
        sample_size=args.sample_size,
        n_draws=args.n_draws,
        seed=args.seed,
        batch_size=args.batch_size,
        t_draw=False,
    ).assign(draw_distribution="normal")
    logger.info("Drawing from t")
    results_t = run_simulation(
        n_repetitions=args.n_repetitions,
        sample_size=args.sample_size,
        n_draws=args.n_draws,
        seed=args.seed,
        batch_size=args.batch_size,
        t_draw=True,
    ).assign(draw_distribution="t")
    results = pd.concat(
        [results_normal, results_t],
        axis=0,
        ignore_index=True,
    )
    results = results[
        ["draw_distribution"]
        + [col for col in results.columns if col != "draw_distribution"]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    display_columns = [
        "setting",
        "L",
        "delta",
        "b_bar",
        "coverage_I_mu",
        "coverage_I_eCI",
        "mean_within_repetition_width_ratio",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(
            results[display_columns].to_string(
                index=False, float_format=lambda x: f"{x:.6f}"
            )
        )
    print(f"\nSaved full results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
