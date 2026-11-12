#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interval estimators for ensemble means of GCM-specific impacts and
exact coverage computations under the linear Gaussian model.

Notation
--------
H_draws : array (J, L)
    impacts H_l(theta^(j)) for coefficient draws j = 1..J and GCMs l = 1..L
H_hat : array (L,)
    impacts H_l(theta_hat) at the point estimate
"""

import numpy as np
from scipy.stats import norm, t as t_dist, chi2
from scipy.special import logsumexp

QUANTILE_METHOD = "linear"  # identical to type 7 in R


# palette for figures (categorical slots in fixed order)

PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


# interval estimators from coefficient draws


def interval_eci(H_draws, alpha):
    """quantiles of the impacts pooled across GCMs and coefficient draws"""
    pooled = np.asarray(H_draws).ravel()
    return np.quantile(pooled, [alpha / 2, 1 - alpha / 2], method=QUANTILE_METHOD)


def interval_mu(H_draws, alpha):
    """quantiles of the ensemble mean computed within each coefficient draw"""
    mu_draws = np.asarray(H_draws).mean(axis=1)
    return np.quantile(mu_draws, [alpha / 2, 1 - alpha / 2], method=QUANTILE_METHOD)


def between_gcm_variance(H_hat):
    """sample variance of the GCM-specific impacts at the point estimate"""
    return np.var(np.asarray(H_hat), ddof=1)


def interval_re_t(H_hat, H_draws, alpha):
    """mu_hat +/- t_{L-1} {V_hat + S^2 / L}^{1/2}"""
    H_hat = np.asarray(H_hat)
    L = H_hat.size
    mu_hat = H_hat.mean()
    v_hat = np.var(np.asarray(H_draws).mean(axis=1), ddof=1)
    half = t_dist.ppf(1 - alpha / 2, L - 1) * np.sqrt(
        v_hat + between_gcm_variance(H_hat) / L
    )
    return np.array([mu_hat - half, mu_hat + half])


def interval_re_sim(H_hat, H_draws, alpha, t_draws):
    """quantiles of mu^(j) + (S / L^{1/2}) U^(j) with U^(j) ~ t_{L-1}"""
    H_hat = np.asarray(H_hat)
    L = H_hat.size
    mu_draws = np.asarray(H_draws).mean(axis=1)
    values = mu_draws + np.sqrt(between_gcm_variance(H_hat) / L) * t_draws
    return np.quantile(values, [alpha / 2, 1 - alpha / 2], method=QUANTILE_METHOD)


def interval_pi_new(H_hat, H_draws, alpha):
    """prediction interval for the impact under a new GCM"""
    H_hat = np.asarray(H_hat)
    L = H_hat.size
    mu_hat = H_hat.mean()
    v_hat = np.var(np.asarray(H_draws).mean(axis=1), ddof=1)
    half = t_dist.ppf(1 - alpha / 2, L - 1) * np.sqrt(
        v_hat + between_gcm_variance(H_hat) * (1 + 1 / L)
    )
    return np.array([mu_hat - half, mu_hat + half])


# mixture quantiles and exact coverage


def _mixture_excess_sign(x, e, p):
    """
    sign of sum_l Phi(x - e_l) - L p, evaluated without cancellation:
    sum_l Phi(a_l) - L p = (|A| - L p) + sum_{l not in A} Phi(a_l) - sum_{l in A} {1 - Phi(a_l)},
    where a_l = x - e_l and A = {l : a_l >= 0}; when |A| - L p = 0, the sign is
    determined by comparing the two sums on the log scale
    """
    a = x[:, None] - e
    in_a = a >= 0
    k = in_a.sum(axis=1) - e.shape[1] * p
    log_p = logsumexp(np.where(in_a, -np.inf, norm.logcdf(a)), axis=1)
    log_n = logsumexp(np.where(in_a, norm.logsf(a), -np.inf), axis=1)
    direct = k + np.exp(log_p) - np.exp(log_n)
    return np.where(np.abs(k) > 1e-9, np.sign(direct), np.sign(log_p - log_n))


def mixture_quantile(p, e, n_iter=200):
    """
    p-quantile of the equally weighted mixture of N(e_l, 1), l = 1..L

    e : array (..., L); the quantile is computed for each row
    """
    e = np.atleast_2d(np.asarray(e, dtype=float))
    lo = e.min(axis=1) + norm.ppf(p) - 1.0
    hi = e.max(axis=1) + norm.ppf(p) + 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        excess = norm.cdf(mid[:, None] - e).mean(axis=1) - p
        sign = np.sign(excess)
        flat = np.abs(excess) < 1e-12
        if np.any(flat):
            sign[flat] = _mixture_excess_sign(mid[flat], e[flat], p)
        below = sign < 0
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
        if np.all(hi - lo <= 1e-12 * np.maximum(1.0, np.abs(mid))):
            break
    return 0.5 * (lo + hi)


def exact_coverage_eci(e, alpha):
    """
    exact coverage of I_eCI for mu_L(theta_0) under the linear Gaussian model
    with J -> infinity, where e_l = {H_l(theta_0) - mu_L(theta_0)} / (c_l sigma)
    """
    q_lo = mixture_quantile(alpha / 2, e)
    q_hi = mixture_quantile(1 - alpha / 2, e)
    return norm.cdf(q_hi) - norm.cdf(q_lo)


def exact_coverage_eci_random_gcm(H0, sd, alpha):
    """
    exact coverage of I_eCI for H_s(theta_0) under the linear Gaussian model
    with J -> infinity, where s is uniform on {1..L}

    H0 : array (L,), the impacts H_l(theta_0)
    sd : array (L,), the standard deviations c_l sigma

    With theta_hat = theta_0 + sigma X, the mixture distribution function at
    H_k(theta_0) is g_k(X) = L^{-1} sum_l Phi(d_kl - X) with
    d_kl = {H_k(theta_0) - H_l(theta_0)} / (c_l sigma), which is decreasing in X,
    so that the coverage given s = k is Phi(x_hi) - Phi(x_lo) with
    g_k(x_hi) = alpha / 2 and g_k(x_lo) = 1 - alpha / 2.
    """
    H0 = np.asarray(H0, dtype=float)
    sd = np.asarray(sd, dtype=float)
    d = (H0[:, None] - H0[None, :]) / sd[None, :]  # (k, l)
    # g_k(X) <= p if and only if X >= -q_p(-d_k), where q_p is the mixture quantile
    x_hi = -mixture_quantile(alpha / 2, -d)
    x_lo = -mixture_quantile(1 - alpha / 2, -d)
    return float(np.mean(norm.cdf(x_hi) - norm.cdf(x_lo)))


def exact_width_ratio_common(e, alpha):
    """width of I_eCI divided by that of I_mu in the location-shift model (c_1 = ... = c_L)"""
    q_lo = mixture_quantile(alpha / 2, e)
    q_hi = mixture_quantile(1 - alpha / 2, e)
    return (q_hi - q_lo) / (2 * norm.ppf(1 - alpha / 2))


def chi2_scaled_grid(df, n_grid):
    """midpoint quantile grid of W = chi^2_df / df"""
    u = (np.arange(n_grid) + 0.5) / n_grid
    return chi2.ppf(u, df) / df


def exact_coverage_re_t(L, omega, alpha, n_grid=20000):
    """
    exact coverage of I_RE,t for mu_inf under the normal random-effects model,
    where omega = (tau^2 / L) / (V + tau^2 / L)
    """
    w = chi2_scaled_grid(L - 1, n_grid)
    tq = t_dist.ppf(1 - alpha / 2, L - 1)
    return np.mean(2 * norm.cdf(tq * np.sqrt(1 - omega + omega * w)) - 1)


def exact_coverage_pi_new(L, omega_new, alpha, n_grid=20000):
    """
    exact coverage of the prediction interval for a new GCM,
    where omega_new = tau^2 (1 + 1/L) / {V + tau^2 (1 + 1/L)}
    """
    w = chi2_scaled_grid(L - 1, n_grid)
    tq = t_dist.ppf(1 - alpha / 2, L - 1)
    return np.mean(2 * norm.cdf(tq * np.sqrt(1 - omega_new + omega_new * w)) - 1)


def exact_coverage_re_sim(L, omega, alpha, n_w=400, n_t=2000, n_iter=80):
    """
    exact coverage of I_RE,sim for mu_inf under the normal random-effects model
    (J -> infinity), computed by numerical integration over W and T
    """
    w = chi2_scaled_grid(L - 1, n_w)
    u = (np.arange(n_t) + 0.5) / n_t
    t_nodes = t_dist.ppf(u, L - 1)
    s = np.sqrt(1 - omega)
    c = np.sqrt(omega * w)
    p = 1 - alpha / 2
    if s == 0:
        h = c * t_dist.ppf(p, L - 1)
    else:
        lo = np.full(n_w, -50.0)
        hi = np.full(n_w, 50.0)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            cdf = norm.cdf((mid[:, None] - c[:, None] * t_nodes[None, :]) / s).mean(
                axis=1
            )
            below = cdf < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        h = 0.5 * (lo + hi)
    return np.mean(2 * norm.cdf(h) - 1)


def coverage_eci_superpopulation(L, rho, alpha, n_mc, rng):
    """
    coverage of I_eCI for mu_inf under the normal random-effects model
    (J -> infinity), integrated over the GCM effects by Monte Carlo;
    rho = tau^2 / V
    """
    delta = rng.normal(0.0, np.sqrt(rho), size=(n_mc, L))
    q_lo = mixture_quantile(alpha / 2, delta)
    q_hi = mixture_quantile(1 - alpha / 2, delta)
    cond = norm.cdf(q_hi) - norm.cdf(q_lo)
    return cond.mean(), cond.std(ddof=1) / np.sqrt(n_mc)


def coverage_mu_superpopulation(L, rho, alpha):
    """coverage of I_mu for mu_inf under the normal random-effects model"""
    z = norm.ppf(1 - alpha / 2)
    return 2 * norm.cdf(z / np.sqrt(1 + rho / L)) - 1
