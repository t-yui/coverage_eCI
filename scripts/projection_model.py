#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data generation and estimation for the simulation study that mimics
projections of heat-related mortality under multiple GCMs.

- daily temperature: seasonal cycle + stationary AR(1) deviations
- daily deaths: Poisson-gamma counts with log-mean
      log D + seasonal term + f(T),
  where f is a natural cubic spline of temperature with fixed knots,
  centered at the reference temperature
- estimation: quasi-Poisson regression with the same spline basis,
  harmonic terms, and day-of-week indicators
- projected impact under GCM l: annual heat-attributable deaths
      H_l(theta) = (1 / F) sum_k n_lk D {1 - exp(-f_theta(t_k))},
  where t_k are 0.1 degC bins above the reference temperature and n_lk is
  the number of days in bin k in the future period of GCM l
"""

import numpy as np
import statsmodels.api as sm
from scipy.signal import lfilter

CLIMATE = {
    "MEAN": 11.5,  # annual mean temperature (degC)
    "AMPLITUDE": 6.5,  # amplitude of the seasonal cycle (degC)
    "PHASE_DAY": 110,  # day of the year at which the seasonal sine term is zero and increasing
    "AR_COEF": 0.75,  # lag-1 autocorrelation of the daily deviations
    "AR_SD": 2.6,  # marginal standard deviation of the daily deviations (degC)
}

EPI = {
    "REFERENCE_TEMPERATURE": 18.0,  # centering temperature of the exposure-response function
    "COLD_COEF": 0.0009,  # f(T) = COLD_COEF (T_ref - T)^2 below the reference temperature (target shape)
    "HEAT_COEF": 0.0045,  # f(T) = HEAT_COEF (T - T_ref)^2 above the reference temperature (target shape)
    "SEASONAL_LOG_RR": 0.08,  # amplitude of the seasonal term of the log-mean of deaths
    "GAMMA_SHAPE": 50.0,  # shape of the gamma frailty (overdispersion)
    "KNOT_PERCENTILES": [10.0, 75.0, 90.0],  # interior knots (percentiles of the climatology)
    "BOUNDARY_PERCENTILES": [0.1, 99.9],  # boundary knots (percentiles of the climatology)
    "N_HARMONICS": 3,  # harmonic terms in the fitted model
    "TEMP_CENTER": 15.0,  # temperature is scaled as (T - TEMP_CENTER) / TEMP_SCALE in the basis
    "TEMP_SCALE": 10.0,
}

HEAT_GRID = np.round(np.arange(EPI["REFERENCE_TEMPERATURE"] + 0.05, 55.0, 0.1), 2)


# temperature


def simulate_temperature(n_series, n_days, rng, warming=None, summer_amplification=0.0):
    """
    daily temperature series, array (n_series, n_days)

    warming : array (n_series,) of the mean warming (degC) of each series
    summer_amplification : additional relative warming in summer
    """
    doy = np.arange(n_days) % 365
    season = np.sin(2 * np.pi * (doy - CLIMATE["PHASE_DAY"]) / 365)
    phi = CLIMATE["AR_COEF"]
    innov = rng.normal(0.0, CLIMATE["AR_SD"] * np.sqrt(1 - phi**2), size=(n_series, n_days))
    innov[:, 0] = rng.normal(0.0, CLIMATE["AR_SD"], size=n_series)  # stationary start
    dev = lfilter([1.0], [1.0, -phi], innov, axis=1)
    temp = CLIMATE["MEAN"] + CLIMATE["AMPLITUDE"] * season[None, :] + dev
    if warming is not None:
        warming = np.asarray(warming, dtype=float)[:, None]
        temp = temp + warming * (1 + summer_amplification * np.clip(season, 0, None))[None, :]
    return temp


def climatology_knots(rng, n_years=500):
    """knots of the spline basis from the percentiles of a long baseline series"""
    temp = simulate_temperature(1, 365 * n_years, rng).ravel()
    inner = np.percentile(temp, EPI["KNOT_PERCENTILES"])
    bound = np.percentile(temp, EPI["BOUNDARY_PERCENTILES"])
    return np.sort(np.concatenate([bound, inner]))


# exposure-response function


def ns_basis(temp, knots):
    """
    natural cubic spline basis without the intercept (linear beyond the boundary knots),
    evaluated at the scaled temperature
    """
    x = (np.asarray(temp, dtype=float) - EPI["TEMP_CENTER"]) / EPI["TEMP_SCALE"]
    kn = (np.asarray(knots, dtype=float) - EPI["TEMP_CENTER"]) / EPI["TEMP_SCALE"]
    K = kn.size

    def d(k):
        return (np.clip(x - kn[k], 0, None) ** 3 - np.clip(x - kn[K - 1], 0, None) ** 3) / (kn[K - 1] - kn[k])

    cols = [x] + [d(k) - d(K - 2) for k in range(K - 2)]
    return np.stack(cols, axis=-1)


def centered_basis(temp, knots):
    return ns_basis(temp, knots) - ns_basis(np.array([EPI["REFERENCE_TEMPERATURE"]]), knots)


def target_shape(temp):
    t_ref = EPI["REFERENCE_TEMPERATURE"]
    return np.where(temp < t_ref, EPI["COLD_COEF"] * (t_ref - temp) ** 2, EPI["HEAT_COEF"] * (temp - t_ref) ** 2)


def true_coefficients(knots):
    """least-squares projection of the target shape onto the spline basis within the boundary knots"""
    grid = np.linspace(knots[0], knots[-1], 1000)
    B = centered_basis(grid, knots)
    return np.linalg.lstsq(B, target_shape(grid), rcond=None)[0]


# epidemiological data and estimation


def simulate_epi_data(n_years, daily_deaths, theta0, knots, rng):
    n = 365 * n_years
    temp = simulate_temperature(1, n, rng).ravel()
    t = np.arange(n)
    doy = t % 365
    B = centered_basis(temp, knots)
    log_mu = np.log(daily_deaths) + EPI["SEASONAL_LOG_RR"] * np.cos(2 * np.pi * doy / 365) + B @ theta0
    frailty = rng.gamma(EPI["GAMMA_SHAPE"], 1.0 / EPI["GAMMA_SHAPE"], size=n)
    y = rng.poisson(np.exp(log_mu) * frailty)
    harmonics = np.column_stack(
        [f(2 * np.pi * k * doy / 365) for k in range(1, EPI["N_HARMONICS"] + 1) for f in (np.sin, np.cos)]
    )
    dow = np.eye(7)[t % 7][:, 1:]
    X = np.column_stack([np.ones(n), B, harmonics, dow])
    return y, X, B.shape[1]


def fit_quasi_poisson(y, X, p):
    """estimates and the covariance matrix (scaled by the Pearson dispersion) of the spline coefficients"""
    res = sm.GLM(y, X, family=sm.families.Poisson()).fit(scale="X2")
    cov = np.asarray(res.cov_params())
    return np.asarray(res.params)[1 : 1 + p], cov[1 : 1 + p, 1 : 1 + p]


# projected impacts


def heat_bin_counts(temp):
    """number of days in each 0.1 degC bin above the reference temperature, array (n_series, K)"""
    temp = np.atleast_2d(temp)
    width = 0.1
    idx = np.floor((temp - EPI["REFERENCE_TEMPERATURE"]) / width).astype(int)
    if np.any(idx >= HEAT_GRID.size):
        raise ValueError("temperature beyond the upper end of the heat grid")
    counts = np.zeros((temp.shape[0], HEAT_GRID.size))
    for i in range(temp.shape[0]):
        k = idx[i][idx[i] >= 0]
        counts[i] = np.bincount(k, minlength=HEAT_GRID.size)
    return counts


def attributable_deaths_per_bin(theta, knots, daily_deaths):
    """D {1 - exp(-f_theta(t_k))} for each coefficient vector, array (J, K)"""
    theta = np.atleast_2d(theta)
    BG = centered_basis(HEAT_GRID, knots)
    return daily_deaths * (1 - np.exp(-(theta @ BG.T)))


def impacts(per_bin, counts, n_future_years):
    """H_l(theta) for each coefficient vector and GCM, array (J, L)"""
    return per_bin @ counts.T / n_future_years
