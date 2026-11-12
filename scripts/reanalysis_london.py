#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reanalysis of the projected heat-related mortality in London under SSP2-4.5
(three GCMs; tutorial of Quijal-Zamorano et al.).

For each age group (<75, >=75, total) and each aggregation (decades, the 21-year
window centered on the 2 degC global warming level of each GCM, and 2079-2099),
we compute the point estimate (mean across GCMs), I_eCI, I_mu, and I_RE,t
from the daily attributable numbers of deaths (point estimate and coefficient draws).
The point estimates and I_eCI are checked against the summaries saved by the tutorial.
"""

import os
import warnings
from pathlib import Path

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import rdata
from eci_intervals import interval_eci, interval_mu, interval_re_t

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "DATA_DIR": ROOT / "data" / "climate_demographic_health_projections",
    "ALPHA_LIST": [0.05, 0.34],
    "PERIOD_YEARS": 21,
    "END_OF_CENTURY": (2079, 2099),
    "WARMING_LEVEL": "2",
    "AGE_ORDER": ["00_74", "75plus", "total"],
    "VALIDATION_TOLERANCE": 1e-6,
    "RESULT_DIR": ROOT / "results_reanalysis",
}


def load_rdata(rel):
    path = Path(CONFIG["DATA_DIR"]) / rel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rdata.conversion.convert(rdata.parser.parse_file(path))


def load_inputs():
    params = load_rdata("indata/processed/study_parameters.RData")["study_param"]
    gcms = [str(g) for g in params["selected_gcms"]]
    ages = [str(a) for a in params["age_groups"]]
    an_raw = load_rdata("outdata/file/03_calibrated_demographic_projections/attributable_number_ssp245.RData")["an"]
    an = {}
    for age in ages:
        an[age] = {}
        for g in gcms:
            da = an_raw[age][g]
            cols = [str(c) for c in da.coords[da.dims[1]].values]
            an[age][g] = pd.DataFrame(np.asarray(da.values, dtype=float), columns=cols)
    an["total"] = {g: sum(an[a][g] for a in ages) for g in gcms}
    years = (
        load_rdata("outdata/file/03_calibrated_demographic_projections/data_proj_mort_spatialcal_tempcal_ssp2.RData")[
            "proj_mort_ldn_daily"
        ]["year"]
        .astype(int)
        .to_numpy()
    )
    gwl = load_rdata("indata/processed/data_global_warming_levels_ssp245.RData")["data_gwl"]
    original = {
        "gwl": load_rdata("outdata/file/04_health_impacts/heat_related_mortality_gwl_ssp245.RData")["an_gwl_summary_age"],
        "endcentury": load_rdata("outdata/file/04_health_impacts/heat_related_mortality_endcentury_ssp245.RData")[
            "an_endcentury_summary_age"
        ],
        "decade": load_rdata("outdata/file/04_health_impacts/heat_related_mortality_decades_ssp245.RData")[
            "an_decade_summary_age"
        ],
    }
    return gcms, an, years, gwl, original


def period_years(gwl, period, gcm):
    if period == "endcentury":
        y1, y2 = CONFIG["END_OF_CENTURY"]
    else:
        row = gwl[(gwl["warming_level"].astype(str) == CONFIG["WARMING_LEVEL"]) & (gwl["gcm"].astype(str) == gcm)]
        if len(row) != 1:
            raise ValueError(f"could not identify the GWL window for {gcm}")
        y1, y2 = int(row["year1"].iloc[0]), int(row["year2"].iloc[0])
    if y2 - y1 + 1 != CONFIG["PERIOD_YEARS"]:
        raise ValueError(f"the period for {gcm} is not {CONFIG['PERIOD_YEARS']} years")
    return y1, y2


def summarize(sums_by_gcm, alpha):
    """sums_by_gcm: list over GCMs of arrays (n_draws + 1,), the first element is 'est'"""
    M = np.column_stack(sums_by_gcm)  # (1 + J, L)
    H_hat, H_draws = M[0], M[1:]
    return {
        "fit": H_hat.mean(),
        "eci": interval_eci(H_draws, alpha),
        "mu": interval_mu(H_draws, alpha),
        "re_t": interval_re_t(H_hat, H_draws, alpha),
    }


def ordered_columns(df):
    sims = sorted([c for c in df.columns if c != "est"], key=lambda c: int(c.replace("sim", "")))
    return ["est"] + sims


def validate(value, ref, label):
    rel = abs(value - ref) / max(1.0, abs(ref))
    if rel > CONFIG["VALIDATION_TOLERANCE"]:
        raise ValueError(f"reconstruction failed for {label}: {value} vs {ref}")


def main():
    out = Path(CONFIG["RESULT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    gcms, an, years, gwl, original = load_inputs()
    n_draws = an["total"][gcms[0]].shape[1] - 1
    print(f"GCMs: {gcms}; coefficient draws: {n_draws}", flush=True)

    rows = []
    # 21-year periods
    for period in ["gwl", "endcentury"]:
        for age in CONFIG["AGE_ORDER"]:
            sums = []
            for g in gcms:
                y1, y2 = period_years(gwl, period, g)
                df = an[age][g]
                sums.append(df.loc[(years >= y1) & (years <= y2), ordered_columns(df)].sum(axis=0).to_numpy())
            for alpha in CONFIG["ALPHA_LIST"]:
                s = summarize(sums, alpha)
                if alpha == 0.05:
                    ref = original[period][age]
                    validate(s["fit"], float(ref["fit"].iloc[0]), f"{period}/{age}/fit")
                    validate(s["eci"][0], float(ref["low"].iloc[0]), f"{period}/{age}/low")
                    validate(s["eci"][1], float(ref["high"].iloc[0]), f"{period}/{age}/high")
                f = CONFIG["PERIOD_YEARS"]
                rows.append(
                    {"aggregation": period, "time": period, "age_group": age, "alpha": alpha, "fit": s["fit"] / f}
                    | {f"{m}_{b}": s[m][i] / f for m in ["eci", "mu", "re_t"] for i, b in enumerate(["low", "high"])}
                )

    # decades
    decades = years // 10 * 10
    for age in CONFIG["AGE_ORDER"]:
        ref = original["decade"][age]
        for dec in np.unique(decades):
            sel = decades == dec
            n_years = np.unique(years[sel]).size
            sums = [an[age][g].loc[sel, ordered_columns(an[age][g])].sum(axis=0).to_numpy() for g in gcms]
            for alpha in CONFIG["ALPHA_LIST"]:
                s = summarize(sums, alpha)
                if alpha == 0.05:
                    r = ref[ref["time"].astype(int) == dec]
                    validate(s["fit"], float(r["fit"].iloc[0]), f"decade {dec}/{age}/fit")
                    validate(s["eci"][0], float(r["low"].iloc[0]), f"decade {dec}/{age}/low")
                    validate(s["eci"][1], float(r["high"].iloc[0]), f"decade {dec}/{age}/high")
                rows.append(
                    {"aggregation": "decade", "time": int(dec), "age_group": age, "alpha": alpha, "fit": s["fit"] / n_years}
                    | {f"{m}_{b}": s[m][i] / n_years for m in ["eci", "mu", "re_t"] for i, b in enumerate(["low", "high"])}
                )
    print("The point estimates and I_eCI agree with the saved summaries of the tutorial.", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "reanalysis_intervals.csv", index=False)
    print(res[res["aggregation"] != "decade"].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
