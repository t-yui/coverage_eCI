#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create the LaTeX tables of the paper from the results.
Cells whose results are not available are filled with \\TBDcell.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "EXACT_DIR": ROOT / "results_exact",
    "SIMULATION_DIR": ROOT / "results_simulation",
    "REANALYSIS_DIR": ROOT / "results_reanalysis",
    "OUTPUT_DIR": ROOT / "results_for_paper" / "tables",
}

TBD = r"\TBDcell"
CONFIG_LABELS = {
    "symmetric, delta=0.0": r"Symmetric, $\delta=0$",
    "symmetric, delta=0.5": r"Symmetric, $\delta=0.5$",
    "symmetric, delta=1.0": r"Symmetric, $\delta=1$",
    "symmetric, delta=2.0": r"Symmetric, $\delta=2$",
    "one distant GCM, L=50": "One distant GCM",
}
CONFIG_L = {"symmetric, delta=0.0": 3, "symmetric, delta=0.5": 3, "symmetric, delta=1.0": 3, "symmetric, delta=2.0": 3, "one distant GCM, L=50": 50}
CONFIG_MU = {"symmetric, delta=0.0": 0, "symmetric, delta=0.5": 0, "symmetric, delta=1.0": 0, "symmetric, delta=2.0": 0, "one distant GCM, L=50": 2}
DESIGN_LABELS = {"spread": "Spread", "hot_model": "Hot model", "large_ensemble": "Large ensemble"}
FIXED_DESIGNS = {"spread": [3, 5, 10, 20, 50], "hot_model": [5, 10, 20, 50], "large_ensemble": [5, 10, 20, 50]}
RANDOM_L = [3, 5, 10, 20]
DAILY_DEATHS = [20, 150]
ALPHAS = [0.05, 0.10, 0.34]
AGE_LABELS = {"00_74": r"$<75$ years", "75plus": r"$\geq 75$ years", "total": "Total"}
PERIOD_LABELS = {"gwl": r"$2\,^{\circ}\mathrm{C}$ warming level", "endcentury": "2079--2099"}


def f3(x):
    return TBD if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return TBD if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def write(name, lines):
    out = Path(CONFIG["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text("\n".join(lines) + "\n")
    print(f"wrote {out / name}")


def read_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


# exact computations


def table_exact_configurations():
    df = read_csv(Path(CONFIG["EXACT_DIR"]) / "configurations.csv")
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & $L$ & $\mu_L(\theta_0)$ & Coverage of $I_{\mu}$ & Coverage of $I_{\mathrm{eCI}}$ & Width ratio \\",
        r"\midrule",
    ]
    for key, label in CONFIG_LABELS.items():
        if df is None:
            cov_mu = cov_eci = ratio = None
        else:
            r = df[df["configuration"] == key].iloc[0]
            cov_mu, cov_eci, ratio = r["coverage_mu"], r["coverage_eci"], r["width_ratio"]
        lines.append(f"{label} & {CONFIG_L[key]} & {CONFIG_MU[key]} & {f3(cov_mu)} & {f3(cov_eci)} & {f3(ratio)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("exact_configurations.tex", lines)


def table_heterogeneous_infimum():
    df = read_csv(Path(CONFIG["EXACT_DIR"]) / "heterogeneous_infimum.csv")
    Ls = [2, 3, 5, 10, 20, 40]
    head = " & ".join(f"${L}$" for L in Ls)
    if df is None:
        num = [TBD] * len(Ls)
        ref = [f3(max(0.0, 1 - L * 0.025)) for L in Ls]
    else:
        num = [f3(float(df.loc[df["L"] == L, "numerical_minimum"].iloc[0])) if (df["L"] == L).any() else TBD for L in Ls]
        ref = [f3(max(0.0, 1 - L * 0.025)) for L in Ls]
    lines = [
        r"\begin{tabular}{l" + "c" * len(Ls) + "}",
        r"\toprule",
        r"$L$ & " + head + r" \\",
        r"\midrule",
        r"Numerical minimum of the coverage & " + " & ".join(num) + r" \\",
        r"$\max(0, 1-L\alpha/2)$ & " + " & ".join(ref) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("exact_heterogeneous_infimum.tex", lines)


# projection-type simulation


def summarize_projection():
    path = Path(CONFIG["SIMULATION_DIR"]) / "projection_raw.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["width"] = df["upper"] - df["lower"]
    df["miss_lower"] = (df["truth"] < df["lower"]).astype(float)
    df["miss_upper"] = (df["truth"] > df["upper"]).astype(float)
    keys = ["ensemble", "daily_deaths", "design", "L", "alpha", "method", "target"]
    s = (
        df.groupby(keys)
        .agg(
            n_rep=("cover", "size"),
            coverage=("cover", "mean"),
            miss_lower=("miss_lower", "mean"),
            miss_upper=("miss_upper", "mean"),
            median_width=("width", "median"),
        )
        .reset_index()
    )
    s.to_csv(Path(CONFIG["SIMULATION_DIR"]) / "projection_summary.csv", index=False)
    return s


def get(s, **kw):
    if s is None:
        return None
    sel = np.ones(len(s), dtype=bool)
    for k, v in kw.items():
        sel &= np.isclose(s[k], v) if isinstance(v, float) else (s[k] == v)
    sub = s[sel]
    return None if sub.empty else sub.iloc[0]


def val(row, col):
    return None if row is None else float(row[col])


def table_projection_fixed(s, D):
    lines = [
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{$1-\alpha=0.95$} & \multicolumn{2}{c}{$1-\alpha=0.90$} & \multicolumn{2}{c}{$1-\alpha=0.66$} & \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"Design & $L$ & $I_{\mathrm{eCI}}$ & $I_{\mu}$ & $I_{\mathrm{eCI}}$ & $I_{\mu}$ & $I_{\mathrm{eCI}}$ & $I_{\mu}$ & Width ratio \\",
        r"\midrule",
    ]
    first = True
    for design, Ls in FIXED_DESIGNS.items():
        if not first:
            lines.append(r"\addlinespace")
        first = False
        for i, L in enumerate(Ls):
            cells = []
            for a in ALPHAS:
                for m in ["eci", "mu"]:
                    r = get(s, ensemble="fixed", daily_deaths=D, design=design, L=L, alpha=a, method=m, target="mu_L")
                    cells.append(f3(val(r, "coverage")))
            re = get(s, ensemble="fixed", daily_deaths=D, design=design, L=L, alpha=0.05, method="eci", target="mu_L")
            rm = get(s, ensemble="fixed", daily_deaths=D, design=design, L=L, alpha=0.05, method="mu", target="mu_L")
            ratio = None if re is None or rm is None else float(re["median_width"] / rm["median_width"])
            lab = DESIGN_LABELS[design] if i == 0 else ""
            lines.append(f"{lab} & {L} & " + " & ".join(cells) + f" & {f2(ratio)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(f"simulation_fixed_D{D}.tex", lines)


def table_projection_hs(s):
    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$D=20$} & \multicolumn{3}{c}{$D=150$} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"Design & $L$ & $0.95$ & $0.90$ & $0.66$ & $0.95$ & $0.90$ & $0.66$ \\",
        r"\midrule",
    ]
    first = True
    for design, Ls in FIXED_DESIGNS.items():
        if not first:
            lines.append(r"\addlinespace")
        first = False
        for i, L in enumerate(Ls):
            cells = []
            for D in DAILY_DEATHS:
                for a in ALPHAS:
                    r = get(s, ensemble="fixed", daily_deaths=D, design=design, L=L, alpha=a, method="eci", target="H_s")
                    cells.append(f3(val(r, "coverage")))
            lab = DESIGN_LABELS[design] if i == 0 else ""
            lines.append(f"{lab} & {L} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("simulation_fixed_Hs.tex", lines)


def table_projection_random(s, alpha, name):
    lines = [
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Coverage for $\mu_{\infty}(\theta_0)$} & \multicolumn{2}{c}{Coverage for $H_{\mathrm{new}}(\theta_0)$} & \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-8}",
        r"$D$ & $L$ & $I_{\mathrm{eCI}}$ & $I_{\mu}$ & $I_{\mathrm{RE},t}$ & $I_{\mathrm{RE,sim}}$ & $I_{\mathrm{eCI}}$ & $I_{\mathrm{new}}$ & Width ratio \\",
        r"\midrule",
    ]
    for j, D in enumerate(DAILY_DEATHS):
        if j > 0:
            lines.append(r"\addlinespace")
        for i, L in enumerate(RANDOM_L):
            cells = []
            for m in ["eci", "mu", "re_t", "re_sim"]:
                cells.append(f3(val(get(s, ensemble="random", daily_deaths=D, L=L, alpha=alpha, method=m, target="mu_inf"), "coverage")))
            for m in ["eci", "pi_new"]:
                cells.append(f3(val(get(s, ensemble="random", daily_deaths=D, L=L, alpha=alpha, method=m, target="H_new"), "coverage")))
            rt = get(s, ensemble="random", daily_deaths=D, L=L, alpha=alpha, method="re_t", target="mu_inf")
            re = get(s, ensemble="random", daily_deaths=D, L=L, alpha=alpha, method="eci", target="mu_inf")
            ratio = None if rt is None or re is None else float(rt["median_width"] / re["median_width"])
            lab = str(D) if i == 0 else ""
            lines.append(f"{lab} & {L} & " + " & ".join(cells) + f" & {f2(ratio)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


# reanalysis


def table_reanalysis():
    df = read_csv(Path(CONFIG["REANALYSIS_DIR"]) / "reanalysis_intervals.csv")

    def iv(r, m):
        if r is None:
            return TBD
        return f"$({r[m + '_low']:.0f}, {r[m + '_high']:.0f})$"

    lines = [
        r"\begin{tabular}{llcrccc}",
        r"\toprule",
        r"Period & Age group & $1-\alpha$ & $\widehat{\mu}$ & $I_{\mathrm{eCI}}$ & $I_{\mu}$ & $I_{\mathrm{RE},t}$ \\",
        r"\midrule",
    ]
    for j, period in enumerate(["gwl", "endcentury"]):
        if j > 0:
            lines.append(r"\addlinespace")
        for i, age in enumerate(["00_74", "75plus", "total"]):
            for k, a in enumerate([0.05, 0.34]):
                r = None
                if df is not None:
                    sub = df[(df["aggregation"] == period) & (df["age_group"] == age) & np.isclose(df["alpha"], a)]
                    r = None if sub.empty else sub.iloc[0]
                per = PERIOD_LABELS[period] if (i == 0 and k == 0) else ""
                agl = AGE_LABELS[age] if k == 0 else ""
                fit = TBD if r is None else f"{r['fit']:.0f}"
                lines.append(f"{per} & {agl} & {1 - a:.2f} & {fit} & {iv(r, 'eci')} & {iv(r, 'mu')} & {iv(r, 're_t')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("reanalysis_periods.tex", lines)


def main():
    table_exact_configurations()
    table_heterogeneous_infimum()
    s = summarize_projection()
    for D in DAILY_DEATHS:
        table_projection_fixed(s, D)
    table_projection_hs(s)
    table_projection_random(s, 0.05, "simulation_random_a05.tex")
    table_projection_random(s, 0.34, "simulation_random_a34.tex")
    table_reanalysis()


if __name__ == "__main__":
    main()
