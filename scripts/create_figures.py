#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create the figures of the paper from the results."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eci_intervals import PALETTE, LINESTYLES, INK, INK_SECONDARY, GRID

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "EXACT_DIR": ROOT / "results_exact",
    "REANALYSIS_DIR": ROOT / "results_reanalysis",
    "OUTPUT_DIR": ROOT / "results_for_paper" / "images",
    # number of GCMs in the panel for the population mean in the figure of the exact coverage
    "POPULATION_PANEL_L": 3,
}

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="bold", color=INK)


def figure_exact():
    d = pd.read_csv(Path(CONFIG["EXACT_DIR"]) / "distant_gcm.csv")
    sp = pd.read_csv(Path(CONFIG["EXACT_DIR"]) / "superpopulation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True)
    axes = axes.ravel()
    for ax, alpha, title in zip(
        axes[:2],
        [0.05, 0.34],
        [
            "a) One distant GCM\n$1-\\alpha=0.95$",
            "b) One distant GCM\n$1-\\alpha=0.66$",
        ],
    ):
        sub = d[np.isclose(d["alpha"], alpha)]
        for k, L in enumerate(sorted(sub["L"].unique())):
            s = sub[sub["L"] == L]
            ax.plot(
                s["beta"],
                s["coverage_eci"],
                color=PALETTE[k],
                ls=LINESTYLES[k],
                lw=1.4,
                label=f"$L={L}$",
            )
        ax.axhline(1 - alpha, color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"Coverage of $I_{\mathrm{eCI}}$ for $\mu_L(\theta_0)$")
        ax.set_ylim(-0.02, 1.02)
        style(ax)
        panel_title(ax, title)
        ax.legend(loc="lower left", fontsize=7.5, handlelength=2.6)
    ax = axes[2]
    L = CONFIG["POPULATION_PANEL_L"]
    sub = sp[(sp["L"] == L) & (sp["rho"] > 0)]
    methods = [
        ("coverage_eci", r"$I_{\mathrm{eCI}}$"),
        ("coverage_mu", r"$I_{\mu}$"),
        ("coverage_re_t", r"$I_{\mathrm{RE},t}$"),
        ("coverage_re_sim", r"$I_{\mathrm{RE,sim}}$"),
    ]
    for k, (col, lab) in enumerate(methods):
        ax.plot(
            sub["rho"], sub[col], color=PALETTE[k], ls=LINESTYLES[k], lw=1.4, label=lab
        )
    ax.axhline(0.95, color=INK_SECONDARY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau^2 / V$")
    ax.set_ylabel(r"Coverage for $\mu_{\infty}(\theta_0)$")
    ax.set_ylim(0.4, 1.01)
    style(ax)
    panel_title(ax, f"c) Random-effects model\n$L={L}$, $1-\\alpha=0.95$")
    ax.legend(loc="lower left", fontsize=7.5, handlelength=2.6)
    out = Path(CONFIG["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exact_coverage.pdf")
    plt.close(fig)


def figure_superpopulation():
    s = pd.read_csv(Path(CONFIG["EXACT_DIR"]) / "superpopulation.csv")
    Ls = sorted(s["L"].unique())
    fig, axes = plt.subplots(
        1,
        len(Ls),
        figsize=(7.2, 2.4),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    methods = [
        ("coverage_eci", r"$I_{\mathrm{eCI}}$"),
        ("coverage_mu", r"$I_{\mu}$"),
        ("coverage_re_t", r"$I_{\mathrm{RE},t}$"),
        ("coverage_re_sim", r"$I_{\mathrm{RE,sim}}$"),
    ]
    for j, (ax, L) in enumerate(zip(axes, Ls)):
        sub = s[(s["L"] == L) & (s["rho"] > 0)]
        for k, (col, lab) in enumerate(methods):
            ax.plot(
                sub["rho"],
                sub[col],
                color=PALETTE[k],
                ls=LINESTYLES[k],
                lw=1.4,
                label=lab,
            )
        ax.axhline(0.95, color=INK_SECONDARY, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau^2 / V$")
        ax.set_ylim(0.4, 1.01)
        style(ax)
        panel_title(ax, f"{chr(97 + j)}) $L={L}$")
    axes[0].set_ylabel(r"Coverage for $\mu_{\infty}(\theta_0)$")
    axes[0].legend(loc="lower left", fontsize=7.5, handlelength=2.6)
    fig.savefig(Path(CONFIG["OUTPUT_DIR"]) / "superpopulation_coverage.pdf")
    plt.close(fig)


def figure_reanalysis():
    path = Path(CONFIG["REANALYSIS_DIR"]) / "reanalysis_intervals.csv"
    if not path.exists():
        print("reanalysis results are not available")
        return
    r = pd.read_csv(path)
    r = r[np.isclose(r["alpha"], 0.05)]
    dec = r[r["aggregation"] == "decade"].copy()
    dec["time"] = dec["time"].astype(int)
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    titles = {
        "total": "a) Total",
        "00_74": r"b) $<75$ years",
        "75plus": r"c) $\geq 75$ years",
    }
    pos = {"total": (0, 0), "00_74": (0, 1), "75plus": (0, 2)}
    col = PALETTE[0]
    for age in ["total", "00_74", "75plus"]:
        ax = fig.add_subplot(gs[pos[age]])
        s = dec[dec["age_group"] == age].sort_values("time")
        ax.fill_between(
            s["time"],
            s["eci_low"],
            s["eci_high"],
            color=col,
            alpha=0.18,
            lw=0,
            label=r"$I_{\mathrm{eCI}}$",
        )
        ax.fill_between(
            s["time"],
            s["mu_low"],
            s["mu_high"],
            color=col,
            alpha=0.45,
            lw=0,
            label=r"$I_{\mu}$",
        )
        ax.plot(s["time"], s["fit"], color=INK, lw=1.2, label=r"$\widehat{\mu}$")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Annual heat-related deaths")
        style(ax)
        panel_title(ax, titles[age])
        if age == "total":
            ax.legend(loc="upper left", fontsize=7.5)
    ax = fig.add_subplot(gs[1, :])
    per = r[r["aggregation"] != "decade"]
    labels, x = [], 0
    ages = ["total", "00_74", "75plus"]
    for period, plabel in [
        ("gwl", r"$2\,^{\circ}$C warming level"),
        ("endcentury", "2079--2099"),
    ]:
        for age in ages:
            s = per[(per["aggregation"] == period) & (per["age_group"] == age)].iloc[0]
            for k, (m, lab) in enumerate(
                [
                    ("eci", r"$I_{\mathrm{eCI}}$"),
                    ("mu", r"$I_{\mu}$"),
                    ("re_t", r"$I_{\mathrm{RE},t}$"),
                ]
            ):
                xx = x + (k - 1) * 0.22
                ax.plot(
                    [xx, xx],
                    [s[f"{m}_low"], s[f"{m}_high"]],
                    color=PALETTE[k],
                    lw=2.0,
                    solid_capstyle="round",
                    label=lab if x == 0 else None,
                )
            ax.plot(
                [x],
                [s["fit"]],
                "o",
                color=INK,
                ms=4,
                label=r"$\widehat{\mu}$" if x == 0 else None,
            )
            labels.append(
                f"{plabel.replace('$2', '$2').replace('warming level', 'WL')}\n"
                + {"total": "Total", "00_74": "<75", "75plus": "≥75"}[age]
            )
            x += 1
    ax.axhline(0, color=INK_SECONDARY, lw=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Annual heat-related deaths")
    style(ax)
    panel_title(ax, "d) 21-year periods")
    ax.legend(loc="upper left", fontsize=7.5, ncol=4)
    fig.savefig(Path(CONFIG["OUTPUT_DIR"]) / "reanalysis_london.pdf")
    plt.close(fig)


def main():
    out = Path(CONFIG["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    figure_exact()
    if (Path(CONFIG["EXACT_DIR"]) / "superpopulation.csv").exists():
        figure_superpopulation()
    figure_reanalysis()


if __name__ == "__main__":
    main()
