"""
Generate the supplementary figures: the sigmoid fits, the urbanisation-group
Fokker-Planck panels, the extended drift/diffusion/potential figure, the annual
Landau parameters and the detrended fluctuation analysis.

These five figures were previously produced only in an exploratory notebook, so
the archive shipped them without a generator that could rebuild them. Anything
the paper prints has to come out of this repository, and that includes figures.

Usage:
    python scripts/make_si_figures.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_merger import load_panel
from src.saai import compute_saai
from src.scaling import run_urbanization_stratified_scaling, fit_sigmoid_crossover
from src.fokker_planck import (estimate_kramers_moyal_all_periods,
                               estimate_km_by_group, compute_fp_potential,
                               find_drift_zeros)
from src.landau import fit_landau_timeseries
from src.dfa import (compute_cross_sectional_dfa_timeseries,
                     compute_cross_sectional_dfa, compute_mfdfa)

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_PRE, C_DUR, C_POST = "#4575b4", "#fee090", "#d73027"
COLORS_PERIOD = {"pre": C_PRE, "during": C_DUR, "post": C_POST}
LABELS_PERIOD = {"pre": "Pre (2000-2012)", "during": "During (2013-2019)",
                 "post": "Post (2020-2023)"}
SECTOR_COLORS = {"Tertiary": "#d73027", "Secondary": "#4575b4",
                 "NonAgri": "#2ca02c"}

# A curve fitted to bins that fail the reliability rule is not an estimate;
# drawing it invites a reading the caption then has to retract.
MIN_RELIABLE_BINS = 3


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 11, "figure.dpi": 150,
    })


def load_state():
    """The analysis panel and the SAAI series both figures below need."""
    panel = load_panel("panel_scaling")
    saai = compute_saai(panel, y_col="GrossOutputValue",
                        x_cols=["PopNum", "CultivateArea"])
    full = load_panel("panel_full")
    extra = [c for c in ("IsPoverty", "ExitYear", "Province", "Region",
                         "RegGDP", "NonAgriGDP", "RegGDP_Secondary",
                         "RegGDP_Tertiary") if c in full.columns]
    saai = saai.merge(full[["Year", "CountyCode"] + extra].drop_duplicates(),
                      on=["Year", "CountyCode"], how="left")
    return panel, saai


def fig_sigmoid(panel):
    """Sigmoid versus linear effective exponent, per sector."""
    sectors = {"Tertiary": "RegGDP_Tertiary", "Secondary": "RegGDP_Secondary",
               "NonAgri": "NonAgriGDP"}
    strat = run_urbanization_stratified_scaling(panel, year="pooled",
                                                n_quantiles=20)
    results = {}
    for name in sectors:
        sub = strat["table"][strat["table"]["Sector"] == name].copy()
        if len(sub) < 5:
            continue
        sub = sub.merge(strat["group_stats"][["mean_ratio"]],
                        left_on="Group", right_index=True, how="left")
        results[name] = fit_sigmoid_crossover(sub["mean_ratio"].values,
                                              sub["Beta"].values,
                                              sub["SE"].values)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, res) in zip(axes, results.items()):
        u, beta = res["u"], res["beta_local"]
        ax.scatter(u, beta, c=SECTOR_COLORS[name], s=40, zorder=5,
                   label="20-quantile $\\beta$")
        grid = np.linspace(u.min(), u.max(), 200)
        lin = res["linear"]
        ax.plot(grid, lin["beta0"] + lin["beta1"] * grid, "k--", alpha=0.5,
                label=f"Linear (AIC={lin['AIC']:.1f})")
        sig = res["sigmoid"]
        if "status" not in sig:
            curve = sig["beta_min"] + (sig["beta_max"] - sig["beta_min"]) / (
                1 + np.exp(-sig["k"] * (grid - sig["u_star"])))
            ax.plot(grid, curve, "-", color=SECTOR_COLORS[name], linewidth=2,
                    label=f"Sigmoid (AIC={sig['AIC']:.1f})")
            ax.axvline(sig["u_star"], color="gray", linestyle=":", alpha=0.5)
        ax.axhline(1, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Urbanization ratio $u$")
        ax.set_ylabel(r"$\beta_{\mathrm{eff}}(u)$")
        ax.set_title(f"{name} GDP")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sigmoid.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_sigmoid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_sigmoid")


def fig_fp_grouped(saai):
    """Drift and potential by urbanisation group, gated on bin reliability."""
    groups = estimate_km_by_group(saai, group_col="auto", saai_col="SAAI")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for row, name in enumerate(["Low", "Medium", "High"]):
        if name not in groups:
            continue
        data = groups[name]
        ax_a, ax_v = axes[row, 0], axes[row, 1]
        for period, km in data["km"].items():
            if km.reliable.sum() < MIN_RELIABLE_BINS:
                continue
            xi = km.bin_centers[km.reliable]
            ax_a.plot(xi, km.A[km.reliable], "-o", color=COLORS_PERIOD[period],
                      markersize=3, label=LABELS_PERIOD[period])
            ax_a.fill_between(xi,
                              km.A[km.reliable] - 1.96 * km.A_se[km.reliable],
                              km.A[km.reliable] + 1.96 * km.A_se[km.reliable],
                              alpha=0.12, color=COLORS_PERIOD[period])
            xi_fp, v_fp = data["fp"].get(period, (np.array([]), np.array([])))
            if len(xi_fp):
                ax_v.plot(xi_fp, v_fp, "-", color=COLORS_PERIOD[period],
                          linewidth=2, label=LABELS_PERIOD[period])
        ax_a.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_a.set_ylabel(r"$A(\xi)$")
        ax_a.set_title(f"{name} urbanization — drift")
        ax_v.set_ylabel(r"$V_{\mathrm{FP}}(\xi)$")
        ax_v.set_title(f"{name} urbanization — potential")
        for ax in (ax_a, ax_v):
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel(r"SAAI ($\xi$)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fp_grouped.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_fp_grouped.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_fp_grouped")


def fig_fp_expanded(saai):
    """Drift, diffusion, potential, and barrier heights with bootstrap CIs."""
    km_all = estimate_kramers_moyal_all_periods(saai, saai_col="SAAI")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, attr, se_attr, title in (
            (axes[0, 0], "A", "A_se", "(a) Drift coefficient"),
            (axes[0, 1], "B", "B_se", "(b) Diffusion coefficient")):
        for period, km in km_all.items():
            if km.reliable.sum() < MIN_RELIABLE_BINS:
                continue
            xi = km.bin_centers[km.reliable]
            y = getattr(km, attr)[km.reliable]
            se = getattr(km, se_attr)[km.reliable]
            ax.plot(xi, y, "-o", color=COLORS_PERIOD[period], markersize=3,
                    label=LABELS_PERIOD[period])
            ax.fill_between(xi, y - 1.96 * se, y + 1.96 * se, alpha=0.12,
                            color=COLORS_PERIOD[period])
        ax.set_xlabel(r"SAAI ($\xi$)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[0, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_ylabel(r"$A(\xi)$")
    axes[0, 1].set_ylabel(r"$B(\xi)$")
    axes[0, 1].axvspan(axes[0, 1].get_xlim()[0], 0, alpha=0.05, color="red")

    ax = axes[1, 0]
    for period, km in km_all.items():
        if km.reliable.sum() < MIN_RELIABLE_BINS:
            continue
        xi_fp, v_fp = compute_fp_potential(km)
        if len(xi_fp):
            ax.plot(xi_fp, v_fp, "-", color=COLORS_PERIOD[period], linewidth=2,
                    label=LABELS_PERIOD[period])
    ax.set_xlabel(r"SAAI ($\xi$)")
    ax.set_ylabel(r"$V_{\mathrm{FP}}(\xi)$")
    ax.set_title("(c) Fokker--Planck potential")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    boot = pd.read_csv("results/tables/km_bootstrap_ci.csv")
    point = pd.read_csv("results/tables/km_point_estimates.csv")
    vals, lo, hi = [], [], []
    for period in ("pre", "during"):
        pt = point[(point["saai_spec"] == "multi")
                   & (point["period"] == period)
                   & (point["binning"] == "equal_width")].iloc[0]["new_delta_V"]
        b = boot[(boot["saai_spec"] == "multi") & (boot["period"] == period)
                 & (boot["quantity"] == "delta_V")].iloc[0]
        vals.append(pt)
        lo.append(pt - b["ci_lo"])
        hi.append(b["ci_hi"] - pt)
    ax.bar([0, 1], vals, width=0.5, color=[C_PRE, C_DUR], alpha=0.85,
           yerr=[lo, hi], capsize=6, zorder=3)
    ax.set_xlim(-0.5, 2.5)
    # An empty third slot reads as a bar of height zero. The post-policy barrier
    # is not zero, it is unestimated: 91 transition pairs, and no bin reaches the
    # 30-pair minimum. The band fills the slot so the label has an anchor and so
    # that no height can be read off the place where a bar would stand.
    ax.axvspan(1.5, 2.5, facecolor="0.95", edgecolor="0.85", hatch="//",
               linewidth=0, zorder=0)
    ax.text(2.0, 0.5, "not estimable\n(91 pairs; no bin reaches\n"
            "the 30-pair minimum)", transform=ax.get_xaxis_transform(),
            ha="center", va="center", fontsize=8.5, color="0.3",
            bbox=dict(boxstyle="square,pad=0.45", facecolor="white",
                      edgecolor="none", alpha=0.85))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Pre\n(2000-2012)", "During\n(2013-2019)",
                        "Post\n(2020-2023)"])
    ax.get_xticklabels()[2].set_color("0.4")
    ax.set_ylabel(r"$\Delta V$ (path barrier to $\xi = 0$)")
    ax.set_title("(d) Barrier height by period (95% county-block CIs)")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fp_expanded.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_fp_expanded.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_fp_expanded")


def fig_landau(saai):
    """Annual static Landau fits: monostable in every year."""
    ts = fit_landau_timeseries(saai, saai_col="SAAI")
    valid = ts.dropna(subset=["a", "b", "h"])
    years = valid["Year"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(years, valid["a"], "ko-", markersize=4)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5,
               label="$a = 0$ (critical point)")
    ax.fill_between(years, valid["a"], 0, where=valid["a"] < 0, alpha=0.15,
                    color="blue", label="Bistable ($a < 0$)")
    ax.fill_between(years, valid["a"], 0, where=valid["a"] >= 0, alpha=0.15,
                    color="green", label="Monostable ($a \\geq 0$)")
    ax.set_ylabel("$a(t)$")
    ax.set_title("(a) Landau parameter $a(t)$")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(years, valid["h"], "ko-", markersize=4)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("$h(t)$")
    ax.set_title("(b) Symmetry-breaking field $h(t)$")

    ax = axes[1, 0]
    ax.plot(years, valid["barrier_height"].fillna(0), "ro-", markersize=4)
    ax.set_ylabel(r"$\Delta V(t)$")
    ax.set_title(r"(c) Static barrier height (annual fits are monostable)")

    ax = axes[1, 1]
    ax.plot(years, valid["R2"], "bs-", markersize=4)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(0, 1)
    ax.set_title("(d) Landau fit quality")

    for ax in axes.flat:
        ax.axvspan(2013, 2020, alpha=0.05, color="orange")
        ax.set_xlabel("Year")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_landau_params.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_landau_params.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_landau_params")


def fig_dfa(saai):
    """Cross-sectional DFA exponent over time, one fluctuation function, h(q)."""
    ts = compute_cross_sectional_dfa_timeseries(saai, saai_col="SAAI",
                                                sort_col="CountyCode")
    rep_year = int(ts.loc[ts["n"].idxmax(), "Year"]) if "n" in ts.columns \
        else int(ts["Year"].iloc[len(ts) // 2])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.errorbar(ts["Year"], ts["alpha"], yerr=ts["alpha_se"], fmt="ko-",
                markersize=4, capsize=3)
    for level, colour, label in ((0.5, "blue", r"$\alpha = 0.5$ (uncorrelated)"),
                                 (1.0, "red", r"$\alpha = 1.0$ ($1/f$)"),
                                 (1.5, "green", r"$\alpha = 1.5$ (Brownian)")):
        ax.axhline(level, color=colour, linestyle="--", alpha=0.5, label=label)
    ax.axvspan(2013, 2020, alpha=0.05, color="orange")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"DFA exponent $\alpha$")
    ax.set_title("(a) Cross-sectional DFA exponent over time")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[1]
    rep = compute_cross_sectional_dfa(saai, rep_year, saai_col="SAAI")
    if "scales" in rep:
        scales = np.asarray(rep["scales"])
        ax.loglog(scales, rep["F"], "ko", markersize=4)
        ax.loglog(scales, np.exp(rep["intercept"]) * scales ** rep["alpha"],
                  "r-", label=f"$\\alpha = {rep['alpha']:.3f}$")
        ax.legend()
    ax.set_xlabel("Scale $n$")
    ax.set_ylabel("$F(n)$")
    ax.set_title(f"(b) DFA fluctuation function ({rep_year})")
    ax.grid(alpha=0.2)

    ax = axes[2]
    mf = compute_mfdfa(saai[saai["Year"] == rep_year]["SAAI"].dropna().values)
    if "status" not in mf:
        ax.plot(mf["q"], mf["h_q"], "ko-", markersize=5)
        ax.fill_between(mf["q"], mf["h_q"] - mf["h_q_se"],
                        mf["h_q"] + mf["h_q_se"], alpha=0.2)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$h(q)$")
    ax.set_title(f"(c) Generalized Hurst exponent ({rep_year})")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dfa.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_dfa.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_dfa")


def main():
    setup_style()
    panel, saai = load_state()
    fig_sigmoid(panel)
    fig_fp_grouped(saai)
    fig_fp_expanded(saai)
    fig_landau(saai)
    fig_dfa(saai)


if __name__ == "__main__":
    main()
