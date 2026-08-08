#!/usr/bin/env python3
"""
Generate all 6 figures for the Results section of the paper.

Fig 1 (fig_spectrum): Sectoral scaling spectrum (2 subplots)
Fig 2 (fig_crossover): Urbanization crossover (3 subplots)
Fig 3 (fig_land): Dual diminishment of agriculture (2 subplots)
Fig 4 (fig_saai): SAAI & poverty traps (2 subplots)
Fig 5 (fig_fp): Distributional dynamics (2 subplots)
Fig 6 (fig_heatmap): Structural transformation heatmap (2 panels)

Usage:
    python scripts/generate_figures.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
import statsmodels.api as sm_api
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from src.config import FP_PERIODS, SCALING_EQUATIONS
from src.scaling import (
    run_scaling_regression,
    run_comparative_scaling,
    run_scaling_timeseries,
    run_interaction_regression,
    run_province_fe_interaction,
    run_interaction_regression_controlled,
    bootstrap_crossover_ci,
    run_urbanization_stratified_scaling,
)
from src.saai import compute_saai, compute_saai_single_factor, compute_order_parameter
from src.distribution import compute_effective_potential, compute_veff_timeseries
from src.fokker_planck import (
    estimate_kramers_moyal_all_periods,
    estimate_kramers_moyal,
    estimate_kramers_moyal_kernel,
    build_transition_pairs,
    find_drift_zeros,
    simulate_langevin,
    validate_km_recovery,
)

# ── Headline values ──
# Figures print the numbers the analysis reports by reading the file those
# numbers are generated into, rather than re-estimating them here.
HEADLINE_CSV = PROJECT_ROOT / "results" / "tables" / "headline_scaling.csv"


def headline(quantity: str, estimator: str = "pooled OLS"):
    """One generated headline value, or None if the file is not built yet."""
    if not HEADLINE_CSV.exists():
        return None
    df = pd.read_csv(HEADLINE_CSV)
    hit = df[(df["quantity"] == quantity) & (df["estimator"] == estimator)]
    return None if hit.empty else hit.iloc[0]


# ── Output directory ──
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──
FONT_SIZE = 11
DPI = 300
C_POVERTY = "#d73027"
C_NONPOV = "#2166ac"
C_PRI = "#2ca02c"
C_SEC = "#ff7f0e"
C_TER = "#9467bd"
C_NONAGRI = "#1f77b4"


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "figure.dpi": DPI,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 2,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })


def load_data():
    """Load and prepare panel data."""
    print("Loading data...")
    panel_scaling = pd.read_parquet(PROJECT_ROOT / "data_clean" / "panel_scaling.parquet")
    panel_full = pd.read_parquet(PROJECT_ROOT / "data_clean" / "panel_full.parquet")

    # Ensure NonAgriGDP exists
    for df in [panel_scaling, panel_full]:
        if "NonAgriGDP" not in df.columns:
            if "RegGDP_Secondary" in df.columns and "RegGDP_Tertiary" in df.columns:
                df["NonAgriGDP"] = df["RegGDP_Secondary"].fillna(0) + df["RegGDP_Tertiary"].fillna(0)
        # Ensure IsPoverty is boolean
        if "IsPoverty" in df.columns:
            df["IsPoverty"] = df["IsPoverty"].fillna(False).astype(bool)

    print(f"  panel_scaling: {len(panel_scaling)} rows")
    print(f"  panel_full: {len(panel_full)} rows")
    return panel_scaling, panel_full


# ════════════════════════════════════════════════════════════
# Figure 1: Sectoral Scaling Spectrum
# ════════════════════════════════════════════════════════════

def fig1_spectrum(panel, ts_df):
    """
    (a) Double-log scatter for 2015: Primary/Secondary/Tertiary + regression lines.
    (b) Year-by-year β time series (2000-2023) with CI bands.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel (a): Cross-section scatter ──
    CROSS_SECTION_YEAR = 2015
    YEAR_LABEL = f"({CROSS_SECTION_YEAR})"
    ax = axes[0]
    d = panel[panel["Year"] == CROSS_SECTION_YEAR].copy()

    sector_cfg = [
        ("RegGDP_Primary", "14j", C_PRI, "Primary", "o"),
        ("RegGDP_Secondary", "14h", C_SEC, "Secondary", "s"),
        ("RegGDP_Tertiary", "14i", C_TER, "Tertiary", "^"),
    ]

    for ycol, eq_key, color, label, marker in sector_cfg:
        mask = (d["PopNum"] > 0) & (d[ycol] > 0)
        dd = d[mask]
        log_x = np.log10(dd["PopNum"])
        log_y = np.log10(dd[ycol])
        is_pov = dd["IsPoverty"].fillna(False)

        # Non-poverty: filled; poverty: hollow
        ax.scatter(log_x[~is_pov], log_y[~is_pov], s=6, alpha=0.2,
                   c=color, marker=marker, rasterized=True)
        ax.scatter(log_x[is_pov], log_y[is_pov], s=12, alpha=0.4,
                   facecolors="none", edgecolors=color, marker=marker,
                   linewidths=0.5, rasterized=True)

        # Regression line
        try:
            r = run_scaling_regression(dd, y_col=ycol, x_cols=["PopNum"], year="pooled")
            b = r.betas["PopNum"]
            ci_lo = r.ci_lower["PopNum"]
            ci_hi = r.ci_upper["PopNum"]
            a_log10 = r.intercept / np.log(10)
            x_range = np.linspace(log_x.min(), log_x.max(), 100)
            # The pooled estimate the analysis reports, read from the
            # generated file rather than re-fitted here; without the year
            # label the two read as rival values for one quantity.
            hv = headline(f"beta_{label.lower()}")
            pooled_str = f" (pooled: {hv['value']:.2f})" if hv is not None else ""
            ax.plot(x_range, a_log10 + b * x_range, "-", color=color, linewidth=2,
                    label=f"{label} {YEAR_LABEL}: $\\beta$={b:.2f} "
                          f"[{ci_lo:.2f}, {ci_hi:.2f}]{pooled_str}")
        except Exception:
            pass

    ax.set_xlabel(r"log$_{10}$(Population, $10^4$ persons)")
    ax.set_ylabel(r"log$_{10}$(Sectoral GDP, $10^4$ CNY)")
    ax.set_title("(a) Cross-section (2015)")
    ax.legend(loc="lower right", fontsize=8)

    # ── Panel (b): Time series ──
    ax = axes[1]
    sector_map = {
        "14j": (C_PRI, "Primary", "-", "o"),
        "14h": (C_SEC, "Secondary", "-", "s"),
        "14i": (C_TER, "Tertiary", "-", "^"),
    }

    for eq, (color, label, ls, marker) in sector_map.items():
        sub = ts_df[(ts_df["Equation"] == eq) & (ts_df["Variable"] == "PopNum")].sort_values("Year")
        if len(sub) == 0:
            continue
        ax.plot(sub["Year"], sub["Beta"], ls, color=color, marker=marker,
                markersize=3, label=label, linewidth=1.5)
        ax.fill_between(sub["Year"], sub["CI_lower"], sub["CI_upper"],
                        alpha=0.12, color=color)

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=r"$\beta=1$")
    ax.axvline(x=2013, color="green", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(2013.3, ax.get_ylim()[1] * 0.98, "2013", color="green", fontsize=8, va="top")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Scaling exponent $\beta$")
    ax.set_title("(b) Year-by-year exponents (2000–2023)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = FIG_DIR / "fig_spectrum.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════
# Figure 2: Urbanization Crossover
# ════════════════════════════════════════════════════════════

def fig2_crossover(panel, sector_interactions, boot_result):
    """
    (a) β_eff(u) lines for 3 sectors + CI + sample distribution.
    (b) 20-quantile LOWESS for tertiary.
    (c) Bootstrap histogram of u*.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Panel (a): Interaction lines ──
    ax = axes[0]
    u_grid = np.linspace(0.1, 1.0, 200)
    colors = {"Primary": C_PRI, "Secondary": C_SEC, "Tertiary": C_TER}
    linestyles = {"Primary": "--", "Secondary": "-.", "Tertiary": "-"}

    # Sample distribution of u as background
    data = panel.copy()
    data = data[(data["RegGDP"] > 0) & (data["NonAgriGDP"] > 0)]
    data["_u"] = (data["NonAgriGDP"] / data["RegGDP"]).clip(0, 1)
    ax2_twin = ax.twinx()
    ax2_twin.hist(data["_u"], bins=50, alpha=0.15, color="gray", density=True)
    ax2_twin.set_ylabel("")
    ax2_twin.set_yticks([])

    for sector, res in sector_interactions.items():
        b0 = res["beta0"]
        b1 = res["beta1"]
        beta_u = b0 + b1 * u_grid
        color = colors.get(sector, "gray")
        ls = linestyles.get(sector, "-")

        # SE band using covariance matrix
        cov = res["cov_matrix"]
        # Indices in cov: 0=const, 1=lnN(β₀), 2=u·lnN(β₁)
        var_b0 = cov[1, 1]
        var_b1 = cov[2, 2]
        cov_b0b1 = cov[1, 2]
        se_u = np.sqrt(var_b0 + u_grid**2 * var_b1 + 2 * u_grid * cov_b0b1)

        ax.plot(u_grid, beta_u, ls, color=color, linewidth=2.5, label=sector)
        ax.fill_between(u_grid, beta_u - 1.96 * se_u, beta_u + 1.96 * se_u,
                         color=color, alpha=0.10)

        u_star = res["crossover_u"]
        if 0 < u_star < 1.2 and sector != "Primary":
            ax.axvline(x=u_star, color=color, linestyle=":", alpha=0.6, linewidth=1)
            ax.text(u_star + 0.01, 0.6, f"$u^*$={u_star:.2f}",
                    color=color, fontsize=8, rotation=90, va="bottom")

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=r"$\beta=1$")
    ax.set_xlabel("Urbanization ratio $u$")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}(u)$")
    ax.set_title("(a) Effective exponent vs urbanization")
    ax.set_xlim(0.1, 1.0)
    ax.legend(fontsize=8, loc="upper left")

    # ── Panel (b): LOWESS non-parametric ──
    ax = axes[1]
    # Compute 20-quantile β for tertiary
    data_q = panel.copy()
    data_q = data_q[(data_q["RegGDP"] > 0) & (data_q["NonAgriGDP"] > 0) &
                     (data_q["RegGDP_Tertiary"] > 0) & (data_q["PopNum"] > 0)]
    data_q["_u"] = (data_q["NonAgriGDP"] / data_q["RegGDP"]).clip(0, 1)

    n_q = 20
    data_q["_q"] = pd.qcut(data_q["_u"], q=n_q, labels=False, duplicates="drop")
    q_betas = []
    q_mids = []
    q_ses = []
    for q_val in sorted(data_q["_q"].unique()):
        grp = data_q[data_q["_q"] == q_val]
        mid_u = grp["_u"].median()
        try:
            r = run_scaling_regression(grp, y_col="RegGDP_Tertiary",
                                        x_cols=["PopNum"], year="pooled")
            q_betas.append(r.betas["PopNum"])
            q_ses.append(r.se["PopNum"])
            q_mids.append(mid_u)
        except Exception:
            pass

    q_betas = np.array(q_betas)
    q_ses = np.array(q_ses)
    q_mids = np.array(q_mids)

    ax.errorbar(q_mids, q_betas, yerr=1.96 * q_ses, fmt="o", color=C_TER,
                 markersize=5, capsize=3, alpha=0.7, label="Quantile $\\beta$")

    # LOWESS smooth
    valid = np.isfinite(q_mids) & np.isfinite(q_betas)
    if valid.sum() >= 5:
        smoothed = sm_lowess(q_betas[valid], q_mids[valid], frac=0.4, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], "-", color=C_TER,
                linewidth=2.5, label="LOWESS")

        # Find LOWESS crossover
        cross_idx = np.where(smoothed[:, 1] >= 1.0)[0]
        if len(cross_idx) > 0:
            cross_u = smoothed[cross_idx[0], 0]
            ax.axvline(x=cross_u, color=C_TER, linestyle=":", alpha=0.6, linewidth=1)
            ax.annotate(f"Crossover = {cross_u:.3f}", xy=(cross_u, 1.0),
                         xytext=(cross_u - 0.15, 1.08),
                         arrowprops=dict(arrowstyle="->", color=C_TER),
                         fontsize=8, color=C_TER)

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Urbanization ratio $u$ (bin median)")
    ax.set_ylabel(r"$\beta$ (Tertiary GDP)")
    ax.set_title("(b) 20-quantile LOWESS (Tertiary)")
    ax.legend(fontsize=8)

    # ── Panel (c): Bootstrap histogram ──
    ax = axes[2]
    u_stars = boot_result["u_stars"]
    ax.hist(u_stars, bins=30, color=C_TER, alpha=0.5, edgecolor="black",
            linewidth=0.5, density=True)
    # OLS point estimate
    ter_res = sector_interactions.get("Tertiary", {})
    if ter_res:
        ax.axvline(x=ter_res["crossover_u"], color="black", linewidth=2,
                    label=f'OLS $u^*$={ter_res["crossover_u"]:.3f}')
    # CI boundaries
    ax.axvline(x=boot_result["ci_lower"], color=C_TER, linestyle="--", linewidth=1.5,
                label=f'2.5% = {boot_result["ci_lower"]:.3f}')
    ax.axvline(x=boot_result["ci_upper"], color=C_TER, linestyle="--", linewidth=1.5,
                label=f'97.5% = {boot_result["ci_upper"]:.3f}')
    ax.set_xlabel("$u^*$ (Tertiary crossover)")
    ax.set_ylabel("Density")
    ax.set_title(f'(c) Bootstrap distribution ($n$={boot_result["n_valid"]})')
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = FIG_DIR / "fig_crossover.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════
# Figure 3: Dual Diminishment of Agriculture
# ════════════════════════════════════════════════════════════

def fig3_land(panel):
    """
    (a) Progressive land control bar chart.
    (b) X-shaped crossover: NonAgri (positive slope) vs GrainOutput (negative slope).
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel (a): Bar chart ──
    ax = axes[0]
    specs = []

    # No control: Agri ~ PopNum
    try:
        r0 = run_scaling_regression(panel, y_col="GrossOutputValue", x_cols=["PopNum"])
        specs.append(("No control", r0.betas["PopNum"], r0.se["PopNum"], C_PRI))
    except Exception:
        pass

    # + Cultivated area
    try:
        r1 = run_scaling_regression(panel, y_col="GrossOutputValue",
                                     x_cols=["PopNum", "CultivateArea"])
        specs.append(("+ Cultivated\narea", r1.betas["PopNum"], r1.se["PopNum"], C_PRI))
    except Exception:
        pass

    # + Sown area
    try:
        r2 = run_scaling_regression(panel, y_col="GrossOutputValue",
                                     x_cols=["PopNum", "TotalSownArea"])
        specs.append(("+ Sown\narea", r2.betas["PopNum"], r2.se["PopNum"], C_PRI))
    except Exception:
        pass

    # Placebo: NonAgri ~ PopNum + CultivateArea
    try:
        rp = run_scaling_regression(panel, y_col="NonAgriGDP",
                                     x_cols=["PopNum", "CultivateArea"])
        specs.append(("NonAgri\n(placebo)", rp.betas["PopNum"], rp.se["PopNum"], "#999999"))
    except Exception:
        pass

    if specs:
        labels, betas, ses, colors = zip(*specs)
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, betas, yerr=[1.96 * s for s in ses],
                       color=colors, alpha=0.7, edgecolor="black", linewidth=0.5,
                       capsize=5)

        # Annotate values
        for i, (b, s) in enumerate(zip(betas, ses)):
            ax.text(i, b + 1.96 * s + 0.02, f"{b:.2f}", ha="center", fontsize=9,
                    fontweight="bold")

        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"$\hat{\beta}_N$ (Population exponent)")
        ax.set_title("(a) Progressive land controls")

    # ── Panel (b): X-shaped crossover ──
    ax = axes[1]
    u_grid = np.linspace(0.1, 1.0, 200)

    # NonAgri crossover (positive slope)
    try:
        res_na = run_interaction_regression(panel, y_col="NonAgriGDP")
        beta_na = res_na["beta0"] + res_na["beta1"] * u_grid
        ax.plot(u_grid, beta_na, "-", color=C_NONAGRI, linewidth=2.5,
                label=f'Non-agricultural ($\\beta_1$={res_na["beta1"]:.2f})')
    except Exception:
        pass

    # Grain output crossover (negative slope)
    if "GrainOutput" in panel.columns:
        try:
            res_gr = run_interaction_regression(panel, y_col="GrainOutput")
            beta_gr = res_gr["beta0"] + res_gr["beta1"] * u_grid
            hv = headline("beta1_grain_interaction", "interaction OLS")
            b1 = res_gr["beta1"] if hv is None else float(hv["value"])
            ax.plot(u_grid, beta_gr, "-", color=C_POVERTY, linewidth=2.5,
                    label=f'Grain output ($\\beta_1$={b1:.2f})')
        except Exception:
            print("    Note: GrainOutput interaction regression failed")
    else:
        print("    Note: GrainOutput column not available, skipping grain crossover")

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=r"$\beta=1$")
    ax.set_xlabel("Urbanization ratio $u$")
    ax.set_ylabel(r"$\beta_{\mathrm{eff}}(u)$")
    ax.set_title("(b) Crossover direction comparison")
    ax.legend(fontsize=9)
    ax.set_xlim(0.1, 1.0)

    plt.tight_layout()
    save_path = FIG_DIR / "fig_land.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════
# Figure 4: SAAI & Poverty Traps
# ════════════════════════════════════════════════════════════

def fig4_saai(saai_df):
    """
    (a) KDE of SAAI: poverty (red) vs non-poverty (blue).
    (b) Grouped bar: Low/Med/High urbanization × poverty/non-poverty.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Use pre-exit data (2000-2015) for clean comparison
    pre = saai_df[(saai_df["Year"] <= 2015) & saai_df["SAAI"].notna()].copy()

    pov_saai = pre[pre["IsPoverty"]]["SAAI"].values
    npov_saai = pre[~pre["IsPoverty"]]["SAAI"].values

    # ── Panel (a): KDE distributions ──
    ax = axes[0]
    xi_grid = np.linspace(-4, 4, 300)

    if len(pov_saai) > 10:
        kde_pov = gaussian_kde(pov_saai)
        ax.fill_between(xi_grid, kde_pov(xi_grid), alpha=0.3, color=C_POVERTY)
        ax.plot(xi_grid, kde_pov(xi_grid), color=C_POVERTY, linewidth=2,
                label=f"Poverty (mean={np.mean(pov_saai):.2f})")

    if len(npov_saai) > 10:
        kde_npov = gaussian_kde(npov_saai)
        ax.fill_between(xi_grid, kde_npov(xi_grid), alpha=0.3, color=C_NONPOV)
        ax.plot(xi_grid, kde_npov(xi_grid), color=C_NONPOV, linewidth=2,
                label=f"Non-poverty (mean={np.mean(npov_saai):.2f})")

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Cohen's d annotation
    if len(pov_saai) > 1 and len(npov_saai) > 1:
        pooled_std = np.sqrt(
            ((len(pov_saai) - 1) * np.std(pov_saai, ddof=1)**2 +
             (len(npov_saai) - 1) * np.std(npov_saai, ddof=1)**2) /
            (len(pov_saai) + len(npov_saai) - 2)
        )
        d = (np.mean(npov_saai) - np.mean(pov_saai)) / pooled_std if pooled_std > 0 else 0
        ax.text(0.95, 0.95, f"Cohen's $d$ = {d:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel(r"SAAI ($\xi$)")
    ax.set_ylabel("Density")
    ax.set_title("(a) SAAI distribution by poverty status")
    ax.legend(fontsize=9)

    # ── Panel (b): Grouped bar by urbanization ──
    ax = axes[1]

    # Compute urbanization ratio
    pre_u = pre.copy()
    pre_u = pre_u[(pre_u["RegGDP"] > 0) & (pre_u["NonAgriGDP"] > 0)]
    pre_u["_u"] = (pre_u["NonAgriGDP"] / pre_u["RegGDP"]).clip(0, 1)

    # Group by urbanization level
    urb_groups = {"Low": (0, 0.50), "Medium": (0.50, 0.80), "High": (0.80, 1.01)}
    bar_data = []
    for grp_name, (lo, hi) in urb_groups.items():
        mask = (pre_u["_u"] >= lo) & (pre_u["_u"] < hi)
        for is_pov, pov_label, color in [(True, "Poverty", C_POVERTY),
                                          (False, "Non-poverty", C_NONPOV)]:
            sub = pre_u[mask & (pre_u["IsPoverty"] == is_pov)]
            if len(sub) > 0:
                bar_data.append({
                    "UrbGroup": grp_name, "PovStatus": pov_label,
                    "mean_saai": sub["SAAI"].mean(),
                    "se_saai": sub["SAAI"].std() / np.sqrt(len(sub)),
                    "n": len(sub), "color": color,
                })

    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        groups = list(urb_groups.keys())
        x = np.arange(len(groups))
        width = 0.35

        for i, (pov_label, offset) in enumerate([("Poverty", -width/2),
                                                   ("Non-poverty", width/2)]):
            sub = bar_df[bar_df["PovStatus"] == pov_label]
            vals = []
            errs = []
            for g in groups:
                row = sub[sub["UrbGroup"] == g]
                vals.append(row["mean_saai"].values[0] if len(row) > 0 else 0)
                errs.append(1.96 * row["se_saai"].values[0] if len(row) > 0 else 0)
            color = C_POVERTY if pov_label == "Poverty" else C_NONPOV
            ax.bar(x + offset, vals, width, yerr=errs, color=color, alpha=0.7,
                   edgecolor="black", linewidth=0.5, capsize=4, label=pov_label)

        # Cohen's d annotations — bracket between each bar pair
        for j, g in enumerate(groups):
            pov_sub = pre_u[(pre_u["_u"] >= urb_groups[g][0]) &
                           (pre_u["_u"] < urb_groups[g][1]) & pre_u["IsPoverty"]]
            npov_sub = pre_u[(pre_u["_u"] >= urb_groups[g][0]) &
                            (pre_u["_u"] < urb_groups[g][1]) & ~pre_u["IsPoverty"]]
            if len(pov_sub) > 1 and len(npov_sub) > 1:
                p_std = np.sqrt(
                    ((len(pov_sub) - 1) * pov_sub["SAAI"].std(ddof=1)**2 +
                     (len(npov_sub) - 1) * npov_sub["SAAI"].std(ddof=1)**2) /
                    (len(pov_sub) + len(npov_sub) - 2)
                )
                d_val = (npov_sub["SAAI"].mean() - pov_sub["SAAI"].mean()) / p_std if p_std > 0 else 0
                # Find the highest point of either bar (mean + CI) in this group
                npov_top = npov_sub["SAAI"].mean() + 1.96 * npov_sub["SAAI"].std() / np.sqrt(len(npov_sub))
                pov_top = pov_sub["SAAI"].mean() + 1.96 * pov_sub["SAAI"].std() / np.sqrt(len(pov_sub))
                y_top = max(npov_top, pov_top)
                # Use annotate with offset in points to clear the error bar cap
                ax.annotate(f"$d$={d_val:.2f}", xy=(j, y_top),
                            xytext=(0, 14), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8,
                            fontweight="bold")

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n($u$$\in$[{urb_groups[g][0]:.1f},{urb_groups[g][1]:.1f}))"
                            for g in groups], fontsize=8)
        ax.set_ylabel("Mean SAAI")
        ax.set_title("(b) SAAI by urbanization $\\times$ poverty status")
        ax.legend(fontsize=9)
        # Add headroom so d-labels don't collide with title
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.15)

    plt.tight_layout()
    save_path = FIG_DIR / "fig_saai.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════
# Figure 5: Distributional Dynamics (FP)
# ════════════════════════════════════════════════════════════

def fig5_fp(saai_df, km_results):
    """
    (a) Drift A(ξ) for pre + during periods, kernel + binning.
    (b) V_eff for representative years.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    fp_colors = {"pre": "#4575b4", "during": "#d73027"}
    fp_labels = {"pre": "Pre (2000–2012)", "during": "During (2013–2019)"}

    # ── Panel (a): Drift function ──
    ax = axes[0]

    kern_results = {}
    for period in ["pre", "during"]:
        if period not in km_results:
            continue
        km = km_results[period]
        mask = km.reliable
        if mask.sum() == 0:
            continue
        xi = km.bin_centers[mask]
        color = fp_colors[period]

        # Binning-based (square markers)
        ax.plot(xi, km.A[mask], "s", color=color, markersize=4, alpha=0.6,
                label=f"{fp_labels[period]} (binning)")

        # Kernel-based (solid line with CI) — compute from same transition pairs
        pairs = build_transition_pairs(saai_df, period)
        if len(pairs) > 100:
            km_kern = estimate_kramers_moyal_kernel(pairs, n_grid=80)
            kern_results[period] = km_kern
            km_mask = km_kern.reliable
            if km_mask.sum() > 0:
                ax.plot(km_kern.bin_centers[km_mask], km_kern.A[km_mask], "-",
                        color=color, linewidth=2,
                        label=f"{fp_labels[period]} (kernel)")
                ax.fill_between(km_kern.bin_centers[km_mask],
                                km_kern.A[km_mask] - 1.96 * km_kern.A_se[km_mask],
                                km_kern.A[km_mask] + 1.96 * km_kern.A_se[km_mask],
                                alpha=0.12, color=color)

    # Find and annotate drift zeros using kernel-based estimator
    if "pre" in kern_results:
        zeros = find_drift_zeros(kern_results["pre"])
        for z in zeros:
            if z["type"] == "stable":
                ax.annotate(f'$\\xi^*$={z["xi"]:.2f}', xy=(z["xi"], 0),
                             xytext=(z["xi"] + 0.3, 0.08),
                             arrowprops=dict(arrowstyle="->", color="black"),
                             fontsize=9, fontweight="bold")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"SAAI ($\xi$)")
    ax.set_ylabel(r"Drift $A(\xi)$")
    ax.set_title("(a) Drift function")
    ax.legend(fontsize=7, loc="upper right")

    # ── Panel (b): V_eff evolution ──
    ax = axes[1]
    veff_ts = compute_veff_timeseries(saai_df, saai_col="SAAI")
    available_years = sorted(veff_ts.keys())

    # Select representative years
    target_years = [2002, 2008, 2015, 2020, 2023]
    rep_years = []
    for ty in target_years:
        closest = min(available_years, key=lambda y: abs(y - ty))
        if closest not in rep_years:
            rep_years.append(closest)

    n_years = len(rep_years)
    cmap = plt.cm.viridis
    colors_veff = [cmap(i / max(n_years - 1, 1)) for i in range(n_years)]

    for i, yr in enumerate(rep_years):
        xi_grid, veff = veff_ts[yr]
        # Clip to reasonable range
        mask = (xi_grid > -3.5) & (xi_grid < 3.5)
        ax.plot(xi_grid[mask], veff[mask], "-", color=colors_veff[i],
                linewidth=2, label=f"{int(yr)}", alpha=0.85)

    ax.set_xlabel(r"SAAI ($\xi$)")
    ax.set_ylabel(r"$V_{\mathrm{eff}}(\xi) = -\ln P(\xi)$")
    ax.set_title("(b) Effective potential evolution")
    ax.legend(fontsize=8, title="Year")

    plt.tight_layout()
    save_path = FIG_DIR / "fig_fp.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════
# Figure 6: Structural Transformation Heatmap
# ════════════════════════════════════════════════════════════

def fig6_heatmap(panel):
    """
    3×3 heatmap for Secondary (left) and Tertiary (right):
    Rows = Pre/During/Post, Cols = Low/Med/High urbanization.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                              gridspec_kw={"wspace": 0.35})

    periods = {
        "Pre (2000–12)": (2000, 2012),
        "During (2013–19)": (2013, 2019),
        "Post (2020–23)": (2020, 2023),
    }
    urb_groups = {"Low": (0, 0.50),
                  "Medium": (0.50, 0.80),
                  "High": (0.80, 1.01)}

    for ax_idx, (sector, ycol) in enumerate([("Secondary GDP", "RegGDP_Secondary"),
                                               ("Tertiary GDP", "RegGDP_Tertiary")]):
        ax = axes[ax_idx]
        matrix = np.full((3, 3), np.nan)
        annot = [["" for _ in range(3)] for _ in range(3)]

        for i, (p_label, (yr_lo, yr_hi)) in enumerate(periods.items()):
            p_data = panel[(panel["Year"] >= yr_lo) & (panel["Year"] <= yr_hi)].copy()
            p_data = p_data[(p_data["RegGDP"] > 0) & (p_data["NonAgriGDP"] > 0)]
            p_data["_u"] = (p_data["NonAgriGDP"] / p_data["RegGDP"]).clip(0, 1)

            for j, (g_label, (u_lo, u_hi)) in enumerate(urb_groups.items()):
                grp = p_data[(p_data["_u"] >= u_lo) & (p_data["_u"] < u_hi)]
                if len(grp) < 50:
                    continue
                try:
                    r = run_scaling_regression(grp, y_col=ycol, x_cols=["PopNum"],
                                               year="pooled")
                    b = r.betas["PopNum"]
                    wald_p = r.wald_beta_eq_1["PopNum"]["pvalue"]
                    matrix[i, j] = b
                    stars = "***" if wald_p < 0.001 else "**" if wald_p < 0.01 else "*" if wald_p < 0.05 else ""
                    annot[i][j] = f"{b:.2f}{stars}"
                except Exception:
                    pass

        # Plot heatmap
        norm = TwoSlopeNorm(vmin=0.70, vcenter=1.00, vmax=1.20)
        im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", norm=norm)

        for i in range(3):
            for j in range(3):
                if annot[i][j]:
                    text_color = "white" if abs(matrix[i, j] - 1) > 0.12 else "black"
                    ax.text(j, i, annot[i][j], ha="center", va="center",
                            fontsize=11, fontweight="bold", color=text_color)

        ax.set_xticks(range(3))
        ax.set_xticklabels(list(urb_groups.keys()), fontsize=9)
        ax.set_xlabel("Urbanization group")
        ax.set_yticks(range(3))
        if ax_idx == 0:
            ax.set_yticklabels(list(periods.keys()), fontsize=9)
            ax.set_ylabel("Period")
        else:
            ax.set_yticklabels(list(periods.keys()), fontsize=9)
        ax.set_title(sector, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=axes, label=r"$\hat{\beta}$ (Population exponent)",
                  shrink=0.85, pad=0.03)
    save_path = FIG_DIR / "fig_heatmap.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════
# Figure S1: Langevin Simulation Validation
# ════════════════════════════════════════════════════════════

def figS1_langevin():
    """
    (a) Sample synthetic trajectories from the bistable potential.
    (b) Drift recovery: true A(ξ) vs KM-estimated A(ξ).
    (c) Diffusion recovery: true B(ξ) vs KM-estimated B(ξ).
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Known dynamics: bistable potential A(ξ) = ξ − ξ³, constant diffusion B = 0.3
    A_func = lambda x: x - x**3
    B_func = lambda x: 0.3

    # ── Panel (a): Sample trajectories ──
    ax = axes[0]
    sim = simulate_langevin(A_func, B_func, n_particles=2000, n_steps=100,
                            dt=0.01, xi0_range=(-2, 2), seed=42)

    # Plot 30 representative trajectories
    rng = np.random.RandomState(0)
    sample_ids = rng.choice(2000, size=30, replace=False)
    cmap = plt.cm.coolwarm
    for idx, pid in enumerate(sample_ids):
        traj = sim[sim["particle_id"] == pid].sort_values("step")
        t = traj["step"].values * 0.01  # physical time
        color = cmap(0.2) if traj["xi"].iloc[-1] < 0 else cmap(0.8)
        ax.plot(t, traj["xi"].values, "-", color=color, alpha=0.35, linewidth=0.6)

    # Mark the two stable fixed points of A(ξ) = ξ − ξ³: ξ = ±1
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=-1, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(1.02, 1.0, "$\\xi^*=+1$", fontsize=8, va="center",
            transform=ax.get_yaxis_transform())
    ax.text(1.02, -1.0, "$\\xi^*=-1$", fontsize=8, va="center",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\xi(t)$")
    ax.set_title("(a) Sample trajectories ($n$=30 of 2,000)")

    # ── Run KM recovery ──
    val = validate_km_recovery(A_func, B_func, n_particles=2000, n_steps=100,
                               dt=0.01, n_bins=25, xi_range=(-2, 2))

    # ── Panel (b): Drift recovery ──
    ax = axes[1]
    xi_fine = np.linspace(val["xi_grid"].min(), val["xi_grid"].max(), 200)
    ax.plot(xi_fine, np.array([A_func(x) for x in xi_fine]), "k-", linewidth=2,
            label=r"True $A(\xi)=\xi-\xi^3$")
    ax.plot(val["xi_grid"], val["A_hat"], "ro", markersize=5, alpha=0.7,
            label=f'KM estimate (RMSE={val["rmse_A"]:.4f})')
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$A(\xi)$")
    ax.set_title("(b) Drift recovery")
    ax.legend(fontsize=8)

    # ── Panel (c): Diffusion recovery ──
    ax = axes[2]
    ax.plot(xi_fine, np.full_like(xi_fine, 0.3), "k-", linewidth=2,
            label=r"True $B(\xi)=0.3$")
    ax.plot(val["xi_grid"], val["B_hat"], "bs", markersize=5, alpha=0.7,
            label=f'KM estimate (RMSE={val["rmse_B"]:.4f})')
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$B(\xi)$")
    ax.set_title("(c) Diffusion recovery")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)

    plt.tight_layout()
    save_path = FIG_DIR / "fig_langevin.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")
    return fig


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    setup_style()
    panel_scaling, panel_full = load_data()

    # ── Compute shared results ──
    print("\n[1/6] Computing scaling time series...")
    ts_df = run_scaling_timeseries(panel_scaling)

    print("\n[2/6] Computing interaction regressions...")
    sector_interactions = {}
    for sector, ycol in [("Primary", "RegGDP_Primary"),
                          ("Secondary", "RegGDP_Secondary"),
                          ("Tertiary", "RegGDP_Tertiary")]:
        try:
            res = run_interaction_regression(panel_scaling, y_col=ycol)
            sector_interactions[sector] = res
            print(f"  {sector}: β₀={res['beta0']:.3f}, β₁={res['beta1']:.3f}, "
                  f"u*={res['crossover_u']:.3f}")
        except Exception as e:
            print(f"  {sector}: FAILED — {e}")

    print("\n[3/6] Running bootstrap for tertiary crossover CI...")
    boot_result = bootstrap_crossover_ci(panel_scaling, y_col="RegGDP_Tertiary",
                                          n_boot=500, seed=42)
    print(f"  u* = {boot_result['mean_u_star']:.3f} "
          f"[{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]")

    print("\n[4/6] Computing SAAI...")
    saai_df = compute_saai(panel_full, y_col="GrossOutputValue",
                            x_cols=["PopNum", "CultivateArea"])
    # Merge IsPoverty back
    if "IsPoverty" not in saai_df.columns and "IsPoverty" in panel_full.columns:
        pov_map = panel_full[["CountyCode", "Year", "IsPoverty"]].drop_duplicates()
        saai_df = saai_df.merge(pov_map, on=["CountyCode", "Year"], how="left")
    if "IsPoverty" in saai_df.columns:
        saai_df["IsPoverty"] = saai_df["IsPoverty"].fillna(False).astype(bool)

    # Merge RegGDP, NonAgriGDP for urbanization grouping
    for col in ["RegGDP", "NonAgriGDP"]:
        if col not in saai_df.columns and col in panel_full.columns:
            merge_cols = panel_full[["CountyCode", "Year", col]].drop_duplicates()
            saai_df = saai_df.merge(merge_cols, on=["CountyCode", "Year"], how="left")

    print(f"  SAAI computed: {len(saai_df)} rows")

    print("\n[5/6] Computing Kramers-Moyal drift/diffusion...")
    km_results = estimate_kramers_moyal_all_periods(saai_df, saai_col="SAAI")

    # ── Generate figures ──
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    print("\nFig 1: Sectoral scaling spectrum...")
    fig1_spectrum(panel_scaling, ts_df)

    print("\nFig 2: Urbanization crossover...")
    fig2_crossover(panel_scaling, sector_interactions, boot_result)

    print("\nFig 3: Dual diminishment of agriculture...")
    fig3_land(panel_scaling)

    print("\nFig 4: SAAI & poverty traps...")
    fig4_saai(saai_df)

    print("\nFig 5: Distributional dynamics...")
    fig5_fp(saai_df, km_results)

    print("\nFig 6: Structural transformation heatmap...")
    fig6_heatmap(panel_scaling)

    print("\nFig S1: Langevin simulation validation...")
    figS1_langevin()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
