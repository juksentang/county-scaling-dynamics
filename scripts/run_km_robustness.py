"""
Crisis-year exclusion and East/Central/West regional stratification of the
Kramers-Moyal estimates.

Input: panel_scaling (the analysis panel).
  multi-factor SAAI  = compute_saai(panel, y_col='GrossOutputValue',
                                    x_cols=['PopNum','CultivateArea'])
  single-factor SAAI = compute_saai_single_factor(panel)

Crisis exclusion: baseline vs crisis-excluded (E = CRISIS_YEARS =
{2008,09,2020,21,22}; pair (t,t+1) dropped iff t in E or t+1 in E) for
pre/during only (post is emptied by construction).  KM on 25 equal-width bins.
-> results/tables/crisis_year_exclusion.csv

Regional stratification: East/Central/West stratified KM.  Primary:
single-factor, all three periods.  Robustness: multi-factor pre (all regions)
+ during (Central/West).
Per-region bins: pre 25 equal-width; during/post 11 equal-count.
estimate_km_by_group does not expose n_bins/binning, so stratification is done
manually with build_transition_pairs per region subset.
-> results/tables/regional_km.csv
-> results/figures/regional_km.png (+ results/figures/fig_si_regional_km.png)

Usage: python3 scripts/run_km_robustness.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_merger import load_panel
from src.saai import compute_saai, compute_saai_single_factor
from src.fokker_planck import (build_transition_pairs, estimate_kramers_moyal,
                               find_drift_zeros, compute_escape_observables,
                               analyze_diffusion_heterogeneity)
from src.config import FP_PERIODS, CRISIS_YEARS
from src.plotting import setup_style

TAB = "results/tables"
FIG = "results/figures"
PAPER_FIG = "results/figures"

# Baseline pair counts pinned on panel_scaling (backward-compat guard)
PINNED_MULTI = {"pre": 5860, "during": 1734, "post": 91}

# Baseline bootstrap CIs for delta_V (multi-factor, panel_scaling pipeline)
G3_DELTAV_CI = {"pre": (-0.002, 0.077), "during": (-0.004, 0.285)}

# Figure style (plotting.py convention; during lifted to #e6a817 for
# visibility over the pale #fee090)
COLORS = {"pre": "#4575b4", "during": "#e6a817", "post": "#d73027"}
MARKERS = {"pre": "o", "during": "s", "post": "^"}
PERIOD_LABELS = {"pre": "Pre (2000-2012)", "during": "During (2013-2019)",
                 "post": "Post (2020-2023)"}

REGIONS = ["East", "Central", "West"]
MIN_PAIRS = 50   # same cell threshold as estimate_km_by_group

# Per-period bin spec for the regional analysis
REGIONAL_BINS = {"pre": (25, "equal_width"),
                 "during": (11, "equal_count"),
                 "post": (11, "equal_count")}


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def _write_meta(path_stem, **kw):
    meta = {"git": _git_hash(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_panel": "panel_scaling", **kw}
    with open(f"{path_stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def u_shape_ratio(km):
    """Diffusion U-shape ratio: mean B(|xi|>=1.5) / mean B(|xi|<=0.5) over reliable bins."""
    mask = km.reliable & ~np.isnan(km.B)
    xi, B = km.bin_centers[mask], km.B[mask]
    tails, center = B[np.abs(xi) >= 1.5], B[np.abs(xi) <= 0.5]
    if len(tails) == 0 or len(center) == 0:
        return np.nan
    return float(tails.mean() / center.mean())


def n_zeros_reliable(km):
    """Count stable/unstable drift zeros restricted to reliable bins."""
    from src.fokker_planck import KMResult
    mask = km.reliable & ~np.isnan(km.A)
    if mask.sum() < 3:
        return np.nan, np.nan
    sub = KMResult(period=km.period, bin_centers=km.bin_centers[mask],
                   A=km.A[mask], B=km.B[mask], A_se=km.A_se[mask],
                   B_se=km.B_se[mask], n_per_bin=km.n_per_bin[mask],
                   n_pairs_total=km.n_pairs_total,
                   reliable=km.reliable[mask])
    zeros = find_drift_zeros(sub)
    return (sum(1 for z in zeros if z["type"] == "stable"),
            sum(1 for z in zeros if z["type"] == "unstable"))


def km_summary_row(km):
    esc = compute_escape_observables(km)
    dh = analyze_diffusion_heterogeneity(km)
    zeros = find_drift_zeros(km)
    ns_rel, nu_rel = n_zeros_reliable(km)
    return {
        "n_pairs": km.n_pairs_total,
        "n_bins_actual": len(km.bin_centers),
        "n_reliable": int(km.reliable.sum()),
        "xi_trap": esc.get("xi_trap", np.nan),
        "delta_V": esc.get("delta_V", np.nan),
        "mean_diffusion": esc.get("mean_diffusion", np.nan),
        "gamma": esc.get("gamma", np.nan),
        "tau_mfpt": esc.get("tau_mfpt", np.nan),
        "id_status": esc.get("id_status", "failed"),
        "B_ratio": dh.get("ratio", np.nan),
        "B_neg_mean": dh.get("B_neg_mean", np.nan),
        "B_pos_mean": dh.get("B_pos_mean", np.nan),
        "B_ratio_t_p": dh.get("t_pvalue", np.nan),
        "B_ratio_mw_p": dh.get("mw_pvalue", np.nan),
        "n_stable_zeros": sum(1 for z in zeros if z["type"] == "stable"),
        "n_unstable_zeros": sum(1 for z in zeros if z["type"] == "unstable"),
        "n_stable_zeros_reliable": ns_rel,
        "n_unstable_zeros_reliable": nu_rel,
        "u_shape_ratio": u_shape_ratio(km),
    }


# ────────────────────────── crisis-year exclusion ───────────────────────────
def run_r16(saai_panels):
    print("\n" + "=" * 72)
    print("Crisis-year exclusion (E = {2008, 2009, 2020, 2021, 2022})")
    print("=" * 72)

    # backward-compatibility guard: baseline counts must equal pinned values
    for period, expected in PINNED_MULTI.items():
        n = len(build_transition_pairs(saai_panels["multi"], period))
        assert n == expected, f"baseline multi {period}: {n} != {expected}"
    print(f"[OK] baseline multi pair counts unchanged: {PINNED_MULTI}")

    rows = []
    for spec in ("multi", "single"):
        for period in ("pre", "during"):        # post emptied by the exclusion
            for variant, excl in (("baseline", frozenset()),
                                  ("excluded", CRISIS_YEARS)):
                pairs = build_transition_pairs(saai_panels[spec], period,
                                               exclude_years=excl)
                km = estimate_kramers_moyal(pairs, n_bins=25,
                                            period=f"{period}_{spec}_{variant}",
                                            binning="equal_width")
                row = {"saai_spec": spec, "period": period, "variant": variant,
                       "n_bins": 25, "binning": "equal_width",
                       **km_summary_row(km)}
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(TAB, exist_ok=True)
    out = f"{TAB}/crisis_year_exclusion.csv"
    df.to_csv(out, index=False)
    _write_meta(f"{TAB}/crisis_year_exclusion",
                crisis_years=sorted(CRISIS_YEARS),
                rule="pair (t,t+1) dropped iff t in E or t+1 in E",
                scoping="pre/during only; post emptied by exclusion",
                note="during window 2013-2019 contains no crisis year -> "
                     "excluded == baseline by construction")
    print(df.to_string(index=False))

    # ── verdict ──
    print("\n--- crisis-exclusion summary ---")
    for spec in ("multi", "single"):
        for period in ("pre", "during"):
            b = df[(df.saai_spec == spec) & (df.period == period)
                   & (df.variant == "baseline")].iloc[0]
            e = df[(df.saai_spec == spec) & (df.period == period)
                   & (df.variant == "excluded")].iloc[0]
            ushape = (e.u_shape_ratio > 1)
            bratio = (e.B_ratio < 1) == (b.B_ratio < 1)
            mono = (e.n_stable_zeros == 1)
            print(f"{spec:6s} {period:6s}: n {b.n_pairs}->{e.n_pairs} | "
                  f"U-shape {b.u_shape_ratio:.2f}->{e.u_shape_ratio:.2f} "
                  f"({'survives' if ushape else 'LOST'}) | "
                  f"B-ratio {b.B_ratio:.3f}->{e.B_ratio:.3f} "
                  f"({'same side of 1' if bratio else 'FLIPS'}) | "
                  f"stable zeros {b.n_stable_zeros}->{e.n_stable_zeros} "
                  f"({'monostable' if mono else 'NOT monostable'}) | "
                  f"dV {b.delta_V:.4f}->{e.delta_V:.4f}")
            if spec == "multi" and period in G3_DELTAV_CI:
                lo, hi = G3_DELTAV_CI[period]
                inside = lo <= e.delta_V <= hi
                print(f"       excluded delta_V {e.delta_V:.4f} "
                      f"{'WITHIN' if inside else 'OUTSIDE'} G3 baseline "
                      f"bootstrap CI [{lo}, {hi}]")
    return df


# ───────────────────────── regional stratification ──────────────────────────
def run_r13(saai_panels, region_map):
    print("\n" + "=" * 72)
    print("Regional stratification (East/Central/West)")
    print("=" * 72)

    cells = [("single", p, r) for p in FP_PERIODS for r in REGIONS]
    cells += [("multi", "pre", r) for r in REGIONS]
    cells += [("multi", "during", r) for r in ("Central", "West")]
    # multi-East-during is thin by design; compute the count for the record
    n_east_multi_during = len(build_transition_pairs(
        saai_panels["multi"][saai_panels["multi"]["Region"] == "East"], "during"))
    print(f"[note] multi-factor East-during pairs = {n_east_multi_during} "
          f"(< {MIN_PAIRS} threshold or too thin -> excluded from robustness "
          f"grid)")

    rows, km_store, km_note = [], {}, {}
    for spec, period, region in cells:
        sub = saai_panels[spec][saai_panels[spec]["Region"] == region]
        pairs = build_transition_pairs(sub, period)
        n_bins, binning = REGIONAL_BINS[period]
        base = {"saai_spec": spec, "period": period, "region": region,
                "n_bins": n_bins, "binning": binning, "note": ""}
        if len(pairs) < MIN_PAIRS:
            rows.append({**base, "n_pairs": len(pairs),
                         "id_status": "insufficient_pairs"})
            continue
        km = estimate_kramers_moyal(pairs, n_bins=n_bins,
                                    period=f"{period}_{region}_{spec}",
                                    binning=binning)
        rows.append({**base, **km_summary_row(km)})
        if spec == "single":
            km_store[(region, period)] = km
        # 10-bin sensitivity when the 11-bin cell has zero reliable
        # bins (n_pairs/11 < MIN_BIN_COUNT)
        if km.reliable.sum() == 0 and n_bins == 11:
            km10 = estimate_kramers_moyal(pairs, n_bins=10,
                                          period=f"{period}_{region}_{spec}_10b",
                                          binning="equal_count")
            rows.append({**base, "n_bins": 10,
                         "note": ("10-bin sensitivity: 11-bin cell has 0 "
                                  "reliable bins (n/11 < 30)"),
                         **km_summary_row(km10)})
            if spec == "single":
                km_store[(region, period)] = km10
                km_note[(region, period)] = " (10 bins)"

    df = pd.DataFrame(rows)
    out = f"{TAB}/regional_km.csv"
    df.to_csv(out, index=False)
    _write_meta(f"{TAB}/regional_km",
                bins_by_period={p: list(v) for p, v in REGIONAL_BINS.items()},
                min_pairs=MIN_PAIRS,
                east_multi_during_pairs=n_east_multi_during,
                note="manual stratification via build_transition_pairs per "
                     "region subset (estimate_km_by_group does not expose "
                     "n_bins/binning)")
    print(df.to_string(index=False))

    make_regional_figure(km_store, km_note)

    # ── verdict ──
    print("\n--- regional consistency summary (single-factor) ---")
    prim = df[df.saai_spec == "single"]
    for period in FP_PERIODS:
        line = []
        for region in REGIONS:
            r = prim[(prim.period == period) & (prim.region == region)]
            if r.empty or r.iloc[0].get("id_status") == "insufficient_pairs":
                line.append(f"{region}: insufficient")
                continue
            # prefer the row with reliable bins (10-bin sensitivity if the
            # 11-bin cell is fully unreliable)
            r = r.sort_values("n_reliable", ascending=False).iloc[0]
            q1 = "mono" if r.n_stable_zeros == 1 else f"{r.n_stable_zeros}-stable"
            q3 = ("U" if r.u_shape_ratio > 1 else
                  ("noU" if np.isfinite(r.u_shape_ratio) else "U:NA"))
            line.append(f"{region}: n={int(r.n_pairs)} xi*={r.xi_trap:.2f} "
                        f"B-ratio={r.B_ratio:.2f} [{q1},{q3}]")
        print(f"  {period:6s} " + " | ".join(line))
    xi_spread = prim.groupby("period")["xi_trap"].agg(lambda s: s.max() - s.min())
    print("  xi* cross-region spread by period:",
          {k: round(v, 3) for k, v in xi_spread.items()})
    return df


def make_regional_figure(km_store, km_note):
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for j, region in enumerate(REGIONS):
        ax_a, ax_b = axes[0, j], axes[1, j]
        for period in FP_PERIODS:
            km = km_store.get((region, period))
            if km is None:
                continue
            m = km.reliable & ~np.isnan(km.A)
            xi = km.bin_centers[m]
            c, mk = COLORS[period], MARKERS[period]
            n = km.n_pairs_total
            lbl = (f"{PERIOD_LABELS[period]}, n={n:,}"
                   + km_note.get((region, period), ""))
            ax_a.errorbar(xi, km.A[m], yerr=km.A_se[m], color=c, marker=mk,
                          ms=4, lw=1.6, capsize=2, elinewidth=0.8,
                          label=lbl, zorder=3)
            mb = km.reliable & ~np.isnan(km.B)
            ax_b.errorbar(km.bin_centers[mb], km.B[mb], yerr=km.B_se[mb],
                          color=c, marker=mk, ms=4, lw=1.6, capsize=2,
                          elinewidth=0.8, label=lbl, zorder=3)
        ax_a.axhline(0, color="0.4", lw=0.8, ls="--", zorder=1)
        ax_a.set_title(region)
        ax_a.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax_b.set_xlabel(r"$\xi$ (SAAI)")
        if j == 0:
            ax_a.set_ylabel(r"Drift $A(\xi)$")
            ax_b.set_ylabel(r"Diffusion $B(\xi)$")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(FIG, exist_ok=True)
    out1 = f"{FIG}/regional_km.png"
    fig.savefig(out1, dpi=300)
    os.makedirs(PAPER_FIG, exist_ok=True)
    out2 = f"{PAPER_FIG}/fig_si_regional_km.png"
    shutil.copyfile(out1, out2)
    plt.close(fig)
    print(f"\n[fig] {out1}\n[fig] {out2}")


def main():
    panel = load_panel("panel_scaling")
    print("Region value_counts (panel_scaling):")
    print(panel["Region"].value_counts().to_string())

    region_map = (panel[["CountyCode", "Region"]].drop_duplicates()
                  .set_index("CountyCode")["Region"])

    saai_panels = {
        "multi": compute_saai(panel, y_col="GrossOutputValue",
                              x_cols=["PopNum", "CultivateArea"]),
        "single": compute_saai_single_factor(panel),
    }
    # compute_saai drops the Region column -> re-attach
    for k in saai_panels:
        saai_panels[k] = saai_panels[k].copy()
        saai_panels[k]["Region"] = saai_panels[k]["CountyCode"].map(region_map)

    run_r16(saai_panels)
    run_r13(saai_panels, region_map)


if __name__ == "__main__":
    main()
