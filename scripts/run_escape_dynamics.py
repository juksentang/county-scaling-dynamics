"""
Driver for the escape-dynamics analyses: pair counts, Kramers-Moyal
re-estimation with bootstrap CIs, grain-tons diffusion robustness, and
quasi-DiD estimates.

All reported pair counts are computed on panel_scaling — the analysis panel.
Counts on panel_full are reported once as a coverage check only.

Usage:
    python3 scripts/run_escape_dynamics.py --steps counts,km,grain,did
        [--n-boot 1000] [--quick]        # --quick: n_boot=200 smoke run
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data_merger import load_panel
from src.saai import compute_saai, compute_saai_single_factor, compute_saai_grain
from src.fokker_planck import (build_transition_pairs, estimate_kramers_moyal,
                               compute_fp_potential, find_drift_zeros,
                               compute_kramers_escape_time,
                               compute_escape_observables,
                               analyze_diffusion_heterogeneity)
from src.robustness import (saai_comparison, bootstrap_km_quantities,
                            bratio_did_test, bootstrap_bratio_did)
from src.config import FP_PERIODS

ROOT_SEED = 20260806
SEEDS = {"did": 101, "km_single": 102, "km_multi": 103,
         "crisis": 104, "grain": 105, "mle": 106}
TAB = "results/tables"
FIG = "results/figures"

SAAI_SPECS = {
    "multi":  dict(y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"]),
    "single": dict(y_col="GrossOutputValue", x_cols=["PopNum"]),
}


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def _write_meta(path_stem: str, **kw):
    meta = {"root_seed": ROOT_SEED, "git": _git_hash(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    with open(f"{path_stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def build_saai_panels() -> dict:
    """SAAI panels computed on panel_scaling, the analysis panel."""
    panel = load_panel("panel_scaling")
    out = {}
    for name, spec in SAAI_SPECS.items():
        out[name] = compute_saai(panel, **spec)
    out["_panel"] = panel
    return out


def count_transition_pairs(saai_panels: dict) -> pd.DataFrame:
    """Pair counts on panel_scaling (reported) plus a panel_full coverage check."""
    rows = []
    for name in ("multi", "single"):
        for period in FP_PERIODS:
            pairs = build_transition_pairs(saai_panels[name], period)
            rows.append({"input_panel": "panel_scaling", "saai_spec": name,
                         "period": period, "n_pairs": len(pairs),
                         "n_counties": pairs["CountyCode"].nunique() if len(pairs) else 0})
    # coverage check: same computation on panel_full
    panel_full = load_panel("panel_full")
    for name, spec in SAAI_SPECS.items():
        sa = compute_saai(panel_full, **spec)
        for period in FP_PERIODS:
            pairs = build_transition_pairs(sa, period)
            rows.append({"input_panel": "panel_full (audit only)", "saai_spec": name,
                         "period": period, "n_pairs": len(pairs),
                         "n_counties": pairs["CountyCode"].nunique() if len(pairs) else 0})
    df = pd.DataFrame(rows)
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(f"{TAB}/transition_pairs_by_period.csv", index=False)
    _write_meta(f"{TAB}/transition_pairs_by_period",
                note="panel_scaling rows are the pinned reporting numbers")
    print("\n=== transition-pair counts ===")
    print(df.to_string(index=False))
    return df


def _escape_row(km, spec, period, n_bins, binning) -> dict:
    old = compute_kramers_escape_time(km)
    new = compute_escape_observables(km)
    dh = analyze_diffusion_heterogeneity(km)
    zeros = find_drift_zeros(km)
    row = {"saai_spec": spec, "period": period,
           "n_bins": n_bins, "binning": binning,
           "n_pairs": km.n_pairs_total,
           "n_bins_actual": len(km.bin_centers),
           "n_reliable": int(km.reliable.sum()),
           "min_per_reliable_bin": int(km.n_per_bin[km.reliable].min()) if km.reliable.any() else 0,
           "n_stable_zeros": sum(1 for z in zeros if z["type"] == "stable"),
           "B_ratio": dh.get("ratio", np.nan),
           "B_ratio_t_p": dh.get("t_pvalue", np.nan),
           "B_ratio_mw_p": dh.get("mw_pvalue", np.nan)}
    for k in ("xi_trap", "xi_barrier", "delta_V", "delta_V_at_target",
              "mean_diffusion", "gamma", "tau_mfpt", "tau_arrhenius", "id_status"):
        row[f"new_{k}"] = new.get(k, np.nan)
    for k_old, k_out in (("xi_trap", "old_xi_trap"), ("xi_barrier", "old_xi_barrier"),
                         ("barrier_height", "old_delta_V"),
                         ("mean_diffusion", "old_mean_diffusion"),
                         ("tau_arrhenius", "old_tau_arrhenius"),
                         ("tau_mfpt", "old_tau_mfpt")):
        row[k_out] = old.get(k_old, np.nan)
    return row


def run_km_estimation(saai_panels: dict, n_boot: int) -> None:
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    # (1) single- vs multi-factor SAAI correlation
    comp = saai_comparison(saai_panels["_panel"])
    print(f"\n===  SAAI single vs multi correlation r = {comp['correlation']:.4f} "
          f"(overlap n = {comp['n_overlap']:,}) ===")

    # (2) point-estimate grid
    grid_specs = {
        "multi":  {"pre": [(25, "equal_width")],
                   "during": [(25, "equal_width")],
                   "post": [(25, "equal_width"), (11, "equal_count")]},
        "single": {"pre": [(25, "equal_width")],
                   "during": [(25, "equal_width")],
                   "post": [(11, "equal_count"), (10, "equal_count"),
                            (12, "equal_count"), (25, "equal_width")]},
    }
    rows = []
    for spec, per_period in grid_specs.items():
        for period, schemes in per_period.items():
            pairs = build_transition_pairs(saai_panels[spec], period)
            if len(pairs) == 0:
                continue
            for (nb, mode) in schemes:
                km = estimate_kramers_moyal(pairs, n_bins=nb, period=period,
                                            binning=mode)
                rows.append(_escape_row(km, spec, period, nb, mode))
    pts = pd.DataFrame(rows)
    pts.insert(0, "saai_corr_single_multi", comp["correlation"])
    pts.to_csv(f"{TAB}/km_point_estimates.csv", index=False)
    _write_meta(f"{TAB}/km_point_estimates",
                input_panel="panel_scaling", xi_target=0.0)
    cols_show = ["saai_spec", "period", "n_bins", "binning", "n_pairs",
                 "n_reliable", "new_xi_trap", "new_delta_V", "new_mean_diffusion",
                 "new_gamma", "new_tau_mfpt", "new_id_status",
                 "old_delta_V", "old_tau_arrhenius", "old_tau_mfpt", "B_ratio"]
    print("\n===  point estimates (new = corrected definitions) ===")
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(pts[cols_show].to_string(index=False))

    # (3) bootstrap: single (primary) + multi (baseline continuity)
    all_summaries = []
    for spec, seed_key in (("single", "km_single"), ("multi", "km_multi")):
        t0 = time.time()
        draws, summary = bootstrap_km_quantities(
            saai_panels[spec], n_boot=n_boot, seed=SEEDS[seed_key])
        summary.insert(0, "saai_spec", spec)
        draws.insert(0, "saai_spec", spec)
        all_summaries.append(summary)
        draws.to_parquet(f"{TAB}/km_bootstrap_draws_{spec}.parquet")
        print(f"\n--- bootstrap [{spec}] B={n_boot} done in {time.time()-t0:.0f}s ---")
    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_all.to_csv(f"{TAB}/km_bootstrap_ci.csv", index=False)
    _write_meta(f"{TAB}/km_bootstrap_ci", n_boot=n_boot,
                seeds={k: SEEDS[k] for k in ("km_single", "km_multi")},
                n_bins_by_period="pre/during 25 equal_width; post 11 equal_count")
    print("\n===  bootstrap 95% CIs ===")
    show = summary_all[summary_all["quantity"].isin(
        ["delta_V", "gamma", "tau_mfpt", "xi_trap", "mean_diffusion", "B_ratio"])]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(show.to_string(index=False))


def run_mle_estimation(saai_panels: dict, n_boot: int) -> None:
    """Parametric Euler pseudo-likelihood estimation of the same dynamics.

    Fits the cubic-drift / log-quadratic-diffusion model to the pre- and
    during-policy transition pairs, tests regime change by likelihood
    ratio, and attaches county-block bootstrap intervals to the derived
    escape observables.  Complements the binned Kramers-Moyal estimates:
    the two use the same data and differ only in whether the coefficient
    functions are smoothed parametrically.
    """
    from src.km_mle import (lrt_periods, derived_observables,
                            bootstrap_mle_quantities)

    sa = saai_panels["multi"]
    pairs = {p: build_transition_pairs(sa, p) for p in ("pre", "during")}

    res = lrt_periods(pairs)
    fit_rows = []
    for period in ("pre", "during"):
        fit = res[f"fit_{period}"]
        obs = derived_observables(fit["params"], pairs[period]["xi_t"].to_numpy())
        row = {"period": period, "n_pairs": fit["n"],
               "converged": fit["converged"], **obs}
        for i, name in enumerate(["a0", "a1", "a2", "a3", "c0", "c1", "c2"]):
            row[name] = fit["params"][i]
        fit_rows.append(row)
        print(f"  MLE {period}: dV={obs['delta_V']:.4f} "
              f"Gamma={obs['gamma']:.3f} tau={obs['tau_mfpt']:.1f}y")
    pd.DataFrame(fit_rows).to_csv(f"{TAB}/mle_fits.csv", index=False)

    lrt_rows = [{"test": k, **{kk: vv for kk, vv in res[k].items()}}
                for k in ("any_change", "drift_change", "diffusion_change")]
    pd.DataFrame(lrt_rows).to_csv(f"{TAB}/mle_lrt.csv", index=False)
    print("  LRT:", {k: f"p={res[k]['p_chi2']:.1e}"
                     for k in ("any_change", "drift_change", "diffusion_change")})

    t0 = time.time()
    draws, summary = bootstrap_mle_quantities(sa, n_boot=n_boot,
                                              seed=SEEDS["mle"])
    draws.to_parquet(f"{TAB}/mle_bootstrap_draws.parquet")
    summary.to_csv(f"{TAB}/mle_bootstrap.csv", index=False)
    _write_meta(f"{TAB}/mle_bootstrap", n_boot=n_boot, seed=SEEDS["mle"])
    print(f"  MLE bootstrap B={n_boot} in {time.time()-t0:.0f}s")


GRAIN_SPECS = {
    "grain_sown":      ["PopNum", "TotalSownArea"],   # main specification
    "grain_single":    ["PopNum"],
    "grain_grainsown": ["PopNum", "GrainSownArea"],
    "grain_cultivate": ["PopNum", "CultivateArea"],
}


def _merge_poverty(sa: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """compute_saai drops the label columns; merge them back onto the SAAI frame."""
    labels = panel[["Year", "CountyCode", "IsPoverty"]].drop_duplicates(
        ["Year", "CountyCode"])
    return sa.merge(labels, on=["Year", "CountyCode"], how="left")


def run_grain_diffusion(n_boot: int) -> None:
    """the grain-diffusion step — grain physical-quantity B-ratio and quasi-DiD estimates.

    Primary panel = panel_scaling (comparability with the value-based
    baseline, which is computed on panel_scaling); the panel_full re-run
    is reported as a larger-sample robustness row.
    """
    from src.km_mle import bootstrap_mle_quantities
    os.makedirs(TAB, exist_ok=True)

    period_rows, did_rows = [], []
    saai_main = {}
    for panel_name in ("panel_scaling", "panel_full"):
        panel = load_panel(panel_name)
        for spec, x_cols in GRAIN_SPECS.items():
            sa = _merge_poverty(compute_saai_grain(panel, x_cols=x_cols), panel)
            if len(sa) == 0:
                continue
            if spec == "grain_sown":
                saai_main[panel_name] = sa
            for period in ("pre", "during"):
                pairs = build_transition_pairs(sa, period)
                if len(pairs) < 50:
                    continue
                km = estimate_kramers_moyal(pairs, period=period)
                dh = analyze_diffusion_heterogeneity(km)
                period_rows.append({
                    "panel": panel_name, "spec": spec, "period": period,
                    "n_pairs": len(pairs),
                    "B_neg": dh.get("B_neg_mean", np.nan),
                    "B_pos": dh.get("B_pos_mean", np.nan),
                    "B_ratio": dh.get("ratio", np.nan),
                    "t_p": dh.get("t_pvalue", np.nan),
                    "mw_p": dh.get("mw_pvalue", np.nan)})
            did = bratio_did_test(sa)
            row = {"panel": panel_name, "spec": spec,
                   "did_estimate": did.get("did_estimate", np.nan)}
            for grp in ("poverty", "nonpoverty"):
                for period in ("pre", "during"):
                    cell = did.get(grp, {}).get(period, {})
                    row[f"{grp}_{period}_Bratio"] = cell.get("B_ratio", np.nan)
                    row[f"{grp}_{period}_n"] = cell.get("n_pairs", np.nan)
            did_rows.append(row)

    pd.DataFrame(period_rows).to_csv(f"{TAB}/grain_bratio_by_period.csv",
                                     index=False)
    pd.DataFrame(did_rows).to_csv(f"{TAB}/grain_bratio_did.csv", index=False)
    _write_meta(f"{TAB}/grain_bratio_by_period",
                note="primary = panel_scaling × grain_sown")
    print("\n===  grain B-ratio by period (binned KM) ===")
    print(pd.DataFrame(period_rows).to_string(index=False))
    print("\n===  quasi-DiD (point estimates) ===")
    print(pd.DataFrame(did_rows).to_string(index=False))

    # main-spec bootstrap: binned (bootstrap_km_quantities) + MLE, primary panel
    boot_rows = []
    for panel_name, sa in saai_main.items():
        seed_base = SEEDS["grain"] + (0 if panel_name == "panel_scaling" else 10)
        draws, _ = bootstrap_km_quantities(sa, n_boot=n_boot, seed=seed_base)
        ok = draws[~draws["id_status"].isin(
            ["failed", "too_few_pairs", "degenerate_path",
             "target_outside_domain"])]
        piv = ok.pivot_table(index="replicate", columns="period",
                             values="B_ratio")
        if {"pre", "during"} <= set(piv.columns):
            diff = (piv["during"] - piv["pre"]).dropna()
            boot_rows.append({
                "panel": panel_name, "estimator": "binned_KM",
                "quantity": "B_ratio during-pre",
                "median": diff.median(), "ci_lo": diff.quantile(0.025),
                "ci_hi": diff.quantile(0.975),
                "P_gt0": float((diff > 0).mean()), "n_draws": len(diff)})
        mdraws, msum = bootstrap_mle_quantities(sa, n_boot=n_boot,
                                                seed=seed_base + 1)
        sel = msum[(msum["quantity"] == "B_ratio_w (during-pre)")]
        for _, r in sel.iterrows():
            boot_rows.append({"panel": panel_name, "estimator": "MLE",
                              "quantity": r["quantity"], "median": r["median"],
                              "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                              "P_gt0": r["P_gt0"], "n_draws": r["n_draws"]})
        mdraws.to_parquet(f"{TAB}/grain_mle_draws_{panel_name}.parquet")
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(f"{TAB}/grain_diffusion_bootstrap.csv", index=False)
    _write_meta(f"{TAB}/grain_diffusion_bootstrap", n_boot=n_boot)
    print("\n===  main-spec bootstrap (B-ratio during-pre) ===")
    print(boot_df.to_string(index=False))


def run_quasi_did(n_boot: int) -> None:
    """the quasi-DiD step — ΔΔB point estimate with county-block bootstrap CI.

    Grid: {multi (primary), single (parallel), grain_sown (targeting check)}
    × {main window (during 2013–2019), COVID-inclusion variant (2013–2020)}.
    All on panel_scaling; grain also on panel_full (larger N).
    """
    os.makedirs(TAB, exist_ok=True)
    panel_s = load_panel("panel_scaling")
    variants = {"main": None,
                "during_incl_2020": {"pre": (2000, 2012), "during": (2013, 2020)}}

    inputs = {
        ("multi", "panel_scaling"): _merge_poverty(
            compute_saai(panel_s, **SAAI_SPECS["multi"]), panel_s),
        ("single", "panel_scaling"): _merge_poverty(
            compute_saai(panel_s, **SAAI_SPECS["single"]), panel_s),
        ("grain_sown", "panel_scaling"): _merge_poverty(
            compute_saai_grain(panel_s), panel_s),
    }
    panel_f = load_panel("panel_full")
    inputs[("grain_sown", "panel_full")] = _merge_poverty(
        compute_saai_grain(panel_f), panel_f)

    rows, draw_frames = [], []
    for (spec, panel_name), sa in inputs.items():
        for var_name, periods in variants.items():
            if var_name == "during_incl_2020" and spec != "multi":
                continue  # sensitivity variant on the primary spec only
            t0 = time.time()
            res = bootstrap_bratio_did(sa, n_boot=n_boot, seed=SEEDS["did"],
                                       periods=periods)
            rows.append({"spec": spec, "panel": panel_name, "variant": var_name,
                         **{k: res[k] for k in
                            ("point", "se", "ci_lo", "ci_hi", "boot_median",
                             "skew", "P_lt0", "n_used", "n_degenerate",
                             "degenerate_frac")},
                         **res["point_cells"],
                         "runtime_s": round(time.time() - t0)})
            draw_frames.append(pd.DataFrame({
                "spec": spec, "panel": panel_name, "variant": var_name,
                "ddb": res["draws"]}))
            print(f"[{spec}/{panel_name}/{var_name}] ΔΔB={res['point']:+.3f} "
                  f"SE={res['se']:.3f} CI=[{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}] "
                  f"P(<0)={res['P_lt0']:.3f} degen={res['degenerate_frac']:.1%} "
                  f"({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/quasi_did_bootstrap.csv", index=False)
    pd.concat(draw_frames, ignore_index=True).to_parquet(
        f"{TAB}/quasi_did_draws.parquet")
    _write_meta(f"{TAB}/quasi_did_bootstrap", n_boot=n_boot,
                seed=SEEDS["did"], estimator="binned KM 25 equal-width (Text S5)")
    print("\n===  ΔΔB summary ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="counts,km,grain,did")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_boot = 200 if args.quick else args.n_boot
    steps = args.steps.split(",")

    if "counts" in steps or "km" in steps:
        saai_panels = build_saai_panels()
        if "counts" in steps:
            count_transition_pairs(saai_panels)
        if "km" in steps:
            run_km_estimation(saai_panels, n_boot)
            run_mle_estimation(saai_panels, n_boot)
    if "grain" in steps:
        run_grain_diffusion(n_boot)
    if "did" in steps:
        run_quasi_did(min(n_boot * 2, 2000) if not args.quick else n_boot)
    if "crisis" in steps:
        print("[r16] not implemented")


if __name__ == "__main__":
    main()
