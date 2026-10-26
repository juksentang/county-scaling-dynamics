"""SAAI is standardised within each year's regression sample, so its yearly
mean is exactly zero and the mean increment of the (t, t+1) stayers is an
accounting identity in who leaves and who enters:

    mean Δξ_stay = (n_L·m_L − n_E·m_E) / n_S

Step 0 regenerates the post-window numbers of an earlier analysis of the panel (leaver mean, mean increment) and checks the identity.  Step 1 tabulates the
decomposition for every transition year.  Step 2 re-estimates the headline
MLE quantities under two composition corrections, pre and during alike:
V1 restandardises each transition year on its stayers; V2 subtracts each
year's mechanical term from the increments.
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
from scipy import stats

from src.config import FP_PERIODS
from src.data_merger import load_panel
from src.fokker_planck import build_transition_pairs
from src.km_mle import bootstrap_mle_quantities
from src.saai import compute_saai

TAB = "results/tables"
SEED = 106
SAAI_SPECS = {
    "multi":  dict(y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"]),
    "single": dict(y_col="GrossOutputValue", x_cols=["PopNum"]),
}
HEADLINE_QUANTITIES = ("xi_trap (during-pre)", "tau_mfpt (during/pre)",
                       "gamma (during-pre)", "mean_diffusion (during/pre)")


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _write_meta(path_stem: str, **kw):
    meta = {"git": _git_hash(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    with open(f"{path_stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _period_of(year: int) -> str:
    for name, (a, b) in FP_PERIODS.items():
        if a <= year and year + 1 <= b:
            return name
    return "cross"


def decompose_year(sa: pd.DataFrame, t: int) -> dict:
    """Leaver/entrant composition of the t → t+1 transition."""
    x_t = sa[sa["Year"] == t].set_index("CountyCode")["SAAI"]
    x_t1 = sa[sa["Year"] == t + 1].set_index("CountyCode")["SAAI"]
    stay = x_t.index.intersection(x_t1.index)
    leave = x_t.index.difference(x_t1.index)
    enter = x_t1.index.difference(x_t.index)
    n_s, n_l, n_e = len(stay), len(leave), len(enter)
    m_l = x_t[leave].mean() if n_l else 0.0
    m_e = x_t1[enter].mean() if n_e else 0.0
    obs = (x_t1[stay] - x_t[stay]).mean() if n_s else np.nan
    mech = (n_l * m_l - n_e * m_e) / n_s if n_s else np.nan
    row = {"year_t": t, "period": _period_of(t), "n_stay": n_s,
           "n_leave": n_l, "n_enter": n_e,
           "mean_leave_t": m_l if n_l else np.nan,
           "mean_stay_t": x_t[stay].mean() if n_s else np.nan,
           "mean_enter_t1": m_e if n_e else np.nan,
           "mean_increment_stay": obs, "mechanical_term": mech,
           "identity_residual": obs - mech}
    if n_l >= 2 and n_s >= 2:
        row["welch_p_leave_vs_stay"] = stats.ttest_ind(
            x_t[leave], x_t[stay], equal_var=False).pvalue
        row["mw_p_leave_vs_stay"] = stats.mannwhitneyu(
            x_t[leave], x_t[stay]).pvalue
    return row


def step0_anchor(saai: dict) -> pd.DataFrame:
    """Regenerate the post-window numbers of an earlier analysis under both specs."""
    rows = []
    for spec, sa in saai.items():
        d = decompose_year(sa, 2020)
        post = build_transition_pairs(sa, "post")
        rows.append({"saai_spec": spec,
                     "leaver_mean_2020_21": d["mean_leave_t"],
                     "n_leave_2020_21": d["n_leave"],
                     "post_mean_increment": post["delta_xi"].mean(),
                     "n_post_pairs": len(post),
                     'earlier_analysis_leaver_mean': -0.46,
                     'earlier_analysis_post_mean_increment': 0.059})
    return pd.DataFrame(rows)


def step1_by_year(sa: pd.DataFrame) -> pd.DataFrame:
    years = sorted(sa["Year"].unique())
    return pd.DataFrame([decompose_year(sa, t) for t in years[:-1]])


def summarise_periods(by_year: pd.DataFrame, sa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in FP_PERIODS:
        sub = by_year[by_year["period"] == period]
        pairs = build_transition_pairs(sa, period)
        w = sub["n_stay"]
        rows.append({"period": period, "n_transitions": len(sub),
                     "n_pairs": len(pairs),
                     "n_leave_total": int(sub["n_leave"].sum()),
                     "mean_leave_t_weighted": np.average(
                         sub["mean_leave_t"].fillna(0), weights=sub["n_leave"])
                     if sub["n_leave"].sum() else np.nan,
                     "mean_increment_pairs": pairs["delta_xi"].mean(),
                     "mechanical_term_weighted": np.average(
                         sub["mechanical_term"], weights=w) if w.sum() else np.nan,
                     "max_abs_mechanical_term": sub["mechanical_term"].abs().max()})
    return pd.DataFrame(rows)


def pairs_v1_restandardised(panel: pd.DataFrame, sa: pd.DataFrame,
                            spec: dict, period: str) -> pd.DataFrame:
    """Per transition year, refit SAAI on that year's stayers only."""
    a, b = FP_PERIODS[period]
    out = []
    for t in range(a, b):
        codes_t = set(sa.loc[sa["Year"] == t, "CountyCode"])
        codes_t1 = set(sa.loc[sa["Year"] == t + 1, "CountyCode"])
        stay = codes_t & codes_t1
        if len(stay) < 10:
            continue
        sub = panel[panel["Year"].isin([t, t + 1]) & panel["CountyCode"].isin(stay)]
        re = compute_saai(sub, **spec)
        out.append(build_transition_pairs(re, period, window=(t, t + 1)))
    return pd.concat(out, ignore_index=True)


def pairs_v2_offset(sa: pd.DataFrame, by_year: pd.DataFrame,
                    period: str) -> pd.DataFrame:
    """Subtract each transition year's mechanical term from its increments."""
    pairs = build_transition_pairs(sa, period).copy()
    mech = by_year.set_index("year_t")["mechanical_term"]
    pairs["delta_xi"] = pairs["delta_xi"] - pairs["Year_t"].map(mech)
    pairs["xi_t1"] = pairs["xi_t"] + pairs["delta_xi"]
    return pairs


def balanced_counts(sa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (a, b) in FP_PERIODS.items():
        s = sa[(sa["Year"] >= a) & (sa["Year"] <= b)].groupby("CountyCode")["Year"].nunique()
        rows.append({"period": period, "n_counties": len(s),
                     "n_balanced": int((s == b - a + 1).sum())})
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame) -> str:
    q = summary.set_index("quantity")
    shift, ratio = q.loc["xi_trap (during-pre)"], q.loc["tau_mfpt (during/pre)"]
    if shift["median"] >= 0 or ratio["median"] <= 1:
        return "overturned"
    if shift["ci_hi"] < 0 and ratio["ci_lo"] > 1:
        return "robust"
    return "attenuated"


def step2_variants(panel, saai, by_year, n_boot,
                   degenerate: str = "nan") -> tuple[pd.DataFrame, pd.DataFrame]:
    sa, spec = saai["multi"], SAAI_SPECS["multi"]
    variants = {
        "V0_baseline": None,
        "V1_stayer_restandardised": {p: pairs_v1_restandardised(panel, sa, spec, p)
                                     for p in ("pre", "during")},
        "V2_mechanical_offset": {p: pairs_v2_offset(sa, by_year, p)
                                 for p in ("pre", "during")},
    }
    rows, levels = [], []
    for name, pairs in variants.items():
        t0 = time.time()
        draws, summary = bootstrap_mle_quantities(sa, n_boot=n_boot, seed=SEED,
                                                  pairs=pairs, degenerate=degenerate)
        v = verdict(summary)
        print(f"  {name}: verdict={v}  ({time.time() - t0:.0f}s)")
        n_pairs = ({p: len(pairs[p]) for p in pairs} if pairs is not None else
                   {p: len(build_transition_pairs(sa, p)) for p in ("pre", "during")})
        for _, r in summary.iterrows():
            rows.append({"variant": name, "verdict": v, "degenerate": degenerate,
                         "n_pairs_pre": n_pairs["pre"],
                         "n_pairs_during": n_pairs["during"], **r.to_dict()})
        for p in ("pre", "during"):
            for k in ("xi_trap", "gamma", "mean_diffusion", "tau_mfpt"):
                col = draws[f"{k}_{p}"].dropna()
                levels.append({"variant": name, "degenerate": degenerate,
                               "period": p, "quantity": k,
                               "median": col.median(), "ci_lo": col.quantile(0.025),
                               "ci_hi": col.quantile(0.975), "n_draws": len(col)})
    return pd.DataFrame(rows), pd.DataFrame(levels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--degenerate", choices=("nan", "limit"), default="nan")
    args = ap.parse_args()
    n_boot = 100 if args.quick else args.n_boot

    panel = load_panel("panel_scaling")
    saai = {name: compute_saai(panel, **spec) for name, spec in SAAI_SPECS.items()}
    if args.degenerate == "limit":
        return main_limit(panel, saai, n_boot, args.quick)

    anchor = step0_anchor(saai)
    anchor.to_csv(f"{TAB}/expi_attrition_anchor.csv", index=False)
    print("\n=== Step 0: anchor ===\n" + anchor.to_string(index=False))

    by_year_all = []
    for spec, sa in saai.items():
        by_year = step1_by_year(sa)
        by_year.insert(0, "saai_spec", spec)
        by_year_all.append(by_year)
    by_year_all = pd.concat(by_year_all, ignore_index=True)
    worst = by_year_all["identity_residual"].abs().max()
    assert worst < 1e-10, f"accounting identity fails: {worst:.2e}"
    by_year_all.to_csv(f"{TAB}/expi_attrition_by_year.csv", index=False)

    by_year = by_year_all[by_year_all["saai_spec"] == "multi"].drop(columns="saai_spec")
    by_period = summarise_periods(by_year, saai["multi"]).merge(
        balanced_counts(saai["multi"]), on="period")
    by_period.to_csv(f"{TAB}/expi_attrition_by_period.csv", index=False)
    print("\n=== Step 1: multi, by period ===\n" + by_period.to_string(index=False))

    print(f"\n=== Step 2: variants (B={n_boot}) ===")
    variants, levels = step2_variants(panel, saai, by_year, n_boot)
    variants.to_csv(f"{TAB}/expi_attrition_variants.csv", index=False)
    levels.to_csv(f"{TAB}/expi_attrition_variant_levels.csv", index=False)
    _write_meta(f"{TAB}/expi_attrition_variants", n_boot=n_boot, seed=SEED,
                identity_max_residual=worst)
    show = variants[variants["quantity"].isin(HEADLINE_QUANTITIES)]
    print(show[["variant", "verdict", "quantity", "median", "ci_lo", "ci_hi"]]
          .to_string(index=False))


def main_limit(panel, saai, n_boot, quick):
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                           capture_output=True, text=True).stdout.strip()
    if dirty or quick:
        sys.exit("refusing: the 'limit' re-run is written only with committed code "
                 "(src/, scripts/ clean) and the full n_boot")
    by_year = step1_by_year(saai["multi"])
    worst = by_year["identity_residual"].abs().max()
    assert worst < 1e-10, f"accounting identity fails: {worst:.2e}"
    print(f"\n=== Step 2 (degenerate='limit'): variants (B={n_boot}) ===")
    variants, levels = step2_variants(panel, saai, by_year, n_boot, degenerate="limit")
    variants.to_csv(f"{TAB}/expi_attrition_variants_limit.csv", index=False)
    levels.to_csv(f"{TAB}/expi_attrition_variant_levels_limit.csv", index=False)
    _write_meta(f"{TAB}/expi_attrition_variants_limit", n_boot=n_boot, seed=SEED,
                degenerate="limit", identity_max_residual=worst,
                companion="expi_attrition_variants.csv (degenerate='nan', unchanged)")
    show = variants[variants["quantity"].isin(HEADLINE_QUANTITIES)]
    print(show[["variant", "verdict", "quantity", "median", "ci_lo", "ci_hi"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
