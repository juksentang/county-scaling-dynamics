"""Usage:
python3 scripts/run_coverage_breaks.py [--n-boot 1000]
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

from src.config import FP_PERIODS
from src.data_merger import load_panel
from src.fokker_planck import build_transition_pairs
from src.km_mle import derived_observables, fit_mle
from src.saai import compute_saai

TAB = "results/tables"
SEED = 106
SPEC = dict(y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"])
INPUTS = ["GrossOutputValue", "PopNum", "CultivateArea"]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _present(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].notna() & (df[col] > 0)


def coverage_table(full: pd.DataFrame, sa: pd.DataFrame) -> pd.DataFrame:
    f = full.assign(**{f"has_{c}": _present(full, c) for c in INPUTS})
    g = f.groupby(["ProvinceCode", "Province", "Year"])
    out = g.agg(n_counties=("CountyCode", "nunique"),
                **{f"n_{c}": (f"has_{c}", "sum") for c in INPUTS}).reset_index()
    multi = sa.groupby(["ProvinceCode", "Year"])["CountyCode"].nunique().rename("n_multi_saai")
    out = out.merge(multi, on=["ProvinceCode", "Year"], how="left")
    out["n_multi_saai"] = out["n_multi_saai"].fillna(0).astype(int)
    return out.sort_values(["ProvinceCode", "Year"])


def leaver_table(full: pd.DataFrame, scaling: pd.DataFrame, sa: pd.DataFrame) -> pd.DataFrame:
    full_idx = full.set_index(["CountyCode", "Year"])
    in_scaling = set(zip(scaling["CountyCode"], scaling["Year"]))
    rows = []
    years = sorted(sa["Year"].unique())
    for t in years[:-1]:
        x_t = sa[sa["Year"] == t].set_index("CountyCode")
        codes_t1 = set(sa.loc[sa["Year"] == t + 1, "CountyCode"])
        leave = x_t.index.difference(list(codes_t1))
        if not len(leave):
            continue
        recs = []
        for c in leave:
            if (c, t + 1) not in full_idx.index:
                reason = 'no county record in year t+1'
            else:
                r = full_idx.loc[(c, t + 1)]
                miss = [v for v in INPUTS if not (pd.notna(r[v]) and r[v] > 0)]
                reason = "missing " + "+".join(miss) if miss else "other"
            if (c, t + 1) not in in_scaling:
                reason += " (not in the scaling panel in year t+1)"
            recs.append({"CountyCode": c, "reason": reason, "xi_t": x_t.loc[c, "SAAI"],
                         "ProvinceCode": x_t.loc[c, "ProvinceCode"], "Province": x_t.loc[c, "Province"]})
        r = pd.DataFrame(recs)
        n_prov_t = x_t.groupby("ProvinceCode").size()
        n_prov_leave = r.groupby("ProvinceCode").size()
        for (pc, pn, reason), g in r.groupby(["ProvinceCode", "Province", "reason"]):
            rows.append({"year_t": t, "ProvinceCode": pc, "Province": pn, "reason": reason,
                         "n_leave": len(g), "n_leave_province": int(n_prov_leave[pc]),
                         "n_province_t": int(n_prov_t.get(pc, 0)),
                         "whole_province": int(n_prov_leave[pc]) == int(n_prov_t.get(pc, 0)),
                         "mean_xi_leave": g["xi_t"].mean(), "n_leave_year": len(r)})
    return pd.DataFrame(rows).sort_values(["year_t", "n_leave"], ascending=[True, False])


def balanced_cohort(panel: pd.DataFrame, sa: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for period, (a, b) in FP_PERIODS.items():
        s = sa[(sa["Year"] >= a) & (sa["Year"] <= b)].groupby("CountyCode")["Year"].nunique()
        codes = s.index[s == b - a + 1]
        row = {"period": period, "n_balanced": len(codes)}
        if period != "during":
            rows.append(row)
            continue
        sub = panel[panel["CountyCode"].isin(codes) & panel["Year"].between(a, b)]
        re = compute_saai(sub, **SPEC)
        pairs = build_transition_pairs(re, period)
        xi, dxi = pairs["xi_t"].to_numpy(), pairs["delta_xi"].to_numpy()
        fit = fit_mle(xi, dxi)
        point = derived_observables(fit["params"], xi)
        by_c = {c: g[["xi_t", "delta_xi"]].to_numpy() for c, g in pairs.groupby("CountyCode")}
        cs = np.array(sorted(by_c))
        rng = np.random.default_rng(SEED)
        draws, n_fail = [], 0
        for _ in range(n_boot):
            m = np.concatenate([by_c[c] for c in cs[rng.integers(0, len(cs), len(cs))]])
            obs = derived_observables(fit_mle(m[:, 0], m[:, 1], x0=fit["params"])["params"], m[:, 0])
            n_fail += obs["id_status"] != "stable_zero"
            draws.append(obs)
        dr = pd.DataFrame(draws)
        dr["two_delta_V"] = 2 * dr["delta_V"]
        point["two_delta_V"] = 2 * point["delta_V"]
        row.update({"n_pairs": len(pairs), "n_boot": n_boot, "n_not_stable_zero": int(n_fail)})
        for k in ("xi_trap", "two_delta_V", "gamma", "mean_diffusion", "tau_mfpt", "B_ratio_w"):
            col = dr[k]
            row.update({f"{k}_point": point[k], f"{k}_lo": col.quantile(0.025),
                        f"{k}_hi": col.quantile(0.975), f"{k}_n_nan": int(col.isna().sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    full = load_panel("panel_full")
    panel = load_panel("panel_scaling")
    sa = compute_saai(panel, **SPEC).merge(
        panel[["CountyCode", "Year", "ProvinceCode", "Province"]], on=["CountyCode", "Year"])

    cov = coverage_table(full, sa)
    cov.to_csv(f"{TAB}/k1_coverage_province_year.csv", index=False)
    lv = leaver_table(full, panel, sa)
    lv.to_csv(f"{TAB}/k1_leaver_provinces.csv", index=False)
    print("=== leavers, transitions with ≥ 50 leavers ===")
    big = lv[lv["n_leave_year"] >= 50]
    print(big[["year_t", "Province", "reason", "n_leave", "n_leave_province", "n_province_t",
               "whole_province",
               "mean_xi_leave"]].to_string(index=False))

    bc = balanced_cohort(panel, sa, args.n_boot)
    bc.to_csv(f"{TAB}/expi_balanced_cohort.csv", index=False)
    print("\n=== balanced cohort (report only) ===\n" + bc.T.to_string())

    have = set(zip(sa["CountyCode"], sa["Year"]))
    mask = panel[["CountyCode", "Year", "ProvinceCode"]].copy()
    mask["available"] = [k in have for k in zip(mask["CountyCode"], mask["Year"])]
    mask.to_parquet("data_clean/k1_mask.parquet", index=False)

    with open(f"{TAB}/k1_coverage_meta.json", "w") as f:
        json.dump({"git": _git_hash(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "seed": SEED, "n_boot": args.n_boot,
                   "mask_rows": int(len(mask)), "mask_available": int(mask["available"].sum())},
                  f, indent=2)


if __name__ == "__main__":
    main()
