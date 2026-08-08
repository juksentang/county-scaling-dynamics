"""
Crossover analysis with the independent nighttime-lights urbanization
measure (u_lit = lit area / county area): urbanization-stratified scaling,
sigmoid crossover fits (raw and quantile-mapped spaces), and interaction
regressions.  Requires data/gee_exports/econp_ntl_*.csv.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data_merger import load_panel
from src.scaling import (run_urbanization_stratified_scaling,
                         fit_sigmoid_crossover, run_interaction_regression)

SECTORS = {"Tertiary": "RegGDP_Tertiary", "Secondary": "RegGDP_Secondary",
           "NonAgri": "NonAgriGDP"}


def build_panel_with_ulit() -> pd.DataFrame:
    panel = load_panel("panel_scaling")
    files = sorted(glob.glob("data/gee_exports/econp_ntl_*.csv"))
    if not files:
        sys.exit("no NTL exports in data/gee_exports/")
    ntl = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ntl = ntl.drop_duplicates(["adcode", "year"])
    ntl["u_lit"] = ntl["lit_area_t1"] / ntl["county_area"]
    panel = panel.merge(
        ntl.rename(columns={"adcode": "CountyCode", "year": "Year"})
        [["CountyCode", "Year", "u_lit"]],
        on=["CountyCode", "Year"], how="left")
    print(f"u_lit coverage: {panel['u_lit'].notna().mean():.2%}")

    m = (panel["RegGDP"] > 0) & (panel["NonAgriGDP"] > 0) & panel["u_lit"].notna()
    sub = panel[m].copy()
    u_gdp = (sub["NonAgriGDP"] / sub["RegGDP"]).clip(0, 1.5)
    sub["u_lit_q"] = np.quantile(u_gdp, sub["u_lit"].rank(pct=True).clip(0, 1))
    return panel.merge(sub[["CountyCode", "Year", "u_lit_q"]],
                       on=["CountyCode", "Year"], how="left")


def crossover_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u_col, tag in [("u_lit", "raw"), ("u_lit_q", "qmapped")]:
        strat = run_urbanization_stratified_scaling(panel, year="pooled",
                                                    n_quantiles=20, u_col=u_col)
        gs = strat["group_stats"]
        for name in SECTORS:
            s2 = strat["table"][strat["table"]["Sector"] == name].merge(
                gs[["mean_ratio"]], left_on="Group", right_index=True)
            if len(s2) < 5:
                continue
            res = fit_sigmoid_crossover(s2["mean_ratio"].values,
                                        s2["Beta"].values, s2["SE"].values)
            sig, lin = res.get("sigmoid", {}), res.get("linear", {})
            blo, bhi = s2.sort_values("mean_ratio")["Beta"].iloc[[0, -1]]
            r = dict(space=tag, sector=name,
                     beta_lo=round(blo, 3), beta_hi=round(bhi, 3))
            if "status" not in sig:
                r.update(k=round(sig["k"], 1), u_star=round(sig["u_star"], 3),
                         R2=round(sig["R2"], 3),
                         dAIC_sig_minus_lin=round(
                             sig["AIC"] - lin.get("AIC", np.nan), 1))
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    panel = build_panel_with_ulit()
    out = crossover_table(panel)
    out.to_csv("results/tables/crossover_nightlights.csv", index=False)
    print(out.to_string(index=False))

    irows = []
    for y in ["RegGDP_Secondary", "RegGDP_Tertiary", "NonAgriGDP",
              "RegGDP_Primary"]:
        r = run_interaction_regression(panel, y_col=y, u_col="u_lit")
        irows.append({"y_col": y, "beta1": r["beta1"], "se": r["se_beta1"]})
        print(f"{y}: beta1={r['beta1']:+.3f} (se {r['se_beta1']:.3f})")
    pd.DataFrame(irows).to_csv("results/tables/interaction_nightlights.csv",
                               index=False)


if __name__ == "__main__":
    main()
