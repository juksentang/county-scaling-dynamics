"""
Attribution designs E1 and E2.

E1: exit-year event study of diffusion within poverty counties
    log((dxi)^2)_it on event-time dummies (k = Year_t - ExitYear, ref k=-1),
    county FE + year FE (within transform), county-clustered SE.
E2: state-conditioned regression DiD on all counties, 2000-2019
    log((dxi)^2)_it = cubic(xi) + b1 (IsPoverty x during) [+ b2 (xi<0 x during)]
    + county FE + year FE, county-clustered SE.
"""
import os
import sys
import json
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data_merger import load_panel
from src.saai import compute_saai
from src.fokker_planck import build_transition_pairs

TAB = "results/tables"
EVENT_BINS = [-4, -3, -2, -1, 0, 1, 2]   # <=-4 and >=+2 grouped; ref = -1


def build_increments() -> pd.DataFrame:
    """Multi-factor SAAI increments on panel_scaling, the analysis panel."""
    panel = load_panel("panel_scaling")
    sa = compute_saai(panel, y_col="GrossOutputValue",
                      x_cols=["PopNum", "CultivateArea"])
    pairs = build_transition_pairs(sa, "pre", window=(2000, 2023))
    labels = panel[["Year", "CountyCode", "IsPoverty", "ExitYear"]]\
        .drop_duplicates(["Year", "CountyCode"])\
        .rename(columns={"Year": "Year_t"})
    df = pairs.merge(labels, on=["Year_t", "CountyCode"], how="left")
    df["log_dxi2"] = np.log(np.clip(df["delta_xi"] ** 2, 1e-12, None))
    df["dxi2"] = df["delta_xi"] ** 2
    return df


def within_ols(y: pd.Series, X: pd.DataFrame, county: pd.Series,
               year: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """County-FE within estimator with explicit year dummies, county-clustered SE."""
    Xf = pd.concat([X, pd.get_dummies(year, prefix="yr", drop_first=True,
                                      dtype=float)], axis=1)
    dat = pd.concat([y.rename("_y"), Xf, county.rename("_c")], axis=1).dropna()
    dm = dat.drop(columns="_c").groupby(dat["_c"]).transform(lambda g: g - g.mean())
    yd = dm["_y"]
    Xd = dm.drop(columns="_y")
    Xd = Xd.loc[:, Xd.std() > 1e-12]  # drop collinear/absorbed columns
    res = sm.OLS(yd, Xd).fit(cov_type="cluster",
                             cov_kwds={"groups": dat["_c"]})
    return res


def run_e1(df: pd.DataFrame) -> None:
    sub = df[(df["IsPoverty"] == True) & df["ExitYear"].notna()].copy()
    sub["k"] = (sub["Year_t"] - sub["ExitYear"]).clip(EVENT_BINS[0],
                                                      EVENT_BINS[-1]).astype(int)
    print(f"E1 sample: {len(sub):,} increments, "
          f"{sub['CountyCode'].nunique()} poverty counties")
    print(sub["k"].value_counts().sort_index().to_string())

    dummies = pd.DataFrame({f"ev_{k}": (sub["k"] == k).astype(float)
                            for k in EVENT_BINS if k != -1}, index=sub.index)
    rows = []
    for yname in ("log_dxi2", "dxi2"):
        res = within_ols(sub[yname], dummies, sub["CountyCode"], sub["Year_t"])
        ev_cols = [c for c in res.params.index if c.startswith("ev_")]
        wald = res.wald_test(np.eye(len(res.params))[
            [list(res.params.index).index(c) for c in ev_cols]], scalar=True)
        for c in ev_cols:
            rows.append({"depvar": yname, "term": c, "coef": res.params[c],
                         "se": res.bse[c], "p": res.pvalues[c],
                         "wald_joint_p": float(wald.pvalue),
                         "n": int(res.nobs)})
        print(f"\n--- E1 [{yname}] joint Wald p = {float(wald.pvalue):.4f} ---")
        for c in ev_cols:
            print(f"  {c:>6}: {res.params[c]:+.4f} (se {res.bse[c]:.4f}, "
                  f"p {res.pvalues[c]:.3f})")
    pd.DataFrame(rows).to_csv(f"{TAB}/exit_year_event_study.csv", index=False)


def run_e2(df: pd.DataFrame) -> None:
    sub = df[(df["Year_t"] >= 2000) & (df["Year_t"] <= 2019)].copy()
    sub["during"] = (sub["Year_t"] >= 2013).astype(float)
    sub["pov_x_during"] = (sub["IsPoverty"] == True).astype(float) * sub["during"]
    sub["neg_x_during"] = (sub["xi_t"] < 0).astype(float) * sub["during"]
    base = {"xi": sub["xi_t"], "xi2": sub["xi_t"] ** 2, "xi3": sub["xi_t"] ** 3}
    print(f"\nE2 sample: {len(sub):,} increments, "
          f"{sub['CountyCode'].nunique()} counties "
          f"({(sub['IsPoverty']==True).sum():,} poverty-county increments)")

    rows = []
    for spec, extra in (("S1", ["pov_x_during"]),
                        ("S2", ["pov_x_during", "neg_x_during"])):
        X = pd.DataFrame({**base, **{c: sub[c] for c in extra}}, index=sub.index)
        res = within_ols(sub["log_dxi2"], X, sub["CountyCode"], sub["Year_t"])
        print(f"\n--- E2 [{spec}] ---")
        for c in ["xi", "xi2", "xi3"] + extra:
            if c in res.params:
                print(f"  {c:>12}: {res.params[c]:+.4f} (se {res.bse[c]:.4f}, "
                      f"p {res.pvalues[c]:.4f})")
                rows.append({"spec": spec, "term": c, "coef": res.params[c],
                             "se": res.bse[c], "p": res.pvalues[c],
                             "n": int(res.nobs)})
    pd.DataFrame(rows).to_csv(f"{TAB}/state_conditioned_did.csv", index=False)


def main():
    os.makedirs(TAB, exist_ok=True)
    df = build_increments()
    run_e1(df)
    run_e2(df)
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git = "n/a"
    with open(f"{TAB}/attribution_designs_meta.json", "w") as f:
        json.dump({"run_commit": git,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "pipeline": "panel_scaling multi-factor SAAI"}, f, indent=2)


if __name__ == "__main__":
    main()
