"""
Compare the quantities the manuscript computes on panel_full against the same
quantities on panel_scaling so that the split between them is stated rather than assumed.

Two panels feed the paper. The scaling and interaction regressions use
panel_scaling, which is panel_full restricted to county-years with a complete
set of scaling regressors. The SAAI group statistics, the mixture models and
the land-control ladder use panel_full, because the SAAI is defined wherever
agricultural output and its inputs are observed and the narrower panel would
discard much of that coverage.

That split is defensible but the manuscript never stated it, and "we will state
it" is only an adequate answer if the two panels agree. This measures the
disagreement against thresholds fixed before the run.

Writes results/tables/panel_comparison.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from src.data_merger import load_panel
from src.saai import (compute_saai, compute_saai_single_factor,
                      compute_saai_sown_area)
from src.scaling import run_land_variable_comparison
from src.distribution import fit_gmm_timeseries

OUT = "results/tables/panel_comparison.csv"

# Fixed before the comparison was run, so a large difference could not be
# rationalised after the fact.
THRESHOLD_D = 0.10          # Cohen's d, absolute change
THRESHOLD_LADDER_SE = 1.0   # land-ladder coefficients, in units of the SE


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def with_poverty(saai_df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if "IsPoverty" in saai_df.columns:
        return saai_df
    keys = panel[["Year", "CountyCode", "IsPoverty"]].drop_duplicates()
    return saai_df.merge(keys, on=["Year", "CountyCode"], how="left")


def saai_effect_sizes(panel: pd.DataFrame) -> dict:
    """Cohen's d between poverty and non-poverty counties, per land control."""
    out = {}
    builders = {
        "population only": lambda p: compute_saai_single_factor(p),
        "+ cultivated area": lambda p: compute_saai(
            p, y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"]),
        "+ sown area": lambda p: compute_saai_sown_area(p),
    }
    for label, build in builders.items():
        try:
            sa = with_poverty(build(panel), panel)
            pov = sa[sa["IsPoverty"] == True]["SAAI"]
            npov = sa[sa["IsPoverty"] == False]["SAAI"]
            out[label] = cohens_d(pov, npov)
        except Exception as exc:                       # reported, never hidden
            out[label] = np.nan
            print(f"    SAAI[{label}] failed: {exc}")
    return out


def land_ladder(panel: pd.DataFrame) -> dict:
    """Population exponent under each land control, with its standard error."""
    land = run_land_variable_comparison(panel, year="pooled")
    sel = land[(land["Specification"] == "Pop + Land")
               & (land["Variable"] == "PopNum")]
    return {r["LandVar"]: (r["Beta"], r["SE"]) for _, r in sel.iterrows()}


def gmm_series(panel: pd.DataFrame) -> pd.Series:
    sa = compute_saai(panel, y_col="GrossOutputValue",
                      x_cols=["PopNum", "CultivateArea"])
    ts = fit_gmm_timeseries(sa, saai_col="SAAI")
    return ts.set_index("Year")["BestK"]


def main():
    full = load_panel("panel_full")
    scaling = load_panel("panel_scaling")
    rows = []

    print("Land-control ladder ...")
    lf, ls = land_ladder(full), land_ladder(scaling)
    for var in sorted(set(lf) | set(ls)):
        bf, sef = lf.get(var, (np.nan, np.nan))
        bs, ses = ls.get(var, (np.nan, np.nan))
        delta = bs - bf
        rows.append(dict(
            family="land ladder", quantity=f"beta_Pop | {var}",
            panel_full=bf, panel_scaling=bs, delta=delta, se=sef,
            threshold=f"|delta| > {THRESHOLD_LADDER_SE:g} SE",
            exceeds=bool(np.isfinite(delta) and np.isfinite(sef) and sef > 0
                         and abs(delta) > THRESHOLD_LADDER_SE * sef)))

    print("SAAI group effect sizes ...")
    df_, ds_ = saai_effect_sizes(full), saai_effect_sizes(scaling)
    for label in df_:
        delta = ds_[label] - df_[label]
        rows.append(dict(
            family="SAAI effect size", quantity=f"Cohen's d | {label}",
            panel_full=df_[label], panel_scaling=ds_[label], delta=delta,
            se=np.nan, threshold=f"|delta| > {THRESHOLD_D}",
            exceeds=bool(np.isfinite(delta) and abs(delta) > THRESHOLD_D)))

    print("GMM component counts by year ...")
    kf, ks = gmm_series(full), gmm_series(scaling)
    for year in sorted(set(kf.index) | set(ks.index)):
        a, b = kf.get(year, np.nan), ks.get(year, np.nan)
        rows.append(dict(
            family="GMM BestK", quantity=f"BestK {int(year)}",
            panel_full=a, panel_scaling=b,
            delta=(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan,
            se=np.nan, threshold="any change in K",
            exceeds=bool(not (np.isfinite(a) and np.isfinite(b) and a == b))))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}")
    print(df.round(4).to_string(index=False))
    breached = df[df["exceeds"]]
    print(f"\n{len(breached)} of {len(df)} quantities exceed the pre-registered "
          f"threshold")
    if len(breached):
        for _, r in breached.iterrows():
            print(f"  {r.family:18s} {r.quantity:34s} "
                  f"{r.panel_full} -> {r.panel_scaling}")


if __name__ == "__main__":
    main()
