"""
GAIA cross-validation of the independent urbanization measure.

u_imp = GAIA impervious-surface area share (imp_area / county_area), 2000-2018.
Cross-validates the u_lit (NTL lit-area share) replication:
  1. correlations u_imp vs u_GDP and u_imp vs u_lit (pooled 2000-2018)
  2. crossover replication (20-quantile stratified scaling + sigmoid fit),
     raw space and quantile-mapped-to-u_GDP space
  3. interaction regressions beta1 per sector
  4. u_lit baselines re-run RESTRICTED to the same 2000-2018 subsample

Outputs:
  results/tables/crossover_impervious.csv       (tidy: measure x space x sector)
  results/tables/urbanisation_measure_correlations.csv  (correlations)
"""
import glob
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy import stats

from src.data_merger import load_panel
from src.scaling import (run_urbanization_stratified_scaling,
                         run_interaction_regression,
                         fit_sigmoid_crossover)

YR_LO, YR_HI = 2000, 2018
SECTORS_CROSS = [("Tertiary", "RegGDP_Tertiary"),
                 ("Secondary", "RegGDP_Secondary"),
                 ("NonAgri", "NonAgriGDP")]
SECTORS_INTER = SECTORS_CROSS + [("Primary", "RegGDP_Primary")]


def load_gee(pattern: str, value_expr, exclude: tuple = ()) -> pd.DataFrame:
    files = sorted(f for f in glob.glob(pattern)
                   if os.path.basename(f) not in exclude)
    if not files:
        raise FileNotFoundError(pattern)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["adcode"] = df["adcode"].astype(int)
    df["year"] = df["year"].astype(float).astype(int)
    df = df.drop_duplicates(["adcode", "year"])
    df["_u"] = value_expr(df)
    print(f"  {pattern}: {len(files)} files, {len(df):,} rows, "
          f"{df['adcode'].nunique():,} counties, years "
          f"{df['year'].min()}-{df['year'].max()}")
    return df[["adcode", "year", "_u"]]


def qmap(u_src: np.ndarray, u_target: np.ndarray) -> np.ndarray:
    """Rank-map u_src onto the empirical distribution of u_target."""
    ranks = (stats.rankdata(u_src) - 0.5) / len(u_src)
    return np.quantile(u_target, ranks)


def crossover_block(panel: pd.DataFrame, u_col: str, measure: str,
                    space: str) -> list[dict]:
    res = run_urbanization_stratified_scaling(
        panel, year="pooled", n_quantiles=20, u_col=u_col)
    tab = res["table"].merge(
        res["group_stats"][["mean_ratio"]], left_on="Group", right_index=True)
    rows = []
    for sector, _col in SECTORS_CROSS:
        sub = tab[tab["Sector"] == sector].sort_values("mean_ratio")
        u = sub["mean_ratio"].values
        beta = sub["Beta"].values
        se = sub["SE"].values
        fit = fit_sigmoid_crossover(u, beta, se)
        sig = fit.get("sigmoid", {})
        lin = fit.get("linear", {})
        ok = "k" in sig
        rows.append({
            "measure": measure, "space": space, "sector": sector,
            "analysis": "crossover",
            "beta_first_q": beta[0], "beta_last_q": beta[-1],
            "rising": bool(beta[-1] > beta[0]),
            "sig_beta_min": sig.get("beta_min") if ok else np.nan,
            "sig_beta_max": sig.get("beta_max") if ok else np.nan,
            "sig_k": sig.get("k") if ok else np.nan,
            "sig_ustar": sig.get("u_star") if ok else np.nan,
            "sig_R2": sig.get("R2") if ok else np.nan,
            "sig_AIC": sig.get("AIC") if ok else np.nan,
            "lin_AIC": lin.get("AIC"),
            "dAIC_sig_minus_lin": (sig.get("AIC") - lin.get("AIC"))
                                  if ok else np.nan,
            "n_bins": fit.get("n"),
        })
    return rows


def interaction_block(panel: pd.DataFrame, u_col: str,
                      measure: str) -> list[dict]:
    rows = []
    for sector, ycol in SECTORS_INTER:
        r = run_interaction_regression(panel, y_col=ycol, u_col=u_col,
                                       year="pooled")
        rows.append({
            "measure": measure, "space": "raw", "sector": sector,
            "analysis": "interaction",
            "int_beta0": r["beta0"], "int_beta1": r["beta1"],
            "int_se_beta1": r["se_beta1"], "int_p_beta1": r["p_beta1"],
            "int_crossover_u": r["crossover_u"],
            "int_R2": r["R2"], "N": r["N"],
        })
    return rows


def main():
    print("[1] load GEE exports")
    # chunk5 exists in two versions: a partial 30 m export and a complete
    # re-export chunk5r100 (scale=100 m, tileScale=8, same 379 western
    # mega-counties, same schema).  Exclude any partial chunk5 CSV and use
    # the 100 m version.  County-level impervious SHARE is a ratio of two
    # areas aggregated at the same scale, so 100 m vs 30 m aggregation is
    # negligible for these mega-counties.
    gaia = load_gee("data/gee_exports/econp_gaia_chunk*.csv",
                    lambda d: d["imp_area"] / d["county_area"],
                    exclude=("econp_gaia_chunk5.csv",)
                    ).rename(columns={"_u": "u_imp"})
    r100_path = "data/gee_exports/econp_gaia_chunk5r100.csv"
    if os.path.exists(r100_path):
        n100 = len(pd.read_csv(r100_path).drop_duplicates(["adcode", "year"]))
        print(f"  scale note: {n100:,} rows (chunk5r100) aggregated at 100 m; "
              f"remaining rows at 30 m")
    ntl = load_gee("data/gee_exports/econp_ntl_*.csv",
                   lambda d: d["lit_area_t1"] / d["county_area"]
                   ).rename(columns={"_u": "u_lit"})

    print("[2] merge into panel_scaling")
    panel = load_panel("panel_scaling")
    panel = panel[(panel["Year"] >= YR_LO) & (panel["Year"] <= YR_HI)].copy()
    n_panel = len(panel)
    panel["CountyCode"] = panel["CountyCode"].astype(int)
    panel["Year"] = panel["Year"].astype(int)
    panel = panel.merge(gaia, left_on=["CountyCode", "Year"],
                        right_on=["adcode", "year"], how="left"
                        ).drop(columns=["adcode", "year"])
    panel = panel.merge(ntl, left_on=["CountyCode", "Year"],
                        right_on=["adcode", "year"], how="left"
                        ).drop(columns=["adcode", "year"])
    cov_imp = panel["u_imp"].notna().mean() * 100
    cov_lit = panel["u_lit"].notna().mean() * 100
    print(f"  panel {YR_LO}-{YR_HI}: {n_panel:,} rows; "
          f"u_imp coverage {cov_imp:.2f}%, u_lit coverage {cov_lit:.2f}%")

    # u_GDP on the same rows
    ok_gdp = (panel["RegGDP"] > 0) & (panel["NonAgriGDP"] > 0)
    panel["u_GDP"] = np.where(
        ok_gdp, (panel["NonAgriGDP"] / panel["RegGDP"]).clip(0, 1.5), np.nan)

    print("[3] correlations (pooled 2000-2018)")
    corr_rows = []
    for a, b in [("u_imp", "u_GDP"), ("u_imp", "u_lit"), ("u_lit", "u_GDP")]:
        sub = panel[[a, b]].dropna()
        pr, pp = stats.pearsonr(sub[a], sub[b])
        sr, sp = stats.spearmanr(sub[a], sub[b])
        corr_rows.append({"pair": f"{a}~{b}", "pearson": pr, "pearson_p": pp,
                          "spearman": sr, "spearman_p": sp, "N": len(sub)})
        print(f"  {a}~{b}: Pearson {pr:.3f}, Spearman {sr:.3f}, N={len(sub):,}")
    corr_df = pd.DataFrame(corr_rows)
    os.makedirs("results/tables", exist_ok=True)
    corr_df.to_csv("results/tables/urbanisation_measure_correlations.csv", index=False)

    # quantile-mapped versions (onto u_GDP distribution, merged sample)
    for src in ["u_imp", "u_lit"]:
        m = panel[src].notna() & panel["u_GDP"].notna()
        panel.loc[m, f"{src}_q"] = qmap(panel.loc[m, src].values,
                                        panel.loc[m, "u_GDP"].values)

    print("[4] crossover replication + [5/6] interactions")
    rows = []
    for measure, ucol in [("u_imp", "u_imp"), ("u_lit", "u_lit"),
                          ("u_GDP", "u_GDP")]:
        rows += crossover_block(panel, ucol, measure, "raw")
        if measure != "u_GDP":
            rows += crossover_block(panel, f"{ucol}_q", measure, "qmapped")
        rows += interaction_block(panel, ucol, measure)

    out = pd.DataFrame(rows)
    out.to_csv("results/tables/crossover_impervious.csv", index=False)
    print("\nsaved results/tables/crossover_impervious.csv")

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\n== crossover ==")
        print(out[out.analysis == "crossover"]
              .drop(columns=[c for c in out.columns if c.startswith("int_")])
              .round(3).to_string(index=False))
        print("\n== interaction ==")
        cols = ["measure", "sector", "int_beta0", "int_beta1",
                "int_se_beta1", "int_p_beta1", "int_crossover_u", "int_R2", "N"]
        print(out[out.analysis == "interaction"][cols]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
