"""
Core statistics of the study, computed from the frozen panels and the
estimators in src/.

What it computes
    * panel coverage: county and row counts, poverty coverage, SAAI coverage;
    * sectoral scaling exponents, pooled and year by year, nominal and deflated;
    * land-controlled scaling specifications;
    * SAAI variants (population-only, cultivated-area, sown-area) and the
      poverty vs non-poverty group differences that go with them;
    * Gaussian-mixture fits of the yearly SAAI distribution;
    * urbanization-stratified scaling, crossover fits, and the interaction
      model with its bootstrap confidence interval;
    * Fokker-Planck transition-pair counts and diffusion heterogeneity;
    * robustness variants (fixed effects, alternative specifications and
      poverty definitions, subsamples, bootstraps);
    * descriptive statistics for both panels.

What it writes (all under results/tables/core/)
    core_statistics.csv     one row per reported quantity: label, source
                            function, value, JSON detail
    plus the intermediate tables named in each section below.

Runtime: about two minutes, dominated by the county-block bootstrap of the
crossover point (n_boot=500) and the bootstrap of the scaling exponents.

Usage: python3 scripts/run_core_statistics.py
"""
import sys, os, json, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from src.data_merger import load_panel
from src.config import SCALING_EQUATIONS
from src.scaling import (
    run_scaling_regression, run_comparative_scaling, run_scaling_timeseries,
    run_scaling_by_subsample, universality_test,
    run_land_variable_comparison, run_sectoral_decomposition,
    run_urbanization_stratified_scaling, run_interaction_regression,
    run_province_fe_scaling, run_panel_fe_scaling,
    run_province_fe_interaction, run_interaction_regression_controlled,
    fit_sigmoid_crossover, bootstrap_crossover_ci,
)
from src.saai import (
    compute_saai, compute_saai_single_factor, compute_saai_sown_area,
    compute_agri_hhi, compute_gdp_hhi, compute_order_parameter,
)
from src.distribution import fit_gmm_timeseries
from src.robustness import (
    saai_comparison, alternative_scaling_specifications,
    alternative_poverty_definitions, subsample_stability,
    sectoral_scaling_check, bootstrap_scaling_exponent,
)
from src.fokker_planck import (
    estimate_kramers_moyal_all_periods, analyze_diffusion_heterogeneity,
)

OUT = os.path.join(REPO, "results/tables/core")
os.makedirs(OUT, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT, "core_statistics.csv")

RECORDS = []
T0 = time.time()


def _conv(o):
    """JSON fallback for numpy / pandas objects in the detail field."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series,)):
        return o.to_dict()
    return str(o)


def add(label, source, value, **detail):
    """Append one quantity record (label, source function, value, JSON detail)."""
    RECORDS.append({
        "quantity_label": label,
        "source": source,
        "value": float(value) if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(float(value)) else value,
        "detail": json.dumps(detail, default=_conv) if detail else "",
    })


def write_records():
    """Write the accumulated records to core_statistics.csv (called after
    every section so a long run leaves a usable table if interrupted)."""
    pd.DataFrame(RECORDS).to_csv(SUMMARY_CSV, index=False)


def tick(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def cohens_d(pov, npov):
    """Cohen's d with pooled standard deviation (non-poverty minus poverty)."""
    pov, npov = pov.dropna(), npov.dropna()
    pooled = np.sqrt(((len(pov)-1)*pov.std()**2 + (len(npov)-1)*npov.std()**2)
                     / (len(pov)+len(npov)-2))
    return (npov.mean() - pov.mean()) / pooled


def saai_group_stats(saai_df, label, source):
    """Poverty vs non-poverty SAAI group statistics: means, Welch t,
    Mann-Whitney U, and Cohen's d. Returns Cohen's d."""
    pov = saai_df[saai_df["IsPoverty"] == True]["SAAI"]
    npov = saai_df[saai_df["IsPoverty"] == False]["SAAI"]
    t_stat, t_p = stats.ttest_ind(pov.dropna(), npov.dropna(), equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(pov.dropna(), npov.dropna(),
                                     alternative="two-sided")
    d = cohens_d(pov, npov)
    add(f"SAAI[{label}] poverty mean", source, round(pov.mean(), 6),
        std=round(pov.std(), 6), n=len(pov.dropna()))
    add(f"SAAI[{label}] nonpoverty mean", source, round(npov.mean(), 6),
        std=round(npov.std(), 6), n=len(npov.dropna()))
    add(f"SAAI[{label}] Welch t", source, round(t_stat, 4), p=t_p)
    add(f"SAAI[{label}] MannWhitney U", source, round(u_stat, 1), p=u_p)
    add(f"SAAI[{label}] Cohens d", source, round(d, 4))
    return d


# ═════════════════════════════════════════════════════════════════════
# PANEL OVERVIEW
# panel_scaling is the analysis panel (counties with the full set of
# scaling variables); panel_full is the wider sample, used for coverage
# counts and for specifications that need the extra rows.
# ═════════════════════════════════════════════════════════════════════
tick("Loading panels")
panel = load_panel("panel_scaling")
full = load_panel("panel_full")

add("counties in the full panel", "src.data_merger.load_panel",
    full["CountyCode"].nunique(), rows=len(full))
add("counties in the scaling panel", "src.data_merger.load_panel",
    panel["CountyCode"].nunique(), rows=len(panel))
add("poverty-designated counties in the full panel", "src.data_merger.load_panel",
    int(full.loc[full["IsPoverty"] == True, "CountyCode"].nunique()))
per_year_n = panel.groupby("Year").size()
add("scaling-panel sample size per year", "src.data_merger.load_panel",
    round(per_year_n.mean(), 1),
    median=per_year_n.median(), min=int(per_year_n.min()), max=int(per_year_n.max()))

# Single-factor SAAI needs positive agricultural output and population.
sf_valid = full[(full["GrossOutputValue"] > 0) & (full["PopNum"] > 0)]
add("single-factor SAAI coverage of the full panel", "src.data_merger.load_panel",
    round(len(sf_valid)/len(full), 4),
    n_valid=len(sf_valid), n_total=len(full))
# The same coverage under the other plausible denominators, since the share
# depends strongly on whether rows, counties or the scaling panel are the base.
cov_rows = len(sf_valid)/len(full)
cov_counties = sf_valid["CountyCode"].nunique()/full["CountyCode"].nunique()
cov_gov_rows = len(sf_valid)/int((full["GrossOutputValue"] > 0).sum())
cov_pop = full["PopNum"].notna().mean()
sf_scaling = panel[(panel["GrossOutputValue"] > 0) & (panel["PopNum"] > 0)]
cov_scaling = len(sf_scaling)/len(panel)
add("single-factor SAAI coverage under alternative denominators",
    "src.data_merger.load_panel",
    f"rows/full={cov_rows*100:.1f}%, counties/full={cov_counties*100:.1f}%, "
    f"rows/GOV>0-rows={cov_gov_rows*100:.1f}%, PopNum-cov={cov_pop*100:.1f}%, "
    f"scaling-panel={cov_scaling*100:.0f}%",
    rows_full=round(cov_rows, 4),
    counties=f"{sf_valid['CountyCode'].nunique()}/{full['CountyCode'].nunique()}")
write_records()


# ═════════════════════════════════════════════════════════════════════
# SECTORAL SCALING
# Writes pooled_scaling_panel_scaling.csv, scaling_timeseries_full.csv,
# yearly_betas_by_sector.csv, universality_test.csv,
# sectoral_decomposition.csv
# ═════════════════════════════════════════════════════════════════════
tick("Pooled comparative scaling (Eq 14a-14j) on the scaling panel")
results_pooled = run_comparative_scaling(panel, year="pooled")
rows = []
for eq_key, r in results_pooled.items():
    for var in r.x_cols:
        rows.append({"Equation": eq_key, "Variable": var,
                     "Beta": r.betas[var], "SE": r.se[var],
                     "Wald_beta1_p": r.wald_beta_eq_1[var]["pvalue"],
                     "R2": r.r_squared, "N": int(r.n_obs)})
        add(f"pooled beta Eq{eq_key} ({SCALING_EQUATIONS[eq_key]['y']}~{var})",
            "src.scaling.run_comparative_scaling",
            round(r.betas[var], 4), SE=round(r.se[var], 4),
            wald_beta1_p=r.wald_beta_eq_1[var]["pvalue"], R2=round(r.r_squared, 4),
            N=int(r.n_obs))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "pooled_scaling_panel_scaling.csv"), index=False)

# The headline equations refit on the wider panel, to show how much of the
# exponent is driven by the scaling-panel selection.
for eq in ["14a", "14b", "14c", "14g", "14j"]:
    cfg = SCALING_EQUATIONS[eq]
    try:
        r = run_scaling_regression(full, y_col=cfg["y"], x_cols=cfg["x"], year="pooled")
        for var in r.x_cols:
            add(f"pooled beta Eq{eq} on the full panel ({cfg['y']}~{var})",
                "src.scaling.run_scaling_regression",
                round(r.betas[var], 4), SE=round(r.se[var], 4), N=int(r.n_obs),
                R2=round(r.r_squared, 4),
                wald_beta1_p=r.wald_beta_eq_1[var]["pvalue"])
    except Exception as e:
        add(f"pooled beta Eq{eq} on the full panel",
            "src.scaling.run_scaling_regression", "FAILED", error=str(e))
write_records()

tick("Year-by-year scaling exponents")
ts_df = run_scaling_timeseries(panel)
ts_df.to_csv(os.path.join(OUT, "scaling_timeseries_full.csv"), index=False)
# Yearly betas for Primary (14j), Secondary (14h), Tertiary (14i), NonAgri (14a)
s3 = ts_df[(ts_df["Equation"].isin(["14j", "14h", "14i", "14a"])) &
           (ts_df["Variable"] == "PopNum")]
piv_raw = s3.pivot(index="Year", columns="Equation", values="Beta")
piv_b = piv_raw.round(4)
piv_se = s3.pivot(index="Year", columns="Equation", values="SE").round(4)
piv = piv_b.join(piv_se, lsuffix="_beta", rsuffix="_se")
piv = piv.rename(columns={
    "14j_beta": "Primary_beta", "14h_beta": "Secondary_beta",
    "14i_beta": "Tertiary_beta", "14a_beta": "NonAgri_beta",
    "14j_se": "Primary_se", "14h_se": "Secondary_se",
    "14i_se": "Tertiary_se", "14a_se": "NonAgri_se"})
piv.to_csv(os.path.join(OUT, "table_s3_yearly_betas.csv"))
add("yearly scaling betas by sector", "src.scaling.run_scaling_timeseries",
    "see table_s3_yearly_betas.csv",
    first_year_2000=piv.iloc[0].to_dict(), last_year=piv.iloc[-1].to_dict())

# Mean over the 24 yearly estimates: the sectoral spectrum.
for eq, lab in [("14j", "Primary"), ("14i", "Tertiary"),
                ("14h", "Secondary"), ("14a", "NonAgri")]:
    sub = ts_df[(ts_df["Equation"] == eq) & (ts_df["Variable"] == "PopNum")]
    add(f"mean of yearly betas {lab}",
        "src.scaling.run_scaling_timeseries (mean over 24 years)",
        round(sub["Beta"].mean(), 4), n_years=len(sub))
    # Same pooled exponent on the CPI-deflated (_real) outcome, to check that
    # the sectoral ordering is not a price-level artefact.
    y_real = f"RegGDP_{lab}_real" if eq != "14a" else "NonAgriGDP_real"
    r_real = run_scaling_regression(panel, y_col=y_real, x_cols=["PopNum"],
                                    year="pooled")
    add(f"pooled beta {lab} on the deflated outcome",
        "src.scaling.run_scaling_regression",
        round(r_real.betas["PopNum"], 4))

# Ordering of the sectoral exponents, year by year.
order_ok = ((piv_b["14j"] < piv_b["14i"]) & (piv_b["14i"] < piv_b["14h"])).sum()
add("years with beta_pri<beta_ter<beta_sec ordering",
    "src.scaling.run_scaling_timeseries", int(order_ok), n_years=len(piv_b))
bad = piv_raw[~((piv_raw["14j"] < piv_raw["14i"]) & (piv_raw["14i"] < piv_raw["14h"]))]
add("years violating the beta_pri<beta_ter<beta_sec ordering",
    "src.scaling.run_scaling_timeseries",
    ", ".join(str(int(y)) for y in bad.index),
    detail_2008={"Primary": float(piv_raw.loc[2008, "14j"]),
                 "Tertiary": float(piv_raw.loc[2008, "14i"]),
                 "Secondary": float(piv_raw.loc[2008, "14h"])})

uni = universality_test(ts_df)
uni.to_csv(os.path.join(OUT, "universality_test.csv"), index=False)
for eq in ["14a", "14b"]:
    sub = ts_df[(ts_df["Equation"] == eq) & (ts_df["Variable"] == "PopNum")]
    cv = sub["SE"] / sub["Beta"].abs()
    add(f"CV of yearly beta Eq{eq}", "src.scaling.universality_test",
        round(cv.mean(), 4),
        cv_std=round(cv.std(), 4), beta_mean=round(sub["Beta"].mean(), 4),
        beta_min=round(sub["Beta"].min(), 4), beta_max=round(sub["Beta"].max(), 4))
write_records()

tick("Sectoral decomposition (pooled)")
decomp = run_sectoral_decomposition(panel, year="pooled")
decomp["table"].to_csv(os.path.join(OUT, "sectoral_decomposition.csv"), index=False)
for _, row in decomp["table"].iterrows():
    add(f"pooled beta sector {row['Sector']}", "src.scaling.run_sectoral_decomposition",
        round(row["Beta_PopNum"], 4), SE=round(row["SE"], 4), N=int(row["N"]),
        R2=round(row["R2"], 4), wald_beta1_p=row["Wald_beta1_p"])
if decomp["wald_diff"] is not None:
    wd = decomp["wald_diff"]
    add("Wald beta_sec=beta_ter z", "src.scaling.run_sectoral_decomposition",
        round(wd["z_stat"], 4), p=wd["pvalue"],
        beta_sec=wd["beta1"], beta_ter=wd["beta2"])
write_records()


# ═════════════════════════════════════════════════════════════════════
# LAND CONTROLS
# Writes land_variable_comparison.csv
# ═════════════════════════════════════════════════════════════════════
tick("Land variable comparison (full panel)")
land = run_land_variable_comparison(full, year="pooled")
land.to_csv(os.path.join(OUT, "land_variable_comparison.csv"), index=False)
pop_land = land[(land["Specification"] == "Pop + Land") & (land["Variable"] == "PopNum")]
for _, row in pop_land.iterrows():
    add(f"beta_Pop controlling {row['LandVar']}",
        "src.scaling.run_land_variable_comparison",
        round(row["Beta"], 4), SE=round(row["SE"], 4), N=int(row["N"]),
        wald_beta1_p=row["Wald_beta1_p"])
land_only = land[(land["Specification"] == "Land only")]
for _, row in land_only.iterrows():
    if row["Variable"] != "N/A":
        add(f"beta land-only {row['LandVar']}",
            "src.scaling.run_land_variable_comparison",
            round(row["Beta"], 4), SE=round(row["SE"], 4), R2=round(row["R2"], 4),
            N=int(row["N"]))
write_records()


# ═════════════════════════════════════════════════════════════════════
# SAAI AND GROUP DIFFERENCES
# ═════════════════════════════════════════════════════════════════════
tick("SAAI variants and poverty/non-poverty group statistics")
# Multi-factor SAAI (population + cultivated area) on the full panel.
saai_multi = compute_saai(full, y_col="GrossOutputValue",
                          x_cols=["PopNum", "CultivateArea"])
saai_multi = saai_multi.merge(
    full[["Year", "CountyCode", "IsPoverty", "ExitYear", "Province", "Region"]].drop_duplicates(),
    on=["Year", "CountyCode"], how="left")
add("SAAI[multi/full] rows", "src.saai.compute_saai", len(saai_multi),
    counties=saai_multi["CountyCode"].nunique(),
    mean=round(saai_multi["SAAI"].mean(), 4), std=round(saai_multi["SAAI"].std(), 4))
d_multi = saai_group_stats(saai_multi, "multi-factor CultivateArea, panel_full",
                           "src.saai.compute_saai")

# Single-factor SAAI (population only): fewer covariates, far wider coverage.
saai_sf = compute_saai_single_factor(full)
saai_sf = saai_sf.merge(full[["Year", "CountyCode", "IsPoverty"]].drop_duplicates(),
                        on=["Year", "CountyCode"], how="left")
add("SAAI[single/full] rows", "src.saai.compute_saai_single_factor", len(saai_sf),
    counties=saai_sf["CountyCode"].nunique())
d_single = saai_group_stats(saai_sf, "single-factor PopNum, panel_full",
                            "src.saai.compute_saai_single_factor")

# Sown-area SAAI: land input measured by sown rather than cultivated area.
saai_sown = compute_saai_sown_area(full)
saai_sown = saai_sown.merge(full[["Year", "CountyCode", "IsPoverty"]].drop_duplicates(),
                            on=["Year", "CountyCode"], how="left")
add("SAAI[sown/full] rows", "src.saai.compute_saai_sown_area", len(saai_sown))
d_sown = saai_group_stats(saai_sown, "multi-factor TotalSownArea, panel_full",
                          "src.saai.compute_saai_sown_area")
add("Cohens d triple (single-factor / cultivated-area / sown-area SAAI)",
    "src.saai", f"{d_single:.4f}/{d_multi:.4f}/{d_sown:.4f}")

# How closely the two SAAI definitions agree on their overlapping rows.
comp = saai_comparison(full)
add("corr(SAAI_multi, SAAI_single)", "src.robustness.saai_comparison",
    round(comp["correlation"], 4), n_overlap=comp["n_overlap"],
    n_multi=comp["n_multi"], n_single=comp["n_single"])
write_records()

tick("HHI structural validation")
# Structural interpretation of SAAI: does a large |SAAI| go with a more
# concentrated sectoral mix? Computed on the scaling panel, where the
# four agricultural sub-sectors are populated.
panel_sc = panel
saai_sc = compute_saai_single_factor(panel_sc)
saai_sc = saai_sc.merge(
    panel_sc[["Year", "CountyCode", "IsPoverty", "Region",
              "Farming_real", "Forestry_real", "AnimalHusbandry_real", "Fishery_real",
              "RegGDP_Primary_real", "RegGDP_Secondary_real",
              "RegGDP_Tertiary_real"]].drop_duplicates(),
    on=["Year", "CountyCode"], how="left")
saai_sc["HHI_agri"] = compute_agri_hhi(saai_sc)
saai_sc["HHI_gdp"] = compute_gdp_hhi(saai_sc)
saai_sc["abs_SAAI"] = saai_sc["SAAI"].abs()
for hhi_col, lab, src_fn in [("HHI_gdp", "GDP 3-sector", "src.saai.compute_gdp_hhi"),
                             ("HHI_agri", "Agri 4-sector", "src.saai.compute_agri_hhi")]:
    sub = saai_sc[["SAAI", "abs_SAAI", hhi_col, "IsPoverty"]].dropna()
    sub["grp"] = pd.qcut(sub["abs_SAAI"], 3, labels=["Low", "Mid", "High"])
    h_stat, h_p = kruskal(*[g[hhi_col].values for _, g in sub.groupby("grp", observed=True)])
    rho_abs, p_abs = spearmanr(sub["abs_SAAI"], sub[hhi_col])
    rho_sgn, p_sgn = spearmanr(sub["SAAI"], sub[hhi_col])
    pov_h = sub.loc[sub["IsPoverty"] == True, hhi_col]
    npov_h = sub.loc[sub["IsPoverty"] == False, hhi_col]
    _, p_mw = mannwhitneyu(pov_h, npov_h, alternative="two-sided")
    add(f"HHI {lab}: Kruskal H", src_fn, round(h_stat, 1), p=h_p, N=len(sub))
    add(f"HHI {lab}: Spearman(|SAAI|,HHI)", src_fn, round(rho_abs, 4), p=p_abs)
    add(f"HHI {lab}: Spearman(SAAI,HHI) signed", src_fn, round(rho_sgn, 4), p=p_sgn)
    add(f"HHI {lab}: poverty vs nonpoverty mean", src_fn,
        f"{pov_h.mean():.4f} vs {npov_h.mean():.4f}", mw_p=p_mw)
write_records()

tick("Cohen's d by urbanization tercile")
# The poverty/non-poverty SAAI gap within terciles of the urbanization
# ratio, so that the gap is not simply a level-of-development difference.
psa = compute_saai_single_factor(full)
psa = psa.merge(full[["Year", "CountyCode", "IsPoverty", "RegGDP", "NonAgriGDP"]].drop_duplicates(),
                on=["Year", "CountyCode"], how="left")
psa["urb_ratio"] = (psa["NonAgriGDP"] / psa["RegGDP"]).clip(0, 1.5)
psa = psa.dropna(subset=["urb_ratio", "SAAI", "IsPoverty"])
psa["UrbGroup"] = pd.qcut(psa["urb_ratio"], 3, labels=["Low", "Med", "High"],
                          duplicates="drop")
for grp in ["Low", "Med", "High"]:
    pov = psa[(psa["UrbGroup"] == grp) & (psa["IsPoverty"] == True)]["SAAI"]
    npov = psa[(psa["UrbGroup"] == grp) & (psa["IsPoverty"] == False)]["SAAI"]
    add(f"Cohens d urb group {grp} (single-factor SAAI, full)",
        "src.saai.compute_saai_single_factor",
        round(cohens_d(pov, npov), 4),
        pov_mean=round(pov.mean(), 4), npov_mean=round(npov.mean(), 4),
        n_pov=len(pov), n_npov=len(npov))
write_records()


# ═════════════════════════════════════════════════════════════════════
# MIXTURE MODELS
# Writes gmm_timeseries_multi.csv, gmm_timeseries_sown.csv,
# order_parameter_multi.csv
# ═════════════════════════════════════════════════════════════════════
tick("Gaussian mixture fits of the yearly SAAI distribution")
gmm_ts = fit_gmm_timeseries(saai_multi, saai_col="SAAI", max_k=3)
gmm_ts.to_csv(os.path.join(OUT, "gmm_timeseries_multi.csv"), index=False)
add("GMM BestK by year (multi)", "src.distribution.fit_gmm_timeseries",
    json.dumps(dict(zip(gmm_ts["Year"].astype(int), gmm_ts["BestK"].astype(int)))),
    years_k2=[int(y) for y, k in zip(gmm_ts["Year"], gmm_ts["BestK"]) if k == 2])
gmm_sown = fit_gmm_timeseries(saai_sown, saai_col="SAAI", max_k=3)
gmm_sown.to_csv(os.path.join(OUT, "gmm_timeseries_sown.csv"), index=False)
add("GMM BestK by year (sown)", "src.distribution.fit_gmm_timeseries", json.dumps(
    dict(zip(gmm_sown["Year"].astype(int), gmm_sown["BestK"].astype(int)))))
# Component means in the early years, where the distribution is bimodal.
early = gmm_ts[gmm_ts["Year"] <= 2005]
if "mu1" in early.columns and "mu2" in early.columns:
    add("GMM early (2000-2005) component means", "src.distribution.fit_gmm_timeseries",
        f"mu1={early['mu1'].mean():.3f}, mu2={early['mu2'].mean():.3f}",
        per_year=early[["Year", "mu1", "mu2", "BestK"]].to_dict("records"))

op = compute_order_parameter(saai_multi)
op.to_csv(os.path.join(OUT, "order_parameter_multi.csv"), index=False)
add("order parameter m(t) multi", "src.saai.compute_order_parameter",
    "see order_parameter_multi.csv",
    m_pov_2000=float(op.iloc[0]["m_poverty"]),
    m_pov_last=float(op.iloc[-1]["m_poverty"]),
    mean_cohens_d=round(op["cohens_d"].mean(), 4))
write_records()


# ═════════════════════════════════════════════════════════════════════
# INTERACTION AND CROSSOVER
# Writes urb_stratified_3group.csv, urb_3group_stats.csv,
# urb_stratified_20q.csv, urb_20q_stats.csv, interaction_ols.csv,
# interaction_prov_fe.csv, bootstrap_u_stars.npy
# ═════════════════════════════════════════════════════════════════════
tick("Urbanization-stratified scaling: 3 groups and 20 quantiles")
strat3 = run_urbanization_stratified_scaling(panel, year="pooled")
strat3["table"].to_csv(os.path.join(OUT, "urb_stratified_3group.csv"), index=False)
strat3["group_stats"].to_csv(os.path.join(OUT, "urb_3group_stats.csv"))
for _, row in strat3["table"].iterrows():
    add(f"urb3 beta {row['Sector']}[{row['Group']}]",
        "src.scaling.run_urbanization_stratified_scaling",
        round(row["Beta"], 4), SE=round(row["SE"], 4), N=int(row["N"]))
for sector, wald in strat3["wald_cross_group"].items():
    add(f"urb3 Wald High-vs-Low {sector}",
        "src.scaling.run_urbanization_stratified_scaling", round(wald["z_stat"], 4),
        p=wald["pvalue"], beta_high=wald["beta1"], beta_low=wald["beta2"])

strat20 = run_urbanization_stratified_scaling(panel, year="pooled", n_quantiles=20)
strat20["table"].to_csv(os.path.join(OUT, "urb_stratified_20q.csv"), index=False)
strat20["group_stats"].to_csv(os.path.join(OUT, "urb_20q_stats.csv"))

# Empirical crossover: the urbanization level u at which beta crosses 1,
# located by the sign change of beta-1 between adjacent quantile bins
# ordered by their median urbanization ratio.
gs = strat20["group_stats"].reset_index()
empirical_crossover = {}
for sector in ["NonAgri", "Tertiary"]:
    rows_ = strat20["table"][strat20["table"]["Sector"] == sector].merge(
        gs[["UrbGroup", "median_ratio"]], left_on="Group", right_on="UrbGroup",
        how="left").sort_values("median_ratio")
    betas = rows_["Beta"].values
    cross_idx = np.where(np.diff(np.sign(betas - 1)))[0]
    u_cross = rows_["median_ratio"].values[cross_idx[0]] if len(cross_idx) else np.nan
    empirical_crossover[sector] = u_cross
    add(f"20q empirical crossover u ({sector})",
        "src.scaling.run_urbanization_stratified_scaling",
        round(float(u_cross), 4) if np.isfinite(u_cross) else "none",
        n_crossings=len(cross_idx))

# Same crossing from a LOWESS smooth (frac=0.4) of beta against the median
# urbanization ratio, linearly interpolated between the bracketing points.
rows_ter = strat20["table"][strat20["table"]["Sector"] == "Tertiary"].merge(
    gs[["UrbGroup", "median_ratio"]], left_on="Group", right_on="UrbGroup",
    how="left").sort_values("median_ratio")
x, y = rows_ter["median_ratio"].values, rows_ter["Beta"].values
sm_curve = sm_lowess(y, x, frac=0.4, return_sorted=True)
cross = np.where(np.diff(np.sign(sm_curve[:, 1] - 1)))[0]
if len(cross):
    x0, x1 = sm_curve[cross[0], 0], sm_curve[cross[0] + 1, 0]
    y0, y1 = sm_curve[cross[0], 1], sm_curve[cross[0] + 1, 1]
    u_lowess = x0 + (1 - y0) * (x1 - x0) / (y1 - y0)
else:
    u_lowess = np.nan
add("LOWESS beta=1 crossing, Tertiary 20q",
    "statsmodels lowess (frac=0.4), crossing interpolated",
    round(float(u_lowess), 4) if np.isfinite(u_lowess) else "none",
    quantile_signchange_crossover=round(float(empirical_crossover["Tertiary"]), 4))
write_records()

tick("Sigmoid / tanh / linear crossover fits")
for sector_name in ["Tertiary", "Secondary", "NonAgri"]:
    sub = strat20["table"][strat20["table"]["Sector"] == sector_name].copy()
    sub = sub.merge(strat20["group_stats"][["mean_ratio"]], left_on="Group",
                    right_index=True, how="left")
    res = fit_sigmoid_crossover(sub["mean_ratio"].values, sub["Beta"].values,
                                sub["SE"].values)
    detail = {}
    for model in ["sigmoid", "tanh", "linear"]:
        m = res.get(model, {})
        detail[model] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                         for k, v in m.items() if not isinstance(v, np.ndarray)}
    s = res.get("sigmoid", {})
    val = (f"u*={s['u_star']:.4f}, k={s['k']:.2f}" if "u_star" in s
           else str(s.get("status", "no-fit")))
    add(f"sigmoid crossover {sector_name}", "src.scaling.fit_sigmoid_crossover",
        val, **detail)
write_records()

tick("Interaction regressions: 4 sectors, OLS, scaling panel")
sectors = {"Tertiary": "RegGDP_Tertiary", "Secondary": "RegGDP_Secondary",
           "Primary": "RegGDP_Primary", "NonAgri": "NonAgriGDP"}
ols_int = {}
irows = []
for name, col in sectors.items():
    r = run_interaction_regression(panel, y_col=col)
    ols_int[name] = r
    irows.append({"Sector": name, "beta0": r["beta0"], "se_beta0": r["se_beta0"],
                  "beta1": r["beta1"], "se_beta1": r["se_beta1"],
                  "p_beta1": r["p_beta1"], "u_star": r["crossover_u"],
                  "R2": r["R2"], "N": r["N"]})
    add(f"interaction OLS {name}: beta1", "src.scaling.run_interaction_regression",
        round(r["beta1"], 4), beta0=round(r["beta0"], 4),
        se_beta0=round(r["se_beta0"], 4), se_beta1=round(r["se_beta1"], 4),
        p_beta1=r["p_beta1"], u_star=round(r["crossover_u"], 4),
        R2=round(r["R2"], 4), N=r["N"])
pd.DataFrame(irows).to_csv(os.path.join(OUT, "interaction_ols.csv"), index=False)

# The non-agricultural slope under alternative outcome definitions
# (nominal vs deflated, non-agricultural vs total GDP).
alt = {}
for y_col in ["NonAgriGDP", "NonAgriGDP_real", "RegGDP", "RegGDP_real"]:
    r = run_interaction_regression(panel, y_col=y_col)
    alt[y_col] = {"beta0": round(r["beta0"], 4), "beta1": round(r["beta1"], 4),
                  "u_star": round(r["crossover_u"], 4), "R2": round(r["R2"], 4)}
add("interaction beta1 under alternative outcome columns",
    "src.scaling.run_interaction_regression",
    "; ".join(f"{k} {v['beta1']:.4f}" for k, v in alt.items()), **alt)

# Interaction model with year fixed effects (HC1 standard errors).
d = panel[(panel["RegGDP"] > 0) & (panel["NonAgriGDP"] > 0)].copy()
d["_u"] = (d["NonAgriGDP"] / d["RegGDP"]).clip(0, 1.5)
yfe = {}
for lab, ycol in [("Tertiary", "RegGDP_Tertiary"), ("Secondary", "RegGDP_Secondary"),
                  ("NonAgri", "NonAgriGDP")]:
    dd = d[(d[ycol] > 0) & (d["PopNum"] > 0)].dropna(subset=[ycol, "PopNum", "_u"])
    yv = np.log(dd[ycol]); ln = np.log(dd["PopNum"])
    yd = pd.get_dummies(dd["Year"].astype(int), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(dd)), ln, dd["_u"] * ln, yd])
    f = sm.OLS(yv, X).fit(cov_type="HC1")
    yfe[lab] = {"beta0": round(f.params.iloc[1], 4), "beta1": round(f.params.iloc[2], 4),
                "R2": round(f.rsquared, 4)}
add("interaction with year fixed effects", "statsmodels OLS (HC1) in this script",
    json.dumps(yfe))
write_records()

tick("Province fixed-effects interaction")
ferows = []
for name, col in sectors.items():
    try:
        r = run_province_fe_interaction(panel, y_col=col)
        ferows.append({"Sector": name, "beta1": r["beta1"], "se_beta1": r["se_beta1"],
                       "p_beta1": r["p_beta1"], "u_star": r["crossover_u"],
                       "R2": r["R2"], "N": r["N"]})
        add(f"interaction ProvFE {name}: beta1",
            "src.scaling.run_province_fe_interaction",
            round(r["beta1"], 4), se_beta1=round(r["se_beta1"], 4),
            p_beta1=r["p_beta1"],
            u_star=(round(r["crossover_u"], 4) if np.isfinite(r["crossover_u"]) else None),
            R2=round(r["R2"], 4), N=r["N"])
    except Exception as e:
        add(f"interaction ProvFE {name}", "src.scaling.run_province_fe_interaction",
            "FAILED", error=str(e))
pd.DataFrame(ferows).to_csv(os.path.join(OUT, "interaction_prov_fe.csv"), index=False)

tick("Interaction with quadratic population control")
for name in ["Tertiary", "Secondary", "NonAgri"]:
    col = sectors[name]
    try:
        ctrl = run_interaction_regression_controlled(panel, y_col=col)
        b_lin = ols_int[name]["beta1"]
        add(f"quadratic control {name}: beta1",
            "src.scaling.run_interaction_regression_controlled",
            round(ctrl["beta1"], 4), beta1_linear=round(b_lin, 4),
            delta=round(ctrl["beta1"] - b_lin, 4), p_beta1=ctrl["p_beta1"],
            beta2=round(ctrl["beta2"], 4), p_beta2=ctrl["p_beta2"],
            u_star=round(ctrl["crossover_u"], 4))
    except Exception as e:
        add(f"quadratic control {name}",
            "src.scaling.run_interaction_regression_controlled", "FAILED", error=str(e))

# Grain output is a physical quantity: it should not respond to the
# urbanization interaction the way nominal sectoral GDP does.
# panel_scaling, like every other interaction regression above: panel_full is
# the coverage panel, and using it here gives a slope that disagrees with the
# figure, which reads the analysis panel.
gr = run_interaction_regression(panel, y_col="GrainOutput")
add("interaction OLS GrainOutput: beta1 (placebo)",
    "src.scaling.run_interaction_regression",
    round(gr["beta1"], 4), p_beta1=gr["p_beta1"], beta0=round(gr["beta0"], 4),
    R2=round(gr["R2"], 4), N=gr["N"])
grain_r = run_scaling_regression(panel, y_col="GrainOutput", x_cols=["PopNum"], year="pooled")
add("pooled beta GrainOutput~PopNum (analysis panel)", "src.scaling.run_scaling_regression",
    round(grain_r.betas["PopNum"], 4), SE=round(grain_r.se["PopNum"], 4),
    N=int(grain_r.n_obs), R2=round(grain_r.r_squared, 4))
write_records()

# County-block bootstrap of the crossover point: resampling is by county,
# so all years of a resampled county enter together.
tick("Bootstrap crossover CI (RegGDP_Tertiary, n_boot=500, seed=42)")
t_bs = time.time()
boot = bootstrap_crossover_ci(panel, y_col="RegGDP_Tertiary", n_boot=500, seed=42)
add("bootstrap crossover CI u*_ter", "src.scaling.bootstrap_crossover_ci",
    f"[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]",
    mean=round(boot["mean_u_star"], 4), median=round(boot["median_u_star"], 4),
    n_valid=int(boot["n_valid"]), n_boot=int(boot["n_boot"]),
    wall_seconds=round(time.time() - t_bs, 1))
np.save(os.path.join(OUT, "bootstrap_u_stars.npy"), boot["u_stars"])
write_records()


# ═════════════════════════════════════════════════════════════════════
# TRANSITION DYNAMICS (FOKKER-PLANCK)
# ═════════════════════════════════════════════════════════════════════
tick("Transition pairs and diffusion heterogeneity")
# Multi-factor SAAI on the scaling panel (the panel used for the drift and
# diffusion estimates), with county metadata merged in from the full panel.
saai_fp = compute_saai(panel, y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"])
saai_fp = saai_fp.merge(
    full[["Year", "CountyCode", "IsPoverty", "ExitYear", "Province", "Region",
          "CountyLng", "CountyLat", "CountyName", "RegGDP", "NonAgriGDP",
          "RegGDP_Secondary", "RegGDP_Tertiary"]].drop_duplicates(),
    on=["Year", "CountyCode"], how="left")
add("SAAI[multi/scaling panel] rows", "src.saai.compute_saai", len(saai_fp),
    counties=saai_fp["CountyCode"].nunique())
km_all = estimate_kramers_moyal_all_periods(saai_fp, saai_col="SAAI")
for period, km in km_all.items():
    add(f"FP transition pairs [{period}]",
        "src.fokker_planck.estimate_kramers_moyal_all_periods", int(km.n_pairs_total),
        reliable_bins=int(km.reliable.sum()), n_bins=len(km.reliable))
    dh = analyze_diffusion_heterogeneity(km)
    if "status" not in dh:
        add(f"diffusion B-ratio [{period}]",
            "src.fokker_planck.analyze_diffusion_heterogeneity", round(dh["ratio"], 3),
            B_neg=round(dh["B_neg_mean"], 4), B_pos=round(dh["B_pos_mean"], 4),
            t=round(dh["t_stat"], 3), t_p=round(dh["t_pvalue"], 4),
            mw_p=round(dh["mw_pvalue"], 4))
    else:
        add(f"diffusion B-ratio [{period}]",
            "src.fokker_planck.analyze_diffusion_heterogeneity", dh["status"])
write_records()


# ═════════════════════════════════════════════════════════════════════
# ROBUSTNESS VARIANTS
# Writes alt_scaling_specs.csv, alt_poverty_definitions.csv,
# subsample_stability.csv, sectoral_scaling_check.csv,
# scaling_by_region.csv, scaling_by_poverty.csv,
# bootstrap_scaling_exponents.csv
# ═════════════════════════════════════════════════════════════════════
tick("Province and county fixed-effects scaling (14a/14b/14c)")
for eq_key in ["14a", "14b", "14c"]:
    cfg = SCALING_EQUATIONS[eq_key]
    ols = run_scaling_regression(panel, y_col=cfg["y"], x_cols=cfg["x"], year="pooled")
    try:
        pfe = run_province_fe_scaling(panel, eq_key, year="pooled")
    except Exception as e:
        pfe = None
        add(f"ProvFE scaling Eq{eq_key}", "src.scaling.run_province_fe_scaling",
            "FAILED", error=str(e))
    try:
        cfe = run_panel_fe_scaling(panel, eq_key)
    except Exception as e:
        cfe = None
        add(f"CountyFE scaling Eq{eq_key}", "src.scaling.run_panel_fe_scaling",
            "FAILED", error=str(e))
    for var in cfg["x"]:
        add(f"scaling Eq{eq_key} {var}: OLS/ProvFE/CountyFE",
            "src.scaling.run_province_fe_scaling, run_panel_fe_scaling",
            f"{ols.betas[var]:.4f}/" +
            (f"{pfe['betas'][var]:.4f}" if pfe else "NA") + "/" +
            (f"{cfe['betas'][var]:.4f}" if cfe else "NA"),
            ols_se=round(ols.se[var], 4),
            provfe_se=(round(pfe["se"][var], 4) if pfe else None),
            countyfe_se=(round(cfe["se"][var], 4) if cfe else None),
            provfe_R2=(round(pfe["R2"], 4) if pfe else None),
            countyfe_R2within=(round(cfe["R2_within"], 4) if cfe else None))
write_records()

tick("Alternative specifications, poverty definitions, subsamples")
alt_specs = alternative_scaling_specifications(panel)
alt_specs.to_csv(os.path.join(OUT, "alt_scaling_specs.csv"), index=False)
add("alternative scaling specs", "src.robustness.alternative_scaling_specifications",
    "see alt_scaling_specs.csv", n_specs=len(alt_specs))

# Poverty status redefined from per-capita income rather than the official
# designation, on the multi-factor SAAI sample.
saai_r = compute_saai(full, y_col="GrossOutputValue", x_cols=["PopNum", "CultivateArea"])
saai_r = saai_r.merge(full[["Year", "CountyCode", "IsPoverty", "PerRegGDP"]].drop_duplicates(),
                      on=["Year", "CountyCode"], how="left")
try:
    alt_pov = alternative_poverty_definitions(saai_r)
    alt_pov.to_csv(os.path.join(OUT, "alt_poverty_definitions.csv"), index=False)
    add("alternative poverty definitions",
        "src.robustness.alternative_poverty_definitions",
        "see alt_poverty_definitions.csv", table=alt_pov.round(4).to_dict("records"))
except Exception as e:
    add("alternative poverty definitions",
        "src.robustness.alternative_poverty_definitions", "FAILED", error=str(e))

sub_st = subsample_stability(panel)
sub_st.to_csv(os.path.join(OUT, "subsample_stability.csv"), index=False)
add("subsample stability", "src.robustness.subsample_stability",
    "see subsample_stability.csv", table=sub_st.round(4).to_dict("records"))

sect = sectoral_scaling_check(panel)
sect.to_csv(os.path.join(OUT, "sectoral_scaling_check.csv"), index=False)

reg_sub = run_scaling_by_subsample(panel, "Region", year="pooled")
reg_sub.to_csv(os.path.join(OUT, "scaling_by_region.csv"), index=False)
pov_sub = run_scaling_by_subsample(panel, "IsPoverty", year="pooled")
pov_sub.to_csv(os.path.join(OUT, "scaling_by_poverty.csv"), index=False)
add("scaling by region/poverty subsample", "src.scaling.run_scaling_by_subsample",
    "see scaling_by_region.csv / scaling_by_poverty.csv")
write_records()

tick("Bootstrap scaling exponents (n_iterations=200)")
try:
    boot_res = bootstrap_scaling_exponent(panel, n_iterations=200)
    boot_res.to_csv(os.path.join(OUT, "bootstrap_scaling_exponents.csv"), index=False)
    add("bootstrap scaling exponents", "src.robustness.bootstrap_scaling_exponent",
        "see bootstrap_scaling_exponents.csv", note="n_iterations=200")
except Exception as e:
    add("bootstrap scaling exponents", "src.robustness.bootstrap_scaling_exponent",
        "FAILED", error=str(e))
write_records()


# ═════════════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS
# Writes descriptive_stats_panel_full.csv, descriptive_stats_panel_scaling.csv
# Both nominal and deflated (_real) GDP columns are reported, because the
# scaling regressions use the nominal series and the deflated series are the
# robustness counterpart.
# ═════════════════════════════════════════════════════════════════════
tick("Descriptive statistics (full panel)")
full_ds = full.copy()
mask_u = (full_ds["RegGDP"] > 0) & (full_ds["NonAgriGDP"] > 0)
full_ds["urb_ratio"] = np.where(mask_u, (full_ds["NonAgriGDP"] / full_ds["RegGDP"]).clip(0, 1.5), np.nan)
desc_vars = ["PopNum",
             "RegGDP_Primary", "RegGDP_Primary_real",
             "RegGDP_Secondary", "RegGDP_Secondary_real",
             "RegGDP_Tertiary", "RegGDP_Tertiary_real",
             "GrossOutputValue", "GrossOutputValue_real",
             "CultivateArea", "TotalSownArea", "GrainOutput", "urb_ratio"]
drecs = []
for v in desc_vars:
    s = full_ds[v]
    drecs.append({"Variable": v,
                  "Coverage": round(s.notna().mean(), 4),
                  "Mean": s.mean(), "SD": s.std(), "Min": s.min(),
                  "Median": s.median(), "Max": s.max(), "N": int(s.notna().sum())})
ddf = pd.DataFrame(drecs)
ddf.to_csv(os.path.join(OUT, "descriptive_stats_panel_full.csv"), index=False)
add("descriptive statistics (full panel)", "scripts/run_core_statistics.py",
    "see descriptive_stats_panel_full.csv",
    PopNum={"cov": float(ddf.loc[0, "Coverage"]), "mean": float(ddf.loc[0, "Mean"]),
            "sd": float(ddf.loc[0, "SD"]), "median": float(ddf.loc[0, "Median"]),
            "max": float(ddf.loc[0, "Max"])},
    urb_ratio=ddf[ddf["Variable"] == "urb_ratio"].iloc[0].to_dict())

tick("Descriptive statistics (scaling panel)")
ps = panel.copy()
mask_u = (ps["RegGDP"] > 0) & (ps["NonAgriGDP"] > 0)
ps["urb_ratio"] = np.where(mask_u, (ps["NonAgriGDP"] / ps["RegGDP"]).clip(0, 1.5), np.nan)
dvars = ["PopNum", "RegGDP_Primary", "RegGDP_Primary_real", "RegGDP_Secondary",
         "RegGDP_Secondary_real", "RegGDP_Tertiary", "RegGDP_Tertiary_real",
         "CultivateArea", "TotalSownArea", "GrainOutput", "urb_ratio"]
drecs = []
for v in dvars:
    s = ps[v]
    drecs.append({"Variable": v, "Coverage": round(s.notna().mean(), 4),
                  "Mean": s.mean(), "SD": s.std(), "Min": s.min(),
                  "Median": s.median(), "Max": s.max()})
pd.DataFrame(drecs).to_csv(os.path.join(OUT, "descriptive_stats_panel_scaling.csv"),
                           index=False)
add("descriptive statistics (scaling panel)", "scripts/run_core_statistics.py",
    "see descriptive_stats_panel_scaling.csv",
    PopNum={"cov": drecs[0]["Coverage"], "mean": round(drecs[0]["Mean"], 1),
            "sd": round(drecs[0]["SD"], 1), "max": round(drecs[0]["Max"], 1)})
write_records()

tick(f"Done. {len(RECORDS)} records written to {SUMMARY_CSV}")
