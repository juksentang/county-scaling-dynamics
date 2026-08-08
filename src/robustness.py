"""
Robustness checks: bootstrap, placebo, alternative specifications.
"""
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from src.scaling import run_scaling_regression, run_comparative_scaling
from src.saai import compute_saai
from src.distribution import fit_gmm, compute_effective_potential, find_potential_extrema
from src.config import BOOTSTRAP_N, SCALING_EQUATIONS


def bootstrap_scaling_exponent(df: pd.DataFrame,
                               n_iterations: int = BOOTSTRAP_N,
                               block_col: str = "Province",
                               seed: int = 42) -> pd.DataFrame:
    """
    Block bootstrap of scaling exponents (by province).

    Resamples provinces with replacement, runs scaling regressions.

    Returns DataFrame with iteration, equation, variable, beta.
    """
    rng = np.random.RandomState(seed)
    provinces = df[block_col].unique()
    n_prov = len(provinces)

    records = []
    for i in tqdm(range(n_iterations), desc="Bootstrap"):
        # Resample provinces
        boot_prov = rng.choice(provinces, size=n_prov, replace=True)
        boot_df = pd.concat(
            [df[df[block_col] == p] for p in boot_prov],
            ignore_index=True,
        )

        try:
            results = run_comparative_scaling(boot_df, year="pooled")
            for eq_key, result in results.items():
                for var in result.x_cols:
                    records.append({
                        "iteration": i,
                        "Equation": eq_key,
                        "Variable": var,
                        "Beta": result.betas[var],
                    })
        except Exception:
            pass

    boot_df = pd.DataFrame(records)

    # Summary
    summary = (
        boot_df.groupby(["Equation", "Variable"])["Beta"]
        .agg(["mean", "std", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
        .reset_index()
    )
    summary.columns = ["Equation", "Variable", "Mean", "SE_boot", "CI_2.5%", "CI_97.5%"]

    return summary


def placebo_phase_transition(df: pd.DataFrame,
                             saai_col: str = "SAAI") -> pd.DataFrame:
    """
    Placebo test: check for phase transition in NON-POVERTY counties.
    Expectation: no bimodal → unimodal transition.

    Returns DataFrame comparing GMM results for poverty vs non-poverty subsamples.
    """
    records = []
    for is_pov, label in [(True, "Poverty"), (False, "Non-poverty")]:
        sub = df[df["IsPoverty"] == is_pov] if "IsPoverty" in df.columns else df
        for year, grp in sub.groupby("Year"):
            xi = grp[saai_col].dropna().values
            if len(xi) < 30:
                continue

            # Fit 1 and 2 component GMM
            try:
                g1 = fit_gmm(xi, n_components=1)
                g2 = fit_gmm(xi, n_components=2)
                records.append({
                    "Group": label,
                    "Year": year,
                    "N": len(xi),
                    "BIC_K1": g1.bic,
                    "BIC_K2": g2.bic,
                    "PreferK2": g2.bic < g1.bic,
                })
            except Exception:
                pass

    return pd.DataFrame(records)


def alternative_scaling_specifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test alternative scaling specifications:
    1. Agricultural population (FFAFEmployeeNum) instead of total population
    2. TotalSownArea instead of CultivateArea
    3. With province fixed effects (dummies)
    """
    records = []

    # Spec 1: FFAFEmployeeNum as size variable
    for spec_name, y, x_list in [
        ("Agri pop only", "GrossOutputValue", ["FFAFEmployeeNum"]),
        ("Total sown area", "GrossOutputValue", ["PopNum", "TotalSownArea"]),
        ("Grain sown area", "GrossOutputValue", ["PopNum", "GrainSownArea"]),
    ]:
        try:
            result = run_scaling_regression(df, y_col=y, x_cols=x_list, year="pooled")
            for var in result.x_cols:
                records.append({
                    "Specification": spec_name,
                    "Variable": var,
                    "Beta": result.betas[var],
                    "SE": result.se[var],
                    "R2": result.r_squared,
                    "N": result.n_obs,
                })
        except Exception as e:
            records.append({"Specification": spec_name, "error": str(e)})

    return pd.DataFrame(records)


def alternative_poverty_definitions(df: pd.DataFrame,
                                    saai_col: str = "SAAI") -> pd.DataFrame:
    """
    Test alternative poverty definitions:
    1. Bottom quintile of per-capita GDP
    2. Bottom quintile of SAAI
    3. Official 832 list (default)

    Compare: mean SAAI, Cohen's d, bimodality strength.
    """
    records = []

    # Official definition
    if "IsPoverty" in df.columns:
        pov = df[df["IsPoverty"]][saai_col].dropna()
        npov = df[~df["IsPoverty"]][saai_col].dropna()
        if len(pov) > 0 and len(npov) > 0:
            records.append(_compare_groups(pov, npov, "Official 832"))

    # Bottom quintile GDP
    if "PerRegGDP" in df.columns:
        q20 = df["PerRegGDP"].quantile(0.2)
        pov_gdp = df[df["PerRegGDP"] <= q20][saai_col].dropna()
        npov_gdp = df[df["PerRegGDP"] > q20][saai_col].dropna()
        if len(pov_gdp) > 0:
            records.append(_compare_groups(pov_gdp, npov_gdp, "Bottom 20% GDP"))

    # Bottom quintile SAAI
    q20_saai = df[saai_col].quantile(0.2)
    pov_saai = df[df[saai_col] <= q20_saai][saai_col].dropna()
    npov_saai = df[df[saai_col] > q20_saai][saai_col].dropna()
    if len(pov_saai) > 0:
        records.append(_compare_groups(pov_saai, npov_saai, "Bottom 20% SAAI"))

    return pd.DataFrame(records)


def _compare_groups(pov: pd.Series, npov: pd.Series, label: str) -> dict:
    """Helper: compare two groups on SAAI metrics."""
    pooled_std = np.sqrt(
        ((len(pov)-1)*pov.std()**2 + (len(npov)-1)*npov.std()**2)
        / (len(pov) + len(npov) - 2)
    )
    d = (npov.mean() - pov.mean()) / pooled_std if pooled_std > 0 else 0
    t_stat, t_p = stats.ttest_ind(pov, npov, equal_var=False)

    return {
        "Definition": label,
        "N_poverty": len(pov),
        "N_nonpoverty": len(npov),
        "Mean_poverty": pov.mean(),
        "Mean_nonpoverty": npov.mean(),
        "Cohens_d": d,
        "t_stat": t_stat,
        "t_pvalue": t_p,
    }


def subsample_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test scaling stability across sub-periods:
    2000-2012 vs 2013-2023.
    """
    records = []
    for period_name, year_range in [("2000-2012", (2000, 2012)),
                                     ("2013-2023", (2013, 2023))]:
        sub = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
        try:
            results = run_comparative_scaling(sub, year="pooled")
            for eq_key, result in results.items():
                for var in result.x_cols:
                    records.append({
                        "Period": period_name,
                        "Equation": eq_key,
                        "Variable": var,
                        "Beta": result.betas[var],
                        "SE": result.se[var],
                        "N": result.n_obs,
                        "R2": result.r_squared,
                    })
        except Exception:
            pass

    return pd.DataFrame(records)


def sectoral_scaling_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check scaling for individual GDP sectors.

    Returns Beta, SE, CI, Wald test β=1 p-value, plus sector GDP share.
    """
    records = []
    for sector, col in [("Primary", "RegGDP_Primary"),
                         ("Secondary", "RegGDP_Secondary"),
                         ("Tertiary", "RegGDP_Tertiary")]:
        if col not in df.columns:
            continue
        try:
            result = run_scaling_regression(
                df, y_col=col, x_cols=["PopNum"], year="pooled"
            )
            # Compute sector's GDP share
            gdp_share = np.nan
            if "RegGDP" in df.columns:
                valid = df[[col, "RegGDP"]].dropna()
                valid = valid[(valid[col] > 0) & (valid["RegGDP"] > 0)]
                if len(valid) > 0:
                    gdp_share = (valid[col] / valid["RegGDP"]).mean()

            records.append({
                "Sector": sector,
                "Beta_PopNum": result.betas["PopNum"],
                "SE": result.se["PopNum"],
                "CI_lower": result.ci_lower["PopNum"],
                "CI_upper": result.ci_upper["PopNum"],
                "Wald_beta1_p": result.wald_beta_eq_1["PopNum"]["pvalue"],
                "R2": result.r_squared,
                "N": int(result.n_obs),
                "GDP_share": gdp_share,
            })
        except Exception:
            pass

    return pd.DataFrame(records)


def missing_data_diagnostic(df: pd.DataFrame,
                             key_var: str = "CultivateArea") -> pd.DataFrame:
    """
    Compare counties with vs without key_var data on multiple dimensions.

    Returns DataFrame with: Dimension, Mean_has, Mean_missing, Diff, StdDiff, t_stat, p_value.
    """
    has_data = df.groupby("CountyCode")[key_var].apply(
        lambda s: s.notna().any() and (s > 0).any()
    )
    codes_has = set(has_data[has_data].index)
    codes_miss = set(has_data[~has_data].index)

    df_has = df[df["CountyCode"].isin(codes_has)]
    df_miss = df[df["CountyCode"].isin(codes_miss)]

    dims = [
        ("RegGDP", "RegGDP"),
        ("PopNum", "PopNum"),
        ("GrossOutputValue", "GrossOutputValue"),
        ("PerRegGDP", "PerRegGDP"),
    ]

    records = []

    for dim_name, col in dims:
        if col not in df.columns:
            continue
        v_has = df_has[col].dropna()
        v_miss = df_miss[col].dropna()
        if len(v_has) < 2 or len(v_miss) < 2:
            continue

        mean_h, mean_m = v_has.mean(), v_miss.mean()
        pooled_std = np.sqrt(
            ((len(v_has)-1)*v_has.std()**2 + (len(v_miss)-1)*v_miss.std()**2)
            / (len(v_has) + len(v_miss) - 2)
        )
        std_diff = (mean_m - mean_h) / pooled_std if pooled_std > 0 else np.nan
        t_stat, t_p = stats.ttest_ind(v_has, v_miss, equal_var=False)

        records.append({
            "Dimension": dim_name,
            "N_has": len(v_has),
            "N_missing": len(v_miss),
            "Mean_has": mean_h,
            "Mean_missing": mean_m,
            "Ratio": mean_m / mean_h if mean_h != 0 else np.nan,
            "StdDiff": std_diff,
            "t_stat": t_stat,
            "p_value": t_p,
        })

    # IsPoverty rate comparison
    if "IsPoverty" in df.columns:
        rate_has = df_has.groupby("CountyCode")["IsPoverty"].first().mean()
        rate_miss = df_miss.groupby("CountyCode")["IsPoverty"].first().mean()
        records.append({
            "Dimension": "IsPoverty_rate",
            "N_has": len(codes_has),
            "N_missing": len(codes_miss),
            "Mean_has": rate_has,
            "Mean_missing": rate_miss,
            "Ratio": rate_miss / rate_has if rate_has > 0 else np.nan,
            "StdDiff": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        })

    # Region distribution
    if "Region" in df.columns:
        for region in ["East", "Central", "West"]:
            rate_h = (df_has.groupby("CountyCode")["Region"].first() == region).mean()
            rate_m = (df_miss.groupby("CountyCode")["Region"].first() == region).mean()
            records.append({
                "Dimension": f"Region_{region}_rate",
                "N_has": len(codes_has),
                "N_missing": len(codes_miss),
                "Mean_has": rate_h,
                "Mean_missing": rate_m,
                "Ratio": rate_m / rate_h if rate_h > 0 else np.nan,
                "StdDiff": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
            })

    return pd.DataFrame(records)


def run_selection_bias_test(df: pd.DataFrame,
                            key_var: str = "CultivateArea") -> dict:
    """
    Heckman-style selection bias detection for missing land data.

    1. Probit: P(has_key_var) ~ log(PopNum) + log(RegGDP) + IsPoverty + Region dummies
    2. Compute inverse Mills ratio (IMR)
    3. Add IMR to scaling regression: log(Y_agri) ~ log(Pop) + log(Land) + IMR
    4. If IMR is significant → selection bias exists

    Returns dict with: probit_summary, imr_significant, scaling_with_imr.
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    # Prepare county-level indicator: has key_var in any year
    has_var = df.groupby("CountyCode")[key_var].apply(
        lambda s: s.notna().any() and (s > 0).any()
    ).astype(int).reset_index()
    has_var.columns = ["CountyCode", "has_var"]

    # Build county-level dataset for probit
    county_info = df.groupby("CountyCode").agg({
        "PopNum": "mean",
        "RegGDP": "mean",
    }).reset_index()

    if "IsPoverty" in df.columns:
        pov = df.groupby("CountyCode")["IsPoverty"].first().astype(int).reset_index()
        county_info = county_info.merge(pov, on="CountyCode", how="left")

    if "Region" in df.columns:
        reg = df.groupby("CountyCode")["Region"].first().reset_index()
        county_info = county_info.merge(reg, on="CountyCode", how="left")

    county_info = county_info.merge(has_var, on="CountyCode", how="left")
    county_info = county_info.dropna(subset=["PopNum", "RegGDP", "has_var"])
    county_info = county_info[(county_info["PopNum"] > 0) & (county_info["RegGDP"] > 0)]

    if "IsPoverty" in county_info.columns:
        county_info = county_info.dropna(subset=["IsPoverty"])

    if "Region" in county_info.columns:
        county_info = county_info.dropna(subset=["Region"])

    county_info = county_info.reset_index(drop=True)

    # Build probit predictors
    probit_X = pd.DataFrame({
        "log_PopNum": np.log(county_info["PopNum"].values),
        "log_RegGDP": np.log(county_info["RegGDP"].values),
    })

    if "IsPoverty" in county_info.columns:
        probit_X["IsPoverty"] = county_info["IsPoverty"].astype(int).values

    if "Region" in county_info.columns:
        region_dummies = pd.get_dummies(county_info["Region"], prefix="Region", drop_first=True)
        probit_X = pd.concat([probit_X, region_dummies.reset_index(drop=True)], axis=1)

    probit_X = sm.add_constant(probit_X)
    probit_y = county_info["has_var"].astype(int).values

    # Fit probit
    probit_model = sm.Probit(probit_y, probit_X.astype(float))
    probit_result = probit_model.fit(disp=False)

    # Compute IMR for counties that HAVE the variable
    xb = probit_result.predict(probit_X, linear=True).astype(float).values
    imr = norm.pdf(xb) / np.clip(norm.cdf(xb), 1e-10, None)
    county_info = county_info.copy()
    county_info["IMR"] = imr

    # Merge IMR into panel-level data
    df_with_imr = df.merge(
        county_info[["CountyCode", "IMR"]],
        on="CountyCode", how="left"
    )

    # Run scaling regression WITH IMR on the subset that has land data
    has_land = df_with_imr[
        (df_with_imr[key_var] > 0) & df_with_imr[key_var].notna()
    ].copy()

    scaling_with_imr = None
    imr_significant = None

    if len(has_land) > 50:
        all_cols = ["GrossOutputValue", "PopNum", key_var, "IMR"]
        valid = has_land[all_cols].dropna()
        valid = valid[(valid["GrossOutputValue"] > 0) & (valid["PopNum"] > 0) & (valid[key_var] > 0)]

        if len(valid) > 10:
            log_y = np.log(valid["GrossOutputValue"].values)
            log_X = np.column_stack([
                np.log(valid["PopNum"].values),
                np.log(valid[key_var].values),
                valid["IMR"].values,
            ])
            log_X = sm.add_constant(log_X)

            ols = sm.OLS(log_y, log_X).fit(cov_type="HC1")

            scaling_with_imr = {
                "beta_PopNum": ols.params[1],
                "se_PopNum": ols.bse[1],
                f"beta_{key_var}": ols.params[2],
                f"se_{key_var}": ols.bse[2],
                "beta_IMR": ols.params[3],
                "se_IMR": ols.bse[3],
                "p_IMR": ols.pvalues[3],
                "R2": ols.rsquared,
                "N": int(ols.nobs),
            }
            imr_significant = ols.pvalues[3] < 0.05

    return {
        "probit_pseudo_r2": probit_result.prsquared,
        "probit_nobs": int(probit_result.nobs),
        "probit_params": dict(zip(probit_X.columns, probit_result.params)),
        "probit_pvalues": dict(zip(probit_X.columns, probit_result.pvalues)),
        "imr_significant": imr_significant,
        "scaling_with_imr": scaling_with_imr,
    }


def placebo_veff_evolution(saai_df, saai_col="SAAI"):
    """
    Compare V_eff evolution for poverty vs non-poverty counties.

    Non-poverty counties should show no barrier (single well),
    while poverty counties show bistable potential in pre-period.

    Returns dict with veff and barrier info for both groups.
    """
    from src.fokker_planck import build_transition_pairs, estimate_kramers_moyal, compute_fp_potential
    from src.config import FP_PERIODS

    results = {}
    for label, is_pov in [("poverty", True), ("nonpoverty", False)]:
        sub = saai_df[saai_df["IsPoverty"] == is_pov] if "IsPoverty" in saai_df.columns else saai_df
        period_results = {}
        for period in FP_PERIODS:
            pairs = build_transition_pairs(sub, period, saai_col=saai_col)
            if len(pairs) < 50:
                continue
            km = estimate_kramers_moyal(pairs, period=f"{label}_{period}")
            xi_fp, V_fp = compute_fp_potential(km)

            # Find barrier height from V_eff
            barrier = 0.0
            if len(V_fp) > 3:
                from scipy.signal import argrelextrema
                minima = argrelextrema(V_fp, np.less, order=2)[0]
                maxima = argrelextrema(V_fp, np.greater, order=2)[0]
                if len(minima) >= 2 and len(maxima) >= 1:
                    # Barrier = max V between two minima
                    left_min = V_fp[minima[0]]
                    max_between = V_fp[maxima[0]]
                    barrier = max_between - left_min

            period_results[period] = {
                "xi": xi_fp,
                "V_fp": V_fp,
                "barrier": barrier,
                "n_pairs": len(pairs),
                "km": km,
            }
        results[label] = period_results

    return results


def bratio_did_test(df: pd.DataFrame,
                    saai_col: str = "SAAI",
                    threshold: float = 0.0,
                    periods: dict[str, tuple[int, int]] | None = None) -> dict:
    """
    Quasi diff-in-diff test for B-ratio reversal.

    Computes B(xi<0)/B(xi>=0) separately for poverty and non-poverty
    counties in each FP period. If the B-ratio change is more
    pronounced in poverty counties, the causal attribution to TPA
    is strengthened.

    Parameters
    ----------
    periods : optional {"pre": (y0, y1), "during": (y0, y1)} overriding
        FP_PERIODS (e.g. a sensitivity variant extending the campaign
        window to include 2020).  Default None uses FP_PERIODS.

    Returns dict with B-ratios by group × period, and DiD estimate.
    """
    from src.fokker_planck import (build_transition_pairs,
                                    estimate_kramers_moyal,
                                    analyze_diffusion_heterogeneity)
    from src.config import FP_PERIODS

    results = {}
    for label, is_pov in [("poverty", True), ("nonpoverty", False)]:
        if "IsPoverty" not in df.columns:
            continue
        sub = df[df["IsPoverty"] == is_pov]
        period_results = {}
        for period in ["pre", "during"]:
            if period not in FP_PERIODS:
                continue
            window = periods.get(period) if periods else None
            pairs = build_transition_pairs(sub, period, saai_col=saai_col,
                                           window=window)
            if len(pairs) < 50:
                period_results[period] = {"status": "insufficient_pairs",
                                          "n_pairs": len(pairs)}
                continue
            km = estimate_kramers_moyal(pairs, period=f"{label}_{period}")
            het = analyze_diffusion_heterogeneity(km, threshold=threshold)
            period_results[period] = {
                "n_pairs": len(pairs),
                "B_neg_mean": het.get("B_neg_mean", np.nan),
                "B_pos_mean": het.get("B_pos_mean", np.nan),
                "B_ratio": het.get("ratio", np.nan),
                "t_pvalue": het.get("t_pvalue", np.nan),
            }
        results[label] = period_results

    # Compute quasi-DiD
    did = np.nan
    try:
        br_pov_pre = results["poverty"]["pre"]["B_ratio"]
        br_pov_dur = results["poverty"]["during"]["B_ratio"]
        br_npov_pre = results["nonpoverty"]["pre"]["B_ratio"]
        br_npov_dur = results["nonpoverty"]["during"]["B_ratio"]
        did = (br_pov_dur - br_pov_pre) - (br_npov_dur - br_npov_pre)
    except (KeyError, TypeError):
        pass

    results["did_estimate"] = did
    results["interpretation"] = (
        "DiD < 0 implies poverty counties show larger B-ratio decrease "
        "(supports TPA attribution)" if did < 0 else
        "DiD >= 0 implies B-ratio change is similar across groups "
        "(secular trends may explain the reversal)"
    ) if np.isfinite(did) else "DiD could not be computed"

    return results


def saai_comparison(df: pd.DataFrame) -> dict:
    """
    Compare single-factor vs multi-factor SAAI:
    correlation, AUC for poverty prediction, bimodality strength.
    """
    from src.saai import compute_saai_single_factor

    # Multi-factor
    multi = compute_saai(df, y_col="GrossOutputValue",
                         x_cols=["PopNum", "CultivateArea"])
    # Single-factor
    single = compute_saai_single_factor(df)

    # Merge
    merged = multi[["Year", "CountyCode", "SAAI"]].merge(
        single[["Year", "CountyCode", "SAAI"]].rename(columns={"SAAI": "SAAI_single"}),
        on=["Year", "CountyCode"],
        how="inner",
    )

    corr = merged["SAAI"].corr(merged["SAAI_single"])

    return {
        "correlation": corr,
        "n_multi": len(multi),
        "n_single": len(single),
        "n_overlap": len(merged),
    }


def bootstrap_bratio_did(df: pd.DataFrame,
                         saai_col: str = "SAAI",
                         threshold: float = 0.0,
                         n_boot: int = 2000,
                         seed: int = 101,
                         periods: dict[str, tuple[int, int]] | None = None,
                         stratified: bool = True) -> dict:
    """
    County-block bootstrap CI for the quasi-DiD estimator ΔΔB (Text S5).

    Resampling contract:
    - Per-county transition pairs are precomputed ONCE per (group, period)
      cell; each replicate resamples counties (stratified: poverty counties
      among poverty, non-poverty among non-poverty, preserving the 2×2
      design) and concatenates precomputed pairs — raw panel rows are never
      resampled.
    - Each replicate: 4 cells → estimate_kramers_moyal (25 equal-width
      bins, the Text S5 estimator) → analyze_diffusion_heterogeneity →
      B-ratio → ΔΔB.  Replicates where any cell has <2 reliable bins on
      either side of the threshold are recorded as degenerate and dropped;
      their fraction is reported.
    """
    from src.fokker_planck import (build_transition_pairs,
                                   estimate_kramers_moyal,
                                   analyze_diffusion_heterogeneity)

    cells = {}          # (group, period) -> {county: ndarray[n,2]}
    group_counties = {}  # group -> ndarray of counties
    for label, is_pov in [("poverty", True), ("nonpoverty", False)]:
        sub = df[df["IsPoverty"] == is_pov]
        codes = set()
        for period in ("pre", "during"):
            window = periods.get(period) if periods else None
            pairs = build_transition_pairs(sub, period, saai_col=saai_col,
                                           window=window)
            grouped = {c: g[["xi_t", "delta_xi"]].to_numpy()
                       for c, g in pairs.groupby("CountyCode")}
            cells[(label, period)] = grouped
            codes.update(grouped.keys())
        group_counties[label] = np.array(sorted(codes))

    def _cell_ratio(mat: np.ndarray) -> float:
        rep = pd.DataFrame(mat, columns=["xi_t", "delta_xi"])
        km = estimate_kramers_moyal(rep)
        het = analyze_diffusion_heterogeneity(km, threshold=threshold)
        return het.get("ratio", np.nan)

    # full-sample point estimate (same estimator as bratio_did_test)
    point_cells = {}
    for key, grouped in cells.items():
        mat = np.concatenate(list(grouped.values()), axis=0) if grouped else np.empty((0, 2))
        point_cells[key] = _cell_ratio(mat) if len(mat) >= 50 else np.nan
    point = ((point_cells[("poverty", "during")] - point_cells[("poverty", "pre")])
             - (point_cells[("nonpoverty", "during")] - point_cells[("nonpoverty", "pre")]))

    rng = np.random.default_rng(seed)
    draws, n_degenerate = [], 0
    for _ in range(n_boot):
        ratios = {}
        for label in ("poverty", "nonpoverty"):
            pool = group_counties[label] if stratified else np.concatenate(
                list(group_counties.values()))
            drawn = pool[rng.integers(0, len(pool), size=len(pool))]
            for period in ("pre", "during"):
                grouped = cells[(label, period)]
                arrs = [grouped[c] for c in drawn if c in grouped]
                ratios[(label, period)] = (_cell_ratio(np.concatenate(arrs, axis=0))
                                           if arrs else np.nan)
        vals = list(ratios.values())
        if any(not np.isfinite(v) for v in vals):
            n_degenerate += 1
            continue
        draws.append((ratios[("poverty", "during")] - ratios[("poverty", "pre")])
                     - (ratios[("nonpoverty", "during")] - ratios[("nonpoverty", "pre")]))
    draws = np.array(draws)

    from scipy.stats import skew as _skew
    return {
        "point": float(point),
        "point_cells": {f"{k[0]}_{k[1]}": float(v) for k, v in point_cells.items()},
        "n_boot": n_boot,
        "n_used": len(draws),
        "n_degenerate": n_degenerate,
        "degenerate_frac": n_degenerate / n_boot,
        "se": float(np.std(draws, ddof=1)) if len(draws) > 1 else np.nan,
        "ci_lo": float(np.quantile(draws, 0.025)) if len(draws) else np.nan,
        "ci_hi": float(np.quantile(draws, 0.975)) if len(draws) else np.nan,
        "boot_median": float(np.median(draws)) if len(draws) else np.nan,
        "skew": float(_skew(draws)) if len(draws) > 2 else np.nan,
        "P_lt0": float((draws < 0).mean()) if len(draws) else np.nan,
        "draws": draws,
    }


def bootstrap_km_quantities(saai_df: pd.DataFrame,
                            saai_col: str = "SAAI",
                            n_boot: int = 1000,
                            seed: int = 102,
                            n_bins_by_period: dict | None = None,
                            xi_target: float = 0.0,
                            min_pairs: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    County-block bootstrap CIs for KM-derived escape observables per period.

    Resampling contract:
    - Transition pairs are precomputed ONCE per period from the fixed SAAI
      panel and grouped by county; each replicate resamples the county list
      with replacement and concatenates the counties' precomputed pairs.
      Raw panel rows are never resampled (duplicate CountyCode would break
      the consecutive-year pairing logic).  Uncertainty from the SAAI
      estimation stage itself is not propagated (conditional bootstrap).
    - Each replicate × period: estimate_kramers_moyal with the period's
      (n_bins, binning) spec → compute_escape_observables (corrected
      definitions) + analyze_diffusion_heterogeneity (B-ratio).
    - id_status is recorded per draw; the primary CI uses all non-failed
      draws, and the summary reports the status distribution.

    Parameters
    ----------
    n_bins_by_period : period → (n_bins, binning). Default:
        {"pre": (25, "equal_width"), "during": (25, "equal_width"),
         "post": (11, "equal_count")}.

    Returns
    -------
    (draws, summary): long DataFrame of per-replicate quantities, and a
    per-period × quantity summary with percentile 95% CIs.
    """
    from src.fokker_planck import (build_transition_pairs,
                                   estimate_kramers_moyal,
                                   compute_escape_observables,
                                   analyze_diffusion_heterogeneity)
    from src.config import FP_PERIODS

    if n_bins_by_period is None:
        n_bins_by_period = {"pre": (25, "equal_width"),
                            "during": (25, "equal_width"),
                            "post": (11, "equal_count")}

    # ── precompute per-county pairs per period ──
    county_pairs: dict[str, dict] = {}
    counties_all: set = set()
    for period in FP_PERIODS:
        pairs = build_transition_pairs(saai_df, period, saai_col=saai_col)
        if len(pairs) == 0:
            continue
        grouped = {code: g[["xi_t", "delta_xi"]].to_numpy()
                   for code, g in pairs.groupby("CountyCode")}
        county_pairs[period] = grouped
        counties_all.update(grouped.keys())

    counties = np.array(sorted(counties_all))
    n_c = len(counties)
    rng = np.random.default_rng(seed)

    quantities = ["xi_trap", "xi_barrier", "delta_V", "delta_V_at_target",
                  "mean_diffusion", "gamma", "tau_mfpt", "tau_arrhenius",
                  "B_ratio"]

    records = []
    for b in range(n_boot):
        drawn = counties[rng.integers(0, n_c, size=n_c)]
        for period, grouped in county_pairs.items():
            arrs = [grouped[c] for c in drawn if c in grouped]
            if not arrs:
                continue
            mat = np.concatenate(arrs, axis=0)
            if len(mat) < min_pairs:
                records.append({"replicate": b, "period": period,
                                "id_status": "too_few_pairs",
                                "n_pairs": len(mat)})
                continue
            rep_pairs = pd.DataFrame(mat, columns=["xi_t", "delta_xi"])
            n_bins, binning = n_bins_by_period.get(period, (25, "equal_width"))
            km = estimate_kramers_moyal(rep_pairs, n_bins=n_bins,
                                        period=period, binning=binning)
            esc = compute_escape_observables(km, xi_target=xi_target)
            dh = analyze_diffusion_heterogeneity(km)
            rec = {"replicate": b, "period": period,
                   "id_status": esc["id_status"], "n_pairs": len(mat)}
            for q in quantities:
                if q == "B_ratio":
                    rec[q] = dh.get("ratio", np.nan)
                else:
                    rec[q] = esc.get(q, np.nan)
            records.append(rec)

    draws = pd.DataFrame(records)

    # ── summary ──
    ok = draws[~draws["id_status"].isin(["failed", "too_few_pairs",
                                         "target_outside_domain",
                                         "degenerate_path"])]
    rows = []
    for period in county_pairs:
        sub = ok[ok["period"] == period]
        n_all = (draws["period"] == period).sum()
        status_counts = (draws[draws["period"] == period]["id_status"]
                         .value_counts().to_dict())
        for q in quantities:
            vals = sub[q].dropna().to_numpy()
            if len(vals) == 0:
                continue
            rows.append({
                "period": period, "quantity": q,
                "n_draws_used": len(vals), "n_draws_total": n_all,
                "boot_median": float(np.median(vals)),
                "boot_mean": float(np.mean(vals)),
                "boot_se": float(np.std(vals, ddof=1)),
                "ci_lo": float(np.quantile(vals, 0.025)),
                "ci_hi": float(np.quantile(vals, 0.975)),
                "id_status_dist": str(status_counts),
            })
    summary = pd.DataFrame(rows)
    return draws, summary
