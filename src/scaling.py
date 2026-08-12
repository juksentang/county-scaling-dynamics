"""
Scaling regression analysis: Eq 14a-14e comparative scaling,
Wald tests, time series estimation, subsample analysis.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import statsmodels.api as sm
from scipy import stats
from src.config import SCALING_EQUATIONS, YEAR_RANGE


@dataclass
class ScalingResult:
    """Result of a single scaling regression."""
    equation_label: str
    y_col: str
    x_cols: list[str]
    year: int | str            # year or "pooled"
    n_obs: int
    betas: dict[str, float]    # coefficient for each x variable
    se: dict[str, float]       # robust SE for each coefficient
    pvalue: dict[str, float]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    r_squared: float
    r_squared_adj: float
    wald_beta_eq_1: dict[str, dict]  # {var: {stat, pvalue}} for H₀: β=1
    residuals: np.ndarray = field(repr=False)
    intercept: float = 0.0


def run_scaling_regression(df: pd.DataFrame,
                           y_col: str,
                           x_cols: list[str],
                           year: int | str = "pooled",
                           robust_se: str = "HC1") -> ScalingResult:
    """
    Run OLS log-log scaling regression with robust SE and Wald test β=1.

    Model: log(Y) = α + Σ βᵢ log(Xᵢ) + ε

    Parameters
    ----------
    df : DataFrame with columns y_col and x_cols.
    y_col : dependent variable column name.
    x_cols : list of independent variable column names.
    year : year label for filtering (or "pooled" for all years).
    robust_se : HC0, HC1, HC2, HC3 for heteroskedasticity-robust SE.

    Returns
    -------
    ScalingResult dataclass.
    """
    # Filter by year if needed
    if year != "pooled" and "Year" in df.columns:
        data = df[df["Year"] == year].copy()
    else:
        data = df.copy()

    # Drop rows with missing or non-positive values
    all_cols = [y_col] + x_cols
    data = data[all_cols].dropna()
    for col in all_cols:
        data = data[data[col] > 0]

    if len(data) < len(x_cols) + 2:
        raise ValueError(f"Insufficient data: {len(data)} rows for {len(x_cols)+1} parameters")

    # Log transform
    log_y = np.log(data[y_col].values)
    log_X = np.column_stack([np.log(data[c].values) for c in x_cols])
    log_X = sm.add_constant(log_X)

    # OLS with robust SE
    model = sm.OLS(log_y, log_X)
    results = model.fit(cov_type=robust_se)

    # Extract coefficients (skip constant at index 0)
    betas = {}
    se = {}
    pvalue = {}
    ci_lower = {}
    ci_upper = {}
    wald_tests = {}

    conf_int = results.conf_int(alpha=0.05)

    for i, var in enumerate(x_cols):
        idx = i + 1  # +1 for constant
        betas[var] = results.params[idx]
        se[var] = results.bse[idx]
        pvalue[var] = results.pvalues[idx]
        ci_lower[var] = conf_int[idx, 0]
        ci_upper[var] = conf_int[idx, 1]

        # Wald test: H₀: β = 1
        t_stat = (results.params[idx] - 1.0) / results.bse[idx]
        wald_p = 2 * stats.t.sf(abs(t_stat), df=results.df_resid)
        wald_tests[var] = {"t_stat": t_stat, "pvalue": wald_p}

    return ScalingResult(
        equation_label=f"log({y_col}) ~ " + " + ".join(f"log({c})" for c in x_cols),
        y_col=y_col,
        x_cols=x_cols,
        year=year,
        n_obs=results.nobs,
        betas=betas,
        se=se,
        pvalue=pvalue,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        r_squared=results.rsquared,
        r_squared_adj=results.rsquared_adj,
        wald_beta_eq_1=wald_tests,
        residuals=results.resid,
        intercept=results.params[0],
    )


def run_comparative_scaling(df: pd.DataFrame,
                            year: int | str = "pooled") -> dict[str, ScalingResult]:
    """
    Run all five scaling equations (Eq 14a-14e) for a given year.

    Returns dict mapping equation key to ScalingResult.
    """
    results = {}
    for eq_key, eq_cfg in SCALING_EQUATIONS.items():
        try:
            result = run_scaling_regression(
                df, y_col=eq_cfg["y"], x_cols=eq_cfg["x"], year=year
            )
            result.equation_label = f"Eq {eq_key}: {eq_cfg['label']}"
            results[eq_key] = result
        except (ValueError, KeyError) as e:
            print(f"  Eq {eq_key} ({year}): FAILED — {e}")
    return results


def run_scaling_timeseries(df: pd.DataFrame,
                           years: list[int] | None = None) -> pd.DataFrame:
    """
    Run scaling regressions year-by-year for all equations.

    Returns DataFrame with columns: Year, Equation, Variable, Beta, SE, CI_lower, CI_upper, N, R2.
    """
    if years is None:
        years = sorted(df["Year"].unique())

    records = []
    for year in years:
        year_results = run_comparative_scaling(df, year=year)
        for eq_key, result in year_results.items():
            for var in result.x_cols:
                records.append({
                    "Year": year,
                    "Equation": eq_key,
                    "Variable": var,
                    "Beta": result.betas[var],
                    "SE": result.se[var],
                    "CI_lower": result.ci_lower[var],
                    "CI_upper": result.ci_upper[var],
                    "N": result.n_obs,
                    "R2": result.r_squared,
                    "Wald_beta1_p": result.wald_beta_eq_1[var]["pvalue"],
                })

    return pd.DataFrame(records)


def run_scaling_by_subsample(df: pd.DataFrame,
                             group_col: str,
                             year: int | str = "pooled") -> pd.DataFrame:
    """
    Run scaling regressions by subsample (province/region/poverty/decile).

    Returns DataFrame with group, equation, variable, beta, SE, N.
    """
    records = []
    for group_val, grp in df.groupby(group_col):
        if len(grp) < 50:
            continue
        year_results = run_comparative_scaling(grp, year=year)
        for eq_key, result in year_results.items():
            for var in result.x_cols:
                records.append({
                    "Group": group_val,
                    "Equation": eq_key,
                    "Variable": var,
                    "Beta": result.betas[var],
                    "SE": result.se[var],
                    "N": result.n_obs,
                    "R2": result.r_squared,
                })

    return pd.DataFrame(records)


def wald_test_equal_betas(r1: ScalingResult, r2: ScalingResult,
                          var: str) -> dict:
    """
    Test H₀: β₁ = β₂ across two independent regressions.

    Uses z-test: z = (β₁ - β₂) / sqrt(SE₁² + SE₂²).
    """
    b1, b2 = r1.betas[var], r2.betas[var]
    se1, se2 = r1.se[var], r2.se[var]
    z = (b1 - b2) / np.sqrt(se1**2 + se2**2)
    p = 2 * stats.norm.sf(abs(z))
    return {"beta1": b1, "beta2": b2, "z_stat": z, "pvalue": p}


def universality_test(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test universality (stability) of scaling exponents over time.

    Returns DataFrame with equation, variable, mean_beta, std_beta, CV,
    and Chow-test-style F-stat.
    """
    records = []
    for (eq, var), grp in ts_df.groupby(["Equation", "Variable"]):
        betas = grp["Beta"].dropna()
        if len(betas) < 3:
            continue

        mean_b = betas.mean()
        std_b = betas.std()
        cv = std_b / abs(mean_b) if mean_b != 0 else np.nan

        # Simple stability test: are betas significantly different from their mean?
        # Using weighted chi-squared: Σ (β_t - β̄)² / SE_t²
        ses = grp["SE"].values
        chi2 = np.sum((betas.values - mean_b)**2 / ses**2)
        dof = len(betas) - 1
        p_stability = stats.chi2.sf(chi2, dof)

        records.append({
            "Equation": eq,
            "Variable": var,
            "MeanBeta": mean_b,
            "StdBeta": std_b,
            "CV": cv,
            "Chi2_stability": chi2,
            "DOF": dof,
            "P_stability": p_stability,
        })

    return pd.DataFrame(records)


def run_land_variable_comparison(df: pd.DataFrame,
                                  year: int | str = "pooled") -> pd.DataFrame:
    """
    Compare three land variables as predictors of agricultural output.

    For each land variable (CultivateArea, TotalSownArea, GrainSownArea):
      - Y_agri ~ Land
      - Y_agri ~ PopNum + Land

    Returns DataFrame with: LandVar, Specification, Variable, Beta, SE, R2, N, AIC.
    """
    land_vars = ["CultivateArea", "TotalSownArea", "GrainSownArea"]
    records = []

    for land_var in land_vars:
        for spec_name, x_list in [
            ("Land only", [land_var]),
            ("Pop + Land", ["PopNum", land_var]),
        ]:
            try:
                result = run_scaling_regression(
                    df, y_col="GrossOutputValue", x_cols=x_list, year=year
                )
                for var in result.x_cols:
                    records.append({
                        "LandVar": land_var,
                        "Specification": spec_name,
                        "Variable": var,
                        "Beta": result.betas[var],
                        "SE": result.se[var],
                        "CI_lower": result.ci_lower[var],
                        "CI_upper": result.ci_upper[var],
                        "Wald_beta1_p": result.wald_beta_eq_1[var]["pvalue"],
                        "R2": result.r_squared,
                        "N": int(result.n_obs),
                    })
            except (ValueError, KeyError) as e:
                records.append({
                    "LandVar": land_var,
                    "Specification": spec_name,
                    "Variable": "N/A",
                    "error": str(e),
                })

    return pd.DataFrame(records)


def run_sectoral_decomposition(df: pd.DataFrame,
                                year: int | str = "pooled") -> dict:
    """
    Decompose β₁ into sectoral components: Secondary and Tertiary GDP ~ PopNum.

    Returns dict with:
      - 'table': DataFrame of per-sector results
      - 'wald_diff': Wald test for H₀: β_secondary = β_tertiary
    """
    results = {}
    records = []

    for sector, col in [("Secondary", "RegGDP_Secondary"),
                         ("Tertiary", "RegGDP_Tertiary"),
                         ("NonAgri", "NonAgriGDP")]:
        if col not in df.columns:
            continue
        try:
            r = run_scaling_regression(
                df, y_col=col, x_cols=["PopNum"], year=year
            )
            results[sector] = r
            records.append({
                "Sector": sector,
                "Beta_PopNum": r.betas["PopNum"],
                "SE": r.se["PopNum"],
                "CI_lower": r.ci_lower["PopNum"],
                "CI_upper": r.ci_upper["PopNum"],
                "Wald_beta1_p": r.wald_beta_eq_1["PopNum"]["pvalue"],
                "R2": r.r_squared,
                "N": int(r.n_obs),
            })
        except (ValueError, KeyError):
            pass

    table = pd.DataFrame(records)

    # Wald test: H₀: β_secondary = β_tertiary
    wald_diff = None
    if "Secondary" in results and "Tertiary" in results:
        wald_diff = wald_test_equal_betas(
            results["Secondary"], results["Tertiary"], "PopNum"
        )

    return {"table": table, "wald_diff": wald_diff}


def run_urbanization_stratified_scaling(df: pd.DataFrame,
                                        year: int | str = "pooled",
                                        n_quantiles: int = 3,
                                        u_col: str | None = None) -> dict:
    """
    Stratify counties by urbanization proxy (NonAgriGDP/RegGDP) and run
    sectoral scaling (Secondary, Tertiary, NonAgri ~ PopNum) per group.

    Parameters
    ----------
    df : DataFrame with RegGDP, NonAgriGDP, RegGDP_Secondary, RegGDP_Tertiary, PopNum.
    year : year filter or "pooled".
    n_quantiles : number of urbanization groups (default 3 = terciles).
    u_col : optional column to use as the urbanization measure instead of
        NonAgriGDP/RegGDP (e.g. a GDP-independent measure such as the
        nighttime-lights lit-area share 'u_lit').  Default None uses the
        NonAgriGDP/RegGDP ratio.

    Returns
    -------
    dict with:
      - 'table': DataFrame (Group × Sector → β, SE, CI, Wald p, N)
      - 'wald_cross_group': dict of Wald tests for β(High) vs β(Low) per sector
      - 'cutoffs': quantile cutoff values
      - 'group_stats': per-group summary (n, mean ratio)
    """
    if year != "pooled" and "Year" in df.columns:
        data = df[df["Year"] == year].copy()
    else:
        data = df.copy()

    # Compute urbanization proxy
    if u_col is not None:
        data = data[data[u_col].notna()].copy()
        data["urb_ratio"] = data[u_col].astype(float)
    else:
        mask = (data["RegGDP"] > 0) & (data["NonAgriGDP"] > 0)
        data = data[mask].copy()
        data["urb_ratio"] = data["NonAgriGDP"] / data["RegGDP"]
        data["urb_ratio"] = data["urb_ratio"].clip(0, 1.5)

    # Quantile groups
    labels = ["Low", "Medium", "High"] if n_quantiles == 3 else \
             [f"Q{i+1}" for i in range(n_quantiles)]
    data["UrbGroup"] = pd.qcut(data["urb_ratio"], q=n_quantiles,
                                labels=labels, duplicates="drop")
    cutoffs = data.groupby("UrbGroup")["urb_ratio"].agg(["min", "max"])

    # Group stats
    group_stats = data.groupby("UrbGroup").agg(
        n=("urb_ratio", "size"),
        mean_ratio=("urb_ratio", "mean"),
        median_ratio=("urb_ratio", "median"),
    )

    # Run scaling per group × sector
    sectors = [("Secondary", "RegGDP_Secondary"),
               ("Tertiary", "RegGDP_Tertiary"),
               ("NonAgri", "NonAgriGDP")]

    records = []
    group_results = {}  # {(group, sector): ScalingResult}
    for grp_name, grp_df in data.groupby("UrbGroup", observed=True):
        for sector_name, col in sectors:
            if col not in grp_df.columns:
                continue
            try:
                r = run_scaling_regression(
                    grp_df, y_col=col, x_cols=["PopNum"], year="pooled"
                )
                group_results[(grp_name, sector_name)] = r
                records.append({
                    "Group": grp_name,
                    "Sector": sector_name,
                    "Beta": r.betas["PopNum"],
                    "SE": r.se["PopNum"],
                    "CI_lower": r.ci_lower["PopNum"],
                    "CI_upper": r.ci_upper["PopNum"],
                    "Wald_beta1_p": r.wald_beta_eq_1["PopNum"]["pvalue"],
                    "R2": r.r_squared,
                    "N": int(r.n_obs),
                })
            except (ValueError, KeyError):
                pass

    table = pd.DataFrame(records)

    # Wald cross-group tests: β(High) vs β(Low) for each sector
    wald_cross = {}
    low_label = labels[0]
    high_label = labels[-1]
    for sector_name, _ in sectors:
        key_low = (low_label, sector_name)
        key_high = (high_label, sector_name)
        if key_low in group_results and key_high in group_results:
            wald_cross[sector_name] = wald_test_equal_betas(
                group_results[key_high], group_results[key_low], "PopNum"
            )

    return {
        "table": table,
        "wald_cross_group": wald_cross,
        "cutoffs": cutoffs,
        "group_stats": group_stats,
    }


def run_interaction_regression(df: pd.DataFrame,
                               y_col: str = "NonAgriGDP",
                               x_pop: str = "PopNum",
                               u_col: str = None,
                               year: int | str = "pooled") -> dict:
    """
    Interaction regression: ln Y = α + (β₀ + β₁·u) ln N + ε
    where u = NonAgriGDP/RegGDP (urbanization proxy).

    β₁ > 0 → urbanization enhances superlinearity.
    Crossover: u* = (1 - β₀) / β₁ where β(u*) = 1.

    Returns dict: beta0, beta1, se_beta0, se_beta1, crossover_u, CI, R2, N.
    """
    if year != "pooled" and "Year" in df.columns:
        data = df[df["Year"] == year].copy()
    else:
        data = df.copy()

    # Compute urbanization proxy
    if u_col is None:
        data = data[(data["RegGDP"] > 0) & (data["NonAgriGDP"] > 0)].copy()
        data["_u"] = (data["NonAgriGDP"] / data["RegGDP"]).clip(0, 1.5)
    else:
        data = data[data[u_col] > 0].copy()
        data["_u"] = data[u_col]

    # Filter valid
    data = data[(data[y_col] > 0) & (data[x_pop] > 0)].dropna(subset=[y_col, x_pop, "_u"])

    if len(data) < 10:
        raise ValueError(f"Insufficient data: {len(data)} rows")

    log_y = np.log(data[y_col].values)
    log_N = np.log(data[x_pop].values)
    u = data["_u"].values

    # Design: [const, ln N, u * ln N]
    X = np.column_stack([np.ones(len(data)), log_N, u * log_N])
    model = sm.OLS(log_y, X)
    fit = model.fit(cov_type="HC1")

    beta0 = fit.params[1]
    beta1 = fit.params[2]
    se_beta0 = fit.bse[1]
    se_beta1 = fit.bse[2]

    # Crossover: β₀ + β₁·u* = 1 → u* = (1 - β₀) / β₁
    crossover_u = (1 - beta0) / beta1 if abs(beta1) > 1e-10 else np.nan

    return {
        "alpha": fit.params[0],
        "beta0": beta0,
        "beta1": beta1,
        "se_beta0": se_beta0,
        "se_beta1": se_beta1,
        "p_beta0": fit.pvalues[1],
        "p_beta1": fit.pvalues[2],
        "crossover_u": crossover_u,
        "cov_matrix": fit.cov_params(),
        "R2": fit.rsquared,
        "N": int(fit.nobs),
    }


def run_province_fe_scaling(df: pd.DataFrame,
                            eq_key: str,
                            year: int | str = "pooled") -> dict:
    """
    Scaling regression with province fixed effects and clustered SE.

    Province inferred from CountyCode[:2] using PROVINCE_CODE_MAP.

    Returns dict: beta, se, CI, R2, N, province_effects.
    """
    from src.config import PROVINCE_CODE_MAP, SCALING_EQUATIONS

    eq_cfg = SCALING_EQUATIONS[eq_key]
    y_col = eq_cfg["y"]
    x_cols = eq_cfg["x"]

    if year != "pooled" and "Year" in df.columns:
        data = df[df["Year"] == year].copy()
    else:
        data = df.copy()

    # Province from county code
    data = data.copy()
    data["_ProvinceCode"] = data["CountyCode"].astype(int) // 10000
    data["_Province"] = data["_ProvinceCode"].map(PROVINCE_CODE_MAP)
    data = data.dropna(subset=["_Province"])

    # Filter valid
    all_cols = [y_col] + x_cols
    data = data.dropna(subset=all_cols)
    for col in all_cols:
        data = data[data[col] > 0]

    if len(data) < 50:
        raise ValueError(f"Insufficient data: {len(data)} rows")

    # Log transform
    log_y = np.log(data[y_col].values)
    log_X_list = [np.log(data[c].values) for c in x_cols]

    # Province dummies (drop first)
    prov_dummies = pd.get_dummies(data["_Province"], prefix="prov", drop_first=True)
    X = np.column_stack(log_X_list + [prov_dummies.values])
    X = sm.add_constant(X)

    model = sm.OLS(log_y, X)
    # Cluster SE by province
    groups = data["_Province"].values
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    betas = {}
    se = {}
    ci = {}
    pvals = {}
    conf = fit.conf_int(alpha=0.05)
    for i, var in enumerate(x_cols):
        idx = i + 1
        betas[var] = fit.params[idx]
        se[var] = fit.bse[idx]
        pvals[var] = fit.pvalues[idx]
        ci[var] = (conf[idx, 0], conf[idx, 1])

    return {
        "betas": betas,
        "se": se,
        "pvalues": pvals,
        "ci": ci,
        "R2": fit.rsquared,
        "R2_adj": fit.rsquared_adj,
        "N": int(fit.nobs),
        "n_provinces": data["_Province"].nunique(),
    }


def run_panel_fe_scaling(df: pd.DataFrame,
                         eq_key: str) -> dict:
    """
    County panel fixed effects (within estimator).

    Demean by county: ln Y_it - mean(ln Y_i) = β(ln N_it - mean(ln N_i)) + ε_it.
    Cluster SE by county.

    Returns dict: betas, se, ci, R2_within, N, n_counties.
    """
    from src.config import SCALING_EQUATIONS

    eq_cfg = SCALING_EQUATIONS[eq_key]
    y_col = eq_cfg["y"]
    x_cols = eq_cfg["x"]

    data = df.copy()
    all_cols = [y_col] + x_cols + ["CountyCode"]
    data = data.dropna(subset=all_cols)
    for col in [y_col] + x_cols:
        data = data[data[col] > 0]

    # Need at least 2 obs per county for within estimator
    county_counts = data.groupby("CountyCode").size()
    valid_counties = county_counts[county_counts >= 2].index
    data = data[data["CountyCode"].isin(valid_counties)]

    if len(data) < 50:
        raise ValueError(f"Insufficient data: {len(data)} rows")

    # Log transform
    data = data.copy()
    data["_log_y"] = np.log(data[y_col])
    for c in x_cols:
        data[f"_log_{c}"] = np.log(data[c])

    # Demean by county
    for col in ["_log_y"] + [f"_log_{c}" for c in x_cols]:
        county_means = data.groupby("CountyCode")[col].transform("mean")
        data[f"{col}_dm"] = data[col] - county_means

    # Regression on demeaned data
    dm_y = data["_log_y_dm"].values
    dm_X = np.column_stack([data[f"_log_{c}_dm"].values for c in x_cols])

    model = sm.OLS(dm_y, dm_X)  # no constant in FE
    groups = data["CountyCode"].values
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    betas = {}
    se = {}
    ci = {}
    pvals = {}
    conf = fit.conf_int(alpha=0.05)
    for i, var in enumerate(x_cols):
        betas[var] = fit.params[i]
        se[var] = fit.bse[i]
        pvals[var] = fit.pvalues[i]
        ci[var] = (conf[i, 0], conf[i, 1])

    # Within R² (on demeaned data)
    ss_res = np.sum(fit.resid ** 2)
    ss_tot = np.sum((dm_y - dm_y.mean()) ** 2)
    r2_within = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "betas": betas,
        "se": se,
        "pvalues": pvals,
        "ci": ci,
        "R2_within": r2_within,
        "N": int(fit.nobs),
        "n_counties": data["CountyCode"].nunique(),
    }


def format_scaling_table(results: dict[str, ScalingResult]) -> pd.DataFrame:
    """
    Format scaling results as a publication-ready table.

    Returns DataFrame with rows = equations, columns = coefficient info.
    """
    rows = []
    for eq_key, result in results.items():
        row = {"Equation": eq_key, "N": int(result.n_obs), "R²": result.r_squared}
        for var in result.x_cols:
            b = result.betas[var]
            se = result.se[var]
            p = result.pvalue[var]
            wald_p = result.wald_beta_eq_1[var]["pvalue"]

            # Stars
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = ""

            row[f"β_{var}"] = f"{b:.4f}{stars}"
            row[f"SE_{var}"] = f"({se:.4f})"
            row[f"β=1_p_{var}"] = f"{wald_p:.4f}"

        rows.append(row)

    return pd.DataFrame(rows)


def run_province_fe_interaction(df: pd.DataFrame,
                                 y_col: str = "NonAgriGDP",
                                 x_pop: str = "PopNum",
                                 u_col: str = None,
                                 year: int | str = "pooled") -> dict:
    """
    Interaction regression with province fixed effects and clustered SE.

    ln Y = α_j + (β₀ + β₁·u) ln N + ε
    where α_j = province FE (dummies, drop first).
    Cluster SE by province.

    Returns dict: beta0, beta1, se, crossover_u, cov_matrix, R2, N, n_provinces.
    """
    from src.config import PROVINCE_CODE_MAP

    if year != "pooled" and "Year" in df.columns:
        data = df[df["Year"] == year].copy()
    else:
        data = df.copy()

    # Province from county code
    data["_ProvinceCode"] = data["CountyCode"].astype(int) // 10000
    data["_Province"] = data["_ProvinceCode"].map(PROVINCE_CODE_MAP)
    data = data.dropna(subset=["_Province"])

    # Compute urbanization proxy
    if u_col is None:
        data = data[(data["RegGDP"] > 0) & (data["NonAgriGDP"] > 0)].copy()
        data["_u"] = (data["NonAgriGDP"] / data["RegGDP"]).clip(0, 1.5)
    else:
        data = data[data[u_col] > 0].copy()
        data["_u"] = data[u_col]

    data = data[(data[y_col] > 0) & (data[x_pop] > 0)].dropna(
        subset=[y_col, x_pop, "_u"])

    if len(data) < 50:
        raise ValueError(f"Insufficient data: {len(data)} rows")

    log_y = np.log(data[y_col].values)
    log_N = np.log(data[x_pop].values)
    u = data["_u"].values

    # Province dummies (drop first)
    prov_dummies = pd.get_dummies(data["_Province"], prefix="prov",
                                   drop_first=True)
    # Design: [prov_dummies, ln N, u * ln N]
    X = np.column_stack([prov_dummies.values, log_N, u * log_N])

    model = sm.OLS(log_y, X)
    groups = data["_Province"].values
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    n_prov_cols = prov_dummies.shape[1]
    beta0 = fit.params[n_prov_cols]      # ln N coefficient
    beta1 = fit.params[n_prov_cols + 1]  # u * ln N coefficient
    se_beta0 = fit.bse[n_prov_cols]
    se_beta1 = fit.bse[n_prov_cols + 1]

    crossover_u = (1 - beta0) / beta1 if abs(beta1) > 1e-10 else np.nan

    # Extract 2x2 covariance for beta0, beta1
    idx0, idx1 = n_prov_cols, n_prov_cols + 1
    full_cov = fit.cov_params()
    cov_sub = full_cov[np.ix_([idx0, idx1], [idx0, idx1])]

    return {
        "beta0": beta0,
        "beta1": beta1,
        "se_beta0": se_beta0,
        "se_beta1": se_beta1,
        "p_beta0": fit.pvalues[n_prov_cols],
        "p_beta1": fit.pvalues[n_prov_cols + 1],
        "crossover_u": crossover_u,
        "cov_matrix": cov_sub,
        "R2": fit.rsquared,
        "R2_adj": fit.rsquared_adj,
        "N": int(fit.nobs),
        "n_provinces": data["_Province"].nunique(),
    }


def run_interaction_regression_controlled(df: pd.DataFrame,
                                           y_col: str = "NonAgriGDP",
                                           x_pop: str = "PopNum",
                                           u_col: str = None,
                                           year: int | str = "pooled") -> dict:
    """
    Interaction regression controlling for population size nonlinearity.

    ln Y = α + β₀ ln N + β₁ u·ln N + β₂ (ln N)² + ε

    β₂ controls for population size itself having nonlinear effects.
    If β₁ remains significant → urbanization effect is independent of size.
    """
    if year != "pooled" and "Year" in df.columns:
        data = df[df["Year"] == year].copy()
    else:
        data = df.copy()

    if u_col is None:
        data = data[(data["RegGDP"] > 0) & (data["NonAgriGDP"] > 0)].copy()
        data["_u"] = (data["NonAgriGDP"] / data["RegGDP"]).clip(0, 1.5)
    else:
        data = data[data[u_col] > 0].copy()
        data["_u"] = data[u_col]

    data = data[(data[y_col] > 0) & (data[x_pop] > 0)].dropna(
        subset=[y_col, x_pop, "_u"])

    if len(data) < 10:
        raise ValueError(f"Insufficient data: {len(data)} rows")

    log_y = np.log(data[y_col].values)
    log_N = np.log(data[x_pop].values)
    u = data["_u"].values

    # Design: [const, ln N, u·ln N, (ln N)²]
    X = np.column_stack([np.ones(len(data)), log_N, u * log_N, log_N**2])
    model = sm.OLS(log_y, X)
    fit = model.fit(cov_type="HC1")

    return {
        "alpha": fit.params[0],
        "beta0": fit.params[1],
        "beta1": fit.params[2],
        "beta2": fit.params[3],
        "se_beta0": fit.bse[1],
        "se_beta1": fit.bse[2],
        "se_beta2": fit.bse[3],
        "p_beta0": fit.pvalues[1],
        "p_beta1": fit.pvalues[2],
        "p_beta2": fit.pvalues[3],
        "crossover_u": (1 - fit.params[1]) / fit.params[2]
            if abs(fit.params[2]) > 1e-10 else np.nan,
        "cov_matrix": fit.cov_params(),
        "R2": fit.rsquared,
        "N": int(fit.nobs),
    }


def fit_sigmoid_crossover(u: np.ndarray, beta_local: np.ndarray,
                          beta_se: np.ndarray | None = None) -> dict:
    """
    Fit sigmoid phase-transition model to effective exponent:
        beta_eff(u) = beta_min + (beta_max - beta_min) / (1 + exp(-k*(u - u*)))

    Also fits tanh variant:
        beta_eff(u) = beta_mid + delta_beta * tanh(k*(u - u*))

    Compares both with the linear model via AIC.

    Parameters
    ----------
    u : array of urbanization ratio values (bin centers).
    beta_local : array of local scaling exponents per bin.
    beta_se : optional standard errors for weighted fitting.

    Returns
    -------
    dict with sigmoid_params, tanh_params, linear_params, AIC comparison.
    """
    from scipy.optimize import curve_fit

    valid = np.isfinite(u) & np.isfinite(beta_local)
    u = u[valid]
    beta_local = beta_local[valid]
    if beta_se is not None:
        beta_se = beta_se[valid]

    n = len(u)
    if n < 5:
        return {"status": "insufficient_data", "n": n}

    sigma = beta_se if beta_se is not None else None

    def _aic(resid, k_params):
        ss = np.sum(resid ** 2)
        return n * np.log(ss / n) + 2 * k_params

    # --- Sigmoid ---
    def sigmoid(u_arr, b_min, b_max, k, u_star):
        return b_min + (b_max - b_min) / (1 + np.exp(-k * (u_arr - u_star)))

    try:
        p0_sig = [beta_local.min(), beta_local.max(), 5.0, np.median(u)]
        bounds_sig = ([-5, -5, 0.1, u.min() - 0.1],
                      [5, 5, 100.0, u.max() + 0.1])
        popt_sig, pcov_sig = curve_fit(sigmoid, u, beta_local, p0=p0_sig,
                                       bounds=bounds_sig, sigma=sigma,
                                       maxfev=10000)
        resid_sig = beta_local - sigmoid(u, *popt_sig)
        aic_sig = _aic(resid_sig, 4)
        sig_result = {
            "beta_min": popt_sig[0], "beta_max": popt_sig[1],
            "k": popt_sig[2], "u_star": popt_sig[3],
            "se": np.sqrt(np.diag(pcov_sig)),
            "AIC": aic_sig, "R2": 1 - np.sum(resid_sig**2) / np.sum((beta_local - beta_local.mean())**2),
        }
    except RuntimeError:
        sig_result = {"status": "fit_failed"}

    # --- Tanh ---
    def tanh_model(u_arr, b_mid, delta_b, k, u_star):
        return b_mid + delta_b * np.tanh(k * (u_arr - u_star))

    try:
        p0_tanh = [beta_local.mean(), (beta_local.max() - beta_local.min()) / 2,
                   5.0, np.median(u)]
        bounds_tanh = ([-5, 0.01, 0.1, u.min() - 0.1],
                       [5, 5, 100.0, u.max() + 0.1])
        popt_tanh, pcov_tanh = curve_fit(tanh_model, u, beta_local, p0=p0_tanh,
                                          bounds=bounds_tanh, sigma=sigma,
                                          maxfev=10000)
        resid_tanh = beta_local - tanh_model(u, *popt_tanh)
        aic_tanh = _aic(resid_tanh, 4)
        # u* where beta_eff = 1: solve b_mid + delta_b * tanh(k*(u-u*)) = 1
        # tanh(k*(u-u*)) = (1 - b_mid) / delta_b
        arg = (1 - popt_tanh[0]) / popt_tanh[1]
        u_cross_tanh = popt_tanh[3] + np.arctanh(np.clip(arg, -0.999, 0.999)) / popt_tanh[2]
        tanh_result = {
            "beta_mid": popt_tanh[0], "delta_beta": popt_tanh[1],
            "k": popt_tanh[2], "u_star": popt_tanh[3],
            "u_crossover_1": u_cross_tanh,
            "se": np.sqrt(np.diag(pcov_tanh)),
            "AIC": aic_tanh, "R2": 1 - np.sum(resid_tanh**2) / np.sum((beta_local - beta_local.mean())**2),
        }
    except RuntimeError:
        tanh_result = {"status": "fit_failed"}

    # --- Linear ---
    slope, intercept, r_val, p_val, se_slope = stats.linregress(u, beta_local)
    resid_lin = beta_local - (intercept + slope * u)
    aic_lin = _aic(resid_lin, 2)
    u_cross_lin = (1 - intercept) / slope if abs(slope) > 1e-10 else np.nan
    lin_result = {
        "beta0": intercept, "beta1": slope, "se_beta1": se_slope,
        "u_crossover_1": u_cross_lin,
        "AIC": aic_lin, "R2": r_val**2,
    }

    return {
        "sigmoid": sig_result,
        "tanh": tanh_result,
        "linear": lin_result,
        "n": n,
        "u": u,
        "beta_local": beta_local,
        "beta_se": beta_se,
    }


def bootstrap_crossover_ci(df: pd.DataFrame,
                            y_col: str = "RegGDP_Tertiary",
                            n_boot: int = 500,
                            block_col: str = None,
                            seed: int = 42) -> dict:
    """
    Block bootstrap CI for crossover u*.

    Block bootstrap by province: resample provinces with replacement,
    run interaction regression on each bootstrap sample, collect u*.

    Returns: mean_u_star, ci_lower, ci_upper, u_stars_array.
    """
    from src.config import PROVINCE_CODE_MAP

    rng = np.random.default_rng(seed)
    data = df.copy()
    data["_ProvinceCode"] = data["CountyCode"].astype(int) // 10000
    data["_Province"] = data["_ProvinceCode"].map(PROVINCE_CODE_MAP)
    data = data.dropna(subset=["_Province"])

    provinces = data["_Province"].unique()
    u_stars = []

    for _ in range(n_boot):
        # Resample provinces with replacement
        boot_provs = rng.choice(provinces, size=len(provinces), replace=True)
        boot_dfs = []
        for p in boot_provs:
            boot_dfs.append(data[data["_Province"] == p])
        boot_data = pd.concat(boot_dfs, ignore_index=True)

        try:
            res = run_interaction_regression(boot_data, y_col=y_col)
            u_star = res["crossover_u"]
            if np.isfinite(u_star) and -1 < u_star < 3:
                u_stars.append(u_star)
        except (ValueError, KeyError):
            continue

    u_stars = np.array(u_stars)
    return {
        "mean_u_star": np.mean(u_stars),
        "median_u_star": np.median(u_stars),
        "ci_lower": np.percentile(u_stars, 2.5),
        "ci_upper": np.percentile(u_stars, 97.5),
        "n_valid": len(u_stars),
        "n_boot": n_boot,
        "u_stars": u_stars,
    }
