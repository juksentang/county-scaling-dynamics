"""
Scaling-Adjusted Anomaly Index (SAAI) computation.

SAAI_i(t) = [ln Y_i(t) - (α_t + β_t ln N_i(t) + γ_t ln A_i(t))] / σ_t

This measures how much a county's agricultural output deviates from
what the scaling law predicts, normalized by the cross-sectional dispersion.
Negative SAAI = underperforming relative to scaling prediction.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_saai(df: pd.DataFrame,
                 y_col: str = "GrossOutputValue",
                 x_cols: list[str] | None = None,
                 by_year: bool = True,
                 pooled_sigma: bool = False) -> pd.DataFrame:
    """
    Compute multi-factor SAAI (Eq 3).

    Parameters
    ----------
    df : DataFrame with Year, CountyCode, y_col, and x_cols.
    y_col : dependent variable (agricultural output).
    x_cols : independent variables. Default: ['PopNum', 'CultivateArea'].
    by_year : if True, run separate regression per year.
    pooled_sigma : if True, normalize by pooled (all-year) σ instead of year-specific σ.

    Returns
    -------
    DataFrame with original columns plus:
    - SAAI: the scaling-adjusted anomaly index
    - SAAI_residual: raw log residual (before normalization)
    """
    if x_cols is None:
        x_cols = ["PopNum", "CultivateArea"]

    all_cols = [y_col] + x_cols + ["Year", "CountyCode"]
    data = df[[c for c in all_cols if c in df.columns]].copy()

    # Keep only positive values
    for col in [y_col] + x_cols:
        if col in data.columns:
            data = data[data[col] > 0]
    data = data.dropna(subset=[y_col] + [c for c in x_cols if c in data.columns])

    results = []

    if by_year:
        groups = data.groupby("Year")
    else:
        groups = [("pooled", data)]

    for year, grp in groups:
        if len(grp) < 10:
            continue

        log_y = np.log(grp[y_col].values)
        available_x = [c for c in x_cols if c in grp.columns]
        log_X = np.column_stack([np.log(grp[c].values) for c in available_x])
        log_X = sm.add_constant(log_X)

        model = sm.OLS(log_y, log_X)
        fit = model.fit(cov_type="HC1")

        residuals = fit.resid
        sigma = residuals.std()

        sub = grp.copy()
        sub["SAAI_residual"] = residuals
        sub["SAAI"] = residuals / sigma if sigma > 0 else 0.0
        results.append(sub)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    # Pooled sigma: re-normalize using global σ across all years
    if pooled_sigma and "SAAI_residual" in combined.columns:
        pooled_std = combined["SAAI_residual"].std()
        if pooled_std > 0:
            combined["SAAI"] = combined["SAAI_residual"] / pooled_std

    return combined


def compute_saai_single_factor(df: pd.DataFrame,
                               y_col: str = "GrossOutputValue") -> pd.DataFrame:
    """
    Compute single-factor SAAI (only PopNum, no CultivateArea).
    For robustness comparison.
    """
    return compute_saai(df, y_col=y_col, x_cols=["PopNum"], by_year=True)


def get_saai_regression_params(df: pd.DataFrame,
                               y_col: str = "GrossOutputValue",
                               x_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Extract year-by-year regression parameters for SAAI computation.

    Returns DataFrame with Year, alpha_t, beta_t, gamma_t (if applicable),
    sigma_t, R2, N.
    """
    if x_cols is None:
        x_cols = ["PopNum", "CultivateArea"]

    all_needed = [y_col] + x_cols
    data = df.copy()
    for col in all_needed:
        if col in data.columns:
            data = data[data[col] > 0]
    data = data.dropna(subset=[c for c in all_needed if c in data.columns])

    records = []
    for year, grp in data.groupby("Year"):
        if len(grp) < 10:
            continue

        log_y = np.log(grp[y_col].values)
        available_x = [c for c in x_cols if c in grp.columns]
        log_X = np.column_stack([np.log(grp[c].values) for c in available_x])
        log_X = sm.add_constant(log_X)

        model = sm.OLS(log_y, log_X)
        fit = model.fit(cov_type="HC1")

        record = {
            "Year": year,
            "alpha": fit.params[0],
            "N_obs": fit.nobs,
            "R2": fit.rsquared,
            "sigma": fit.resid.std(),
        }
        for i, var in enumerate(available_x):
            record[f"beta_{var}"] = fit.params[i + 1]
            record[f"se_{var}"] = fit.bse[i + 1]

        records.append(record)

    return pd.DataFrame(records)


def compute_saai_grain(df: pd.DataFrame,
                       x_cols: list[str] | None = None) -> pd.DataFrame:
    """Physical-quantity SAAI: y = GrainOutput (tons).

    Default x_cols = ["PopNum", "TotalSownArea"] — the flow measure of land
    input is the natural regressor for physical output; the physical
    tonnage is immune to price effects and transfer accounting, which is
    the point of this robustness variant.
    """
    if x_cols is None:
        x_cols = ["PopNum", "TotalSownArea"]
    return compute_saai(df, y_col="GrainOutput", x_cols=x_cols)


def compute_saai_sown_area(df: pd.DataFrame,
                           y_col: str = "GrossOutputValue") -> pd.DataFrame:
    """Compute multi-factor SAAI using TotalSownArea instead of CultivateArea."""
    return compute_saai(df, y_col=y_col, x_cols=["PopNum", "TotalSownArea"])


def get_saai_sown_area_params(df: pd.DataFrame,
                              y_col: str = "GrossOutputValue") -> pd.DataFrame:
    """Extract year-by-year regression params for TotalSownArea-based SAAI."""
    return get_saai_regression_params(df, y_col=y_col,
                                      x_cols=["PopNum", "TotalSownArea"])


def compute_order_parameter(saai_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the order parameter m(t) = mean SAAI of poverty counties per year.

    Also computes:
    - m_all: mean SAAI of all counties (should be ≈ 0)
    - m_poverty: mean SAAI of poverty counties
    - m_nonpoverty: mean SAAI of non-poverty counties
    - effect_size: Cohen's d for poverty vs non-poverty
    """
    if "IsPoverty" not in saai_df.columns:
        raise ValueError("Need IsPoverty column to compute order parameter")

    records = []
    for year, grp in saai_df.groupby("Year"):
        pov = grp[grp["IsPoverty"]]["SAAI"]
        npov = grp[~grp["IsPoverty"]]["SAAI"]

        record = {
            "Year": year,
            "m_all": grp["SAAI"].mean(),
            "m_poverty": pov.mean() if len(pov) > 0 else np.nan,
            "m_nonpoverty": npov.mean() if len(npov) > 0 else np.nan,
            "n_poverty": len(pov),
            "n_nonpoverty": len(npov),
            "std_poverty": pov.std() if len(pov) > 1 else np.nan,
            "std_nonpoverty": npov.std() if len(npov) > 1 else np.nan,
        }

        # Cohen's d
        if len(pov) > 1 and len(npov) > 1:
            pooled_std = np.sqrt(
                ((len(pov)-1)*pov.std()**2 + (len(npov)-1)*npov.std()**2)
                / (len(pov) + len(npov) - 2)
            )
            record["cohens_d"] = (npov.mean() - pov.mean()) / pooled_std if pooled_std > 0 else 0

        records.append(record)

    return pd.DataFrame(records)


# ──────────────────────────── HHI (mechanism verification) ────────────────────
def compute_hhi(df: pd.DataFrame,
                sector_cols: list[str],
                hhi_col: str = "HHI") -> pd.Series:
    """
    Compute Herfindahl-Hirschman Index from sector columns.

    HHI = Σ (share_i)^2 where share_i = sector_i / Σ sectors.
    Range: [1/n, 1]. Higher = more concentrated.

    Returns a Series aligned with df.index; rows with missing/zero total → NaN.
    """
    sectors = df[sector_cols].copy()
    sectors[sectors < 0] = np.nan
    total = sectors.sum(axis=1)
    shares = sectors.div(total, axis=0)
    hhi = (shares ** 2).sum(axis=1)
    hhi[total <= 0] = np.nan
    hhi.name = hhi_col
    return hhi


def compute_agri_hhi(df: pd.DataFrame) -> pd.Series:
    """Agricultural HHI from 4 sub-sectors (Farming/Forestry/AnimalHusbandry/Fishery)."""
    return compute_hhi(df,
                       ["Farming_real", "Forestry_real",
                        "AnimalHusbandry_real", "Fishery_real"],
                       hhi_col="HHI_agri")


def compute_gdp_hhi(df: pd.DataFrame) -> pd.Series:
    """GDP HHI from 3 sectors (Primary/Secondary/Tertiary)."""
    return compute_hhi(df,
                       ["RegGDP_Primary_real", "RegGDP_Secondary_real",
                        "RegGDP_Tertiary_real"],
                       hhi_col="HHI_gdp")
