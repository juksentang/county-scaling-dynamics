"""
Province-level GDP deflator construction and monetary variable deflation.

Strategy:
- RegGDPIdx is a year-on-year REAL GDP growth index (previous year = 100).
- We have nominal GDP (RegGDP) for each county-year.
- Implicit GDP deflator = NominalGrowth / RealGrowth per province.
- Use SUM of county GDPs as province-level nominal GDP (robust to county count changes).
- Deflation: value_real = value_nominal / Deflator
"""
import pandas as pd
import numpy as np
from src.config import BASE_YEAR


def build_province_gdp_deflator(gdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build province-level implicit GDP deflator.

    Method:
    1. For each province-year: sum nominal GDP across counties → ProvNomGDP(t)
    2. For each province-year: median RegGDPIdx across counties → growth rate
    3. Build cumulative real index: CumReal(t) = CumReal(t-1) × MedianIdx(t)/100
    4. Deflator(t) = [ProvNomGDP(t)/ProvNomGDP(base)] / CumReal(t)
    5. Normalize so Deflator(base_year) = 1.0

    Parameters
    ----------
    gdp_df : DataFrame
        Must contain Year, Province, RegGDP, RegGDPIdx columns.

    Returns
    -------
    DataFrame with columns: Province, Year, Deflator
    """
    df = gdp_df[["Year", "Province", "RegGDP", "RegGDPIdx"]].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["RegGDP"] = pd.to_numeric(df["RegGDP"], errors="coerce")
    df["RegGDPIdx"] = pd.to_numeric(df["RegGDPIdx"], errors="coerce")
    df = df.dropna(subset=["Year", "Province"])

    # Province-year aggregation
    agg = (
        df.groupby(["Province", "Year"])
        .agg(
            SumGDP=("RegGDP", "sum"),
            NCountyGDP=("RegGDP", "count"),
            MedianIdx=("RegGDPIdx", "median"),
        )
        .reset_index()
    )

    # Compute average GDP per reporting county (handles changing county counts)
    agg["AvgGDP"] = agg["SumGDP"] / agg["NCountyGDP"]

    results = []
    for prov, grp in agg.groupby("Province"):
        grp = grp.sort_values("Year").copy()
        years = grp["Year"].values.astype(int)
        avg_gdp = grp["AvgGDP"].values
        idx_vals = grp["MedianIdx"].values

        # Find base year position
        base_pos = np.searchsorted(years, BASE_YEAR)
        if base_pos >= len(years) or years[base_pos] != BASE_YEAR:
            # Base year not available — use earliest year as base
            base_pos = 0

        # 1. Cumulative real index (from RegGDPIdx)
        cum_real = np.ones(len(years))
        for i in range(base_pos + 1, len(years)):
            rate = idx_vals[i] / 100.0 if not np.isnan(idx_vals[i]) else 1.0
            # Clamp extreme values
            rate = np.clip(rate, 0.5, 2.0)
            cum_real[i] = cum_real[i-1] * rate

        for i in range(base_pos - 1, -1, -1):
            rate = idx_vals[i+1] / 100.0 if not np.isnan(idx_vals[i+1]) else 1.0
            rate = np.clip(rate, 0.5, 2.0)
            cum_real[i] = cum_real[i+1] / rate

        # 2. Nominal growth index (using avg GDP per county for robustness)
        base_gdp = avg_gdp[base_pos]
        if np.isnan(base_gdp) or base_gdp <= 0:
            base_gdp = np.nanmedian(avg_gdp)
        nom_idx = avg_gdp / base_gdp

        # 3. Implicit deflator = nominal growth / real growth
        deflator = np.where(cum_real > 0, nom_idx / cum_real, np.nan)

        # Normalize so base year = 1.0
        base_val = deflator[base_pos]
        if not np.isnan(base_val) and base_val > 0:
            deflator = deflator / base_val

        for yr, d in zip(years, deflator):
            if not np.isnan(d) and d > 0:
                results.append({"Province": prov, "Year": int(yr), "Deflator": d})

    deflator_df = pd.DataFrame(results)

    # Sanity check: deflator should typically be 1-5x for 2000-2023
    med = deflator_df.groupby("Year")["Deflator"].median()
    print(f"  Deflator median by year (sample): "
          f"{med.iloc[0]:.2f} ({int(med.index[0])}) → "
          f"{med.iloc[-1]:.2f} ({int(med.index[-1])})")

    return deflator_df


def deflate_all_monetary(df: pd.DataFrame,
                         deflator_df: pd.DataFrame,
                         var_list: list[str],
                         suffix: str = "_real") -> pd.DataFrame:
    """
    Deflate all monetary variables in var_list.

    Parameters
    ----------
    df : DataFrame with Province, Year, and monetary columns.
    deflator_df : DataFrame with Province, Year, Deflator.
    var_list : list of column names to deflate.
    suffix : str, suffix for real-value columns.

    Returns
    -------
    DataFrame with additional {col}{suffix} columns.
    """
    df = df.merge(
        deflator_df[["Province", "Year", "Deflator"]],
        on=["Province", "Year"],
        how="left",
    )

    for col in var_list:
        if col in df.columns:
            df[f"{col}{suffix}"] = df[col] / df["Deflator"]

    df = df.drop(columns=["Deflator"], errors="ignore")
    return df
