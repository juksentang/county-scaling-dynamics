"""
Data cleaning: type conversion, county code standardization, district filtering,
province lookup, and panel validation.
"""
import pandas as pd
import numpy as np
from src.config import PROVINCE_CODE_MAP, get_region


def coerce_numeric_columns(df: pd.DataFrame,
                           exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Convert all non-excluded columns to numeric where possible.

    Parameters
    ----------
    df : DataFrame
    exclude_cols : list of column names to leave as-is.
        Defaults to common identifier columns.

    Returns
    -------
    DataFrame with numeric columns coerced.
    """
    if exclude_cols is None:
        exclude_cols = ["CountyName", "PrefLevCity", "Province", "AreaID",
                        "Region", "City"]

    df = df.copy()
    for col in df.columns:
        if col in exclude_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_county_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize CountyCode to integer, validate 6 digits,
    and extract ProvinceCode (first 2 digits).

    Returns
    -------
    DataFrame with CountyCode (Int64) and ProvinceCode (int) columns.
    """
    df = df.copy()
    df["CountyCode"] = pd.to_numeric(df["CountyCode"], errors="coerce").astype("Int64")

    # Validate: should be 6-digit codes (100000 - 999999)
    valid = df["CountyCode"].between(100000, 999999)
    n_invalid = (~valid & df["CountyCode"].notna()).sum()
    if n_invalid > 0:
        print(f"WARNING: {n_invalid} rows with invalid CountyCode (not 6-digit)")
        df.loc[~valid & df["CountyCode"].notna(), "CountyCode"] = pd.NA

    # Extract province code (first 2 digits)
    df["ProvinceCode"] = (df["CountyCode"] // 10000).astype("Int64")

    return df


def filter_county_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove municipal districts and prefecture-level entries.

    Chinese admin code: AABBCC (AA=province, BB=prefecture, CC=county type).
    CC = 01-20: municipal districts → remove
    CC = 21-80: counties / autonomous counties → keep
    CC = 81-99: county-level cities → keep
    CC = 00: prefecture-level aggregate → remove
    """
    df = df.copy()
    county_type = df["CountyCode"] % 100
    is_county = county_type >= 21
    n_removed = (~is_county).sum()
    n_total = len(df)
    df = df[is_county].reset_index(drop=True)
    print(f"District filter: {n_total} → {len(df)} rows "
          f"(removed {n_removed} municipal district rows)")
    return df


def add_province_from_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Province column from CountyCode's first 2 digits.
    Only fills where Province is missing.
    """
    df = df.copy()
    if "ProvinceCode" not in df.columns:
        df["ProvinceCode"] = (df["CountyCode"] // 10000).astype("Int64")

    code_province = df["ProvinceCode"].map(PROVINCE_CODE_MAP)

    if "Province" not in df.columns:
        df["Province"] = code_province
    else:
        df["Province"] = df["Province"].fillna(code_province)

    return df


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    """Add Region column (East / Central / West) from Province."""
    df = df.copy()
    if "Province" not in df.columns:
        df = add_province_from_code(df)
    df["Region"] = df["Province"].map(get_region)
    return df


def validate_panel(df: pd.DataFrame, name: str = "panel") -> dict:
    """
    Report panel structure: county count, year range, balance,
    duplicate keys, and missing rates.

    Returns
    -------
    dict with validation results.
    """
    info = {"name": name}

    # Basic shape
    info["n_rows"] = len(df)
    info["n_cols"] = len(df.columns)

    # Year range
    if "Year" in df.columns:
        info["year_min"] = int(df["Year"].min())
        info["year_max"] = int(df["Year"].max())
        info["n_years"] = df["Year"].nunique()

    # County count
    if "CountyCode" in df.columns:
        info["n_counties"] = df["CountyCode"].nunique()

    # Balance check
    if "Year" in df.columns and "CountyCode" in df.columns:
        expected = info.get("n_counties", 0) * info.get("n_years", 0)
        info["expected_rows"] = expected
        info["balance_ratio"] = info["n_rows"] / expected if expected > 0 else 0

        # Check duplicates
        dup = df.duplicated(subset=["Year", "CountyCode"], keep=False)
        info["n_duplicates"] = dup.sum()

    # Missing rates
    missing = df.isnull().mean()
    info["missing_rates"] = missing[missing > 0].to_dict()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Panel Validation: {name}")
    print(f"{'='*60}")
    print(f"Shape: {info['n_rows']:,} rows × {info['n_cols']} columns")
    if "year_min" in info:
        print(f"Years: {info['year_min']}-{info['year_max']} ({info['n_years']} unique)")
    if "n_counties" in info:
        print(f"Counties: {info['n_counties']:,} unique")
    if "expected_rows" in info:
        print(f"Balance: {info['balance_ratio']:.1%} "
              f"({info['n_rows']:,} / {info['expected_rows']:,})")
    if "n_duplicates" in info:
        print(f"Duplicate (Year, CountyCode): {info['n_duplicates']}")
    if info.get("missing_rates"):
        print(f"Missing rates (top 10):")
        sorted_missing = sorted(info["missing_rates"].items(), key=lambda x: -x[1])
        for col, rate in sorted_missing[:10]:
            print(f"  {col}: {rate:.1%}")

    return info


def deduplicate_panel(df: pd.DataFrame,
                      keep: str = "first") -> pd.DataFrame:
    """Remove duplicate (Year, CountyCode) rows, keeping first/last."""
    n_before = len(df)
    df = df.drop_duplicates(subset=["Year", "CountyCode"], keep=keep)
    n_after = len(df)
    if n_before != n_after:
        print(f"Dedup: {n_before} → {n_after} (removed {n_before - n_after})")
    return df.reset_index(drop=True)
