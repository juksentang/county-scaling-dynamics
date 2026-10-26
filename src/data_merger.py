"""
Panel construction: merge themed datasets, add poverty status, spatial info,
derived variables, and save/load parquet.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    DATA_CLEAN, DEFLATE_VARS, FP_PERIODS, get_region, PROVINCE_CODE_MAP
)
from src.data_loader import (
    load_themed_file, load_long_format_file, load_poverty_list,
    load_spatial_coordinates,
)
from src.data_cleaner import (
    coerce_numeric_columns, standardize_county_code, filter_county_type,
    add_province_from_code, add_region, validate_panel, deduplicate_panel,
)
from src.deflator import build_province_gdp_deflator, deflate_all_monetary


def _load_and_clean(dataset_key: str, columns: list[str] | None = None,
                    long_format: bool = False) -> pd.DataFrame:
    """Load, coerce numeric, standardize county code."""
    if long_format:
        df = load_long_format_file(dataset_key)
    else:
        df = load_themed_file(dataset_key, columns=columns)
    df = coerce_numeric_columns(df)
    df = standardize_county_code(df)
    df["Year"] = df["Year"].astype("Int64")
    df = deduplicate_panel(df)
    return df


def merge_core_panel() -> pd.DataFrame:
    """
    Merge all core datasets into a single panel.
    Population is the base table; all others left-joined.

    Returns
    -------
    DataFrame: merged panel with Year, CountyCode, and all value columns.
    """
    merge_on = ["Year", "CountyCode"]

    print("Loading population...")
    pop = _load_and_clean("population")
    base = pop[merge_on + ["CountyName", "PrefLevCity", "Province", "PopNum"]].copy()
    base = deduplicate_panel(base)
    print(f"  Base: {len(base):,} rows")

    print("Loading GDP...")
    gdp = _load_and_clean("gdp")
    gdp_cols = merge_on + [c for c in gdp.columns
                           if c not in merge_on + ["CountyName", "PrefLevCity",
                                                    "Province", "AreaID",
                                                    "ProvinceCode"]]
    gdp = gdp[gdp_cols].copy()
    gdp = deduplicate_panel(gdp)
    base = base.merge(gdp, on=merge_on, how="left")
    print(f"  After GDP merge: {len(base):,} rows")

    print("Loading agricultural output...")
    agri = _load_and_clean("agri_output")
    agri_keep = merge_on + ["GrossOutputValue", "Farming", "Forestry",
                            "AnimalHusbandry", "Fishery", "FFAFServices"]
    agri_keep = [c for c in agri_keep if c in agri.columns]
    agri = agri[agri_keep].copy()
    agri = deduplicate_panel(agri)
    base = base.merge(agri, on=merge_on, how="left")
    print(f"  After agri merge: {len(base):,} rows")

    print("Loading land area...")
    land = _load_and_clean("land_area")
    land_keep = merge_on + ["AdmArea", "CultivateArea"]
    land_keep = [c for c in land_keep if c in land.columns]
    land = land[land_keep].copy()
    land = deduplicate_panel(land)
    base = base.merge(land, on=merge_on, how="left")
    print(f"  After land merge: {len(base):,} rows")

    print("Loading agricultural conditions...")
    cond = _load_and_clean("agri_conditions")
    cond_keep = merge_on + ["AgrMachPower", "FertilizerConsm",
                            "ElectricityConsm", "IrrigatedArea"]
    cond_keep = [c for c in cond_keep if c in cond.columns]
    cond = cond[cond_keep].copy()
    cond = deduplicate_panel(cond)
    base = base.merge(cond, on=merge_on, how="left")
    print(f"  After conditions merge: {len(base):,} rows")

    print("Loading crop area (total)...")
    crop_area = _load_and_clean("crop_area", long_format=True)
    crop_area_keep = merge_on + ["TotalSownArea"]
    crop_area_keep = [c for c in crop_area_keep if c in crop_area.columns]
    crop_area = crop_area[crop_area_keep].copy()
    crop_area = deduplicate_panel(crop_area)
    base = base.merge(crop_area, on=merge_on, how="left")
    print(f"  After crop area merge: {len(base):,} rows")

    print("Loading grain sown area...")
    grain_area = _load_and_clean("grain_area", long_format=True)
    grain_area_keep = merge_on + ["GrainSownArea"]
    grain_area_keep = [c for c in grain_area_keep if c in grain_area.columns]
    grain_area = grain_area[grain_area_keep].copy()
    grain_area = deduplicate_panel(grain_area)
    base = base.merge(grain_area, on=merge_on, how="left")
    print(f"  After grain area merge: {len(base):,} rows")

    print("Loading crop yield...")
    crop_yield = _load_and_clean("crop_yield", long_format=True)
    crop_yield_keep = merge_on + ["GrainOutput"]
    crop_yield_keep = [c for c in crop_yield_keep if c in crop_yield.columns]
    crop_yield = crop_yield[crop_yield_keep].copy()
    crop_yield = deduplicate_panel(crop_yield)
    base = base.merge(crop_yield, on=merge_on, how="left")
    print(f"  After crop yield merge: {len(base):,} rows")

    print("Loading income...")
    income = _load_and_clean("income")
    income_keep = merge_on + ["RurPerNetInc", "RurPerDisposableInc"]
    income_keep = [c for c in income_keep if c in income.columns]
    income = income[income_keep].copy()
    income = deduplicate_panel(income)
    base = base.merge(income, on=merge_on, how="left")
    print(f"  After income merge: {len(base):,} rows")

    print("Loading finance...")
    fin = _load_and_clean("finance")
    fin_keep = merge_on + ["Loan", "Deposit"]
    fin_keep = [c for c in fin_keep if c in fin.columns]
    fin = fin[fin_keep].copy()
    fin = deduplicate_panel(fin)
    base = base.merge(fin, on=merge_on, how="left")
    print(f"  After finance merge: {len(base):,} rows")

    print("Loading employment...")
    emp = _load_and_clean("employment")
    emp_keep = merge_on + ["EmployeeNum", "RuralEmployeeNum", "FFAFEmployeeNum"]
    emp_keep = [c for c in emp_keep if c in emp.columns]
    emp = emp[emp_keep].copy()
    emp = deduplicate_panel(emp)
    base = base.merge(emp, on=merge_on, how="left")
    print(f"  After employment merge: {len(base):,} rows")

    print("Loading fixed assets...")
    fix = _load_and_clean("fixed_assets")
    fix_keep = merge_on + ["FixAst"]
    fix_keep = [c for c in fix_keep if c in fix.columns]
    fix = fix[fix_keep].copy()
    fix = deduplicate_panel(fix)
    base = base.merge(fix, on=merge_on, how="left")
    print(f"  After fixed assets merge: {len(base):,} rows")

    return base


def add_poverty_status(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add poverty county indicators to panel.

    Adds columns:
    - IsPoverty (bool): whether county is in the 832 list
    - ExitYear (int): year of poverty delisting
    - PreExit (bool): year < ExitYear (still in poverty)
    - PostExit (bool): year >= ExitYear (after delisting)
    """
    poverty = load_poverty_list()

    panel = panel.copy()
    poverty_map = poverty.set_index("CountyCode")["ExitYear"].to_dict()

    panel["IsPoverty"] = panel["CountyCode"].isin(poverty["CountyCode"].values)
    panel["ExitYear"] = panel["CountyCode"].map(poverty_map)
    panel["PreExit"] = panel["Year"] < panel["ExitYear"]
    panel["PostExit"] = panel["Year"] >= panel["ExitYear"]

    panel.loc[~panel["IsPoverty"], "PreExit"] = False
    panel.loc[~panel["IsPoverty"], "PostExit"] = False

    n_poverty = panel[panel["IsPoverty"]]["CountyCode"].nunique()
    print(f"Poverty status added: {n_poverty} poverty counties tagged")

    return panel


def add_spatial_info(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add longitude/latitude from spatial file.
    For years < 2017, use 2017 coordinates as approximation.
    """
    spatial = load_spatial_coordinates()
    spatial = coerce_numeric_columns(spatial)
    spatial = standardize_county_code(spatial)
    spatial["Year"] = spatial["Year"].astype("Int64")

    earliest = (
        spatial.sort_values("Year")
        .drop_duplicates(subset=["CountyCode"], keep="first")
        [["CountyCode", "CountyLng", "CountyLat"]]
    )

    panel = panel.copy()

    spatial_slim = spatial[["Year", "CountyCode", "CountyLng", "CountyLat"]].copy()
    spatial_slim = deduplicate_panel(spatial_slim)
    panel = panel.merge(spatial_slim, on=["Year", "CountyCode"], how="left")

    missing_lng = panel["CountyLng"].isna()
    if missing_lng.any():
        fill = panel.loc[missing_lng, ["CountyCode"]].merge(
            earliest, on="CountyCode", how="left", suffixes=("", "_fill")
        )
        panel.loc[missing_lng, "CountyLng"] = fill["CountyLng"].values
        panel.loc[missing_lng, "CountyLat"] = fill["CountyLat"].values

    n_with_coords = panel["CountyLng"].notna().sum()
    print(f"Spatial info added: {n_with_coords:,}/{len(panel):,} rows with coordinates")

    return panel


def add_derived_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Add derived variables needed for analysis."""
    panel = panel.copy()

    if "RegGDP_Secondary" in panel.columns and "RegGDP_Tertiary" in panel.columns:
        panel["NonAgriGDP"] = (
            panel["RegGDP_Secondary"].fillna(0) + panel["RegGDP_Tertiary"].fillna(0)
        )
        both_missing = (
            panel["RegGDP_Secondary"].isna() & panel["RegGDP_Tertiary"].isna()
        )
        panel.loc[both_missing, "NonAgriGDP"] = np.nan

    panel = add_province_from_code(panel)

    panel = add_region(panel)

    return panel


def apply_deflation(panel: pd.DataFrame) -> pd.DataFrame:
    """Build deflator and deflate all monetary variables."""
    print("Building GDP deflator...")

    deflator = build_province_gdp_deflator(panel)

    print(f"  Deflator coverage: {len(deflator):,} province-year combinations")
    print(f"  Deflator range: {deflator['Deflator'].min():.2f} - {deflator['Deflator'].max():.2f}")

    vars_to_deflate = [v for v in DEFLATE_VARS if v in panel.columns]
    if "NonAgriGDP" in panel.columns:
        vars_to_deflate.append("NonAgriGDP")

    print(f"  Deflating {len(vars_to_deflate)} variables...")
    panel = deflate_all_monetary(panel, deflator, vars_to_deflate, suffix="_real")

    return panel


def build_scaling_panel(panel: pd.DataFrame,
                        require_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Filter panel to rows where core scaling variables are non-null and positive.

    Parameters
    ----------
    require_cols : list of column names that must be non-null.
        Defaults to PopNum + GrossOutputValue + RegGDP (minimum for Eq 14a-14b).
        CultivateArea is NOT required by default since it limits coverage.

    Returns
    -------
    Filtered DataFrame.
    """
    if require_cols is None:
        require_cols = ["PopNum", "GrossOutputValue", "RegGDP"]

    df = panel.copy()
    n_before = len(df)

    for col in require_cols:
        if col in df.columns:
            df = df[df[col].notna() & (df[col] > 0)]

    n_after = len(df)
    print(f"Scaling panel filter: {n_before:,} → {n_after:,} rows "
          f"(require: {require_cols})")
    return df.reset_index(drop=True)


def save_panel(df: pd.DataFrame, name: str = "panel_scaling") -> Path:
    """Save DataFrame as parquet to DATA_CLEAN."""
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    path = DATA_CLEAN / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved: {path} ({len(df):,} rows)")
    return path


def load_panel(name: str = "panel_scaling") -> pd.DataFrame:
    """Load panel from parquet."""
    path = DATA_CLEAN / f"{name}.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded: {path} ({len(df):,} rows, {len(df.columns)} cols)")
    return df


def run_full_pipeline() -> pd.DataFrame:
    """
    Execute the complete data pipeline:
    1. Merge all themed datasets
    2. Filter county types (remove municipal districts)
    3. Add poverty status
    4. Add derived variables
    5. Apply deflation
    6. Add spatial info
    7. Save panel

    Returns
    -------
    Complete panel DataFrame.
    """
    print("=" * 60)
    print("RUNNING FULL DATA PIPELINE")
    print("=" * 60)

    panel = merge_core_panel()

    print("\nFiltering municipal districts...")
    panel = filter_county_type(panel)

    print("\nAdding derived variables...")
    panel = add_derived_variables(panel)

    print("\nApplying deflation...")
    panel = apply_deflation(panel)

    print("\nAdding poverty status...")
    panel = add_poverty_status(panel)

    print("\nAdding spatial info...")
    panel = add_spatial_info(panel)

    print("\n")
    validate_panel(panel, "Full Panel")

    save_panel(panel, "panel_full")

    print("\nBuilding scaling panel...")
    scaling = build_scaling_panel(panel)
    validate_panel(scaling, "Scaling Panel")
    save_panel(scaling, "panel_scaling")

    return panel
