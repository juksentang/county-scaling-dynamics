"""
Excel data loading for three header formats + long-format pivot.
All loaders return DataFrames with standardized Year (int) and CountyCode (int) columns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_RAW, THEMED_FILES, LONG_FORMAT_FILES, POVERTY_FILE, MAIN_DATABASE


def _resolve_file(filename: str) -> Path:
    """Resolve filename to full path in DATA_RAW."""
    p = DATA_RAW / filename
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    return p


def load_themed_file(dataset_key: str, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load a themed Excel file with multi-row headers.

    Parameters
    ----------
    dataset_key : str
        Key in THEMED_FILES (e.g., 'agri_output', 'gdp').
    columns : list[str] or None
        Specific key_cols to keep. If None, keeps all key_cols.

    Returns
    -------
    DataFrame with Year (int), CountyCode (str), and requested value columns.
    """
    cfg = THEMED_FILES[dataset_key]
    fpath = _resolve_file(cfg["file"])

    df = pd.read_excel(
        fpath,
        sheet_name=cfg["sheet"],
        skiprows=cfg["skiprows"],
        dtype=str,
    )

    year_col = cfg["year_col"]
    if year_col in df.columns:
        df = df.rename(columns={year_col: "Year"})
    elif "Year" not in df.columns:
        raise KeyError(f"Year column '{year_col}' not found in {cfg['file']}. "
                       f"Columns: {df.columns.tolist()}")

    id_cols = ["Year", "CountyCode"]
    optional_id = ["CountyName", "PrefLevCity", "Province", "AreaID"]
    keep_ids = [c for c in id_cols + optional_id if c in df.columns]

    if columns is not None:
        val_cols = [c for c in columns if c in df.columns]
    else:
        val_cols = [c for c in cfg["key_cols"] if c in df.columns]

    df = df[keep_ids + val_cols].copy()

    if "year_id" in df.columns:
        df = df.drop(columns=["year_id"])

    return df


def load_long_format_file(dataset_key: str) -> pd.DataFrame:
    """
    Load a long-format Excel file, filter by type, and pivot to wide format.

    Parameters
    ----------
    dataset_key : str
        Key in LONG_FORMAT_FILES (e.g., 'crop_area', 'crop_yield').

    Returns
    -------
    DataFrame with Year, CountyCode, and one column per filter value.
    """
    cfg = LONG_FORMAT_FILES[dataset_key]
    fpath = _resolve_file(cfg["file"])

    df = pd.read_excel(
        fpath,
        sheet_name=cfg["sheet"],
        skiprows=cfg["skiprows"],
        dtype=str,
    )

    year_col = cfg["year_col"]
    if year_col in df.columns and year_col != "Year":
        df = df.rename(columns={year_col: "Year"})

    type_col = cfg["type_col"]
    value_col = cfg["value_col"]

    if "CountyCode" not in df.columns:
        raise KeyError(f"CountyCode not found in {cfg['file']}. Columns: {df.columns.tolist()}")

    pieces = []
    for new_col, filter_val in cfg["filters"].items():
        sub = df[df[type_col] == filter_val].copy()
        sub = sub.rename(columns={value_col: new_col})
        sub = sub[["Year", "CountyCode", new_col]].copy()
        pieces.append(sub)

    if len(pieces) == 1:
        return pieces[0]

    result = pieces[0]
    for p in pieces[1:]:
        result = result.merge(p, on=["Year", "CountyCode"], how="outer")

    return result


def load_poverty_list() -> pd.DataFrame:
    """
    Load the 832 national poverty county list.

    Returns
    -------
    DataFrame with CountyCode (int), CountyName, Region, Province, City, ExitYear (int).
    """
    cfg = POVERTY_FILE
    fpath = _resolve_file(cfg["file"])

    df = pd.read_excel(fpath, sheet_name=cfg["sheet"], dtype=str)

    df = df.rename(columns=cfg["col_map"])

    df = df[[c for c in cfg["col_map"].values() if c in df.columns]].copy()

    df["CountyCode"] = pd.to_numeric(df["CountyCode"], errors="coerce").astype("Int64")
    df["ExitYear"] = pd.to_numeric(df["ExitYear"], errors="coerce").astype("Int64")

    n = len(df)
    if n != cfg["expected_rows"]:
        print(f"WARNING: Expected {cfg['expected_rows']} poverty counties, got {n}")

    exit_dist = df["ExitYear"].value_counts().sort_index()
    print(f"Poverty county list loaded: {n} counties")
    print(f"Exit year distribution:\n{exit_dist.to_string()}")

    return df


def load_main_database(columns: list[str] | None = None,
                       use_interpolated: bool = False) -> pd.DataFrame:
    """
    Load the main county database 6.0 (backup / cross-validation).

    Parameters
    ----------
    columns : list[str] or None
        Chinese column names to keep. If None, keeps all.
    use_interpolated : bool
        If True, use the linearly interpolated sheet.

    Returns
    -------
    DataFrame with Chinese column names (use MAIN_DATABASE['col_map'] to rename).
    """
    cfg = MAIN_DATABASE
    fpath = _resolve_file(cfg["file"])
    sheet = cfg["sheets"]["interpolated"] if use_interpolated else cfg["sheets"]["raw"]

    read_kwargs = {"sheet_name": sheet}
    if columns is not None:
        read_kwargs["usecols"] = columns

    df = pd.read_excel(fpath, **read_kwargs)
    return df


def load_spatial_coordinates() -> pd.DataFrame:
    """Returns
    -------
    DataFrame with Year, CountyCode, CountyLng, CountyLat.
    """
    df = load_themed_file("spatial", columns=["CountyLng", "CountyLat"])
    return df
