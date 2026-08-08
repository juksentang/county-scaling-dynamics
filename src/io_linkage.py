"""Input-output linkage intensity measures for the two-channel mechanism test.

Parses China's national input-output tables (competitive-imports type, current
producers' prices) for the vintages 2002-2020, computes four sector-level
interaction-intensity indicators, aggregates them to the three broad industries
(primary / secondary / tertiary), and runs the ordering test

    I_sec > I_ter > I_pri            (strong form)
    min(I_sec, I_ter) > I_pri        (weak form: interaction sectors > resource sector)

Indicators:

    m_j  = sum_i Z_ij / x_j          intermediate-input rate (column sum of A)
    BL_j = n * colsum(L)_j / sum(L)  Rasmussen backward linkage, L = (I - A)^-1
    FL_i = n * rowsum(G)_i / sum(G)  forward linkage (Ghosh), G = (I - B)^-1
    o_j  = sum_{i != j} Z_ij / x_j   non-self-supplied intermediate share
                                     (computed at the DISAGGREGATED sector level,
                                     then output-weighted -- never on the 3x3
                                     aggregate, to avoid diagonal inflation)

Aggregation: route B (output-weighted sector indicators) is the headline;
route A (flows aggregated to 3x3 first, indicators recomputed) is reported as a
cross-check for m / BL / FL.

Data: archive of national IO tables, one file per vintage; competitive-imports
sheets are used for 2017/2018/2020 for consistency across vintages.
All indicators are within-year ratios, so the monetary unit cancels.

Outputs:
    results/tables/io_linkage_indicators.csv
    results/tables/io_ordering_test.csv
    results/tables/io_table_validation.csv
    results/tables/io_sector_mapping_{year}.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import PROJECT_ROOT, TABLES
except ImportError:  # allow running as a plain script from repo root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TABLES = PROJECT_ROOT / "results" / "tables"

# ---------------------------------------------------------------- data files
IO_DIR = (
    PROJECT_ROOT
    / "data"
    / "1987-2020年中国全国各部门42部门投入产出表"
    / "1987-2020年中国全国各部门42部门投入产出表(非竞争性投入等"
    / "中国全国各部门42部门投入产出表（1987-2020年）"
    / "历年投入产出表"
    / "历年投入产出表"
)

# vintage -> (file name, sheet). Competitive-imports tables throughout.
IO_FILES: dict[int, tuple[str, int | str]] = {
    2002: ("2002.xls", 0),                       # benchmark, 122 sectors
    2005: ("2005.xls", 0),                       # extension, 42 sectors
    2007: ("2007.xls", 0),                       # benchmark, 135 sectors
    2010: ("2010.xls", 0),                       # extension, 41 sectors
    2012: ("2012.xlsx", 0),                      # benchmark, 139 sectors
    2015: ("2015.xlsx", 0),                      # extension, 42 sectors
    2017: ("2017竞争型投入产出表.xlsx", 0),        # benchmark, 149 sectors
    2018: ("2018竞争型投入产出表.xlsx", "生产价投入产出表"),  # extension, 153 sectors
    2020: ("2020竞争型投入产出表.xlsx", 0),        # extension, 153 sectors
}

# primary vintage set for the headline ordering test (survey benchmarks + 2020)
PRIMARY_VINTAGES = [2002, 2007, 2012, 2017, 2020]
ALL_VINTAGES = sorted(IO_FILES)

BENCHMARK = {2002, 2007, 2012, 2017}  # survey-based benchmark tables; rest are extensions

INDICATORS = ["m", "BL", "FL", "o"]

# name patterns for the positional industry rule and its cross-checks
_RE_MINING = re.compile(r"煤炭|石油和天然气|采选|开采|采矿")
_RE_CONSTRUCTION = re.compile(r"建筑")
_RE_PRIMARY_NAME = re.compile(r"农|林|畜牧|牧业|渔|木材")
# knowledge-intensive services, matched on sector names
_RE_KIS = re.compile(
    r"电信|信息传输|软件|信息技术|信息服务|互联网|货币金融|资本市场|保险|其他金融|金融业"
    r"|租赁和商务|租赁业|商务服务|科学研究|专业技术|研究和试验|研究与试验|技术服务|科技|教育"
)

# plain digit codes ('01001', '001', '01', '1') plus merged codes like '21/22'
# (2010 merges two handicrafts/other-manufacturing sectors) and split codes like
# '34072a'/'34072b' (2018/2020 split some machinery and construction sectors
# with letter suffixes)
_CODE_DIGITS = re.compile(r"^\d{1,5}(/\d{1,5})?[a-z]?$")
_CODE_5DIGIT = re.compile(r"^\d{5}[a-z]?$")


# ================================================================ parsing
def _norm_code(v) -> str | None:
    """Normalize a cell to a sector-code string ('01001', '001', '01', '1'), else None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else None
    s = str(v).strip()
    return s if _CODE_DIGITS.match(s) else None


def _modal_len(codes: list[str]) -> int:
    """Modal string length of a vintage's codes (robust to '21/22'-style outliers)."""
    lens = [len(c) for c in codes]
    return max(set(lens), key=lens.count)


def load_io_table(year: int) -> dict:
    """Parse one vintage file -> dict(Z, x, va, codes, names, meta).

    Z : DataFrame (n x n) intermediate-flow matrix, rows/cols indexed by code
    x : Series total output (from the TI total-input row; cross-checked
        against the GO total-output column)
    va: Series value added (TVA row)
    """
    fname, sheet = IO_FILES[year]
    path = IO_DIR / fname
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    # --- locate the code column (row labels); it also carries TII/TVA/TI markers
    ncols_scan = min(6, raw.shape[1])
    code_col = max(
        range(ncols_scan),
        key=lambda c: sum(_norm_code(v) is not None for v in raw.iloc[:, c]),
    )
    col_labels = raw.iloc[:, code_col].astype(str).str.strip()
    if "TII" not in set(col_labels):
        raise ValueError(f"{year}: TII marker not found in code column {code_col}")

    # --- sector rows and their codes (authoritative code set of the vintage)
    tii_r = int(np.where(col_labels.values == "TII")[0][-1])
    sec_rows = [
        r for r in range(tii_r) if _norm_code(raw.iat[r, code_col]) is not None
    ]
    raw_row_codes = [_norm_code(raw.iat[r, code_col]) for r in sec_rows]
    target_len = _modal_len(raw_row_codes)
    row_codes = [c.zfill(target_len) for c in raw_row_codes]

    # --- code row for columns: the header row with the largest overlap with the
    # row-code set (raw counts would be fooled by data rows full of integer zeros)
    code_set = set(row_codes)

    def _overlap(r: int) -> int:
        vals = [_norm_code(v) for v in raw.iloc[r, code_col + 1 :]]
        return sum(v is not None and v.zfill(target_len) in code_set for v in vals)

    code_row = max(range(min(sec_rows[0] + 1, raw.shape[0])), key=_overlap)
    if _overlap(code_row) < len(row_codes) * 0.9:
        raise ValueError(f"{year}: could not locate the column-code header row")

    sec_cols = [
        c
        for c in range(code_col + 1, raw.shape[1])
        if (v := _norm_code(raw.iat[code_row, c])) is not None
        and v.zfill(target_len) in code_set
    ]
    col_codes = [
        _norm_code(raw.iat[code_row, c]).zfill(target_len) for c in sec_cols
    ]
    if set(row_codes) != set(col_codes):
        raise ValueError(
            f"{year}: row/column sector codes differ "
            f"({len(row_codes)} rows vs {len(col_codes)} cols)"
        )

    # --- Z, aligned to the row-code order on both axes
    Z = raw.iloc[sec_rows, sec_cols].apply(pd.to_numeric, errors="coerce")
    Z.index = row_codes
    Z.columns = col_codes
    Z = Z.loc[row_codes, row_codes].fillna(0.0).astype(float)

    # --- totals: TI/TO1 row (total input = total output), TVA row, GO column
    def _marker_row(*names):
        for nm in names:
            hit = np.where(col_labels.values == nm)[0]
            if len(hit):
                return int(hit[-1])
        raise ValueError(f"{year}: none of {names} found in code column")

    ti_r = _marker_row("TI", "TO1")
    tva_r = _marker_row("TVA")
    x = pd.Series(
        pd.to_numeric(raw.iloc[ti_r, sec_cols], errors="coerce").values,
        index=col_codes,
        dtype=float,
    ).loc[row_codes]
    va = pd.Series(
        pd.to_numeric(raw.iloc[tva_r, sec_cols], errors="coerce").values,
        index=col_codes,
        dtype=float,
    ).loc[row_codes]

    # GO column: 'GO' marker in the code row if present (most vintages), else
    # the total-output header cell (2012 has no alphabetic column markers)
    go_c = None
    code_row_vals = raw.iloc[code_row].astype(str).str.strip()
    hit = np.where(code_row_vals.values == "GO")[0]
    if len(hit):
        go_c = int(hit[-1])
    else:
        for r in range(code_row + 1):
            for c in range(raw.shape[1] - 1, code_col, -1):
                v = raw.iat[r, c]
                if isinstance(v, str) and re.sub(r"\s+", "", v) == "总产出":
                    go_c = c
                    break
            if go_c is not None:
                break
    if go_c is None:
        raise ValueError(f"{year}: total-output (GO/总产出) column not found")
    go = pd.Series(
        pd.to_numeric(raw.iloc[sec_rows, go_c], errors="coerce").values,
        index=row_codes,
        dtype=float,
    )

    # --- sector names: rightmost non-null text left of the code column
    names = []
    for r in sec_rows:
        nm = None
        for c in range(code_col - 1, -1, -1):
            v = raw.iat[r, c]
            if isinstance(v, str) and v.strip():
                nm = re.sub(r"\s+", "", v)
                break
        names.append(nm or "")
    names = pd.Series(names, index=row_codes)

    return {
        "Z": Z,
        "x": x,
        "va": va,
        "go": go,
        "codes": row_codes,
        "names": names,
        "meta": {
            "year": year,
            "file": fname,
            "sheet": sheet,
            "n_sectors": len(row_codes),
            "table_type": "benchmark" if year in BENCHMARK else "extension",
        },
    }


# ============================================================ industry mapping
def map_sectors_to_industries(codes: list[str], names: pd.Series) -> pd.DataFrame:
    """Sector -> {pri, sec, ter} (+ KIS flag), derived from the vintage's own list.

    Rule 1 (5-digit codes, e.g. 01001/23068/90139): 2-digit prefix = GB/T 4754
    division; 01-05 -> primary, 06-50 (mining/manufacturing/utilities/
    construction) -> secondary, >=51 -> tertiary.
    Rule 2 (sequence codes): positional -- sectors before the first mining
    sector are primary, through the last construction sector secondary,
    after it tertiary. Applied as a cross-check when 5-digit codes exist.
    """
    # positional rule
    idx_min = next(
        (i for i, c in enumerate(codes) if _RE_MINING.search(names[c] or "")), None
    )
    idx_con = max(
        (i for i, c in enumerate(codes) if _RE_CONSTRUCTION.search(names[c] or "")),
        default=None,
    )
    if idx_min is None or idx_con is None or idx_min > idx_con:
        raise ValueError("positional industry rule failed: mining/construction not found")
    pos = [
        "pri" if i < idx_min else ("sec" if i <= idx_con else "ter")
        for i in range(len(codes))
    ]

    five_digit = all(_CODE_5DIGIT.match(c) for c in codes)
    if five_digit:
        pref = [int(c[:2]) for c in codes]
        ind = [
            "pri" if p <= 5 else ("sec" if p <= 50 else "ter") for p in pref
        ]
        if ind != pos:
            diff = [codes[i] for i in range(len(ind)) if ind[i] != pos[i]]
            print(f"  WARNING: prefix vs positional rule disagree on {diff}; using prefix rule")
    else:
        ind = pos

    out = pd.DataFrame(
        {
            "code": codes,
            "name": [names[c] for c in codes],
            "industry": ind,
            "kis": [
                bool(ind[i] == "ter" and _RE_KIS.search(names[codes[i]] or ""))
                for i in range(len(codes))
            ],
        }
    ).set_index("code")

    # sanity: every primary sector name must look agricultural
    bad = [
        (c, out.loc[c, "name"])
        for c in out.index[out["industry"] == "pri"]
        if not _RE_PRIMARY_NAME.search(out.loc[c, "name"])
    ]
    if bad:
        raise ValueError(f"primary block contains non-agricultural sectors: {bad}")
    return out


# ============================================================ indicators
def technical_coefficients(Z: pd.DataFrame, x: pd.Series) -> pd.DataFrame:
    """A = Z x_hat^-1 (column coefficients); columns with x=0 are set to 0."""
    xs = x.replace(0.0, np.nan)
    return Z.div(xs, axis=1).fillna(0.0)


def allocation_coefficients(Z: pd.DataFrame, x: pd.Series) -> pd.DataFrame:
    """B = x_hat^-1 Z (row coefficients); rows with x=0 are set to 0."""
    xs = x.replace(0.0, np.nan)
    return Z.div(xs, axis=0).fillna(0.0)


def leontief_inverse(A: pd.DataFrame) -> pd.DataFrame:
    n = A.shape[0]
    L = np.linalg.inv(np.eye(n) - A.values)
    return pd.DataFrame(L, index=A.index, columns=A.columns)


def sector_linkage_indicators(Z: pd.DataFrame, x: pd.Series) -> pd.DataFrame:
    """Disaggregated-sector indicators m_j, BL_j, FL_j, o_j, o_share_j."""
    A = technical_coefficients(Z, x)
    B = allocation_coefficients(Z, x)
    L = leontief_inverse(A)
    G = leontief_inverse(B)  # Ghosh inverse: (I - B)^-1

    n = Z.shape[0]
    colsum_Z = Z.sum(axis=0)
    xs = x.replace(0.0, np.nan)

    m = colsum_Z / xs
    BL = n * L.sum(axis=0) / L.values.sum()
    FL = n * G.sum(axis=1) / G.values.sum()
    off = colsum_Z - pd.Series(np.diag(Z.values), index=Z.index)
    o = off / xs
    o_share = off / colsum_Z.replace(0.0, np.nan)

    return pd.DataFrame({"m": m, "BL": BL, "FL": FL, "o": o, "o_share": o_share})


def aggregate_to_industries(
    ind: pd.DataFrame, x: pd.Series, mapping: pd.DataFrame, kis_split: bool = False
) -> pd.DataFrame:
    """Route B: output-weighted aggregation of sector indicators to industries."""
    groups = {g: mapping.index[mapping["industry"] == g] for g in ("pri", "sec", "ter")}
    if kis_split:
        ter = mapping.index[mapping["industry"] == "ter"]
        groups["ter_kis"] = ter[mapping.loc[ter, "kis"]]
        groups["ter_routine"] = ter[~mapping.loc[ter, "kis"]]

    rows = {}
    for g, idx in groups.items():
        w = x.loc[idx]
        sub = ind.loc[idx]
        ok = w > 0
        rows[g] = (sub[ok].mul(w[ok], axis=0)).sum() / w[ok].sum()
        rows[g]["x_share"] = w.sum() / x.sum()
        rows[g]["n_sectors"] = len(idx)
    return pd.DataFrame(rows).T


def aggregate_flows_first(
    Z: pd.DataFrame, x: pd.Series, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Route A cross-check: aggregate flows to 3x3 first, recompute m/BL/FL.

    o is intentionally NOT computed here (diagonal inflation on the 3x3 aggregate).
    """
    order = ["pri", "sec", "ter"]
    S = pd.DataFrame(
        [(mapping["industry"] == g).astype(float).values for g in order],
        index=order,
        columns=Z.index,
    )
    Z3 = S.values @ Z.values @ S.values.T
    x3 = S.values @ x.values
    Z3 = pd.DataFrame(Z3, index=order, columns=order)
    x3 = pd.Series(x3, index=order)
    return sector_linkage_indicators(Z3, x3)[["m", "BL", "FL"]]


# ============================================================ validation
def validate_table(tab: dict, tol: float = 0.005) -> dict:
    """Accounting-identity and sanity checks for one parsed vintage."""
    Z, x, va, go = tab["Z"], tab["x"], tab["va"], tab["go"]
    year = tab["meta"]["year"]

    checks = {"year": year, "n_sectors": tab["meta"]["n_sectors"]}
    pos = x > 0
    checks["n_zero_output"] = int((~pos).sum())
    checks["x_positive"] = bool((x[pos] > 0).all())

    # column identity: colsum(Z) + VA = x  (total input)
    resid = (Z.sum(axis=0) + va - x).abs() / x.where(pos)
    checks["max_col_identity_err"] = float(resid.max())
    checks["col_identity_ok"] = bool(resid.max() < tol)

    # TI row (total input) vs GO column (total output) per sector
    diff = (x - go).abs() / x.where(pos)
    checks["max_TI_vs_GO_err"] = float(diff.max())
    checks["TI_eq_GO_ok"] = bool(diff.max() < tol)

    A = technical_coefficients(Z, x)
    colsums = A.sum(axis=0)[pos]
    checks["max_A_colsum"] = float(colsums.max())
    checks["A_colsums_lt_1"] = bool((colsums < 1).all())

    L = leontief_inverse(A)
    checks["leontief_min"] = float(L.values.min())
    checks["leontief_nonneg"] = bool(L.values.min() > -1e-9)

    checks["all_ok"] = all(
        checks[k]
        for k in ("x_positive", "col_identity_ok", "TI_eq_GO_ok", "A_colsums_lt_1", "leontief_nonneg")
    )
    return checks


# ============================================================ ordering test
def ordering_rows(year: int, agg: pd.DataFrame, route_a: pd.DataFrame) -> list[dict]:
    rows = []
    for indicator in INDICATORS:
        vals = {g: float(agg.loc[g, indicator]) for g in ("pri", "sec", "ter")}
        row = {
            "vintage": year,
            "indicator": indicator,
            "I_pri": vals["pri"],
            "I_sec": vals["sec"],
            "I_ter": vals["ter"],
            "sec_gt_ter": vals["sec"] > vals["ter"],
            "ter_gt_pri": vals["ter"] > vals["pri"],
            "sec_gt_pri": vals["sec"] > vals["pri"],
        }
        row["strong_ok"] = row["sec_gt_ter"] and row["ter_gt_pri"]
        row["weak_ok"] = min(vals["sec"], vals["ter"]) > vals["pri"]
        if indicator in route_a.columns:
            row["strong_ok_routeA"] = bool(
                route_a.loc["sec", indicator]
                > route_a.loc["ter", indicator]
                > route_a.loc["pri", indicator]
            )
        rows.append(row)
    return rows


# ============================================================ driver
def run_all(vintages: list[int] | None = None) -> dict[str, pd.DataFrame]:
    vintages = vintages or ALL_VINTAGES
    TABLES.mkdir(parents=True, exist_ok=True)

    ind_rows, ord_rows, val_rows = [], [], []
    parsed, skipped = [], []

    for year in vintages:
        try:
            tab = load_io_table(year)
            checks = validate_table(tab)
            val_rows.append(checks)
            if not checks["all_ok"]:
                print(f"[{year}] VALIDATION FAILED -> skipped: {checks}")
                skipped.append(year)
                continue

            mapping = map_sectors_to_industries(tab["codes"], tab["names"])
            mapping.assign(x=tab["x"]).to_csv(
                TABLES / f"io_sector_mapping_{year}.csv", encoding="utf-8-sig"
            )

            ind = sector_linkage_indicators(tab["Z"], tab["x"])
            agg = aggregate_to_industries(ind, tab["x"], mapping, kis_split=True)
            route_a = aggregate_flows_first(tab["Z"], tab["x"], mapping)

            block = agg.copy()
            block.insert(0, "vintage", year)
            block.insert(1, "table_type", tab["meta"]["table_type"])
            block.insert(2, "n_sectors_table", tab["meta"]["n_sectors"])
            for c in ("m", "BL", "FL"):
                block[f"{c}_routeA"] = route_a[c].reindex(block.index)
            ind_rows.append(block.rename_axis("industry").reset_index())

            ord_rows += ordering_rows(year, agg, route_a)
            parsed.append(year)
            print(
                f"[{year}] ok: {tab['meta']['n_sectors']} sectors "
                f"({tab['meta']['table_type']}), "
                f"max col-identity err {checks['max_col_identity_err']:.2e}"
            )
        except Exception as e:  # skip unparseable vintages, never fabricate
            print(f"[{year}] PARSE FAILED -> skipped: {e}")
            skipped.append(year)

    if not ind_rows:
        raise FileNotFoundError(
            "No input-output tables were parsed. Place the national "
            "input-output workbooks under data/ as described in "
            "DATA_SOURCES.md, then rerun.")
    indicators = pd.concat(ind_rows, ignore_index=True)
    ordering = pd.DataFrame(ord_rows)
    validation = pd.DataFrame(val_rows)

    indicators.to_csv(TABLES / "io_linkage_indicators.csv", index=False, encoding="utf-8-sig")
    ordering.to_csv(TABLES / "io_ordering_test.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(TABLES / "io_table_validation.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ summary report
    print("\n==================== Sector linkage summary ====================")
    print(f"parsed vintages : {parsed}")
    print(f"skipped         : {skipped or 'none'}")
    pre = ordering[ordering["vintage"].isin(PRIMARY_VINTAGES)]
    n_cells = len(pre)
    hits = int(pre["strong_ok"].sum())
    # illustrative binomial p (cells are not independent):
    # random chance of the full 3-way ordering in one cell is 1/6
    from math import comb

    p_binom = sum(
        comb(n_cells, k) * (1 / 6) ** k * (5 / 6) ** (n_cells - k)
        for k in range(hits, n_cells + 1)
    )
    print(
        f"\nPrimary vintages {PRIMARY_VINTAGES}: "
        f"strong ordering (sec>ter>pri) holds in {hits}/{n_cells} "
        f"indicator-vintage cells; weak form in {int(pre['weak_ok'].sum())}/{n_cells}."
        f"\n  (illustrative binomial P(>= {hits}/{n_cells} | chance 1/6) = {p_binom:.2e}; "
        f"cells are not independent)"
    )
    allv = ordering
    print(
        f"All parsed vintages: strong {int(allv['strong_ok'].sum())}/{len(allv)}, "
        f"weak {int(allv['weak_ok'].sum())}/{len(allv)}."
    )
    for indicator in INDICATORS:
        sub = allv[allv["indicator"] == indicator]
        stat = "stable" if sub["strong_ok"].all() else (
            "weak-stable" if sub["weak_ok"].all() else "UNSTABLE"
        )
        fails = sub.loc[~sub["strong_ok"], "vintage"].tolist()
        print(
            f"  {indicator:>2}: strong {int(sub['strong_ok'].sum())}/{len(sub)}"
            + (f" (strong fails: {fails})" if fails else "")
            + f", weak {int(sub['weak_ok'].sum())}/{len(sub)} -> {stat}"
        )
    return {"indicators": indicators, "ordering": ordering, "validation": validation}


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or None
    run_all(years)
