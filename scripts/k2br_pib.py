"""Files (downloaded by scripts/fetch_public_data.py brazil; URLs and sha256 in
results/tables/k2br_coverage_sources.csv), read straight from the zips:

  series 'ref2010' (reference-2010 series, D2 a primary), fixed-width txt,
    layout from the PDF shipped in each zip (1-based position, width):
      Ano (1, 4)   Codigo do Municipio (47, 7)   Codigo da UF (21, 2)
      VAB Agropecuaria, precos correntes, R$ 1 000 (821, 18)
      PIB, precos correntes, R$ 1 000             (935, 18)
    line length 971 (2002-2009) / 1256 (2010-2023), ISO-8859-1, CRLF.
    Layout note 2 (2010-2023 file): for 2022 and 2023 IBGE published only
    PIB and PIB per capita, so vab_agro is NOT PUBLISHED for 2022-2023 and
    is returned as NaN for those years whatever the field holds: on those
    lines the field is never sliced or parsed
    (``VAB_NOT_PUBLISHED``; fixed from the layout PDF before any value was
    read).
  series 'ref2002' (reference-2002 series 1999-2012, D2 a secondary),
    xlsx (the txt version of this series ships without a layout); columns by
    the header row: A 'Ano de referencia', D 'Codigo do Municipio',
    K 'Valor Adicionado Bruto, a precos correntes, da Agropecuaria (R$ 1.000)',
    P 'Produto Interno Bruto a precos correntes (R$ 1.000)'.  Parsed from the
    sheet XML (no openpyxl dependency).

Unit key: the first six digits of the 7-digit IBGE code (as the coverage
tables).  Units of both outcomes: R$ 1 000, current prices (D1: nominal).

Existence check (kill rule 3; codes and years only):
  python3 -P scripts/k2br_pib.py --existence
writes results/tables/k2br_existence.csv (the records, which the real run
re-verifies), k2br_existence_by_year.csv (per series x year: records vs the
population list) and k2br_existence_discrepancies.csv (every code-year in
one list but not the other, classified against D3's known one-year lag), and
k2br_existence_meta.json with the verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('data/brazil')
OUT = ROOT / "results" / "tables"

TXT_2002_2009 = "raw/pib/base_de_dados_2002_2009_txt.zip"
TXT_2010_2023 = "raw/pib/base_de_dados_2010_2023_txt.zip"
XLSX_1999_2012 = "raw/pib/base_1999_2012_xlsx.zip"
RAW_FILES = (TXT_2002_2009, TXT_2010_2023, XLSX_1999_2012)

SERIES = {"ref2010": ((TXT_2002_2009, TXT_2010_2023), 2002, 2023),
          "ref2002": ((XLSX_1999_2012,), 1999, 2012)}

TXT_FIELDS = {"year": (0, 4), "uf": (20, 22), "code7": (46, 53),
              "vab_agro": (820, 838), "pib": (934, 952)}
TXT_LINE_LEN = {TXT_2002_2009: 971, TXT_2010_2023: 1256}
VAB_NOT_PUBLISHED = {"ref2010": (2022, 2023)}

XLSX_COLS = {"year": ("A", "Ano de referência"),
             "code7": ("D", "Código do Município"),
             "vab_agro": ("K", "Valor Adicionado Bruto, a preços correntes, da Agropecuária"),
             "pib": ("P", "Produto Interno Bruto a preços correntes")}
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_NUM = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
_NUM_COMMA = re.compile(r"-?\d+(,\d+)?")


def _member(z: zipfile.ZipFile, suffix: str) -> str:
    names = [n for n in z.namelist() if n.lower().endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"expected one *{suffix} member, found {len(names)}")
    return names[0]


def parse_number(s: str) -> float:
    """Blank -> NaN; '123.4' or '123,4' -> float; anything else ValueError
    (the message never carries the field)."""
    s = s.strip()
    if not s:
        return np.nan
    if _NUM.fullmatch(s):
        return float(s)
    if _NUM_COMMA.fullmatch(s):
        return float(s.replace(",", "."))
    raise ValueError("non-numeric outcome field")


def _parse_column(fields: list, what: str) -> np.ndarray:
    """A ``None`` entry is a field that is not published: NaN, never parsed."""
    out = np.empty(len(fields))
    bad = 0
    for i, s in enumerate(fields):
        if s is None:
            out[i] = np.nan
            continue
        try:
            out[i] = parse_number(s)
        except ValueError:
            bad += 1
    if bad:
        raise ValueError(f"{what}: {bad} of {len(fields)} fields are not numeric")
    return out


def _check_keys(year: list[str], code7: list[str], what: str) -> None:
    by = sum(not re.fullmatch(r"\d{4}", y) for y in year)
    bc = sum(not re.fullmatch(r"\d{7}", c) for c in code7)
    if by or bc:
        raise ValueError(f"{what}: {by} malformed year and {bc} malformed code fields")


def _txt_lines(path: Path, rel: str) -> list[str]:
    with zipfile.ZipFile(path) as z:
        raw = z.read(_member(z, ".txt"))
    lines = raw.decode("latin-1").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    want = TXT_LINE_LEN.get(rel)
    if want is not None:
        nbad = sum(len(ln) != want for ln in lines)
        if nbad:
            raise ValueError(f"{rel}: {nbad} lines differ from the layout length {want}")
    return lines


def _txt(path: Path, rel: str, values: bool, unpublished: dict | None = None) -> pd.DataFrame:
    """``unpublished``: outcome -> years whose field is not published; on those
    lines the field is never sliced or parsed (NaN), whatever it holds."""
    lines = _txt_lines(path, rel)
    sl = {k: slice(*v) for k, v in TXT_FIELDS.items()}
    year = [ln[sl["year"]] for ln in lines]
    code7 = [ln[sl["code7"]] for ln in lines]
    _check_keys(year, code7, rel)
    uf = [ln[sl["uf"]] for ln in lines]
    if any(u != c[:2] for u, c in zip(uf, code7)):
        raise ValueError(f"{rel}: UF code field disagrees with the municipality code")
    df = pd.DataFrame({"code6": [c[:6] for c in code7], "year": np.array(year, int)})
    if values:
        unpublished = unpublished or {}
        for o in ("pib", "vab_agro"):
            skip = {str(y) for y in unpublished.get(o, ())}
            df[o] = _parse_column([None if y in skip else ln[sl[o]]
                                   for y, ln in zip(year, lines)], f"{rel} {o}")
    return df


def _xlsx_rows(path: Path, cols: dict[str, str]):
    """Yield (row number, {key: text}) for the requested column letters only;
    every other cell is skipped without being decoded."""
    with zipfile.ZipFile(path) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read(_member(outer, ".xlsx"))))
    ss = [''.join(t.text or '' for t in si.iter(_NS + "t"))
          for si in ET.parse(inner.open("xl/sharedStrings.xml")).getroot().iter(_NS + "si")]
    sheets = sorted(n for n in inner.namelist() if n.startswith("xl/worksheets/sheet"))
    if len(sheets) != 1:
        raise ValueError(f"expected one worksheet, found {len(sheets)}")
    letter_key = {v: k for k, v in cols.items()}
    for _, el in ET.iterparse(inner.open(sheets[0])):
        if el.tag != _NS + "row":
            continue
        rec = {}
        for c in el.iter(_NS + "c"):
            letter = re.match(r"[A-Z]+", c.get("r")).group(0)
            key = letter_key.get(letter)
            if key is None:
                continue
            t = c.get("t")
            if t == "s":
                txt = ss[int(c.find(_NS + "v").text)]
            elif t == "inlineStr":
                txt = ''.join(x.text or '' for x in c.iter(_NS + "t"))
            else:
                v = c.find(_NS + "v")
                txt = "" if v is None or v.text is None else v.text
            rec[key] = txt
        yield int(el.get("r")), rec
        el.clear()


def _xlsx(path: Path, rel: str, values: bool, unpublished: dict | None = None) -> pd.DataFrame:
    keys = ("year", "code7") + (("pib", "vab_agro") if values else ())
    cols = {k: XLSX_COLS[k][0] for k in keys}
    rows = _xlsx_rows(path, cols)
    _, head = next(rows)
    for k in keys:
        name = XLSX_COLS[k][1]
        if not head.get(k, "").startswith(name):
            raise ValueError(f"{rel}: column {XLSX_COLS[k][0]} header is not {name!r}")
    recs = [r for _, r in rows if r.get("year", "").strip() or r.get("code7", "").strip()]

    def key_text(s: str) -> str:
        s = s.strip()
        return s[:-2] if s.endswith(".0") else s
    year = [key_text(r.get("year", "")) for r in recs]
    code7 = [key_text(r.get("code7", "")) for r in recs]
    _check_keys(year, code7, rel)
    df = pd.DataFrame({"code6": [c[:6] for c in code7], "year": np.array(year, int)})
    if values:
        unpublished = unpublished or {}
        for o in ("pib", "vab_agro"):
            skip = {str(y) for y in unpublished.get(o, ())}
            df[o] = _parse_column([None if y in skip else r.get(o, "")
                                   for y, r in zip(year, recs)], f"{rel} {o}")
    return df


def _read_series(data: Path, values: bool) -> pd.DataFrame:
    parts = []
    for series, (files, y0, y1) in SERIES.items():
        for rel in files:
            p = Path(data) / rel
            unpub = {"vab_agro": VAB_NOT_PUBLISHED.get(series, ())}
            reader = _xlsx if rel.endswith("xlsx.zip") else _txt
            df = reader(p, rel, values, unpub)
            df.insert(0, "series", series)
            parts.append(df)
        got = pd.concat([d for d in parts if d["series"].iat[0] == series])
        if got["year"].min() != y0 or got["year"].max() != y1:
            raise ValueError(f"{series}: years {got['year'].min()}-{got['year'].max()}, "
                             f"expected {y0}-{y1}")
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["series", "code6", "year"]).any():
        raise ValueError("a (series, code6, year) record occurs twice")
    if values:
        for series, yrs in VAB_NOT_PUBLISHED.items():
            out.loc[(out["series"] == series) & out["year"].isin(yrs), "vab_agro"] = np.nan
    return out.sort_values(["series", "code6", "year"], ignore_index=True)


def outcome_records(data: Path = DATA) -> pd.DataFrame:
    """Existence only: DataFrame[series, code6, year]; no value field is read."""
    return _read_series(data, values=False)


def outcome_values(data: Path = DATA) -> pd.DataFrame:
    """DataFrame[series, code6, year, pib, vab_agro] (R$ 1 000, current
    prices).  Only for the registered single run."""
    return _read_series(data, values=True)


def compare_existence(rec: pd.DataFrame, pop_codes: dict[int, set],
                      first_pop: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per series x year: records vs population list (codes and years only).

    Every code-year in one list but not the other is a discrepancy.  D3's
    known case ('entry_lag'): an entrant listed by the population estimate
    one year before its first PIB year, i.e. a 'pop_only' code-year y with
      y = the code's first population year, y after the window's first year,
      first PIB year = y + 1 (in this series; if y is the window's last year,
        in any series -- the lag is then censored by the window end), and
      present in both lists in every later year of the window.
    Anything else is 'unexplained' and hands the decision to the author
    (kill rule 3)."""
    first_any = rec.groupby("code6")["year"].min().to_dict()
    by_year, disc = [], []
    for series, (_, y0, y1) in SERIES.items():
        r = rec[rec["series"] == series]
        pib_codes = {y: set(g["code6"]) for y, g in r.groupby("year")}
        first_pib = r.groupby("code6")["year"].min().to_dict()

        def both_from(c, t0):
            return all(c in pib_codes.get(t, set()) and c in pop_codes.get(t, set())
                       for t in range(t0, y1 + 1))
        for y in range(y0, y1 + 1):
            P, N = pib_codes.get(y, set()), pop_codes.get(y, set())
            rows = ([(c, "pib_only") for c in sorted(P - N)] +
                    [(c, "pop_only") for c in sorted(N - P)])
            n_lag = 0
            for c, kind in rows:
                fp, fb = first_pop.get(c), first_pib.get(c)
                cls = "unexplained"
                if kind == "pop_only" and fp == y and y > y0:
                    if fb == y + 1 and both_from(c, fb):
                        cls = "entry_lag"
                    elif fb is None and y == y1 and first_any.get(c) == y + 1:
                        cls = "entry_lag_censored"
                n_lag += cls != "unexplained"
                lag = (fb - fp) if fb is not None and fp is not None else (
                    first_any[c] - fp if cls == "entry_lag_censored" else None)
                disc.append({"series": series, "year": y, "code6": c, "kind": kind,
                             "first_pop_year": fp, "first_pib_year": fb, "lag": lag,
                             "cls": cls})
            by_year.append({"series": series, "year": y, "n_pib_records": len(P),
                            "n_pop": len(N), "n_both": len(P & N),
                            "n_pib_only": len(P - N), "n_pop_only": len(N - P),
                            "n_known_entry_lag": n_lag,
                            "n_unexplained": len(rows) - n_lag,
                            "n_new_codes_pop": (sum(first_pop.get(c) == y for c in N)
                                                if y > y0 else np.nan),
                            "n_new_codes_pib": (sum(first_pib.get(c) == y for c in P)
                                                if y > y0 else np.nan)})
    disc = pd.DataFrame(disc, columns=["series", "year", "code6", "kind", "first_pop_year",
                                       "first_pib_year", "lag", "cls"])
    by_year = pd.DataFrame(by_year)
    n_un = int((disc["cls"] == "unexplained").sum())
    verdict = {"n_records": int(len(rec)),
               "n_discrepancies": int(len(disc)),
               "n_known_entry_lag": int(len(disc) - n_un),
               "n_unexplained": n_un,
               "lags_seen": sorted({int(x) for x in disc["lag"].dropna()}),
               "kill_rule_3": ("hand to author: existence differs from the population list "
                               "beyond D3's known one-year entry lag" if n_un else
                               "not triggered: every difference is D3's known one-year entry lag"
                               if len(disc) else "not triggered: lists identical")}
    return by_year, disc, verdict


def _run_k2br():
    spec = importlib.util.spec_from_file_location("run_k2br", ROOT / "scripts" / "run_k2br.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_k2br", mod)
    spec.loader.exec_module(mod)
    return mod


def existence_main(data: Path, out: Path) -> dict:
    rec = outcome_records(data)
    struct = _run_k2br().load_structure(data)
    pop = struct["pop"]
    pop_codes = {y: set(s.index) for y, s in pop.items()}
    first_pop = {}
    for y in sorted(k for k in pop if k != 2023):
        for c in pop[y].index:
            first_pop.setdefault(c, y)
    by_year, disc, verdict = compare_existence(rec, pop_codes, first_pop)
    out.mkdir(parents=True, exist_ok=True)
    rec[["series", "code6", "year"]].to_csv(out / "k2br_existence.csv", index=False)
    by_year.to_csv(out / "k2br_existence_by_year.csv", index=False)
    disc.to_csv(out / "k2br_existence_discrepancies.csv", index=False)
    meta = {"script": "scripts/k2br_pib.py --existence",
            "git_hash": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
            "mode": "codes and years only (outcome_records); no value field read",
            "raw_sha256": {f: hashlib.sha256((Path(data) / f).read_bytes()).hexdigest()
                           for f in RAW_FILES},
            "vab_agro_not_published": {k: list(v) for k, v in VAB_NOT_PUBLISHED.items()},
            **verdict}
    (out / "k2br_existence_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existence", action="store_true", required=True)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    meta = existence_main(Path(args.data), Path(args.out))
    print(json.dumps({k: meta[k] for k in ("n_records", "n_discrepancies", "n_known_entry_lag",
                                            "n_unexplained", "lags_seen", "kill_rule_3")}))


if __name__ == "__main__":
    main()
