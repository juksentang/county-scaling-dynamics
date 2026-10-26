"""SYNTHETIC files only, written in the layouts of the IBGE PIB dos Municipios
files (fixed-width txt for the reference-2010 series, xlsx for the
reference-2002 series).  No Brazilian outcome is read by any test.

Run:  python3 -m pytest -q tests/test_k2br_pib.py
"""
from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("k2br_pib", ROOT / "scripts" / "k2br_pib.py")
P = importlib.util.module_from_spec(_spec)
sys.modules["k2br_pib"] = P
_spec.loader.exec_module(P)

CODES = ["1100015", "1100023", "3550308"]


def _put(line: list, key: str, text: str) -> None:
    a, b = P.TXT_FIELDS[key]
    text = text.rjust(b - a)
    assert len(text) == b - a
    line[a:b] = list(text)


def txt_line(rel, year, code7, pib="1234.567", vab="12.5", uf=None):
    line = [" "] * P.TXT_LINE_LEN[rel]
    _put(line, "year", str(year))
    _put(line, "uf", uf or code7[:2])
    _put(line, "code7", code7)
    _put(line, "pib", pib)
    _put(line, "vab_agro", vab)
    return "".join(line)


def write_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("layout.pdf", b"%PDF synthetic")
        z.writestr("PIB dos Municipios - base de dados.txt",
                   ("\r\n".join(lines) + "\r\n").encode("latin-1"))


def _cell(ref, val, shared):
    if isinstance(val, str):
        shared.append(val)
        return f'<c r="{ref}" t="s"><v>{len(shared) - 1}</v></c>'
    if val is None:
        return f'<c r="{ref}"/>'
    return f'<c r="{ref}"><v>{val}</v></c>'


def write_xlsx(path: Path, rows: list[dict], header: dict | None = None) -> None:
    """rows: dicts with keys year, code7, pib, vab_agro (numbers or str);
    columns as P.XLSX_COLS plus filler columns that must never be read."""
    head = {k: v[1] + "\n(R$ 1.000 )" for k, v in P.XLSX_COLS.items()} | (header or {})
    shared: list[str] = []
    xml_rows = []
    letters = {k: v[0] for k, v in P.XLSX_COLS.items()}

    def row(r, rec):
        cells = [_cell(f"{letters[k]}{r}", rec.get(k), shared) for k in letters]
        cells.append(_cell(f"E{r}", "Nome", shared))
        cells.append(_cell(f"Q{r}", "not a number", shared))
        cells.sort(key=lambda c: (len(c.split('"')[1].rstrip("0123456789")), c.split('"')[1]))
        return f'<row r="{r}">' + "".join(cells) + "</row>"
    xml_rows.append(row(1, head))
    for i, rec in enumerate(rows, start=2):
        xml_rows.append(row(i, rec))
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = f'<worksheet xmlns="{ns}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    sst = (f'<sst xmlns="{ns}">' + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("PIB dos Municipios 1999-2012 Base Dados.xlsx", buf.getvalue())


def make_data(tmp: Path, txt_kw=None, xlsx_rows=None, header=None) -> Path:
    txt_kw = txt_kw or {}
    a = [txt_line(P.TXT_2002_2009, y, c, **txt_kw.get((y, c), {}))
         for y in range(2002, 2010) for c in CODES]
    b = [txt_line(P.TXT_2010_2023, y, c, **txt_kw.get((y, c), {}))
         for y in range(2010, 2024) for c in CODES]
    write_txt(tmp / P.TXT_2002_2009, a)
    write_txt(tmp / P.TXT_2010_2023, b)
    if xlsx_rows is None:
        xlsx_rows = [{"year": y, "code7": int(c), "pib": 100.25, "vab_agro": 3}
                     for y in range(1999, 2013) for c in CODES]
    write_xlsx(tmp / P.XLSX_1999_2012, xlsx_rows, header)
    return tmp


def test_records_mode_returns_codes_and_years_only(tmp_path):
    rec = P.outcome_records(make_data(tmp_path))
    assert list(rec.columns) == ["series", "code6", "year"]
    assert len(rec) == 3 * 22 + 3 * 14
    r10 = rec[rec.series == "ref2010"]
    assert set(r10.year) == set(range(2002, 2024))
    assert set(r10.code6) == {c[:6] for c in CODES}
    assert set(rec[rec.series == "ref2002"].year) == set(range(1999, 2013))


def test_records_mode_never_parses_a_value_field(tmp_path, monkeypatch):
    """Garbage in every value field and a parser that raises: the records
    mode still works (it never slices or parses a value), the values mode
    refuses with counts only (the field text is not in the message)."""
    kw = {(y, c): {"pib": "SECRETVALUE", "vab": "SECRETVALUE"}
          for y in range(2002, 2024) for c in CODES}
    rows = [{"year": y, "code7": c, "pib": "SECRETVALUE", "vab_agro": "SECRETVALUE"}
            for y in range(1999, 2013) for c in CODES]
    data = make_data(tmp_path, kw, rows)
    with pytest.raises(ValueError) as e:
        P.outcome_values(data)
    assert "SECRETVALUE" not in str(e.value) and "not numeric" in str(e.value)

    def boom(*a, **k):
        raise AssertionError("a value field was parsed in records mode")
    monkeypatch.setattr(P, "parse_number", boom)
    monkeypatch.setattr(P, "_parse_column", boom)
    assert len(P.outcome_records(data)) == 3 * 22 + 3 * 14


def test_values_mode_parses_and_blanks(tmp_path):
    kw = {(2003, CODES[0]): {"pib": "1,5", "vab": ""},
          (2022, CODES[1]): {"vab": "99"},
          (2015, CODES[2]): {"pib": "-0.000", "vab": "7"}}
    df = P.outcome_values(make_data(tmp_path, kw))
    assert list(df.columns) == ["series", "code6", "year", "pib", "vab_agro"]
    g = df.set_index(["series", "code6", "year"])
    assert g.loc[("ref2010", CODES[0][:6], 2003), "pib"] == 1.5
    assert np.isnan(g.loc[("ref2010", CODES[0][:6], 2003), "vab_agro"])
    assert g.loc[("ref2010", CODES[0][:6], 2004), "pib"] == 1234.567
    assert g.loc[("ref2010", CODES[2][:6], 2015), "vab_agro"] == 7
    r10 = df[df.series == "ref2010"]
    assert r10[r10.year.isin([2022, 2023])]["vab_agro"].isna().all()
    assert r10[r10.year.isin([2022, 2023])]["pib"].notna().all()
    assert r10[r10.year == 2021]["vab_agro"].notna().all()
    r02 = df[df.series == "ref2002"]
    assert (r02.pib == 100.25).all() and (r02.vab_agro == 3).all()
    assert r02.code6.str.len().eq(6).all()


@pytest.mark.parametrize("ph", ["...", "-", "X", "1.234,5", "abc"])
def test_values_mode_never_parses_unpublished_vab(tmp_path, ph):
    """vab_agro 2022-2023 (ref2010) is not published: a placeholder there must
    not raise; it is NaN and pib in the same lines still parses."""
    kw = {(y, c): {"vab": ph} for y in (2022, 2023) for c in CODES}
    df = P.outcome_values(make_data(tmp_path, kw))
    r10 = df[df.series == "ref2010"]
    assert r10[r10.year.isin([2022, 2023])]["vab_agro"].isna().all()
    assert r10[r10.year.isin([2022, 2023])]["pib"].notna().all()
    assert r10[r10.year == 2021]["vab_agro"].notna().all()


def test_values_mode_still_rejects_published_placeholder(tmp_path):
    kw = {(2021, CODES[0]): {"vab": "..."}}
    with pytest.raises(ValueError, match="vab_agro: 1 of"):
        P.outcome_values(make_data(tmp_path, kw))


def test_values_mode_matches_records_mode(tmp_path):
    data = make_data(tmp_path)
    pd.testing.assert_frame_equal(P.outcome_values(data)[["series", "code6", "year"]],
                                  P.outcome_records(data))


def test_run_k2br_accepts_the_reader(tmp_path):
    spec = importlib.util.spec_from_file_location("run_k2br_t", ROOT / "scripts" / "run_k2br.py")
    K = importlib.util.module_from_spec(spec)
    sys.modules["run_k2br_t"] = K
    spec.loader.exec_module(K)
    mod = K._pib_module()
    assert set(mod.RAW_FILES) == {P.TXT_2002_2009, P.TXT_2010_2023, P.XLSX_1999_2012}
    df = K.check_outcome_frame(mod.outcome_values(make_data(tmp_path)))
    assert set(df.series) == {"ref2010", "ref2002"}


def test_layout_guards(tmp_path):
    data = make_data(tmp_path / "a")
    with zipfile.ZipFile(data / P.TXT_2010_2023) as z:
        txt = z.read([n for n in z.namelist() if n.endswith(".txt")][0]).decode("latin-1")
    write_txt(data / P.TXT_2010_2023, [ln + "X" for ln in txt.splitlines()])
    with pytest.raises(ValueError, match="layout length"):
        P.outcome_records(data)
    data = make_data(tmp_path / "b", {(2005, CODES[0]): {"uf": "99"}})
    with pytest.raises(ValueError, match="UF code"):
        P.outcome_records(data)
    data = make_data(tmp_path / "c", header={"pib": "Something else"})
    with pytest.raises(ValueError, match="header"):
        P.outcome_values(data)
    rows = [{"year": y, "code7": c, "pib": 1, "vab_agro": 1}
            for y in range(1999, 2013) for c in CODES] + [{"year": 2000, "code7": CODES[0],
                                                          "pib": 1, "vab_agro": 1}]
    with pytest.raises(ValueError, match="twice"):
        P.outcome_records(make_data(tmp_path / "d", xlsx_rows=rows))
    rows = [{"year": y, "code7": c, "pib": 1, "vab_agro": 1}
            for y in range(2000, 2013) for c in CODES]
    with pytest.raises(ValueError, match="expected 1999-2012"):
        P.outcome_records(make_data(tmp_path / "e", xlsx_rows=rows))


def test_parse_number():
    assert P.parse_number(" 12.5 ") == 12.5 and P.parse_number("3,25") == 3.25
    assert np.isnan(P.parse_number("   "))
    for bad in ("1.234,5", "x", "1 000", "--1"):
        with pytest.raises(ValueError):
            P.parse_number(bad)


def _toy_lists():
    """Window ref2010 2002-2023 and ref2002 1999-2012 on toy codes."""
    base = {f"{i:06d}" for i in range(10)}
    pop = {y: set(base) for y in range(1991, 2027)}
    rec = []
    for s, (_, y0, y1) in P.SERIES.items():
        for y in range(y0, y1 + 1):
            rec += [(s, c, y) for c in base]
    return pop, rec


def _cmp(pop, rec):
    first = {}
    for y in sorted(pop):
        for c in pop[y]:
            first.setdefault(c, y)
    df = pd.DataFrame(rec, columns=["series", "code6", "year"])
    return P.compare_existence(df, pop, first)


def test_existence_identical_lists():
    pop, rec = _toy_lists()
    by, disc, v = _cmp(pop, rec)
    assert len(disc) == 0 and v["kill_rule_3"].startswith("not triggered: lists identical")
    assert (by.n_both == 10).all()


def test_existence_one_year_lag_is_known():
    pop, rec = _toy_lists()
    for y in range(2004, 2027):
        pop[y].add("900001")
    rec += [(s, "900001", y) for s, (_, y0, y1) in P.SERIES.items()
            for y in range(max(y0, 2005), y1 + 1)]
    for y in range(2012, 2027):
        pop[y].add("900002")
    rec += [("ref2010", "900002", y) for y in range(2013, 2024)]
    by, disc, v = _cmp(pop, rec)
    assert v["n_unexplained"] == 0, disc
    assert set(disc.cls) == {"entry_lag", "entry_lag_censored"}
    assert len(disc) == 4
    assert v["lags_seen"] == [1]


@pytest.mark.parametrize("case", ["lag2", "pib_only", "gap", "nonentrant_missing"])
def test_existence_other_differences_hand_to_author(case):
    pop, rec = _toy_lists()
    if case == "lag2":
        for y in range(2004, 2027):
            pop[y].add("900001")
        rec += [("ref2010", "900001", y) for y in range(2006, 2024)]
    elif case == "pib_only":
        rec.append(("ref2010", "900003", 2010))
    elif case == "gap":
        rec.remove(("ref2010", "000003", 2015))
    else:
        rec.remove(("ref2002", "000004", 1999))
    by, disc, v = _cmp(pop, rec)
    assert v["n_unexplained"] >= 1
    assert v["kill_rule_3"].startswith("hand to author")
