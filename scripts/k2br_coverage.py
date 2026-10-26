"""B  open panel as published: a municipality is a unit from the first year
     IBGE publishes a population for it (entrant); its parent(s) -- the
     municipalities it was split from ("desmembrado de", IBGE genealogy
     1872-2010, plus six post-2010 cases documented in POST2010_PARENTS) --
     stay in the panel with a changed territory ("changed definition").
  A  fixed geography: minimum comparable areas (AMC) built as connected
     components of the entrant--parent graph for all entrants after a base
     year (1991 or 2000); AMC population = sum of members.  Composition is
     constant by construction (n_E = n_L = 0).
  A0 constant set: municipalities present in every year of the window that
     are neither entrants nor parents within it (fixed territory, but the
     set drops the split-affected units).

Usage:  python3 -P scripts/k2br_coverage.py [--data DIR]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"

LICENCE = ('IBGE public statistical information; reproduction permitted with citation of IBGE (Lei 12.527/2011; Decreto 8.777/2016)')

SIDRA = "https://apisidra.ibge.gov.br/values"
FTP = "https://ftp.ibge.gov.br/Estimativas_de_Populacao"
GEO = "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial"
GEN_PDF = "raw/ibge_evolucao/municipios_genealogia_e_alteracao_de_nome_e_grafia.pdf"

POP_FILES: dict[int, tuple[str, str, str]] = {
    1991: ("census", "raw/sidra/t202_p1991.json", f"{SIDRA}/t/202/n6/all/v/93/p/1991"),
    2000: ("census", "raw/sidra/t202_p2000.json", f"{SIDRA}/t/202/n6/all/v/93/p/2000"),
    2007: ("contagem", "raw/sidra/t793_p2007.json", f"{SIDRA}/t/793/n6/all/v/93/p/2007"),
    2010: ("census", "raw/sidra/t202_p2010.json", f"{SIDRA}/t/202/n6/all/v/93/p/2010"),
    2022: ("census", "raw/sidra/t4709_p2022.json", f"{SIDRA}/t/4709/n6/all/v/93/p/2022"),
}
for _y in (1992, 1993, 1994, 1995, 1997, 1998, 1999):
    POP_FILES[_y] = ("estimate_ftp", f"raw/ftp_estimativas/estimativa_populacao_{_y}_xls.zip",
                     f"{FTP}/Estimativas_{_y}/estimativa_populacao_{_y}_xls.zip")
for _y in (2001, 2002, 2003, 2004, 2005, 2006, 2008, 2009, 2011, 2012, 2013, 2014, 2015,
           2016, 2017, 2018, 2019, 2020, 2021, 2024, 2025, 2026):
    POP_FILES[_y] = ("estimate_sidra6579", f"raw/sidra/t6579_p{_y}.json",
                     f"{SIDRA}/t/6579/n6/all/v/9324/p/{_y}")
YEARS = sorted(POP_FILES)

OTHER_FILES = [
    ("raw/ftp_estimativas/estimativa_2000_UF_Municipio.zip", f"{FTP}/Estimativas_2000/UF_Municipio.zip"),
    ("raw/dtb/dtb_1994_brasil.zip", f"{GEO}/divisao_territorial/1994/brasil.zip"),
    ("raw/dtb/dtb_2000.zip", f"{GEO}/divisao_territorial/2000/dtb_2000.zip"),
    ("raw/dtb/dtb_2013_municipio.zip", f"{GEO}/divisao_territorial/2013/dtb_2013_municipio.zip"),
    ("raw/dtb/DTB_2025.zip", f"{GEO}/divisao_territorial/2025/DTB_2025.zip"),
    (GEN_PDF, f"{GEO}/evolucao_da_divisao_territorial_do_brasil/"
              "evolucao_da_divisao_territorial_do_brasil_1872_2010/"
              "municipios_genealogia_e_alteracao_de_nome_e_grafia.pdf"),
    ("raw/ibge_evolucao/notas_tecnicas.pdf", f"{GEO}/evolucao_da_divisao_territorial_do_brasil/"
     "evolucao_da_divisao_territorial_do_brasil_1872_2010/notas_tecnicas.pdf"),
]

PIB = "https://ftp.ibge.gov.br/Pib_Municipios"
PIB_FILES = [
    ("raw/pib/base_de_dados_2002_2009_txt.zip", f"{PIB}/2022_2023/base/base_de_dados_2002_2009_txt.zip",
     "PIB dos Municipios, reference-2010 series, 2002-2009 (outcome)"),
    ("raw/pib/base_de_dados_2010_2023_txt.zip", f"{PIB}/2022_2023/base/base_de_dados_2010_2023_txt.zip",
     "PIB dos Municipios, reference-2010 series, 2010-2023 (outcome)"),
    ("raw/pib/base_1999_2012_xlsx.zip", f"{PIB}/2012/base/base_1999_2012_xlsx.zip",
     "PIB dos Municipios, reference-2002 series, 1999-2012 (outcome)"),
]

POST2010_PARENTS = {
    "Mojuí dos Campos - PA": ["Santarém"],
    "Pescaria Brava - SC": ["Laguna"],
    "Balneário Rincão - SC": ["Içara"],
    "Pinto Bandeira - RS": ["Bento Gonçalves"],
    "Paraíso das Águas - MS": ["Água Clara", "Costa Rica", "Chapadão do Sul"],
    "Boa Esperança do Norte - MT": ["Sorriso", "Nova Ubiratã"],
}

CHINA_TABLE = ROOT / "results" / "tables" / "expi_attrition_by_year.csv"
US_REF = {
    "US metro 2003-2024": 0.008267665676269216,
    "US metro 1993-2024": 0.01611201687925578,
}

WINDOWS = {
    "all 1991-2026": (1991, 2026),
    "pre-GDP 1991-1999": (1991, 1999),
    "GDP ref2002 1999-2012": (1999, 2012),
    "GDP ref2010 2002-2023": (2002, 2023),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace("'", "").replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


def load_population(data: Path) -> dict[int, pd.Series]:
    import zipfile
    pop: dict[int, pd.Series] = {}
    for y, (kind, rel, _) in POP_FILES.items():
        p = data / rel
        if p.suffix == ".json":
            rows = json.load(open(p, encoding="utf-8"))[1:]
            s = {r["D1C"][:6]: float(r["V"]) for r in rows if r["V"].isdigit()}
        else:
            with zipfile.ZipFile(p) as z:
                name = [n for n in z.namelist() if n.lower().endswith(".xls")][0]
                df = pd.read_excel(z.open(name), sheet_name=0, header=None, dtype=str)
            df = df[df[1].fillna("").str.fullmatch(r"\d{2}") & df[2].fillna("").str.fullmatch(r"\d{5}")]
            s = {r[1] + r[2][-4:]: float(r[4]) for _, r in df.iterrows()}
        pop[y] = pd.Series(s, dtype=float).sort_index()
    return pop


def load_names(data: Path) -> dict[str, str]:
    rows = json.load(open(data / POP_FILES[2026][1], encoding="utf-8"))[1:]
    rows += json.load(open(data / POP_FILES[2000][1], encoding="utf-8"))[1:]
    return {r["D1C"][:6]: r["D1N"] for r in rows}


HEADER_RE = re.compile(r"Evolução da divisão|Municípios, genealogia|Município, genealogia|Código e ano|"
                       r"censitário|\(continua|\(conclus|^Fonte|^\(1\)|^anexado|^extint|^elevado|^suprimido")


def parse_genealogy(data: Path) -> dict[str, dict]:
    """Parse IBGE 'Municípios, genealogia e alteração de nome e grafia' (4 columns/page).

    Each line is cut into segments separated by >= 3 spaces; a segment is an
    entry head (7-digit code + name), a parent line ("desmembrado de ..."),
    a name-history line ("1991 a 2010 Name"), or other text.  Segments are
    assigned to the page column whose entry-name x position is nearest and
    read column by column.  Other text directly after a parent line
    is taken as its continuation (wrapped line) unless it is page-header or
    annotation text (HEADER_RE).
    """
    txt = subprocess.run(["pdftotext", "-layout", str(data / GEN_PDF), "-"],
                         capture_output=True, text=True, check=True).stdout
    seg_re = re.compile(r"\S.*?(?=\s{3,}|$)")
    code_re = re.compile(r"^(\d{7})\s{2,}(\S.*)$")
    hist_re = re.compile(r"^(\d{4})(?: [ae] (\d{4}))? (\S.*)$")
    events = []
    for pi, pg in enumerate(txt.split("\f")):
        lines = pg.split("\n")
        segs = []
        for li, l in enumerate(lines):
            for m in re.finditer(r"(\d{7})\s{2,}(\S.*?)(?=\s{3,}|$)|\S.*?(?=\s{3,}|$)", l):
                segs.append((li, m.start() if not m.group(1) else m.start(2), m.group(0)))
        heads = [s for s in segs if code_re.match(s[2])]
        if not heads:
            continue
        cl: list[list[int]] = []
        for x in sorted(h[1] for h in heads):
            if cl and x - cl[-1][-1] <= 12:
                cl[-1].append(x)
            else:
                cl.append([x])
        centers = [sum(c) / len(c) for c in cl]

        def col(x: int) -> int:
            return min(range(len(centers)), key=lambda k: abs(centers[k] - x))
        for li, x, s in segs:
            m = code_re.match(s)
            if m:
                events.append((pi, col(x), li, "E", m.group(1), m.group(2).strip()))
                continue
            k = s.find("desmembrado de ")
            if k >= 0:
                events.append((pi, col(x), li, "D", s[k + len("desmembrado de "):].strip(), None))
                continue
            m = hist_re.match(s)
            if m:
                events.append((pi, col(x), li, "H", (m.group(1), m.group(2)), m.group(3).strip()))
                continue
            events.append((pi, col(x), li, "O", s.strip(), None))
    events.sort(key=lambda e: e[:3])
    rec: dict[str, dict] = {}
    cur, last = None, None
    for e in events:
        kind = e[3]
        if kind == "E":
            cur = e[4][:6]
            rec.setdefault(cur, {"name": e[5], "parents_raw": [], "hist": []})
            last = "E"
        elif cur is None:
            continue
        elif kind == "D":
            rec[cur]["parents_raw"].append(e[4])
            last = "D"
        elif kind == "H":
            rec[cur]["hist"].append((int(e[4][0]), int(e[4][1] or e[4][0]), e[5]))
            last = "H"
        else:
            if HEADER_RE.search(e[4]):
                continue
            prev = rec[cur]["parents_raw"][-1] if (last == "D" and rec[cur]["parents_raw"]) else None
            if prev is not None:
                rec[cur]["parents_raw"][-1] = f"{prev} {e[4]}"
                continue
            last = "O"
    return rec


UF_SIGLA = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
            "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
            "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
            "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}
SIGLA_UF = {v: k for k, v in UF_SIGLA.items()}


def resolve_parents(gen: dict[str, dict], existing: set[str]) -> tuple[dict, list]:
    """Map each genealogy entry's raw parent names to 6-digit codes.

    Names are matched within the entry's state (or the state given as ", UF").
    A parent string is split on " e " / "," only where the pieces are not
    themselves one name (e.g. "Pontes e Lacerda").  Spelling errors in the
    PDF are matched with difflib (ratio >= 0.85) and logged as 'fuzzy'.
    """
    import difflib
    idx: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for c, r in gen.items():
        idx[(c[:2], norm(r["name"]))].add(c)
        for y0, y1, n in r["hist"]:
            if y1 >= 1970:
                idx[(c[:2], norm(re.sub(r",\s*[A-Z]{2}$", "", n)))].add(c)
    old: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for c, r in gen.items():
        for y0, y1, n in r["hist"]:
            old[(c[:2], norm(re.sub(r",\s*[A-Z]{2}$", "", n)))].add(c)
    by_uf: dict[str, list[str]] = collections.defaultdict(list)
    for uf, nm in idx:
        by_uf[uf].append(nm)

    def lookup(uf: str, nm: str, self_code: str):
        cand = idx.get((uf, nm), set()) - {self_code}
        if not cand:
            cand = old.get((uf, nm), set()) - {self_code}
        if len(cand) > 1:
            alive = cand & existing
            cand = alive if alive else cand
        return cand

    out, issues = {}, []
    for c, r in gen.items():
        codes, flags = [], []
        for raw in r["parents_raw"]:
            if "(1)" in raw:
                flags.append("footnote(1): sources diverge or genealogy missing")
            s = raw.replace("(1)", "").strip().rstrip(",")
            toks = [t for t in re.split(r"(,\s*(?![A-Z]{2}\b)|\s+e\s+)", s)]
            pieces = [t for t in toks[::2]]
            seps = toks[1::2]
            i = 0
            while i < len(pieces):
                found = None
                for j in range(len(pieces), i, -1):
                    name = "".join(p + (seps[k] if k < j - 1 else "") for k, p in enumerate(pieces[i:j], start=i))
                    m = re.match(r"(.*?),\s*([A-Z]{2})$", name.strip())
                    uf = SIGLA_UF.get(m.group(2), c[:2]) if m else c[:2]
                    nm = norm(m.group(1) if m else name)
                    cand = lookup(uf, nm, c)
                    if len(cand) == 1:
                        found = (j, next(iter(cand)), None)
                        break
                if found is None:
                    name = pieces[i].strip()
                    m = re.match(r"(.*?),\s*([A-Z]{2})$", name)
                    uf = SIGLA_UF.get(m.group(2), c[:2]) if m else c[:2]
                    nm = norm(m.group(1) if m else name)
                    close = difflib.get_close_matches(nm, by_uf[uf], n=2, cutoff=0.85)
                    cand = lookup(uf, close[0], c) if close else set()
                    if len(cand) == 1:
                        found = (i + 1, next(iter(cand)), f"fuzzy '{name}' -> '{close[0]}'")
                    elif name:
                        issues.append({"code6": c, "name": r["name"], "parent_raw": name,
                                       "problem": "unresolved" if not cand else f"ambiguous {sorted(cand)}"})
                        i += 1
                        continue
                    else:
                        i += 1
                        continue
                j, code, note = found
                codes.append(code)
                if note:
                    flags.append(note)
                    issues.append({"code6": c, "name": r["name"], "parent_raw": pieces[i].strip(),
                                   "problem": note})
                i = j
        out[c] = {"parents": sorted(set(codes)), "flags": sorted(set(flags))}
    return out, issues


class UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, a: str) -> str:
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default='data/brazil')
    args = ap.parse_args()
    data = Path(args.data)
    OUT.mkdir(parents=True, exist_ok=True)

    src = []
    for y, (kind, rel, url) in sorted(POP_FILES.items()):
        src.append({"file": rel, "url": url, "content": f"municipal population {y} ({kind})",
                    "sha256": sha256(data / rel), "licence": LICENCE})
    for rel, url in OTHER_FILES:
        src.append({"file": rel, "url": url, "content": "territorial division / genealogy",
                    "sha256": sha256(data / rel), "licence": LICENCE})
    for rel, url, content in PIB_FILES:
        src.append({"file": rel, "url": url, "content": content,
                    "sha256": sha256(data / rel), "licence": LICENCE})
    pd.DataFrame(src).to_csv(OUT / "k2br_coverage_sources.csv", index=False)

    pop = load_population(data)
    names = load_names(data)
    gen = parse_genealogy(data)
    ever = set().union(*[set(s.index) for s in pop.values()])
    par, issues = resolve_parents(gen, ever)
    by_name = {norm(n): c for c, n in names.items()}
    for en, pnames in POST2010_PARENTS.items():
        c = by_name[norm(en)]
        uf_s = en.rsplit(" - ", 1)[1]
        ps = [by_name[norm(f"{pn} - {uf_s}")] for pn in pnames]
        par[c] = {"parents": ps, "flags": ["post-2010: parents from creating law (IBGE Cidades@), not genealogy PDF"]}

    ub = []
    for y in YEARS:
        ub.append({"year": y, "source": POP_FILES[y][0], "n_units": len(pop[y]),
                   "pop_total": pop[y].sum(), "pop_median": pop[y].median()})
    pd.DataFrame(ub).to_csv(OUT / "k2br_coverage_units_by_year.csv", index=False)

    rows, ent_rows = [], []
    first_year = {}
    for y in YEARS:
        for c in pop[y].index:
            first_year.setdefault(c, y)
    for t, t1 in zip(YEARS[:-1], YEARS[1:]):
        s0, s1 = set(pop[t].index), set(pop[t1].index)
        S, E, L = s0 & s1, sorted(s1 - s0), sorted(s0 - s1)
        P = sorted({p for e in E for p in par.get(e, {"parents": []})["parents"]} & S)
        unres = [e for e in E if not (set(par.get(e, {"parents": []})["parents"]) & s0)]
        Sl = sorted(S)
        lnS0 = np.log(pop[t].loc[Sl])
        lnS1 = np.log(pop[t1].loc[Sl])
        dln = lnS1 - lnS0
        nonP = [c for c in Sl if c not in set(P)]
        med_d_nonP = float(dln.loc[nonP].median())
        logS1 = np.log(pop[t1].loc[Sl])
        r = {"t": t, "t1": t1, "gap_years": t1 - t, "n_t": len(s0), "n_t1": len(s1),
             "n_S": len(S), "n_E": len(E), "n_L": len(L), "n_parents_changed_definition": len(P),
             "n_entrants_parent_unresolved": len(unres),
             "comp_shock_(nE+nL)/nS": (len(E) + len(L)) / len(S),
             "parents_share_nP/nS": len(P) / len(S),
             "entrant_pop_median": float(pop[t1].loc[E].median()) if E else np.nan,
             "stayer_pop_median_t1": float(pop[t1].loc[Sl].median()),
             "entrant_to_stayer_median_ratio": (float(pop[t1].loc[E].median()) / float(pop[t1].loc[Sl].median())) if E else np.nan,
             "entrant_minus_stayer_mean_logN": (float(np.log(pop[t1].loc[E]).mean()) - float(logS1.mean())) if E else np.nan,
             "entrant_median_rank_in_stayers": (float(np.median([(logS1 < np.log(pop[t1][e])).mean() for e in E]))
                                                if E else np.nan),
             "entrant_pop_share_of_total_t1": float(pop[t1].loc[E].sum() / pop[t1].sum()) if E else 0.0,
             "parent_median_dlogN": float(dln.loc[P].median()) if P else np.nan,
             "nonparent_stayer_median_dlogN": med_d_nonP,
             "parent_median_abs_dlogN_excess": float((dln.loc[P] - med_d_nonP).abs().median()) if P else np.nan,
             "stayer_p99_abs_dlogN_excess": float((dln.loc[nonP] - med_d_nonP).abs().quantile(0.99)),
             }
        rows.append(r)
        for e in E:
            ps = [p for p in par.get(e, {"parents": []})["parents"]]
            ps_in = [p for p in ps if p in s0]
            ent_rows.append({
                "code6": e, "name": names.get(e, gen.get(e, {}).get("name", "")),
                "entry_transition": f"{t}->{t1}", "first_pop_year": t1,
                "parents": ";".join(ps), "parents_present_at_t": len(ps_in),
                "flags": ";".join(par.get(e, {"flags": ["no genealogy entry"]})["flags"]),
                "entrant_pop_t1": pop[t1][e],
                "parents_pop_t": float(pop[t].loc[ps_in].sum()) if ps_in else np.nan,
                "entrant_over_parents_pop_t": float(pop[t1][e] / pop[t].loc[ps_in].sum()) if ps_in else np.nan,
                "parent_dlogN_minus_stayer_median": (float(np.log(pop[t1].loc[[p for p in ps_in if p in s1]].sum())
                                                           - np.log(pop[t].loc[[p for p in ps_in if p in s1]].sum()))
                                                     - med_d_nonP) if ps_in else np.nan,
            })
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "k2br_coverage_panelB_by_transition.csv", index=False)
    ENT = pd.DataFrame(ent_rows)
    ENT.to_csv(OUT / "k2br_coverage_entrants.csv", index=False)

    arow = []
    amc_maps: dict[int, dict[str, str]] = {}
    for base in (1991, 2000):
        uf = UF()
        ys = [y for y in YEARS if y >= base]
        for c in set().union(*[set(pop[y].index) for y in ys]):
            uf.find(c)
        for e in set().union(*[set(pop[y].index) for y in ys]) - set(pop[base].index):
            for p in par.get(e, {"parents": []})["parents"]:
                uf.union(e, p)
        amc = {c: uf.find(c) for c in uf.p}
        amc_maps[base] = amc
        n_orphan = len({a for a in set(amc.values())
                        if not any(m in pop[base].index for m in [k for k, v in amc.items() if v == a])})
        sizes = collections.Counter(amc.values())
        for y in ys:
            a = pd.Series(pop[y].values, index=[amc[c] for c in pop[y].index]).groupby(level=0).sum()
            arow.append({"panel": f"A_AMC{base}", "year": y, "n_units": len(a),
                         "n_amc_multi_member": sum(1 for v in sizes.values() if v > 1),
                         "max_members": max(sizes.values()), "n_amc_without_base_member": n_orphan,
                         "pop_total": a.sum(), "pop_median": a.median()})
        touched = set()
        for e in set().union(*[set(pop[y].index) for y in ys]) - set(pop[base].index):
            touched |= {e, *par.get(e, {"parents": []})["parents"]}
        const = set.intersection(*[set(pop[y].index) for y in ys]) - touched
        for y in ys:
            arow.append({"panel": f"A0_untouched{base}", "year": y, "n_units": len(const),
                         "n_amc_multi_member": 0, "max_members": 1, "n_amc_without_base_member": 0,
                         "pop_total": pop[y].loc[sorted(const)].sum(),
                         "pop_median": pop[y].loc[sorted(const)].median()})
    pd.DataFrame(arow).to_csv(OUT / "k2br_coverage_panelA_by_year.csv", index=False)

    summ = []
    for w, (a, b) in WINDOWS.items():
        sub = B[(B.t >= a) & (B.t1 <= b)]
        ratio = sub["comp_shock_(nE+nL)/nS"]
        summ.append({"dataset": "Brazil municipalities (panel B)", "window": w,
                     "n_transitions": len(sub), "n_update_transitions": int((sub.n_E + sub.n_L > 0).sum()),
                     "sum_E": int(sub.n_E.sum()), "sum_L": int(sub.n_L.sum()),
                     "sum_parents_changed_definition": int(sub.n_parents_changed_definition.sum()),
                     "nS_weighted_comp_shock": float((ratio * sub.n_S).sum() / sub.n_S.sum()) if len(sub) else np.nan,
                     "nS_weighted_parent_share": float((sub["parents_share_nP/nS"] * sub.n_S).sum() / sub.n_S.sum()) if len(sub) else np.nan,
                     "max_comp_shock": float(ratio.max()) if len(sub) else np.nan,
                     "transition_of_max": (f"{int(sub.loc[ratio.idxmax(), 't'])}->{int(sub.loc[ratio.idxmax(), 't1'])}" if len(sub) else "")})
    ch = pd.read_csv(CHINA_TABLE)
    ch = ch[ch.saai_spec == "multi"]
    for per in ("pre", "during"):
        sub = ch[ch.period == per]
        ratio = (sub.n_enter.fillna(0) + sub.n_leave) / sub.n_stay
        summ.append({"dataset": "China counties multi-SAAI (expi_attrition_by_year.csv)",
                     "window": per, "n_transitions": len(sub),
                     "n_update_transitions": int(((sub.n_enter.fillna(0) + sub.n_leave) > 0).sum()),
                     "sum_E": int(sub.n_enter.fillna(0).sum()), "sum_L": int(sub.n_leave.sum()),
                     "sum_parents_changed_definition": np.nan,
                     "nS_weighted_comp_shock": float((ratio * sub.n_stay).sum() / sub.n_stay.sum()),
                     "nS_weighted_parent_share": np.nan, "max_comp_shock": float(ratio.max()),
                     "transition_of_max": f"{int(sub.loc[ratio.idxmax(), 'year_t'])}->{int(sub.loc[ratio.idxmax(), 'year_t']) + 1}"})
    for k, v in US_REF.items():
        summ.append({"dataset": k, "window": k.split()[2], "nS_weighted_comp_shock": v})
    pd.DataFrame(summ).to_csv(OUT / "k2br_coverage_summary.csv", index=False)

    pd.DataFrame(issues, columns=["code6", "name", "parent_raw", "problem"]).to_csv(
        OUT / "k2br_coverage_issues.csv", index=False)

    import zipfile
    dtb_check = {}
    for y, rel, member in ((2013, "raw/dtb/dtb_2013_municipio.zip", "dtb_2013_municipio.xls"),
                           (2025, "raw/dtb/DTB_2025.zip", "RELATORIO_DTB_BRASIL_2025_MUNICIPIOS.xls")):
        with zipfile.ZipFile(data / rel) as z:
            x = pd.read_excel(z.open(member), header=None, dtype=str)
        codes = set()
        for c in x.columns:
            v = x[c].dropna().astype(str)
            codes |= {k[:6] for k in v[v.str.fullmatch(r"\d{7}")]}
        dtb_check[str(y)] = {"dtb_n": len(codes), "pop_n": len(pop[y]),
                             "dtb_minus_pop": sorted(codes - set(pop[y].index)),
                             "pop_minus_dtb": sorted(set(pop[y].index) - codes)}

    with zipfile.ZipFile(data / "raw/dtb/DTB_2025.zip") as z:
        x = pd.read_excel(z.open("RELATORIO_DTB_BRASIL_2025_MUNICIPIOS.xls"), header=6, dtype=str)
    x = x[x["Código Município Completo"].fillna("").str.fullmatch(r"\d{7}")]
    x["code6"] = x["Código Município Completo"].str[:6]
    blocks = {}
    for lab, colname in (("uf", "UF"), ("intermediate", "Região Geográfica Intermediária"),
                         ("immediate", "Região Geográfica Imediata")):
        m = dict(zip(x.code6, x[colname]))
        d = {"n_blocks": int(x[colname].nunique())}
        for base, amc in amc_maps.items():
            g = collections.defaultdict(set)
            for c, a in amc.items():
                g[a].add(m.get(c))
            d[f"amc{base}_straddling"] = sum(1 for v in g.values() if len(v) > 1)
            d[f"amc{base}_n"] = len(g)
        blocks[lab] = d
    pd.Series({f"{k}.{kk}": vv for k, v in blocks.items() for kk, vv in v.items()}).to_csv(
        OUT / "k2br_coverage_blocks.csv", header=["value"])

    meta = {
        "script": "scripts/k2br_coverage.py",
        "git_hash": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "years_with_population": YEARS,
        "years_missing": [1996, 2023],
        "unit_key": "first 6 digits of the IBGE 7-digit municipality code",
        "entry_rule": "first year with a published population (IBGE estimates list a municipality "
                      "from the year after its creation law, which can precede installation by up to a year)",
        "parent_rule": "genealogy 'desmembrado de' (IBGE 1872-2010) resolved by name within state; "
                       "post-2010 cases from POST2010_PARENTS",
        "genealogy_entries_parsed": len(gen),
        "dtb_crosscheck": dtb_check,
        "outcome_data_read": "none (no PIB dos Municipios file downloaded or read)",
    }
    json.dump(meta, open(OUT / "k2br_coverage_meta.json", "w"), indent=1)
    print(B[["t", "t1", "n_S", "n_E", "n_L", "n_parents_changed_definition", "n_entrants_parent_unresolved",
             "comp_shock_(nE+nL)/nS", "entrant_to_stayer_median_ratio", "entrant_median_rank_in_stayers",
             "parent_median_abs_dlogN_excess"]].to_string())
    print(pd.DataFrame(summ).to_string())
    print("issues:", len(issues))


if __name__ == "__main__":
    main()
