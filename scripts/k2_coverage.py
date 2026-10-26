"""Coverage only.  This script reads exactly one numeric series, BEA CAINC1
LineCode 2 (county mid-year population, the scaling covariate), plus the OMB
delineation files.  It never reads personal income, GDP or any other outcome
value (CAINC1 lines 1/3 and CAGDP* rows are filtered out before conversion to
numbers; see `load_population`).  No SAMI, residual, increment or drift is
computed here.

  A  back-cast panel: one delineation (the "current" one) applied to every
     year -> same metro set every year, mechanical (composition) term 0.
  B  vintage panel: each data year uses the delineation in force that year
     (latest delineation dated on or before 1 July of the year, matching BEA
     mid-year population), so metros enter when designated, leave when
     demoted, and change definition at updates.

Outputs (results/tables/):
  k2_coverage_vintages.csv          one row per delineation vintage
  k2_coverage_panelA_by_year.csv    per year, panel A
  k2_coverage_panelB_by_year.csv    per transition t->t+1, panel B (two link rules)
  k2_coverage_crosswalk_issues.csv  OMB->BEA county matching problems
  k2_coverage_sources.csv           raw files used, sha256
  k2_coverage_meta.json             git hash, rules

Usage:  python3 -P scripts/k2_coverage.py [--data DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"

NEW_ENGLAND = {"09", "23", "25", "33", "44", "50"}
EXCLUDE_STATES = {"72", "09"}
YEARS = list(range(1993, 2025))

VINTAGES = [
    ("1993-06", "1993-06-30", "raw/omb/1993/93mfips.txt", "pre2003"),
    ("1999-06", "1999-06-30", "raw/omb/1999/99mfips.txt", "pre2003"),
    ("2003-06", "2003-06-06", "raw/omb/2003/030606omb-cbsa-csa.xls", "cbsa"),
    ("2003-12", "2003-12-31", "raw/omb/200312/0312cbsas-csas.xls", "cbsa"),
    ("2004-11", "2004-11-30", "raw/omb/2004/list3.xls", "cbsa"),
    ("2005-12", "2005-12-05", "raw/omb/2005/list3.xls", "cbsa"),
    ("2006-12", "2006-12-18", "raw/omb/2006/list3.xls", "cbsa"),
    ("2007-11", "2007-11-20", "raw/omb/2007/list3.xls", "cbsa"),
    ("2008-11", "2008-11-20", "raw/omb/2008/list3.xls", "cbsa"),
    ("2009-12", "2009-12-01", "raw/omb/2009/list3.xls", "cbsa"),
    ("2013-02", "2013-02-28", "raw/omb/2013/list1.xls", "cbsa"),
    ("2015-07", "2015-07-15", "raw/omb/2015/list1.xls", "cbsa"),
    ("2017-08", "2017-08-15", "raw/omb/2017/list1.xls", "cbsa"),
    ("2018-04", "2018-04-10", "raw/omb/201804/list1.xls", "cbsa"),
    ("2018-09", "2018-09-14", "raw/omb/201809/list1_Sep_2018.xls", "cbsa"),
    ("2020-03", "2020-03-06", "raw/omb/2020/list1_2020.xls", "cbsa"),
    ("2023-07", "2023-07-21", "raw/omb/2023/list1_2023.xlsx", "cbsa"),
]
NECMA = {"1993-06": "raw/omb/1993/93nfips.txt", "1999-06": "raw/omb/1999/99nfips.txt"}

VA_COMBOS = {
    "51901": ["51003", "51540"], "51903": ["51005", "51580", "51560"],
    "51907": ["51015", "51790", "51820"], "51911": ["51031", "51680"],
    "51913": ["51035", "51640"], "51918": ["51053", "51570", "51730"],
    "51919": ["51059", "51600", "51610"], "51921": ["51069", "51840"],
    "51923": ["51081", "51595"], "51929": ["51089", "51690"],
    "51931": ["51095", "51830"], "51933": ["51121", "51750"],
    "51939": ["51143", "51590"], "51941": ["51149", "51670"],
    "51942": ["51153", "51683", "51685"], "51944": ["51161", "51775"],
    "51945": ["51163", "51530", "51678"], "51947": ["51165", "51660"],
    "51949": ["51175", "51620"], "51951": ["51177", "51630"],
    "51953": ["51191", "51520"], "51955": ["51195", "51720"],
    "51958": ["51199", "51735"], "15901": ["15009", "15005"],
}
RECODE = {c: combo for combo, comps in VA_COMBOS.items() for c in comps}
RECODE.update({
    "51515": "51019",
    "51780": "51083",
    "12025": "12086",
    "46113": "46102",
    "02270": "02158",
})
COMBO_LEAD = {combo: comps[0] for combo, comps in VA_COMBOS.items()}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def default_data_dir() -> Path:
    cand = ROOT / 'data/us'
    if cand.exists():
        return cand
    try:
        common = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute",
             "--git-common-dir"], text=True).strip()
        return Path(common).parent / 'data/us'
    except Exception:
        return cand


def load_population(data: Path) -> pd.DataFrame:
    """BEA CAINC1 LineCode 2 only (persons).  Returns wide frame, index=BEA FIPS."""
    f = data / "raw/bea/x/CAINC1__ALL_AREAS_1969_2024.csv"
    d = pd.read_csv(f, encoding="latin1", dtype=str)
    d = d[d["LineCode"].str.strip() == "2"].copy()
    assert d["Description"].str.contains("Population").all()
    d["fips"] = d["GeoFIPS"].str.strip().str.strip('"').str.strip()
    d = d[(d["fips"].str.len() == 5) & (d["fips"].str[2:] != "000")]
    yrs = [str(y) for y in YEARS]
    pop = d.set_index("fips")[yrs].apply(pd.to_numeric, errors="coerce")
    pop.columns = YEARS
    return pop


def read_cbsa_xls(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, dtype=str)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "CBSA Code"][0]
    x = pd.read_excel(path, header=hdr, dtype=str)
    cols = {c: str(c).strip() for c in x.columns}
    x = x.rename(columns=cols)
    if "FIPS" in x.columns:
        fips = x["FIPS"].str.strip()
    else:
        fips = x["FIPS State Code"].str.strip().str.zfill(2) + \
            x["FIPS County Code"].str.strip().str.zfill(3)
    lvl_col = [c for c in x.columns if c.startswith("Level of CBSA")
               or c.startswith("Metropolitan/Micropolitan")][0]
    out = pd.DataFrame({
        "code": x["CBSA Code"].str.strip(),
        "title": x["CBSA Title"].str.strip(),
        "metro": x[lvl_col].str.contains("Metropolitan", na=False),
        "fips": fips,
    })
    out = out[out["code"].str.fullmatch(r"\d{5}", na=False)
              & out["fips"].str.fullmatch(r"\d{5}", na=False)]
    return out.drop_duplicates()


def read_pre2003(path: Path, necma: Path) -> pd.DataFrame:
    """MSA / CMSA (top-level code) with county components; New England from NECMAs."""
    rows, titles = [], {}
    for line in path.read_bytes().decode("latin1").splitlines():
        if not re.match(r"^\d{4}\s", line):
            continue
        code = line[0:4]
        cty = line[24:29].strip()
        ent = line[40:45].strip()
        name = line[48:].strip()
        if not cty:
            if line[8:12].strip() == "" and name:
                titles.setdefault(code, name)
            continue
        if ent or cty[:2] in NEW_ENGLAND or not re.fullmatch(r"\d{5}", cty):
            continue
        rows.append((code, cty))
    for line in necma.read_bytes().decode("latin1").splitlines():
        m = re.match(r"^(\d{4}) (\d{2}) (\d{3})\s", line)
        if m:
            rows.append(("N" + m.group(1), m.group(2) + m.group(3)))
        else:
            m = re.match(r"^(\d{4})\s{3,}(\S.*)$", line)
            if m:
                titles.setdefault("N" + m.group(1), m.group(2).strip())
    out = pd.DataFrame(rows, columns=["code", "fips"]).drop_duplicates()
    out["title"] = out["code"].map(titles).fillna("")
    out["metro"] = True
    return out[["code", "title", "metro", "fips"]]


def load_vintage(data: Path, label: str, rel: str, kind: str) -> pd.DataFrame:
    p = data / rel
    if kind == "cbsa":
        v = read_cbsa_xls(p)
    else:
        v = read_pre2003(p, data / NECMA[label])
    v = v[~v["fips"].str[:2].isin(EXCLUDE_STATES)].copy()
    v["unit"] = v["fips"].map(lambda f: RECODE.get(f, f))
    return v


def resolve_units(v: pd.DataFrame, issues: list, label: str) -> pd.DataFrame:
    """One BEA unit belongs to one area.  Split combos go with their lead county."""
    g = v.groupby("unit")["code"].nunique()
    for unit in g[g > 1].index:
        sub = v[v["unit"] == unit]
        lead = COMBO_LEAD.get(unit)
        keep = sub[sub["fips"] == lead]["code"]
        keep = keep.iloc[0] if len(keep) else sub["code"].iloc[0]
        issues.append(dict(vintage=label, issue="combo_split_across_areas", unit=unit,
                           detail=f"codes={sorted(sub['code'].unique())}; kept={keep}"))
        v = v[~((v["unit"] == unit) & (v["code"] != keep))]
    return v.drop_duplicates(["code", "unit"])


def in_force(year: int) -> str:
    """Latest delineation dated on or before 1 July of `year` (mid-year pop)."""
    cut = pd.Timestamp(f"{year}-07-01")
    ok = [lab for lab, date, _, _ in VINTAGES if pd.Timestamp(date) <= cut]
    return ok[-1]


def area_sets(v: pd.DataFrame, metro_only: bool) -> dict[str, frozenset]:
    w = v[v["metro"]] if metro_only else v
    return {c: frozenset(s) for c, s in w.groupby("code")["unit"]}


def area_pop(units: frozenset, pop: pd.DataFrame, year: int) -> tuple[float, int]:
    s = pop.reindex(sorted(units))[year]
    return float(s.sum(skipna=True)), int(s.isna().sum())


def link(prev: dict, nxt: dict, pop: pd.DataFrame, year: int, rule: str) -> dict:
    """Map next-area code -> prev-area code (stayers)."""
    if rule == "code":
        return {c: c for c in nxt if c in prev}
    pv = pop[year]
    out = {}
    unit_prev = {u: c for c, s in prev.items() for u in s}
    for c, s in nxt.items():
        cand = {}
        for u in s:
            if u in unit_prev:
                cand[unit_prev[u]] = cand.get(unit_prev[u], 0.0) + float(np.nan_to_num(pv.get(u, 0.0)))
        if not cand:
            continue
        best = max(cand, key=cand.get)
        shared = cand[best]
        n_new = sum(float(np.nan_to_num(pv.get(u, 0.0))) for u in s)
        n_old = sum(float(np.nan_to_num(pv.get(u, 0.0))) for u in prev[best])
        if n_new > 0 and n_old > 0 and shared > 0.5 * n_new and shared > 0.5 * n_old:
            out[c] = best
    return out


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()
    data = args.data or default_data_dir()
    OUT.mkdir(parents=True, exist_ok=True)

    pop = load_population(data)
    issues: list = []
    vint = {}
    for lab, date, rel, kind in VINTAGES:
        v = load_vintage(data, lab, rel, kind)
        miss = sorted(set(v["unit"]) - set(pop.index))
        for u in miss:
            issues.append(dict(vintage=lab, issue="unit_not_in_BEA", unit=u,
                               detail=";".join(v.loc[v["unit"] == u, "code"].unique())))
        v = v[v["unit"].isin(pop.index)]
        vint[lab] = resolve_units(v, issues, lab)

    vrows = []
    prev_lab = None
    for lab, date, rel, kind in VINTAGES:
        v = vint[lab]
        m = area_sets(v, True)
        a = area_sets(v, False)
        yrs = [y for y in YEARS if in_force(y) == lab]
        r = dict(vintage=lab, date=date, kind=kind,
                 years_in_force=f"{yrs[0]}-{yrs[-1]}" if yrs else "none (superseded before 1 July)",
                 n_metro=len(m), n_cbsa_all=len(a) if kind == "cbsa" else np.nan,
                 n_metro_units=sum(len(s) for s in m.values()))
        if prev_lab is not None:
            pm = area_sets(vint[prev_lab], True)
            same = [c for c in m if c in pm]
            r.update(vs_previous=prev_lab,
                     new_codes=len(set(m) - set(pm)), dropped_codes=len(set(pm) - set(m)),
                     same_code_changed_units=sum(m[c] != pm[c] for c in same))
        vrows.append(r)
        prev_lab = lab
    pd.DataFrame(vrows).to_csv(OUT / "k2_coverage_vintages.csv", index=False)

    arows = []
    for cur in ["2023-07", "2020-03"]:
        m = area_sets(vint[cur], True)
        for y in YEARS:
            n_pop = [area_pop(s, pop, y) for s in m.values()]
            sizes = np.array([p for p, _ in n_pop if p > 0])
            arows.append(dict(current_delineation=cur, year=y, n_metro=int((np.array([p for p, _ in n_pop]) > 0).sum()),
                              n_metro_any_unit_missing=sum(k > 0 for _, k in n_pop),
                              entrants=0, leavers=0, changed_definitions=0,
                              pop_median=q(sizes, .5), pop_min=sizes.min() if len(sizes) else np.nan,
                              log10_pop_mean=float(np.log10(sizes).mean()) if len(sizes) else np.nan))
    pd.DataFrame(arows).to_csv(OUT / "k2_coverage_panelA_by_year.csv", index=False)

    brows = []
    for scope in ["metro", "cbsa_all"]:
        for rule in ["code", "overlap"]:
            for t in YEARS[:-1]:
                lt, ln = in_force(t), in_force(t + 1)
                if scope == "cbsa_all" and (VINTAGES[[l for l, *_ in VINTAGES].index(lt)][3] != "cbsa"):
                    continue
                Pt = area_sets(vint[lt], scope == "metro")
                Pn = area_sets(vint[ln], scope == "metro")
                lk = link(Pt, Pn, pop, t, rule) if lt != ln else {c: c for c in Pn}
                stay_next = sorted(lk)
                stay_prev = set(lk.values())
                ent = [c for c in Pn if c not in stay_next]
                lev = [c for c in Pt if c not in stay_prev]
                changed = [c for c in stay_next if Pn[c] != Pt[lk[c]]]
                recoded = [c for c in stay_next if c != lk[c]]
                sz_e = np.array([area_pop(Pn[c], pop, t + 1)[0] for c in ent])
                sz_l = np.array([area_pop(Pt[c], pop, t)[0] for c in lev])
                sz_s = np.array([area_pop(Pt[lk[c]], pop, t)[0] for c in stay_next])
                dlog = np.array([np.log(area_pop(Pn[c], pop, t)[0] / area_pop(Pt[lk[c]], pop, t)[0])
                                 for c in changed]) if changed else np.array([])
                med_s = q(sz_s, .5)
                brows.append(dict(
                    scope=scope, link_rule=rule, transition=f"{t}->{t + 1}",
                    vintage_t=lt, vintage_t1=ln, delineation_update=lt != ln,
                    n_t=len(Pt), n_t1=len(Pn), n_S=len(stay_next), n_E=len(ent), n_L=len(lev),
                    n_changed_definition=len(changed), n_recoded=len(recoded),
                    frac_E=len(ent) / max(len(stay_next), 1), frac_L=len(lev) / max(len(stay_next), 1),
                    frac_changed=len(changed) / max(len(stay_next), 1),
                    stayer_pop_median=med_s,
                    entrant_pop_median=q(sz_e, .5), leaver_pop_median=q(sz_l, .5),
                    entrant_to_stayer_median_ratio=q(sz_e, .5) / med_s if len(sz_e) else np.nan,
                    leaver_to_stayer_median_ratio=q(sz_l, .5) / med_s if len(sz_l) else np.nan,
                    entrant_mean_log_pop_minus_stayer=(np.log(sz_e).mean() - np.log(sz_s).mean()) if len(sz_e) else np.nan,
                    leaver_mean_log_pop_minus_stayer=(np.log(sz_l).mean() - np.log(sz_s).mean()) if len(sz_l) else np.nan,
                    entrant_pop_rank_pct_median=(float(np.median([(sz_s < e).mean() for e in sz_e])) if len(sz_e) and len(sz_s) else np.nan),
                    changed_abs_dlogN_median=q(np.abs(dlog), .5), changed_abs_dlogN_max=float(np.abs(dlog).max()) if len(dlog) else np.nan,
                    changed_dlogN_mean=float(dlog.mean()) if len(dlog) else np.nan,
                    mech_bound_per_unit_m=((len(ent) + len(lev)) / len(stay_next)) if stay_next else np.nan,
                ))
    bdf = pd.DataFrame(brows)
    bdf.to_csv(OUT / "k2_coverage_panelB_by_year.csv", index=False)

    srows = []
    t0 = bdf["transition"].str[:4].astype(int)
    for (scope, rule), g in bdf.groupby(["scope", "link_rule"]):
        for per, (a, b) in {"1993-2024": (1993, 2024), "2001-2024": (2001, 2024),
                            "2003-2024": (2003, 2024)}.items():
            h = g[(t0[g.index] >= a) & (t0[g.index] + 1 <= b)]
            if scope == "cbsa_all" and a < 2003:
                continue
            srows.append(dict(scope=scope, link_rule=rule, period=per, n_transitions=len(h),
                              n_update_transitions=int(h["delineation_update"].sum()),
                              n_pairs=int(h["n_S"].sum()), sum_E=int(h["n_E"].sum()),
                              sum_L=int(h["n_L"].sum()),
                              sum_changed_definition=int(h["n_changed_definition"].sum()),
                              nS_weighted_mech_bound_per_unit_m=float(
                                  (h["mech_bound_per_unit_m"].fillna(0) * h["n_S"]).sum() / h["n_S"].sum()),
                              max_mech_bound_per_unit_m=float(h["mech_bound_per_unit_m"].max()),
                              transition_of_max=h.loc[h["mech_bound_per_unit_m"].idxmax(), "transition"],
                              transitions_with_no_stayers=";".join(h.loc[h["n_S"] == 0, "transition"])))
    pd.DataFrame(srows).to_csv(OUT / "k2_coverage_panelB_summary.csv", index=False)

    pd.DataFrame(issues, columns=["vintage", "issue", "unit", "detail"]).to_csv(
        OUT / "k2_coverage_crosswalk_issues.csv", index=False)

    used = ["raw/bea/x/CAINC1__ALL_AREAS_1969_2024.csv"] + [r for _, _, r, _ in VINTAGES] + list(NECMA.values())
    pd.DataFrame([dict(file=u, sha256=sha256(data / u)) for u in used]).to_csv(
        OUT / "k2_coverage_sources.csv", index=False)
    meta = dict(script="scripts/k2_coverage.py", git_hash=git_hash(),
                series_read="BEA CAINC1 LineCode 2 (population) only",
                in_force_rule="latest delineation dated on or before 1 July of the data year",
                excluded_states=sorted(EXCLUDE_STATES),
                pre2003_unit="MSA or CMSA (top-level code), New England replaced by NECMAs",
                overlap_link_rule="next area linked to the previous area sharing the most year-t population, "
                                  "if the shared population exceeds 50% of both",
                years=[YEARS[0], YEARS[-1]])
    (OUT / "k2_coverage_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
