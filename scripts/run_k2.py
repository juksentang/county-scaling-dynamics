"""US metropolitan-area analysis: V0 and V1 attractor contrasts on the BEA county
panel aggregated to OMB metropolitan areas (Panel A, Panel B), with lineage
bootstrap intervals and the K1 composition predictions.

    python3 scripts/run_k2.py --synthetic OUT_DIR   (toy world, any machine)
    python3 scripts/run_k2.py --preflight           (lineage tables)
    python3 scripts/run_k2.py --real [--jobs N] [--allow-rerun REASON]

Inputs: data/us/ (make us). Outputs: results/tables/k2/.
"""
from __future__ import annotations

def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.km_mle import derived_observables, fit_mle


def _load_coverage_module():
    spec = importlib.util.spec_from_file_location("k2_coverage", ROOT / "scripts" / "k2_coverage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


COV = _load_coverage_module()

SEED = 20261024
N_BOOT = 1000
N_MIN = 250
BETA_RANGE = (0.9, 1.3)
H1_TOL = 1e-10
H2_TOL = 1e-12
H3A_TOL = 1e-8
H3B_GATE = 0.5
H3B_REL = 0.30
SD_DDOF = 1
PERIODS = {"main": (2003, 2024), "secondary": (1993, 2024)}
OUTCOMES = {
    "pi": dict(file="raw/bea/x/CAINC1__ALL_AREAS_1969_2024.csv", line="1",
               desc="Personal income", years=(1969, 2024)),
    "gdp": dict(file="raw/bea/x/CAGDP1__ALL_AREAS_2001_2024.csv", line="3",
                desc="Current-dollar GDP", years=(2001, 2024)),
}
PRIMARY_OUTCOME = "pi"
PANEL_A_VINTAGE = "2023-07"
PRIMARY_SCOPE, PRIMARY_LINK = "metro", "overlap"
SCOPES = ("metro", "cbsa_all")
LINK_RULES = ("overlap", "code")
VARIANTS = {"A": ("V0", "V1", "V1d"), "B": ("V0", "V1", "V1d", "V1d_stay")}
CONV_ARG = {"limit": "limit", "nanpath": "nan"}
CONVENTIONS = tuple(CONV_ARG)
PRIMARY_CONV = "limit"
SCHEMES = ("lineage", "state")
PRIMARY_SCHEME = "lineage"
CI_RULES = {"tol0": 0.0, "tol5": 0.05, "literal": None}
CI_TAIL = 0.025
PRIMARY_CI = "tol0"
OBS_KEYS = ("xi_trap", "rho1", "mean_increment", "mean_diffusion", "two_delta_V",
            "tau_mfpt", "gamma", "B_ratio_w")
LINEAGE_MAX_SHARE = 0.10
STATE_POP_YEAR = 2023
LITERAL_CHECK_DRAWS = 3
LITERAL_PAIR_TOL = 1e-10
LITERAL_OBS_TOL = 1e-4
OUT_REAL = ROOT / "results" / "tables" / "k2"
REGISTRATION = ROOT / 'results' / 'records' / "k2_registration.json"
RUN_LOG = ROOT / 'results' / 'records' / "k2_run_log.jsonl"
REGISTERED_FILES = ("scripts/run_k2.py", "scripts/k2_coverage.py", "src/km_mle.py",
                    "src/k1_theory.py", "src/k1_contract.py", "src/config.py",
                    "tests/test_k2.py")


class KillRule(RuntimeError):

    def __init__(self, rule: int, msg: str):
        super().__init__(f"kill rule {rule}: {msg}")
        self.rule = rule


@dataclass
class World:
    """County units, their population and outcomes, and the delineations.

    pop / outcomes[name]: DataFrame, index = unit id (str), columns = years.
    vintages: [(label, date, kind)] in date order; kind 'pre2003' or 'cbsa'.
    areas[(scope, label)]: {area code: frozenset(units)}.
    """
    years: list
    pop: pd.DataFrame
    outcomes: dict
    vintages: list
    areas: dict
    panel_a: str = PANEL_A_VINTAGE

    def in_force(self, year: int) -> str:
        cut = pd.Timestamp(f"{year}-07-01")
        ok = [lab for lab, date, _ in self.vintages if pd.Timestamp(date) <= cut]
        if not ok:
            raise ValueError(f"no delineation in force in {year}")
        return ok[-1]

    def kind(self, label: str) -> str:
        return {lab: k for lab, _, k in self.vintages}[label]


def load_outcome(data: Path, name: str) -> pd.DataFrame:
    """BEA county outcome rows (one LineCode), index = BEA FIPS, years 1993+."""
    spec = OUTCOMES[name]
    d = pd.read_csv(data / spec["file"], encoding="latin1", dtype=str)
    d = d[d["LineCode"].str.strip() == spec["line"]].copy()
    if not d["Description"].str.contains(spec["desc"], regex=False).all():
        raise ValueError(f"{name}: LineCode {spec['line']} is not {spec['desc']!r}")
    d["fips"] = d["GeoFIPS"].str.strip().str.strip('"').str.strip()
    d = d[(d["fips"].str.len() == 5) & (d["fips"].str[2:] != "000")]
    yrs = [y for y in COV.YEARS if str(y) in d.columns]
    out = d.set_index("fips")[[str(y) for y in yrs]].apply(pd.to_numeric, errors="coerce")
    out.columns = yrs
    return out


def load_world(data: Path, with_outcomes: bool) -> World:
    """Real world from data_k2 via the committed coverage loaders (same unit
    crosswalk, exclusions and in-force rule as scripts/k2_coverage.py)."""
    pop = COV.load_population(data)
    issues: list = []
    areas, vintages = {}, []
    for lab, date, rel, kind in COV.VINTAGES:
        v = COV.load_vintage(data, lab, rel, kind)
        v = v[v["unit"].isin(pop.index)]
        v = COV.resolve_units(v, issues, lab)
        for scope in SCOPES:
            areas[(scope, lab)] = COV.area_sets(v, scope == "metro")
        vintages.append((lab, date, kind))
    if issues:
        raise RuntimeError(f"crosswalk issues reappeared: {issues[:3]}")
    outcomes = {n: load_outcome(data, n) for n in OUTCOMES} if with_outcomes else {}
    return World(years=list(COV.YEARS), pop=pop, outcomes=outcomes, vintages=vintages,
                 areas=areas)


@dataclass
class Agg:
    codes: list
    N: np.ndarray
    Y: np.ndarray
    present: np.ndarray
    n_missing: np.ndarray
    years: list


def aggregate(world: World, scope: str, label: str, outcome: str) -> Agg:
    """Area totals of N and Y over the units with finite positive N and Y."""
    amap = world.areas[(scope, label)]
    codes = sorted(amap)
    years = world.years
    units = sorted({u for s in amap.values() for u in s})
    P = world.pop.reindex(units)[years].to_numpy(float)
    Yo = world.outcomes[outcome].reindex(units).reindex(columns=years).to_numpy(float)
    valid = np.isfinite(P) & np.isfinite(Yo) & (P > 0) & (Yo > 0)
    uidx = {u: i for i, u in enumerate(units)}
    M = np.zeros((len(codes), len(units)))
    for i, c in enumerate(codes):
        M[i, [uidx[u] for u in amap[c]]] = 1.0
    Nn = M @ np.where(valid, P, 0.0)
    Yn = M @ np.where(valid, Yo, 0.0)
    cnt = M @ valid.astype(float)
    present = (cnt > 0) & (Nn > 0) & (Yn > 0)
    return Agg(codes, Nn, Yn, present, M.sum(1)[:, None] - cnt, list(years))


def lineages(world: World, scope: str) -> tuple[dict, dict]:
    """Connected components of units that share an area in any delineation
    of `scope` (all vintages, Panel A's included).  Returns
    ({unit: lineage id}, {(label, code): lineage id}); ids are 0..L-1 in the
    order of the smallest unit id of the component."""
    parent: dict = {}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for (sc, lab), amap in world.areas.items():
        if sc != scope:
            continue
        for s in amap.values():
            s = sorted(s)
            for u in s:
                parent.setdefault(u, u)
            r0 = find(s[0])
            for u in s[1:]:
                r = find(u)
                if r != r0:
                    parent[max(r, r0)] = min(r, r0)
                    r0 = min(r, r0)
    roots = sorted({find(u) for u in parent})
    rid = {r: i for i, r in enumerate(roots)}
    unit_lin = {u: rid[find(u)] for u in parent}
    area_lin = {}
    for (sc, lab), amap in world.areas.items():
        if sc != scope:
            continue
        for c, s in amap.items():
            ids = {unit_lin[u] for u in s}
            assert len(ids) == 1
            area_lin[(lab, c)] = ids.pop()
    return unit_lin, area_lin


def lineage_states(world: World, unit_lin: dict, year: int = STATE_POP_YEAR,
                   scope: str = PRIMARY_SCOPE) -> np.ndarray:
    n = max(unit_lin.values()) + 1
    in_a = {u for s in world.areas.get((scope, world.panel_a), {}).values() for u in s}
    best = [(False, -np.inf, "")] * n
    pv = world.pop[year] if year in world.pop.columns else world.pop.iloc[:, -1]
    for u in sorted(unit_lin):
        p = float(pv.get(u, np.nan))
        p = -np.inf if not np.isfinite(p) else p
        lid = unit_lin[u]
        cand = (u in in_a, p)
        if best[lid][2] == "" or cand > best[lid][:2]:
            best[lid] = (u in in_a, p, u)
    return np.array([u[:2] for _, _, u in best])


def lineage_table(world: World, scope: str) -> pd.DataFrame:
    unit_lin, area_lin = lineages(world, scope)
    states = lineage_states(world, unit_lin, scope=scope)
    rows = []
    a_codes = world.areas[(scope, world.panel_a)]
    for lid in range(len(states)):
        ar = [(lab, c) for (lab, c), l in area_lin.items() if l == lid]
        rows.append(dict(scope=scope, lineage=lid, state=states[lid],
                         n_units=sum(1 for u, l in unit_lin.items() if l == lid),
                         n_area_vintages=len(ar),
                         n_distinct_codes=len({c for _, c in ar}),
                         n_panelA_areas=sum(1 for c in a_codes if area_lin[(world.panel_a, c)] == lid)))
    return pd.DataFrame(rows)


def lineage_summary(tab: pd.DataFrame, world: World) -> dict:
    scope = tab["scope"].iloc[0]
    n_a = len(world.areas[(scope, world.panel_a)])
    n_av = int(tab["n_area_vintages"].sum())
    i = tab["n_panelA_areas"].idxmax()
    j = tab["n_area_vintages"].idxmax()
    return dict(scope=scope, n_lineages=len(tab), n_states=tab["state"].nunique(),
                n_panelA_areas=n_a, n_area_vintages=n_av,
                max_panelA_areas_in_one_lineage=int(tab.at[i, "n_panelA_areas"]),
                max_share_panelA=float(tab.at[i, "n_panelA_areas"] / n_a),
                max_share_area_vintages=float(tab.at[j, "n_area_vintages"] / n_av),
                exceeds_10pct=bool(max(tab.at[i, "n_panelA_areas"] / n_a,
                                       tab.at[j, "n_area_vintages"] / n_av) > LINEAGE_MAX_SHARE))


@dataclass
class Skeleton:
    key: tuple
    years: list
    transitions: list
    recs: dict
    trans: dict
    dropped: list = field(default_factory=list)


def period_years(outcome: str, period: str) -> tuple[int, int]:
    a, b = PERIODS[period]
    lo, hi = OUTCOMES[outcome]["years"] if outcome in OUTCOMES else (a, b)
    return max(a, lo), min(b, hi)


def spec_list(outcomes=tuple(OUTCOMES)) -> list[tuple]:
    """(scope, outcome, period, panel, link).  The code rule and cbsa_all only
    in the CBSA era (main period); see the module docstring."""
    out = []
    for scope in SCOPES:
        for oc in outcomes:
            for period in PERIODS:
                if scope == "cbsa_all" and period != "main":
                    continue
                out.append((scope, oc, period, "A", "none"))
                for link in LINK_RULES:
                    if link == "code" and period != "main":
                        continue
                    out.append((scope, oc, period, "B", link))
    return out


def _rec(agg: Agg, year: int, area_lin: dict, label: str) -> dict:
    j = agg.years.index(year)
    idx = np.where(agg.present[:, j])[0]
    codes = [agg.codes[i] for i in idx]
    return dict(codes=codes, logN=np.log(agg.N[idx, j]), logY=np.log(agg.Y[idx, j]),
                lin=np.array([area_lin[(label, c)] for c in codes], dtype=int),
                vintage=label, pos={c: k for k, c in enumerate(codes)})


def build_skeleton(world: World, aggs: dict, key: tuple, area_lin: dict,
                   drop_years: set | None = None) -> Skeleton:
    """Area records per year and the stayer / leaver / entrant structure of
    every transition of the period (Panel B linked by `link`)."""
    scope, outcome, period, panel, link = key
    y0, y1 = period_years(outcome, period)
    years = list(range(y0, y1 + 1))
    lab_of = {y: (world.panel_a if panel == "A" else world.in_force(y)) for y in years}
    recs = {y: _rec(aggs[(scope, lab_of[y], outcome)], y, area_lin, lab_of[y]) for y in years}
    drop_years = set(drop_years or ())
    trans, ts, dropped = {}, [], []
    for t in years[:-1]:
        if t in drop_years or t + 1 in drop_years:
            dropped.append(t)
            continue
        rt, rn = recs[t], recs[t + 1]
        lt, ln = lab_of[t], lab_of[t + 1]
        Pt, Pn = world.areas[(scope, lt)], world.areas[(scope, ln)]
        if lt == ln:
            lk = {c: c for c in Pn if c in Pt}
        else:
            lk = COV.link(Pt, Pn, world.pop, t, link)
        stay = sorted((rt["pos"][lk[c]], rn["pos"][c]) for c in lk
                      if c in rn["pos"] and lk[c] in rt["pos"])
        ia = np.array([a for a, _ in stay], dtype=int)
        ib = np.array([b for _, b in stay], dtype=int)
        if len(ia) and not np.array_equal(rt["lin"][ia], rn["lin"][ib]):
            raise AssertionError(f"{key} {t}: a stayer link crosses lineages")
        lev = np.setdiff1d(np.arange(len(rt["codes"])), ia)
        ent = np.setdiff1d(np.arange(len(rn["codes"])), ib)
        changed = np.array([Pn[rn["codes"][b]] != Pt[rt["codes"][a]] for a, b in stay], dtype=bool)
        ag = aggs[(scope, ln, outcome)]
        jt, jn = ag.years.index(t), ag.years.index(t + 1)
        both = np.where(ag.present[:, jt] & ag.present[:, jn])[0]
        dcodes = [ag.codes[i] for i in both]
        stay_codes = {rn["codes"][b] for b in ib}
        trans[t] = dict(
            ia=ia, ib=ib, lev=lev, ent=ent, changed=changed,
            vintage_t=lt, vintage_t1=ln,
            v1d=dict(logN_t=np.log(ag.N[both, jt]), logY_t=np.log(ag.Y[both, jt]),
                     logN_t1=np.log(ag.N[both, jn]), logY_t1=np.log(ag.Y[both, jn]),
                     lin=np.array([area_lin[(ln, c)] for c in dcodes], dtype=int),
                     stay=np.array([c in stay_codes for c in dcodes], dtype=bool)))
        ts.append(t)
    return Skeleton(key, years, ts, recs, trans, dropped)


def sami(logN: np.ndarray, logY: np.ndarray, w: np.ndarray) -> tuple:
    """Weighted (integer multiplicity) OLS log Y ~ 1 + log N; xi = r / sd with
    sd² = Σ w r² / (Σ w − SD_DDOF).  Equal to OLS on the expanded sample."""
    W = w.sum()
    mx, my = (w * logN).sum() / W, (w * logY).sum() / W
    dx, dy = logN - mx, logY - my
    sxx = (w * dx * dx).sum()
    beta = (w * dx * dy).sum() / sxx
    alpha = my - beta * mx
    r = logY - alpha - beta * logN
    ssr = (w * r * r).sum()
    sd = np.sqrt(ssr / (W - SD_DDOF))
    r2 = 1.0 - ssr / (w * dy * dy).sum()
    return r / sd, float(beta), float(r2), float(sd)


def _wmean(x, w):
    W = w.sum()
    return float((w * x).sum() / W) if W > 0 else np.nan


def _wvar(x, w):
    W = w.sum()
    if W - SD_DDOF <= 0:
        return np.nan
    m = (w * x).sum() / W
    return float((w * (x - m) ** 2).sum() / (W - SD_DDOF))


def run_spec(sk: Skeleton, lin_w: np.ndarray | None = None) -> dict:
    """One pass of the whole pipeline on one skeleton: yearly SAMI, the V0 /
    V1 / V1d / V1d_stay pairs of every transition, the Lemma 1 decomposition.
    lin_w: multiplicity per lineage (None = every lineage once)."""
    def wts(lin):
        return np.ones(len(lin)) if lin_w is None else lin_w[lin].astype(float)

    xi0, wy, beta_rows = {}, {}, []
    for y in sk.years:
        r = sk.recs[y]
        w = wts(r["lin"])
        m = w > 0
        xi = np.full(len(w), np.nan)
        if m.sum() > 2:
            xi[m], b, r2, sd = sami(r["logN"][m], r["logY"][m], w[m])
        else:
            b = r2 = sd = np.nan
        xi0[y], wy[y] = xi, w
        beta_rows.append(dict(year=y, n_areas=float(w.sum()), beta=b, r2=r2, sd_resid=sd))
    pairs = {v: [] for v in VARIANTS[sk.key[3]]}
    dec = []
    for t in sk.transitions:
        tr = sk.trans[t]
        ia, ib = tr["ia"], tr["ib"]
        ws = wy[t][ia]
        k = ws > 0
        ia, ib, ws, ch = ia[k], ib[k], ws[k], tr["changed"][k]
        wl = wy[t][tr["lev"]]
        we = wy[t + 1][tr["ent"]]
        wl, lev = wl[wl > 0], tr["lev"][wy[t][tr["lev"]] > 0]
        we, ent = we[we > 0], tr["ent"][wy[t + 1][tr["ent"]] > 0]
        n_S, n_L, n_E = ws.sum(), wl.sum(), we.sum()
        x_t, x_t1 = xi0[t][ia], xi0[t + 1][ib]
        m_L = _wmean(xi0[t][lev], wl) if n_L else 0.0
        m_E = _wmean(xi0[t + 1][ent], we) if n_E else 0.0
        mech = (n_L * m_L - n_E * m_E) / n_S if n_S else np.nan
        obs_inc = _wmean(x_t1 - x_t, ws) if n_S else np.nan
        row = dict(year_t=t, vintage_t=tr["vintage_t"], vintage_t1=tr["vintage_t1"],
                   n_S=n_S, n_E=n_E, n_L=n_L, n_changed=float(ws[ch].sum()),
                   m_E=m_E if n_E else np.nan, m_L=m_L if n_L else np.nan, M_t=mech,
                   mean_inc_V0=obs_inc, identity_residual=obs_inc - mech,
                   bound_per_unit_m=(n_E + n_L) / n_S if n_S else np.nan,
                   upper_bound=((n_E + n_L) / n_S * max(abs(m_E) if n_E else 0.0,
                                                       abs(m_L) if n_L else 0.0)) if n_S else np.nan,
                   xbar_stay_t=_wmean(x_t, ws), var_stay_t=_wvar(x_t, ws),
                   xbar_stay_t1=_wmean(x_t1, ws), var_stay_t1=_wvar(x_t1, ws))
        pairs["V0"].append((x_t, x_t1, ws))
        rt, rn = sk.recs[t], sk.recs[t + 1]
        if n_S > 2:
            v1t = sami(rt["logN"][ia], rt["logY"][ia], ws)[0]
            v1n = sami(rn["logN"][ib], rn["logY"][ib], ws)[0]
        else:
            v1t = v1n = np.full(len(ia), np.nan)
        pairs["V1"].append((v1t, v1n, ws))
        row["mean_inc_V1"] = _wmean(v1n - v1t, ws) if n_S else np.nan
        d = tr["v1d"]
        wd = wts(d["lin"])
        for name, sel in (("V1d", wd > 0), ("V1d_stay", (wd > 0) & d["stay"])):
            if name not in pairs:
                continue
            if sel.sum() > 2:
                a = sami(d["logN_t"][sel], d["logY_t"][sel], wd[sel])[0]
                b = sami(d["logN_t1"][sel], d["logY_t1"][sel], wd[sel])[0]
            else:
                a = b = np.full(int(sel.sum()), np.nan)
            pairs[name].append((a, b, wd[sel]))
            row[f"n_{name}"] = float(wd[sel].sum())
            row[f"mean_inc_{name}"] = _wmean(b - a, wd[sel]) if sel.any() else np.nan
        dec.append(row)
    pooled = {}
    for v, lst in pairs.items():
        if lst:
            x = np.concatenate([p[0] for p in lst])
            x1 = np.concatenate([p[1] for p in lst])
            w = np.concatenate([p[2] for p in lst])
        else:
            x = x1 = w = np.array([])
        pooled[v] = (x, x1 - x, w)
    return dict(beta=pd.DataFrame(beta_rows), dec=pd.DataFrame(dec), pairs=pooled)


def _expand(x, dx, w):
    """The expanded sample (each pair repeated by its integer weight), in
    lexicographic (x, dx) order, so that the estimator's input does not depend
    on the order in which areas or lineage copies were listed."""
    k = np.round(w).astype(int)
    xe, de = np.repeat(x, k), np.repeat(dx, k)
    o = np.lexsort((de, xe))
    return xe[o], de[o]


def estimate(x, dx, w, x0=None) -> dict:
    """{convention: observables} of the pooled pairs (expanded by weight)."""
    xe, de = _expand(x, dx, w)
    nan = {c: {**dict.fromkeys(OBS_KEYS, np.nan), "id_status": "failed",
               "converged": False, "n_pairs": int(len(xe))} for c in CONVENTIONS}
    if len(xe) < 20 or not (np.isfinite(xe).all() and np.isfinite(de).all()):
        return dict(nan, params=None)
    try:
        fit = fit_mle(xe, de, x0=x0)
    except Exception:
        return dict(nan, params=None)
    xc = xe - xe.mean()
    slope = float((xc * (de - de.mean())).sum() / (xc * xc).sum())
    out = {"params": fit["params"]}
    for c in CONVENTIONS:
        try:
            o = derived_observables(fit["params"], xe, xi_target=0.0, degenerate=CONV_ARG[c])
        except Exception:
            out[c] = dict(nan[c])
            continue
        out[c] = dict(xi_trap=o["xi_trap"], rho1=1.0 + slope, mean_increment=float(de.mean()),
                      mean_diffusion=o["mean_diffusion"], two_delta_V=2.0 * o["delta_V"],
                      tau_mfpt=o["tau_mfpt"], gamma=o["gamma"], B_ratio_w=o["B_ratio_w"],
                      id_status=o["id_status"], converged=fit["converged"], n_pairs=int(len(xe)))
    return out


def estimate_spec(res: dict, panel: str, warm: dict | None = None,
                  reuse_a: bool = False) -> dict:
    """Estimates of every variant of one run_spec result.  On Panel A with
    reuse_a, V1 / V1d reuse the V0 fit when their pairs are identical
    (H2; checked here, a fresh fit otherwise)."""
    out = {}
    for v in VARIANTS[panel]:
        x, dx, w = res["pairs"][v]
        if reuse_a and panel == "A" and v != "V0":
            x0_, dx0_, w0_ = res["pairs"]["V0"]
            if (len(x) == len(x0_) and np.array_equal(x, x0_) and np.array_equal(dx, dx0_)
                    and np.array_equal(w, w0_)):
                out[v] = out["V0"]
                continue
        out[v] = estimate(x, dx, w, None if warm is None else warm.get(v))
    return out


def _obs_gap(a: float, b: float) -> float:
    """|a − b|; 0 when both are NaN (or equal infinities); inf when only one
    is finite."""
    if np.isnan(a) and np.isnan(b):
        return 0.0
    if a == b:
        return 0.0
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.inf
    return abs(a - b)


def k1_guard(v1_pairs: tuple, rho1_v1: float) -> tuple | None:
    """The guard K1 registers for a cubic fitted to V1 pairs of real data
    (src/k1_contract.registered_true_params, processes (ii)/(iii)): the 1–99 %
    support of the V1 xi_t (scripts/k1_calibrate.py: np.percentile of the
    pairs' xi_t), slope_floor = −|a1| of the V1 OU fit (its Gaussian MLE
    slope = the pooled OLS slope = rho1 − 1).  None when not defined."""
    xe, _ = _expand(*v1_pairs)
    if len(xe) < 20 or not np.isfinite(rho1_v1) or rho1_v1 - 1.0 == 0.0:
        return None
    return (float(np.percentile(xe, 1)), float(np.percentile(xe, 99)), -abs(rho1_v1 - 1.0))


def k1_prediction(theta_v1: np.ndarray, dec: pd.DataFrame, guard: tuple | None = None) -> dict:
    """H3b prediction with the frozen K1 pooled projection (src/k1_theory.py
    of the second freeze, `_pooled` called as is).

    Latent frame = the V1 frame of each transition (stayers: mean 0, var 1,
    ddof SD_DDOF); the V0 frame of the stayers is affine in it,
    xi0 = (x − mu)/sigma, so sigma_t = 1/sqrt(var_stay_t), mu_t = −xbar_t·sigma_t
    (and the same at t+1).  Open rows: n_S, M_t, the observed stayer design
    (xbar_stay_t, var_stay_t); complete rows: no composition (M = 0, xbar 0,
    V 1, mu 0, sigma 1).  wH = 0 (no common-shock share), cp = (0, 1, 1).
    Predicted V0 − V1 = projection(open) − projection(complete).

    guard: the latent chain's guard (k1_guard: K1's registered rule for a
    fitted cubic; primary); None = the unguarded polynomial (report only;
    the frozen theory refuses a chain whose stationary law is not confined,
    reported as status projection_undefined)."""
    from src import k1_theory as T
    out = dict(status="ok", pred_xi_trap=np.nan, pred_rho1=np.nan,
               proj_open_xi_trap=np.nan, proj_complete_xi_trap=np.nan)
    if theta_v1 is None or not np.all(np.isfinite(theta_v1)):
        return {**out, "status": "no_V1_fit"}
    d = dec[dec["n_S"] > 0]
    open_rows, comp_rows = [], []
    for r in d.itertuples(index=False):
        if not (r.var_stay_t > 0 and r.var_stay_t1 > 0):
            return {**out, "status": f"var_stay<=0 at {r.year_t}"}
        st, st1 = 1.0 / np.sqrt(r.var_stay_t), 1.0 / np.sqrt(r.var_stay_t1)
        open_rows.append(dict(n_stay=float(r.n_S), mu_t=-r.xbar_stay_t * st, sigma_t=st,
                              mu_t1=-r.xbar_stay_t1 * st1, sigma_t1=st1, mech=float(r.M_t),
                              xbar=float(r.xbar_stay_t), V=float(r.var_stay_t), wH=0.0,
                              cp=(0.0, 1.0, 1.0)))
        comp_rows.append(dict(n_stay=float(r.n_S), mu_t=0.0, sigma_t=1.0, mu_t1=0.0,
                              sigma_t1=1.0, mech=0.0, xbar=0.0, V=1.0, wH=0.0,
                              cp=(0.0, 1.0, 1.0)))
    try:
        po = T._pooled(np.asarray(theta_v1, float), open_rows, guard)
        pc = T._pooled(np.asarray(theta_v1, float), comp_rows, guard)
    except Exception as e:
        return {**out, "status": f"projection_undefined: {type(e).__name__}: {str(e)[:160]}"}
    return dict(status="ok",
                pred_xi_trap=float(po["obs"]["xi_trap"] - pc["obs"]["xi_trap"]),
                pred_rho1=float(po["obs"]["rho1"] - pc["obs"]["rho1"]),
                proj_open_xi_trap=float(po["obs"]["xi_trap"]),
                proj_complete_xi_trap=float(pc["obs"]["xi_trap"]),
                proj_open_id=po["obs"].get("id_status"),
                proj_complete_id=pc["obs"].get("id_status"))


def zeroth_order(dec: pd.DataFrame, rho1_v1: float) -> float:
    """Report-only zeroth-order shift M̄/|a1| (M̄ n_S-weighted, a1 = rho1(V1) − 1)."""
    d = dec[dec["n_S"] > 0]
    mbar = float((d["n_S"] * d["M_t"]).sum() / d["n_S"].sum())
    return mbar / abs(rho1_v1 - 1.0)


def draw_weights(rng: np.random.Generator, n_lin: int, block_of_lin: np.ndarray) -> np.ndarray:
    """Multiplicity per lineage of one block draw (blocks with replacement)."""
    blocks = np.unique(block_of_lin)
    picks = rng.choice(blocks, size=len(blocks), replace=True)
    cnt = pd.Series(picks).value_counts()
    bw = np.array([cnt.get(b, 0) for b in blocks])
    pos = {b: i for i, b in enumerate(blocks)}
    return np.array([bw[pos[b]] for b in block_of_lin], dtype=float)


def draw_plan(seed: int, n_boot: int, scope_i: int, scheme_i: int, n_lin: int,
              block_of_lin: np.ndarray) -> np.ndarray:
    """All draws of one (scope, scheme), fixed before any is computed:
    rng = default_rng(SeedSequence(seed, spawn_key=(scope_i, scheme_i)))."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(scope_i, scheme_i)))
    return np.stack([draw_weights(rng, n_lin, block_of_lin) for _ in range(n_boot)])


_G: dict = {}


def _draw_records(b: int) -> list:
    """Worker: one draw over every skeleton of the scope (fork-shared _G)."""
    sks, warm, W = _G["sks"], _G["warm"], _G["W"]
    rows, ident = [], 0.0
    for key, sk in sks.items():
        res = run_spec(sk, W[b])
        if len(res["dec"]):
            ident = max(ident, float(np.nanmax(np.abs(res["dec"]["identity_residual"]))))
        est = estimate_spec(res, key[3], warm.get(key), reuse_a=True)
        for v, e in est.items():
            for c in CONVENTIONS:
                rows.append(dict(draw=b, scope=key[0], outcome=key[1], period=key[2],
                                 panel=key[3], link=key[4], variant=v, conv=c,
                                 **{q: e[c][q] for q in OBS_KEYS},
                                 id_status=e[c]["id_status"], n_pairs=e[c]["n_pairs"]))
    rows.append(dict(draw=b, scope="__identity__", identity_max=ident))
    return rows


def bootstrap(sks: dict, warm: dict, W: np.ndarray, n_jobs: int) -> pd.DataFrame:
    _G.update(sks=sks, warm=warm, W=W)
    idx = list(range(len(W)))
    if n_jobs > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            parts = pool.map(_draw_records, idx, chunksize=max(1, len(idx) // (n_jobs * 8)))
    else:
        parts = [_draw_records(b) for b in idx]
    return pd.DataFrame([r for p in parts for r in p])


def order_ci(v: np.ndarray) -> tuple[float, float]:
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    if n == 0:
        return np.nan, np.nan
    k = max(1, int(np.ceil(CI_TAIL * n - 1e-12)))
    return float(v[k - 1]), float(v[n - k])


def ci(values: np.ndarray, rule: str) -> tuple[float, float, bool]:
    """95% percentile CI (lo, hi, usable) under one K1 NaN rule; failed
    draws (NaN) always count in n."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return np.nan, np.nan, False
    isnan = np.isnan(v)
    if CI_RULES[rule] is None:
        lo = order_ci(np.where(isnan, -np.inf, v))[0]
        hi = order_ci(np.where(isnan, np.inf, v))[1]
        return lo, hi, True
    if isnan.mean() > CI_RULES[rule] or isnan.all():
        lo, hi = order_ci(v[~isnan]) if (~isnan).any() else (np.nan, np.nan)
        return lo, hi, False
    lo, hi = order_ci(v[~isnan])
    return lo, hi, True


def literal_county_draw(world: World, scope: str, unit_lin: dict, lin_w: np.ndarray,
                        keys: list, drop: dict, warm: dict | None = None) -> dict:
    """Rebuild one draw literally: every unit of a lineage drawn k times is
    copied k times (ids 'u#j'), areas likewise ('code#j'), then the whole
    pipeline runs from county aggregation with each record once.  Returns
    {key: estimate_spec output}; must equal the weighted fast path."""
    areas, new_pop_idx = {}, []
    for u, lid in sorted(unit_lin.items()):
        for j in range(int(lin_w[lid])):
            new_pop_idx.append((u, f"{u}#{j}"))
    src = [u for u, _ in new_pop_idx]
    dst = [n for _, n in new_pop_idx]
    pop = world.pop.reindex(src)
    pop.index = dst
    outcomes = {}
    for n, df in world.outcomes.items():
        o = df.reindex(src)
        o.index = dst
        outcomes[n] = o
    for (sc, lab), amap in world.areas.items():
        if sc != scope:
            continue
        na = {}
        for c, s in amap.items():
            lid = unit_lin[next(iter(s))]
            for j in range(int(lin_w[lid])):
                na[f"{c}#{j}"] = frozenset(f"{u}#{j}" for u in s)
        areas[(sc, lab)] = na
    w2 = World(years=world.years, pop=pop, outcomes=outcomes, vintages=world.vintages,
               areas=areas, panel_a=world.panel_a)
    _, area_lin2 = lineages(w2, scope)
    aggs = {(scope, lab, oc): aggregate(w2, scope, lab, oc)
            for (sc, lab) in w2.areas for oc in w2.outcomes if sc == scope}
    out = {}
    for key in keys:
        sk = build_skeleton(w2, aggs, key, area_lin2, drop.get((key[1], key[2])))
        res = run_spec(sk)
        out[key] = (res, estimate_spec(res, key[3], (warm or {}).get(key), reuse_a=True))
    return out


def pair_multiset_diff(a: tuple, b: tuple) -> float:
    """Max abs difference of two pooled pair sets after expansion and sorting
    (inf if their sizes differ)."""
    xa, da = _expand(*a)
    xb, db = _expand(*b)
    if len(xa) != len(xb):
        return np.inf
    if not len(xa):
        return 0.0
    with np.errstate(invalid="ignore"):
        d = np.concatenate([np.abs(xa - xb), np.abs(da - db)])
    both_nan = np.concatenate([np.isnan(xa) & np.isnan(xb), np.isnan(da) & np.isnan(db)])
    d[both_nan] = 0.0
    return float(np.nanmax(np.where(np.isnan(d), np.inf, d)))


def n_areas_by_year(world: World, aggs: dict, outcome: str, scope: str = PRIMARY_SCOPE) -> dict:
    out = {}
    for y in world.years:
        if outcome in world.outcomes:
            lab = world.in_force(y)
            ag = aggs[(scope, lab, outcome)]
            out[y] = int(ag.present[:, ag.years.index(y)].sum())
    return out


def min_n_drops(world: World, aggs: dict, n_min: int) -> tuple[dict, pd.DataFrame]:
    """Kill rule 4: panel-B metro years with fewer than n_min areas leave the
    main analysis with both adjacent transitions (every panel / variant of
    that outcome, so the pooled transitions stay comparable)."""
    drops, rows = {}, []
    for oc in world.outcomes:
        n = n_areas_by_year(world, aggs, oc)
        for period in PERIODS:
            y0, y1 = period_years(oc, period)
            bad = {y for y in range(y0, y1 + 1) if n.get(y, 0) < n_min}
            drops[(oc, period)] = bad
            for y in range(y0, y1 + 1):
                rows.append(dict(outcome=oc, period=period, year=y, n_metro_B=n.get(y, 0),
                                 below_n_min=y in bad))
    return drops, pd.DataFrame(rows)


def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def run_pipeline(world: World, out: Path, *, n_boot: int = N_BOOT, seed: int = SEED,
                 n_jobs: int = 1, n_min: int = N_MIN, beta_range=BETA_RANGE,
                 literal_checks: int = LITERAL_CHECK_DRAWS, meta: dict | None = None) -> dict:
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.update(seed=seed, n_boot=n_boot, n_min=n_min, beta_range=list(beta_range),
                sd_ddof=SD_DDOF, periods=PERIODS, panel_a=world.panel_a,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))

    def finish(status: str, **kw):
        meta.update(status=status, runtime_s=round(time.time() - t_start, 1), **kw)
        (out / "k2_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    lin = {s: lineages(world, s) for s in SCOPES}
    ltabs = pd.concat([lineage_table(world, s) for s in SCOPES], ignore_index=True)
    ltabs.to_csv(out / "k2_lineages.csv", index=False)
    lsum = pd.DataFrame([lineage_summary(ltabs[ltabs.scope == s], world) for s in SCOPES])
    lsum.to_csv(out / "k2_lineage_summary.csv", index=False)

    aggs = {(sc, lab, oc): aggregate(world, sc, lab, oc)
            for (sc, lab) in world.areas for oc in world.outcomes}
    miss = [dict(scope=sc, vintage=lab, outcome=oc, year=y, n_area_years_any_unit_missing=int(
        ((ag.n_missing[:, j] > 0) & ag.present[:, j]).sum()), n_areas_absent=int((~ag.present[:, j]).sum()))
        for (sc, lab, oc), ag in aggs.items() for j, y in enumerate(ag.years)]
    pd.DataFrame(miss).to_csv(out / "k2_aggregation_missing.csv", index=False)

    drops, nmin_tab = min_n_drops(world, aggs, n_min)
    nmin_tab.to_csv(out / "k2_n_min.csv", index=False)
    keys = [k for k in spec_list(tuple(world.outcomes))]
    sks = {k: build_skeleton(world, aggs, k, lin[k[0]][1], drops[(k[1], k[2])]) for k in keys}

    point = {k: run_spec(sk) for k, sk in sks.items()}
    betas = pd.concat([p["beta"].assign(scope=k[0], outcome=k[1], period=k[2], panel=k[3], link=k[4])
                       for k, p in point.items()], ignore_index=True)
    betas = betas.drop_duplicates(["scope", "outcome", "panel", "year"])
    betas = betas.drop(columns=["link", "period"]).sort_values(["scope", "outcome", "panel", "year"])
    betas["in_range"] = betas["beta"].between(*beta_range)
    betas.to_csv(out / "k2_beta_by_year.csv", index=False)
    bad = betas[(betas.scope == PRIMARY_SCOPE) & ~betas.in_range]
    if len(bad):
        finish("killed_rule_3", beta_out_of_range=bad.to_dict("records"))
        raise KillRule(3, f"{len(bad)} metro year(s) with beta_t outside {beta_range}; "
                          "check aggregation and units; H3/H4 not entered")

    dec = pd.concat([p["dec"].assign(scope=k[0], outcome=k[1], period=k[2], panel=k[3], link=k[4])
                     for k, p in point.items()], ignore_index=True)
    dec["registered"] = (dec.scope == PRIMARY_SCOPE) & dec.link.isin(["overlap", "none"])
    dec.to_csv(out / "k2_decomposition.csv", index=False)
    h1_max = float(dec["identity_residual"].abs().max())
    h1_by = dec.groupby(["scope", "outcome", "period", "panel", "link"])["identity_residual"] \
        .apply(lambda s: float(s.abs().max())).reset_index(name="max_abs_residual")
    h1_by["pass"] = h1_by["max_abs_residual"] < H1_TOL

    est = {k: estimate_spec(point[k], k[3]) for k in keys}
    h2_rows = []
    for k in keys:
        if k[3] != "A":
            continue
        d = point[k]["dec"]
        x0, dx0, w0 = point[k]["pairs"]["V0"]
        x1, dx1, w1 = point[k]["pairs"]["V1"]
        pair_diff = (float(max(np.max(np.abs(x0 - x1)), np.max(np.abs(dx0 - dx1)),
                               np.max(np.abs(w0 - w1)))) if len(x0) == len(x1) and len(x0)
                     else np.inf)
        obs_diff = max(_obs_gap(est[k]["V0"][c][q], est[k]["V1"][c][q])
                       for c in CONVENTIONS for q in OBS_KEYS)
        ok = bool((d["n_E"] == 0).all() and (d["n_L"] == 0).all() and (d["M_t"] == 0).all()
                  and pair_diff < H2_TOL and obs_diff < H2_TOL)
        h2_rows.append(dict(scope=k[0], outcome=k[1], period=k[2], max_n_E=float(d["n_E"].max()),
                            max_n_L=float(d["n_L"].max()), max_abs_M=float(d["M_t"].abs().max()),
                            max_pair_diff_V0_V1=pair_diff, max_obs_diff_V0_V1=float(obs_diff),
                            n_transitions=len(d), **{"pass": ok}))
    h2 = pd.DataFrame(h2_rows)
    h1_by.to_csv(out / "k2_h1.csv", index=False)
    h2.to_csv(out / "k2_h2.csv", index=False)
    if not h1_by["pass"].all() or not h2["pass"].all():
        finish("killed_rule_1", h1_max=h1_max, h2_fail=h2[~h2["pass"]].to_dict("records"))
        raise KillRule(1, f"H1 max residual {h1_max:.3g} / H2 failures "
                          f"{int((~h2['pass']).sum())}: code error, fix and re-register")

    h3a = dec[(dec.panel == "B") & ((dec.n_E + dec.n_L) > 0)].copy()
    h3a["M_nonzero"] = h3a["M_t"].abs() > H3A_TOL
    h3a["M_over_bound"] = h3a["M_t"].abs() / h3a["upper_bound"]
    h3a["M_over_count_bound"] = h3a["M_t"] / h3a["bound_per_unit_m"]
    h3a.to_csv(out / "k2_h3a.csv", index=False)

    preds = []
    for k in keys:
        if k[3] != "B":
            continue
        e1 = est[k]["V1"]
        r1 = e1[PRIMARY_CONV]["rho1"]
        guard = k1_guard(point[k]["pairs"]["V1"], r1)
        p = k1_prediction(e1["params"], point[k]["dec"], guard)
        pu = k1_prediction(e1["params"], point[k]["dec"], None)
        preds.append(dict(scope=k[0], outcome=k[1], period=k[2], link=k[4], **p,
                          guard=None if guard is None else list(guard),
                          unguarded_status=pu["status"],
                          unguarded_pred_xi_trap=pu["pred_xi_trap"],
                          unguarded_pred_rho1=pu["pred_rho1"],
                          zeroth_order_xi_trap=(zeroth_order(point[k]["dec"], r1)
                                                if np.isfinite(r1) else np.nan),
                          M_bar=float((point[k]["dec"]["n_S"] * point[k]["dec"]["M_t"]).sum()
                                      / point[k]["dec"]["n_S"].sum()),
                          a1_V1=r1 - 1.0))
    preds = pd.DataFrame(preds)

    warm = {k: {v: (e["params"] if e["params"] is not None else None) for v, e in est[k].items()}
            for k in keys}
    draws_all, plan_meta, literal_rows = [], {}, []
    for si, scope in enumerate(SCOPES):
        unit_lin, area_lin = lin[scope]
        n_lin = max(unit_lin.values()) + 1
        states = lineage_states(world, unit_lin, scope=scope)
        sk_s = {k: v for k, v in sks.items() if k[0] == scope}
        for hi_, scheme in enumerate(SCHEMES):
            block = np.arange(n_lin) if scheme == "lineage" else states
            W = draw_plan(seed, n_boot, si, hi_, n_lin, block)
            plan_meta[f"{scope}/{scheme}"] = dict(spawn_key=[si, hi_], n_blocks=int(len(np.unique(block))),
                                                  n_lineages=int(n_lin),
                                                  plan_sha256=hashlib.sha256(W.tobytes()).hexdigest())
            dr = bootstrap(sk_s, warm, W, n_jobs).assign(scheme=scheme)
            draws_all.append(dr)
            for b in range(min(literal_checks, n_boot)):
                lit = literal_county_draw(world, scope, unit_lin, W[b], list(sk_s), drops, warm)
                for key, (lres, e) in lit.items():
                    fres = run_spec(sk_s[key], W[b])
                    for v in VARIANTS[key[3]]:
                        literal_rows.append(dict(scope=scope, scheme=scheme, draw=b, outcome=key[1],
                                                 period=key[2], panel=key[3], link=key[4], variant=v,
                                                 conv="pairs", quantity="pairs", fast=np.nan,
                                                 literal=np.nan, abs_diff=pair_multiset_diff(
                                                     fres["pairs"][v], lres["pairs"][v]),
                                                 tol=LITERAL_PAIR_TOL))
                    fast = dr[(dr.draw == b) & (dr.outcome == key[1]) & (dr.period == key[2])
                              & (dr.panel == key[3]) & (dr.link == key[4])]
                    for v, ev in e.items():
                        for c in CONVENTIONS:
                            f = fast[(fast.variant == v) & (fast.conv == c)].iloc[0]
                            for q in OBS_KEYS:
                                a, bq = float(f[q]), float(ev[c][q])
                                literal_rows.append(dict(scope=scope, scheme=scheme, draw=b,
                                                         outcome=key[1], period=key[2], panel=key[3],
                                                         link=key[4], variant=v, conv=c, quantity=q,
                                                         fast=a, literal=bq, abs_diff=abs(a - bq)
                                                         if np.isfinite(a) and np.isfinite(bq) else
                                                         (0.0 if (np.isnan(a) and np.isnan(bq)) or a == bq
                                                          else np.inf),
                                                         tol=LITERAL_OBS_TOL * max(1.0, abs(a))
                                                         if np.isfinite(a) else LITERAL_OBS_TOL))
    draws = pd.concat(draws_all, ignore_index=True)
    ident = draws[draws.scope == "__identity__"]
    draws = draws[draws.scope != "__identity__"].drop(columns=["identity_max"], errors="ignore")
    draws.to_csv(out / "k2_boot_draws.csv.gz", index=False)
    lit = pd.DataFrame(literal_rows, columns=["scope", "scheme", "draw", "outcome", "period", "panel",
                                              "link", "variant", "conv", "quantity", "fast",
                                              "literal", "abs_diff", "tol"])
    lit["over_tol"] = lit["abs_diff"].astype(float) > lit["tol"].astype(float)
    lit.to_csv(out / "k2_literal_draw_check.csv", index=False)
    is_pair = lit["quantity"] == "pairs"
    fin = np.isfinite(lit["abs_diff"].astype(float))
    lit_max = float(lit.loc[is_pair, "abs_diff"].max()) if is_pair.any() else 0.0
    lit_bad = int(lit.loc[is_pair, "over_tol"].sum())
    lit_obs_flag = int(lit.loc[~is_pair, "over_tol"].sum())
    lit_obs_max = float(lit.loc[~is_pair & fin, "abs_diff"].max()) if (~is_pair & fin).any() else 0.0
    if lit_bad:
        finish("killed_literal_check", literal_max_abs_diff=lit_max, literal_n_bad=lit_bad)
        raise KillRule(1, f"fast bootstrap path pairs != literal county re-aggregation "
                          f"({lit_bad} variant(s) over {LITERAL_PAIR_TOL})")
    boot_ident_max = float(ident["identity_max"].max()) if len(ident) else 0.0

    obs_rows = []
    for k in keys:
        for v, e in est[k].items():
            for c in CONVENTIONS:
                base = dict(scope=k[0], outcome=k[1], period=k[2], panel=k[3], link=k[4],
                            variant=v, conv=c, registered=(k[0] == PRIMARY_SCOPE and k[4] in ("overlap", "none")
                                                           and v in ("V0", "V1", "V1d")))
                sub = draws[(draws.scope == k[0]) & (draws.outcome == k[1]) & (draws.period == k[2])
                            & (draws.panel == k[3]) & (draws.link == k[4]) & (draws.variant == v)
                            & (draws.conv == c)]
                for q in OBS_KEYS:
                    r = dict(base, quantity=q, estimate=e[c][q], id_status=e[c]["id_status"],
                             n_pairs=e[c]["n_pairs"])
                    for scheme in SCHEMES:
                        vals = sub[sub.scheme == scheme][q].to_numpy(float)
                        r[f"{scheme}_n_draws"] = len(vals)
                        r[f"{scheme}_n_nan"] = int((~np.isfinite(vals)).sum())
                        for rule in CI_RULES:
                            lo, hi, ok = ci(vals, rule)
                            r[f"{scheme}_{rule}_lo"], r[f"{scheme}_{rule}_hi"] = lo, hi
                            r[f"{scheme}_{rule}_usable"] = ok
                    obs_rows.append(r)
    obs = pd.DataFrame(obs_rows)
    obs.to_csv(out / "k2_observables.csv", index=False)

    def paired(k_b, var_b, k_a, var_a, q, conv, scheme):
        da = draws[(draws.scope == k_a[0]) & (draws.outcome == k_a[1]) & (draws.period == k_a[2])
                   & (draws.panel == k_a[3]) & (draws.link == k_a[4]) & (draws.variant == var_a)
                   & (draws.conv == conv) & (draws.scheme == scheme)].set_index("draw")[q]
        db = draws[(draws.scope == k_b[0]) & (draws.outcome == k_b[1]) & (draws.period == k_b[2])
                   & (draws.panel == k_b[3]) & (draws.link == k_b[4]) & (draws.variant == var_b)
                   & (draws.conv == conv) & (draws.scheme == scheme)].set_index("draw")[q]
        return (db - da.reindex(db.index)).to_numpy(float)

    diff_rows = []
    for k in keys:
        if k[3] != "B":
            continue
        ka = (k[0], k[1], k[2], "A", "none")
        comps = [("V0-V1", k, "V0", k, "V1"), ("V1(B)-A", k, "V1", ka, "V0"),
                 ("V0(B)-A", k, "V0", ka, "V0"), ("V1d(B)-A", k, "V1d", ka, "V0"),
                 ("V1d_stay(B)-A", k, "V1d_stay", ka, "V0"), ("V1-V1d", k, "V1", k, "V1d")]
        for name, kb, vb, kx, vx in comps:
            for c in CONVENTIONS:
                for q in ("xi_trap", "rho1"):
                    pt = est[kb][vb][c][q] - est[kx][vx][c][q]
                    r = dict(scope=k[0], outcome=k[1], period=k[2], link=k[4], contrast=name,
                             conv=c, quantity=q, estimate=pt)
                    for scheme in SCHEMES:
                        vals = paired(kb, vb, kx, vx, q, c, scheme)
                        r[f"{scheme}_n_nan"] = int((~np.isfinite(vals)).sum())
                        for rule in CI_RULES:
                            lo, hi, ok = ci(vals, rule)
                            r[f"{scheme}_{rule}_lo"], r[f"{scheme}_{rule}_hi"] = lo, hi
                            r[f"{scheme}_{rule}_usable"] = ok
                    diff_rows.append(r)
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(out / "k2_paired_differences.csv", index=False)

    verdicts = verdict_table(keys, est, preds, obs, diffs, h1_by, h2, h3a)
    verdicts.to_csv(out / "k2_verdicts.csv", index=False)
    preds_out = preds.merge(verdicts[verdicts.hypothesis == "H3b"][
        ["scope", "outcome", "period", "link", "observed", "half_width_V1", "gate", "verdict"]],
        on=["scope", "outcome", "period", "link"], how="left")
    preds_out.to_csv(out / "k2_h3b.csv", index=False)
    finish("complete", h1_max_point=h1_max, h1_max_bootstrap=boot_ident_max,
           literal_check_pairs_max_abs_diff=lit_max,
           literal_check_obs_max_abs_diff=lit_obs_max, literal_check_obs_n_over_tol=lit_obs_flag,
           literal_check_draws=literal_checks,
           draw_plans=plan_meta, lineages=lsum.to_dict("records"),
           dropped_transitions={f"{k[1]}/{k[2]}/{k[0]}/{k[3]}/{k[4]}": sk.dropped
                                for k, sk in sks.items() if sk.dropped})
    return dict(verdicts=verdicts, obs=obs, diffs=diffs, preds=preds_out, dec=dec, meta=meta)


def _diff_ci(diffs, k, contrast, q, scheme=PRIMARY_SCHEME, rule=PRIMARY_CI, conv=PRIMARY_CONV):
    r = diffs[(diffs.scope == k[0]) & (diffs.outcome == k[1]) & (diffs.period == k[2])
              & (diffs.link == k[4]) & (diffs.contrast == contrast) & (diffs.quantity == q)
              & (diffs.conv == conv)].iloc[0]
    return (r["estimate"], r[f"{scheme}_{rule}_lo"], r[f"{scheme}_{rule}_hi"],
            bool(r[f"{scheme}_{rule}_usable"]))


def h3b_verdict(pred: float, observed: float, half_width: float, usable: bool) -> tuple[str, float]:
    if not np.isfinite(pred):
        return "not_evaluable_prediction_undefined", np.nan
    if not usable or not np.isfinite(half_width):
        return "not_evaluable_ci_unusable", np.nan
    gate = H3B_GATE * half_width
    if abs(pred) < gate:
        return "not_testable", gate
    if not np.isfinite(observed):
        return "fail_nonfinite_observed", gate
    ok = np.sign(observed) == np.sign(pred) and abs(observed - pred) <= H3B_REL * abs(pred)
    return ("pass" if ok else "fail"), gate


def h4_verdict(ci_xi: tuple, ci_rho: tuple) -> str:
    """Both 95% CIs (lo, hi, usable) contain 0 -> consistent."""
    if not (ci_xi[2] and ci_rho[2]):
        return "not_evaluable_ci_unusable"
    inside = ci_xi[0] <= 0 <= ci_xi[1] and ci_rho[0] <= 0 <= ci_rho[1]
    return "consistent" if inside else "inconsistent"


def verdict_table(keys, est, preds, obs, diffs, h1_by, h2, h3a) -> pd.DataFrame:
    rows = []
    for r in h1_by.to_dict("records"):
        rows.append(dict(hypothesis="H1", scope=r["scope"], outcome=r["outcome"],
                         period=r["period"], link=r["link"], panel=r["panel"],
                         value=r["max_abs_residual"], verdict="pass" if r["pass"] else "fail",
                         registered=r["scope"] == PRIMARY_SCOPE
                         and r["link"] in ("overlap", "none")))
    for r in h2.to_dict("records"):
        rows.append(dict(hypothesis="H2", scope=r["scope"], outcome=r["outcome"], period=r["period"],
                         link="none", panel="A", value=r["max_pair_diff_V0_V1"],
                         verdict="pass" if r["pass"] else "fail",
                         registered=r["scope"] == PRIMARY_SCOPE))
    for (sc, oc, per, link), g in h3a.groupby(["scope", "outcome", "period", "link"]):
        rows.append(dict(hypothesis="H3a", scope=sc, outcome=oc, period=per, link=link, panel="B",
                         value=float(g["M_t"].abs().min()), n_transitions=len(g),
                         verdict=("all_nonzero" if g["M_nonzero"].all()
                                  else f"{int((~g['M_nonzero']).sum())}_near_zero"),
                         registered=False))
    for k in keys:
        if k[3] != "B":
            continue
        reg = k[0] == PRIMARY_SCOPE and k[4] == PRIMARY_LINK
        p = preds[(preds.scope == k[0]) & (preds.outcome == k[1]) & (preds.period == k[2])
                  & (preds.link == k[4])].iloc[0]
        o = obs[(obs.scope == k[0]) & (obs.outcome == k[1]) & (obs.period == k[2]) & (obs.panel == "B")
                & (obs.link == k[4]) & (obs.variant == "V1") & (obs.conv == PRIMARY_CONV)
                & (obs.quantity == "xi_trap")].iloc[0]
        lo, hi = o[f"{PRIMARY_SCHEME}_{PRIMARY_CI}_lo"], o[f"{PRIMARY_SCHEME}_{PRIMARY_CI}_hi"]
        hw = (hi - lo) / 2.0
        observed = est[k]["V0"][PRIMARY_CONV]["xi_trap"] - est[k]["V1"][PRIMARY_CONV]["xi_trap"]
        v, gate = h3b_verdict(p["pred_xi_trap"], observed,
                              hw, bool(o[f"{PRIMARY_SCHEME}_{PRIMARY_CI}_usable"]))
        rows.append(dict(hypothesis="H3b", scope=k[0], outcome=k[1], period=k[2], link=k[4], panel="B",
                         value=p["pred_xi_trap"], observed=observed, half_width_V1=hw, gate=gate,
                         zeroth_order=p["zeroth_order_xi_trap"], prediction_status=p["status"],
                         verdict=v, registered=reg))
        est_r, lo_r, hi_r, ok_r = _diff_ci(diffs, k, "V0-V1", "rho1")
        rows.append(dict(hypothesis="H3c", scope=k[0], outcome=k[1], period=k[2], link=k[4], panel="B",
                         value=est_r, ci_lo=lo_r, ci_hi=hi_r, ci_usable=ok_r,
                         predicted=p["pred_rho1"], verdict="report_only", registered=False))
        for scheme in SCHEMES:
            cx = _diff_ci(diffs, k, "V1(B)-A", "xi_trap", scheme)
            cr = _diff_ci(diffs, k, "V1(B)-A", "rho1", scheme)
            rows.append(dict(hypothesis="H4" if scheme == PRIMARY_SCHEME else "H4_state",
                             scope=k[0], outcome=k[1], period=k[2], link=k[4], panel="B-A",
                             value=cx[0], ci_lo=cx[1], ci_hi=cx[2], value_rho1=cr[0],
                             ci_lo_rho1=cr[1], ci_hi_rho1=cr[2], scheme=scheme,
                             verdict=h4_verdict(cx[1:], cr[1:]),
                             registered=reg and scheme == PRIMARY_SCHEME))
    v = pd.DataFrame(rows)
    h4 = v[v.hypothesis == "H4"].set_index(["scope", "outcome", "period", "link"])["verdict"]
    h4s = v[v.hypothesis == "H4_state"].set_index(["scope", "outcome", "period", "link"])["verdict"]
    v["state_changes_H4"] = pd.Series([None] * len(v), dtype=object)
    for idx in h4.index:
        m = (v.hypothesis == "H4") & (v.scope == idx[0]) & (v.outcome == idx[1]) & \
            (v.period == idx[2]) & (v.link == idx[3])
        v.loc[m, "state_changes_H4"] = bool(h4s.get(idx) != h4[idx])
    return v


def make_toy_world(seed: int = 0, n_core: int = 40, a1: float = -0.15, beta: float = 1.08,
                   entrant_shift: float = -1.0, leaver_shift: float = -0.8,
                   years=range(1993, 2025)) -> World:
    rng = np.random.default_rng(seed)
    years = list(years)
    T = len(years)
    states = ["01", "06", "17", "36", "48"]
    units, k = [], 0

    def new_unit():
        nonlocal k
        k += 1
        st = states[k % len(states)]
        return f"{st}{k:03d}"

    core = []
    for i in range(n_core):
        core.append([new_unit() for _ in range(1 + (i % 3))])
    entr = [[new_unit()] for _ in range(5)]
    early_entr = [[new_unit()] for _ in range(4)]
    added = [new_unit() for _ in range(4)]
    micro = [[new_unit()] for _ in range(6)]
    late = new_unit()
    units = sorted({u for a in core + entr + early_entr + micro for u in a} | set(added) | {late})
    lnN0 = {u: rng.normal(12.5, 1.0) for u in units}
    for a in entr + early_entr:
        lnN0[a[0]] = rng.normal(11.3, 0.3)
    for a in micro:
        lnN0[a[0]] = rng.normal(10.6, 0.3)
    for u in added:
        lnN0[u] = rng.normal(10.0, 0.2)
    g = {u: rng.normal(0.01, 0.005) for u in units}
    shift = {u: 0.0 for u in units}
    for a in entr:
        shift[a[0]] = entrant_shift
    x = np.zeros((len(units), T))
    ui = {u: i for i, u in enumerate(units)}
    x[:, 0] = rng.normal(0, 0.4, len(units))
    for j in range(1, T):
        x[:, j] = x[:, j - 1] + a1 * x[:, j - 1] + rng.normal(0, 0.15, len(units))
    leavers_core = list(range(3))
    for i in leavers_core:
        for u in core[i]:
            x[ui[u], :] += leaver_shift
    for a in entr:
        x[ui[a[0]], :] += shift[a[0]]
    lnN = np.array([[lnN0[u] + g[u] * (y - years[0]) for y in years] for u in units])
    price = np.array([0.03 * (y - years[0]) for y in years])
    lnY = 5.0 + beta * lnN + x + price[None, :]
    pop = pd.DataFrame(np.exp(lnN), index=units, columns=years)
    pi = pd.DataFrame(np.exp(lnY), index=units, columns=years)
    gdp = pd.DataFrame(np.exp(lnY + rng.normal(0, 0.05, lnY.shape) + 0.2), index=units, columns=years)
    pre02 = [y for y in years if y < 2002]
    pop.loc[late, pre02] = np.nan
    pi.loc[late, pre02] = np.nan
    gdp = gdp.loc[:, [y for y in years if y >= 2001]]

    def amap(pairs):
        return {c: frozenset(s) for c, s in pairs}

    pre = [(f"{1000 + i}", core[i]) for i in range(n_core)]
    pre[5] = (pre[5][0], core[5] + [late])
    v93 = amap(pre)
    i99 = n_core - 2
    assert len(core[i99]) == 3
    keep99 = sorted(core[i99], key=lambda u: lnN0[u])[1:]
    v99 = amap(pre[:i99] + [(pre[i99][0], keep99)] + pre[i99 + 1:])
    cb = [(f"{10000 + 20 * i}", core[i]) for i in range(n_core)]
    cb[5] = (cb[5][0], core[5] + [late])
    cb += [(f"{30000 + 20 * i}", a) for i, a in enumerate(early_entr)]
    v03 = amap(cb)
    v08 = dict(v03)
    v08.pop(cb[3][0])
    cb13 = [c for c in cb if c[0] != cb[3][0]]
    cb13 = [c for c in cb13 if c[0] not in {cb[0][0], cb[1][0], cb[2][0]}]
    cb13 = [(c, list(s) + [added[i]]) if i < 4 else (c, s) for i, (c, s) in enumerate(cb13)]
    c_old, s_old = cb13[10]
    cb13[10] = ("49990", s_old)
    i_sp = next(i for i, (c, s) in enumerate(cb13) if len(s) == 3 and i > 10)
    c_sp, s_sp = cb13[i_sp]
    big = max(s_sp, key=lambda u: lnN0[u])
    rest = [u for u in s_sp if u != big]
    cb13[i_sp] = (c_sp, [big] + rest[:1])
    cb13.append(("48880", rest[1:]))
    i_m = [i for i, (c, s) in enumerate(cb13) if len(s) == 1 and i > i_sp][:2]
    (ca, sa), (cbb, sbb) = cb13[i_m[0]], cb13[i_m[1]]
    keep = ca if lnN0[sa[0]] >= lnN0[sbb[0]] else cbb
    cb13 = [c for j, c in enumerate(cb13) if j not in i_m] + [(keep, list(sa) + list(sbb))]
    cb13 += [(f"{40000 + 20 * i}", a) for i, a in enumerate(entr)]
    v13 = amap(cb13)
    v23 = dict(v13)
    micro_a = {f"{20000 + 20 * i}": frozenset(a) for i, a in enumerate(micro)}
    micro13 = dict(micro_a)
    promoted = sorted(micro13)[0]
    prom_units = micro13.pop(promoted)

    def all_scope(metro: dict, micro_: dict) -> dict:
        return {**metro, **micro_}

    v13_m = dict(v13)
    v13_m[promoted] = prom_units
    vintages = [("1993-06", "1993-06-30", "pre2003"), ("1999-06", "1999-06-30", "pre2003"),
                ("2003-06", "2003-06-06", "cbsa"), ("2008-11", "2008-11-20", "cbsa"),
                ("2013-02", "2013-02-28", "cbsa"), ("2023-07", "2023-07-21", "cbsa")]
    metro = {"1993-06": v93, "1999-06": v99, "2003-06": v03, "2008-11": v08, "2013-02": v13_m,
             "2023-07": dict(v13_m)}
    allc = {"1993-06": v93, "1999-06": v99, "2003-06": all_scope(v03, micro_a),
            "2008-11": all_scope(v08, micro_a), "2013-02": all_scope(v13_m, micro13),
            "2023-07": all_scope(dict(v13_m), micro13)}
    areas = {}
    for lab in metro:
        areas[("metro", lab)] = metro[lab]
        areas[("cbsa_all", lab)] = allc[lab]
    del v23
    return World(years=years, pop=pop, outcomes={"pi": pi, "gdp": gdp}, vintages=vintages,
                 areas=areas, panel_a="2023-07")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_registration() -> dict:
    """Rule 3: the real run needs the committed registration (seed, B, code
    commit, data sha256), a clean tree, and the registered files unchanged
    since the code commit."""
    if not REGISTRATION.exists():
        raise SystemExit(f"no registration at {REGISTRATION}: register seeds and code commit first")
    rel = str(REGISTRATION.relative_to(ROOT))
    if git("ls-files", rel) != rel or git("status", "--porcelain", "--", rel):
        raise SystemExit("the registration file is not committed as is")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files differ from HEAD: commit or stash before the real run")
    reg = json.loads(REGISTRATION.read_text())
    for kname, want in (("seed", SEED), ("n_boot", N_BOOT), ("n_min", N_MIN), ("sd_ddof", SD_DDOF)):
        if reg.get(kname) != want:
            raise SystemExit(f"registered {kname} {reg.get(kname)!r} != script {want!r}")
    code = reg["code_commit"]
    subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", code, "HEAD"])
    for f in REGISTERED_FILES:
        if git("rev-parse", f"{code}:{f}") != git("rev-parse", f"HEAD:{f}"):
            raise SystemExit(f"{f} changed since the registered code commit {code}")
    return reg


def check_single_run(allow_rerun: str | None) -> None:
    """Threat model: no selective re-runs.  A second real run needs a stated
    reason, which is logged with the digest of the previous meta."""
    prev = OUT_REAL / "k2_meta.json"
    if prev.exists():
        if not allow_rerun:
            raise SystemExit(f"{prev} exists: the registered single real run already happened; "
                             "a re-run needs --allow-rerun REASON (logged)")
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps(dict(event="rerun_requested", reason=allow_rerun,
                                    previous_meta_sha256=sha256(prev),
                                    time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                    head=git("rev-parse", "HEAD"))) + "\n")


def check_integrity(data: Path, reg: dict) -> dict:
    """Kill rule 2: BEA / OMB files as registered, and scripts/k2_coverage.py
    reproduces the committed coverage tables byte for byte."""
    src = pd.read_csv(ROOT / "results" / "tables" / "k2_coverage_sources.csv")
    bad = [r.file for r in src.itertuples() if sha256(data / r.file) != r.sha256]
    for f, want in reg.get("data_sha256", {}).items():
        if sha256(data / f) != want:
            bad.append(f)
    if OUTCOMES["gdp"]["file"] not in reg.get("data_sha256", {}):
        raise KillRule(2, "the registration lacks the CAGDP1 sha256")
    if bad:
        raise KillRule(2, f"sha256 differs from the registration for {bad}: a new BEA/OMB release? "
                          "author decides old or new files")
    with tempfile.TemporaryDirectory() as td:
        old_out, old_argv = COV.OUT, sys.argv
        try:
            COV.OUT = Path(td)
            sys.argv = ["k2_coverage.py", "--data", str(data)]
            COV.main()
        finally:
            COV.OUT, sys.argv = old_out, old_argv
        diff = []
        for p in sorted((ROOT / "results" / "tables").glob("k2_coverage_*.csv")):
            if (Path(td) / p.name).read_bytes() != p.read_bytes():
                diff.append(p.name)
    if diff:
        raise KillRule(2, f"k2_coverage.py does not reproduce {diff}")
    return dict(sources_checked=len(src) + len(reg.get("data_sha256", {})), coverage_reproduced=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=Path, metavar="OUT_DIR")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--real", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--allow-rerun", default=None, metavar="REASON")
    a = ap.parse_args(argv)
    if a.synthetic:
        w = make_toy_world()
        res = run_pipeline(w, a.synthetic, n_boot=a.n_boot or 20, n_jobs=a.jobs, n_min=10,
                           meta=dict(mode="synthetic"))
        print(res["verdicts"][["hypothesis", "scope", "outcome", "period", "link", "verdict"]]
              .to_string(index=False))
        return 0
    data = a.data or COV.default_data_dir()
    if a.preflight:
        w = load_world(data, with_outcomes=False)
        out = ROOT / "results" / "tables"
        tabs = pd.concat([lineage_table(w, s) for s in SCOPES], ignore_index=True)
        tabs.to_csv(out / "k2_lineages.csv", index=False)
        summ = pd.DataFrame([lineage_summary(tabs[tabs.scope == s], w) for s in SCOPES])
        summ.to_csv(out / "k2_lineage_summary.csv", index=False)
        print(summ.to_string(index=False))
        return 0
    if a.n_boot not in (None, N_BOOT):
        raise SystemExit("the real run uses the registered B")
    reg = public_guard_skipped({'seed': SEED, 'n_boot': N_BOOT, 'n_min': N_MIN, 'sd_ddof': SD_DDOF, 'lineage_split_decision': None, 'data_sha256': json.loads((ROOT / 'results' / 'tables' / 'k2' / 'k2_meta.json').read_text())['data_sha256']})
    check_single_run(a.allow_rerun)
    from src import k1_theory as T
    frozen = public_guard_skipped({'frozen_commit': None, 'theory_version': T.THEORY_VERSION, 'environment_at_run': T.environment(), 'environment_differs': None})
    head = git("rev-parse", "HEAD")
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(dict(event="attempt", head=head,
                                time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    try:
        integ = check_integrity(data, reg)
    except KillRule as e:
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps(dict(event="end", head=head, status=str(e), outcomes_read=False,
                                    time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
        print(e)
        return 3
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(dict(event="start", head=head, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                registration=reg)) + "\n")
    w = load_world(data, with_outcomes=True)
    lsum = pd.DataFrame([lineage_summary(lineage_table(w, s), w) for s in SCOPES])
    if lsum["exceeds_10pct"].any() and not reg.get("lineage_split_decision"):
        raise SystemExit("a lineage holds > 10% of the areas: a decision of the authors is needed")
    meta = dict(mode="real", git_hash=head, registration=reg, integrity=integ,
                k1_freeze=dict(frozen_commit=frozen["frozen_commit"],
                               environment_at_run=frozen["environment_at_run"],
                               environment_differs=frozen["environment_differs"]),
                data_sha256={**{r.file: r.sha256 for r in pd.read_csv(
                    ROOT / "results" / "tables" / "k2_coverage_sources.csv").itertuples()},
                    **reg.get("data_sha256", {})})
    try:
        run_pipeline(w, OUT_REAL, n_boot=N_BOOT, seed=SEED, n_jobs=a.jobs, meta=meta)
        status = "complete"
    except KillRule as e:
        status = str(e)
        print(status)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(dict(event="end", head=head, status=status,
                                time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
