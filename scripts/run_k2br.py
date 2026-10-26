"""Brazilian municipal analysis: V0 and V1 attractor contrasts on IBGE municipal
GDP (PIB dos Municipios) over minimum comparable areas, with bootstrap
intervals and the K1 composition predictions.

    python3 scripts/run_k2br.py --synthetic --out DIR   (toy panel)
    python3 scripts/run_k2br.py --real [--workers N]

--n-boot and --out apply to --synthetic only: --real always draws the
registered 1,000 bootstrap samples and writes to results/tables/.

Inputs: data/brazil/ (make brazil). Outputs: results/tables/k2br_*.
"""
from __future__ import annotations

def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.km_mle import derived_observables, fit_mle

WINDOWS = {
    "ref2010": (2002, 2023, "ref2010"),
    "ref2002": (1999, 2012, "ref2002"),
}
PRIMARY_WINDOW = "ref2010"
OUTCOMES = ("pib", "vab_agro")
PRIMARY_OUTCOME = "pib"
SEED = 20261024
N_BOOT = 1000
SCHEMES = ("amc", "immediate", "uf")
PRIMARY_SCHEME = "amc"
N_MIN = 5000
BETA_RANGE = (0.9, 1.3)
DDOF = 1
H1_TOL = 1e-10
H2_TOL = 1e-12
H3A_TOL = 1e-8
GATE_FACTOR = 0.5
H3B_REL_TOL = 0.30
NONFINITE_MAX_FRAC = 0.05
CONVENTIONS = ("limit", "nan")
PANELS = ("A", "A0", "B")
VARIANTS = ("V0", "V1", "V1p", "V1m")
QUANTS = ("xi_trap", "rho1", "mean_increment", "mean_diffusion", "two_delta_V",
          "tau_mfpt", "gamma", "B_ratio_w")
MIN_REG_N = 10

REG_FILE = ROOT / 'results' / 'records' / "k2br_registration.json"
ATTEMPT_LOG = ROOT / 'results' / 'records' / "k2br_run_attempts.jsonl"
EXISTENCE_TABLE = ROOT / "results" / "tables" / "k2br_existence.csv"
DATA = Path('data/brazil')
ALLOWED_AFTER_CODE = {"results/records/k2br_registration.json",
                      "results/records/k2br_run_notes.md"}


class KillRule(RuntimeError):

    def __init__(self, msg: str, partial: dict | None = None):
        super().__init__(msg)
        self.partial = partial or {}


@dataclass(frozen=True)
class Config:
    windows: dict = field(default_factory=lambda: dict(WINDOWS))
    outcomes: tuple = OUTCOMES
    n_min: int = N_MIN
    beta_range: tuple = BETA_RANGE
    beta_outcomes: tuple = ("pib",)
    ddof: int = DDOF
    min_reg_n: int = MIN_REG_N


@dataclass
class Muni:
    """Municipal panel of one window.  Rows are municipalities (or their
    copies in a bootstrap draw), columns the window's years.
    """
    years: np.ndarray
    Y: dict
    N: np.ndarray
    edges: np.ndarray
    amc: np.ndarray
    code: np.ndarray
    pop_base: np.ndarray


@dataclass
class Panel:
    name: str
    years: np.ndarray
    Y: np.ndarray
    N: np.ndarray
    present: np.ndarray
    edges: np.ndarray
    n_d1_dropped: int = 0
    d1_dropped_by_year: np.ndarray | None = None
    lag_amc_years: np.ndarray | None = None


def _components(n: int, edges: np.ndarray) -> np.ndarray:
    """Union-find labels (smallest row index of each component)."""
    parent = np.arange(n)

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r
    for a, b in np.asarray(edges, int).reshape(-1, 2):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def _record(m: Muni, o: str) -> np.ndarray:
    return np.isfinite(m.Y[o]) & np.isfinite(m.N)


def panel_B(m: Muni, o: str) -> Panel:
    """Open panel as published: a municipality is in year t's set iff it has
    both an outcome and a population record (D3 a).  D1: a unit with Y <= 0
    in any year of the window is removed in every year."""
    rec = _record(m, o)
    bad = (rec & (np.nan_to_num(m.Y[o], nan=1.0) <= 0)).any(axis=1)
    present = rec & ~bad[:, None]
    return Panel("B", m.years, m.Y[o], m.N, present, m.edges,
                 n_d1_dropped=int(bad.sum()),
                 d1_dropped_by_year=(rec & bad[:, None]).sum(axis=0))


def panel_A(m: Muni, o: str) -> Panel:
    """Fixed geography: AMC sums.  Y is summed over the members with an
    outcome record and N over the members with a population record, each
    separately: a territory is reported once in either list (inside its
    parent before it is listed on its own), so each sum covers the whole AMC
    even in a D3 lag year in which an entrant already has a population
    record but no outcome record yet (summing over 'both records' would drop
    that entrant's population while its output is still inside the parent).
    An AMC is present in a year iff it has at least one record of each.
    D1 on the AMC sums.  ``lag_amc_years`` counts AMC-years whose Y-member
    set differs from their N-member set (reported)."""
    ry = np.isfinite(m.Y[o])
    rn = np.isfinite(m.N)
    ids, inv = np.unique(m.amc, return_inverse=True)
    n_a, T = len(ids), len(m.years)
    Y = np.zeros((n_a, T))
    N = np.zeros((n_a, T))
    cy = np.zeros((n_a, T))
    cn = np.zeros((n_a, T))
    lag = np.zeros(T, int)
    for j in range(T):
        a, b = ry[:, j], rn[:, j]
        Y[:, j] = np.bincount(inv[a], weights=m.Y[o][a, j], minlength=n_a)
        N[:, j] = np.bincount(inv[b], weights=m.N[b, j], minlength=n_a)
        cy[:, j] = np.bincount(inv[a], minlength=n_a)
        cn[:, j] = np.bincount(inv[b], minlength=n_a)
        lag[j] = int((np.bincount(inv[a ^ b], minlength=n_a) > 0).sum())
    rec_a = (cy > 0) & (cn > 0)
    bad = (rec_a & (Y <= 0)).any(axis=1)
    present = rec_a & ~bad[:, None]
    Y = np.where(rec_a, Y, np.nan)
    N = np.where(rec_a, N, np.nan)
    return Panel("A", m.years, Y, N, present, np.zeros((0, 2), int),
                 n_d1_dropped=int(bad.sum()),
                 d1_dropped_by_year=(rec_a & bad[:, None]).sum(axis=0),
                 lag_amc_years=lag)


def panel_A0(m: Muni, o: str, B: Panel, keep_years: np.ndarray | None = None) -> Panel:
    """Report only: municipalities in B in every year of the window that are
    neither entrants nor parents after the base year.  'Every year' = every
    year kept by kill rule 5 (``keep_years``; None = all window years): a
    year that kill rule 5 removes (e.g. vab_agro 2022-2023, not published by
    IBGE) is not part of the analysis, and requiring presence in it would
    empty A0."""
    touched = np.zeros(len(m.code), bool)
    if len(m.edges):
        touched[np.unique(m.edges.ravel())] = True
    yrs = np.ones(len(m.years), bool) if keep_years is None else np.isin(m.years, keep_years)
    keep = B.present[:, yrs].all(axis=1) & ~touched
    present = np.repeat(keep[:, None], len(m.years), axis=1)
    return Panel("A0", m.years, m.Y[o], m.N, present, np.zeros((0, 2), int))


def apply_year_exclusion(p: Panel, keep_years: np.ndarray) -> Panel:
    """Kill rule 5: a year with fewer than N_min B units is removed (both
    adjacent transitions then vanish: pairs need consecutive years)."""
    k = np.isin(p.years, keep_years)
    return replace(p, years=p.years[k], Y=p.Y[:, k], N=p.N[:, k],
                   present=p.present[:, k],
                   d1_dropped_by_year=(None if p.d1_dropped_by_year is None
                                       else p.d1_dropped_by_year[k]),
                   lag_amc_years=(None if p.lag_amc_years is None
                                  else p.lag_amc_years[k]))


def standardise(logY: np.ndarray, logN: np.ndarray, mask: np.ndarray,
                ddof: int = DDOF, min_n: int = MIN_REG_N):
    """OLS log Y = a + b log N on ``mask``; xi = resid / sd(resid, ddof).
    Returns (xi with NaN off the mask, stats)."""
    xi = np.full(len(logY), np.nan)
    n = int(mask.sum())
    if n < max(min_n, ddof + 3):
        return xi, {"n": n, "alpha": np.nan, "beta": np.nan, "r2": np.nan, "sd": np.nan}
    y, x = logY[mask], logN[mask]
    xm, ym = x.mean(), y.mean()
    xc = x - xm
    beta = float(xc @ (y - ym) / (xc @ xc))
    alpha = float(ym - beta * xm)
    r = y - alpha - beta * x
    sd = float(np.std(r, ddof=ddof))
    xi[mask] = r / sd
    tss = float(((y - ym) ** 2).sum())
    return xi, {"n": n, "alpha": alpha, "beta": beta,
                "r2": 1.0 - float(r @ r) / tss if tss > 0 else np.nan, "sd": sd}


def _logs(p: Panel):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where(p.present, p.Y, np.nan)), np.log(np.where(p.present, p.N, np.nan))


def decompose(xi_t, xi_t1, Ct, Ct1, S) -> dict:
    """Lemma 1 bookkeeping for one standardisation and one pair set S:
    L = Ct minus S, E = Ct1 minus S; M = (n_L m_L - n_E m_E) / n_S."""
    L, E = Ct & ~S, Ct1 & ~S
    n_s, n_l, n_e = int(S.sum()), int(L.sum()), int(E.sum())
    m_l = float(xi_t[L].mean()) if n_l else 0.0
    m_e = float(xi_t1[E].mean()) if n_e else 0.0
    obs = float((xi_t1[S] - xi_t[S]).mean()) if n_s else np.nan
    mech = (n_l * m_l - n_e * m_e) / n_s if n_s else np.nan
    return {"n_S": n_s, "n_L": n_l, "n_E": n_e,
            "m_L": m_l if n_l else np.nan, "m_E": m_e if n_e else np.nan,
            "M": mech, "mean_increment": obs, "identity_residual": obs - mech,
            "bound_nE_over_nS_absmE": (n_e / n_s * abs(m_e)) if (n_s and n_e) else 0.0,
            "M_over_bound": (abs(mech) / (n_e / n_s * abs(m_e))
                             if (n_s and n_e and m_e != 0) else np.nan)}


def _merge_back(Ct, Ct1, edges, logs_lv):
    """V1m geography of one entry transition: entrants merged into their
    parents (union over genealogy edges whose entrant enters at t+1; a parent
    that is itself a same-wave entrant chains).  Returns group labels and the
    merged level sums for years t and t+1."""
    Y, N = logs_lv
    n = len(Ct)
    E = Ct1 & ~Ct
    e = edges[E[edges[:, 0]]] if len(edges) else edges
    grp = _components(n, e) if len(e) else np.arange(n)
    ids, inv = np.unique(grp, return_inverse=True)
    out = {}
    for key, C, j in (("t", Ct, 0), ("t1", Ct1, 1)):
        k = len(ids)
        yv = np.bincount(inv[C], weights=Y[C, j], minlength=k)
        nv = np.bincount(inv[C], weights=N[C, j], minlength=k)
        pres = np.bincount(inv[C], minlength=k) > 0
        out[key] = (yv, nv, pres)
    return ids, out


def transitions(p: Panel):
    return [j for j in range(len(p.years) - 1) if p.years[j + 1] - p.years[j] == 1]


def build_variants(p: Panel, cfg: Config, collect: bool = True) -> dict:
    """Pairs of every variant for every transition of one panel, plus the
    decomposition rows (collect=True) and the K1 inputs of V0/V1."""
    logY, logN = _logs(p)
    T = len(p.years)
    xi0, yearly = np.full(p.Y.shape, np.nan), []
    for j in range(T):
        xi0[:, j], st = standardise(logY[:, j], logN[:, j], p.present[:, j],
                                    cfg.ddof, cfg.min_reg_n)
        yearly.append({"year": int(p.years[j]), **st,
                       "d1_dropped": (int(p.d1_dropped_by_year[j])
                                      if p.d1_dropped_by_year is not None else 0),
                       "lag_amc_years": (int(p.lag_amc_years[j])
                                         if p.lag_amc_years is not None else 0)})
    pairs = {v: [] for v in VARIANTS}
    rows, k1rows = [], []
    edges = p.edges
    for j in transitions(p):
        Ct, Ct1 = p.present[:, j], p.present[:, j + 1]
        S = Ct & Ct1
        E = Ct1 & ~Ct
        if S.sum() < cfg.min_reg_n:
            continue
        lt, lt1, nt, nt1 = logY[:, j], logY[:, j + 1], logN[:, j], logN[:, j + 1]
        if len(edges):
            ee = edges[E[edges[:, 0]]]
            P = np.zeros(len(S), bool)
            P[ee[:, 1][S[ee[:, 1]]]] = True
        else:
            P = np.zeros(len(S), bool)
        n_e = int(E.sum())
        base = {"year_t": int(p.years[j]), "n_P": int(P.sum())}
        a0, a1 = xi0[:, j], xi0[:, j + 1]
        pairs["V0"].append((a0[S], a1[S] - a0[S]))
        d_v0 = decompose(a0, a1, Ct, Ct1, S)
        b0, s0 = standardise(lt, nt, S, cfg.ddof, cfg.min_reg_n)
        b1, s1 = standardise(lt1, nt1, S, cfg.ddof, cfg.min_reg_n)
        pairs["V1"].append((b0[S], b1[S] - b0[S]))
        d_v1 = decompose(b0, b1, S, S, S)
        if n_e > 0:
            Sp = S & ~P
            c0, _ = standardise(lt, nt, Sp, cfg.ddof, cfg.min_reg_n)
            c1, _ = standardise(lt1, nt1, Sp, cfg.ddof, cfg.min_reg_n)
            d_v0p = decompose(a0, a1, Ct, Ct1, Sp)
        else:
            Sp, c0, c1, d_v0p = S, b0, b1, None
        pairs["V1p"].append((c0[Sp], c1[Sp] - c0[Sp]))
        d_v1p = decompose(c0, c1, Sp, Sp, Sp)
        if n_e > 0:
            lv = (np.where(p.present, p.Y, 0.0)[:, [j, j + 1]],
                  np.where(p.present, p.N, 0.0)[:, [j, j + 1]])
            _, mg = _merge_back(Ct, Ct1, edges, lv)
            (yt, nt_, pt), (yt1, nt1_, pt1) = mg["t"], mg["t1"]
            Sm = pt & pt1
            with np.errstate(divide="ignore"):
                m0, _ = standardise(np.log(np.where(Sm, yt, np.nan)),
                                    np.log(np.where(Sm, nt_, np.nan)), Sm, cfg.ddof,
                                    cfg.min_reg_n)
                m1, _ = standardise(np.log(np.where(Sm, yt1, np.nan)),
                                    np.log(np.where(Sm, nt1_, np.nan)), Sm, cfg.ddof,
                                    cfg.min_reg_n)
            pairs["V1m"].append((m0[Sm], m1[Sm] - m0[Sm]))
            d_v1m = decompose(m0, m1, Sm, Sm, Sm) | {
                "n_groups_t": int(pt.sum()), "n_groups_t1": int(pt1.sum())}
        else:
            pairs["V1m"].append((b0[S], b1[S] - b0[S]))
            d_v1m = d_v1
        if collect:
            for fam, d in (("V0", d_v0), ("V1", d_v1), ("V0p", d_v0p), ("V1p", d_v1p),
                           ("V1m", d_v1m)):
                if d is not None:
                    rows.append({**base, "family": fam, **d})
            xt = (lt - s0["alpha"] - s0["beta"] * nt) / s0["sd"]
            xt1 = (lt1 - s1["alpha"] - s1["beta"] * nt1) / s1["sd"]
            k1rows.append({"year_t": int(p.years[j]), "n_stay": int(S.sum()),
                           "n_leave": int((Ct & ~S).sum()), "n_enter": n_e,
                           "mu_x_t": float(xt[Ct].mean()), "sigma_x_t": float(xt[Ct].std()),
                           "mu_x_t1": float(xt1[Ct1].mean()),
                           "sigma_x_t1": float(xt1[Ct1].std()),
                           "cp_mu_t": float(xt[S].mean()), "cp_sigma_t": float(xt[S].std()),
                           "cp_mu_t1": float(xt1[S].mean()),
                           "cp_sigma_t1": float(xt1[S].std()),
                           "mech_term": d_v0["M"],
                           "xi_bar_stay": float((xt[S].mean() - xt[Ct].mean()) / xt[Ct].std()),
                           "var_stay": float(xt[S].var() / xt[Ct].var()),
                           "xi_bar_stay_V0": float(a0[S].mean())})
    cat = {v: (np.concatenate([a for a, _ in pairs[v]]) if pairs[v] else np.zeros(0),
               np.concatenate([b for _, b in pairs[v]]) if pairs[v] else np.zeros(0))
           for v in VARIANTS}
    return {"pairs": cat, "decomposition": rows, "yearly": yearly, "k1": k1rows}


def rho1_from_pairs(xi_t, dxi) -> float:
    xc = xi_t - xi_t.mean()
    return float(1.0 + np.sum(xc * (dxi - dxi.mean())) / np.sum(xc * xc))


def estimate(xi_t, dxi, x0=None) -> dict:
    """fit_mle + derived_observables under both conventions."""
    out = {"n_pairs": int(len(xi_t))}
    if len(xi_t) < 20:
        for c in CONVENTIONS:
            out[c] = {q: np.nan for q in QUANTS} | {"id_status": "too_few_pairs"}
        out["params"], out["converged"] = np.full(7, np.nan), False
        return out
    fit = fit_mle(xi_t, dxi, x0=x0)
    rho1, mi = rho1_from_pairs(xi_t, dxi), float(dxi.mean())
    for c in CONVENTIONS:
        o = derived_observables(fit["params"], xi_t, xi_target=0.0, degenerate=c)
        out[c] = {"xi_trap": o["xi_trap"], "rho1": rho1, "mean_increment": mi,
                  "mean_diffusion": o["mean_diffusion"], "two_delta_V": 2.0 * o["delta_V"],
                  "tau_mfpt": o["tau_mfpt"], "gamma": o["gamma"],
                  "B_ratio_w": o["B_ratio_w"], "id_status": o["id_status"]}
    out["params"], out["converged"] = fit["params"], fit["converged"]
    return out


def analyse(m: Muni, cfg: Config, keep_years: dict | None = None, warm: dict | None = None,
            collect: bool = True, cache: bool = False) -> dict:
    """keep_years : {outcome: years kept by kill rule 5} (data: computed here
        and returned; draws: the data's, passed in).
    warm : {(outcome, panel, variant): params} warm starts (draws).
    cache : reuse a fit when the pairs of another variant are identical
        (draws only; the data path always fits every variant afresh, so H2
        compares independent fits).
    """
    res = {"panels": {}, "keep_years": {}, "n_units": {}}
    for o in cfg.outcomes:
        B = panel_B(m, o)
        A = panel_A(m, o)
        nB = B.present.sum(axis=0)
        if keep_years is None:
            ky = B.years[nB >= cfg.n_min]
        else:
            ky = np.asarray(keep_years[o])
        A0 = panel_A0(m, o, B, ky)
        res["keep_years"][o] = ky
        res["n_units"][o] = {"B": nB, "A": A.present.sum(axis=0), "A0": A0.present.sum(axis=0),
                             "years": B.years}
        for pname, p in (("A", A), ("A0", A0), ("B", B)):
            p = apply_year_exclusion(p, ky)
            bv = build_variants(p, cfg, collect=collect)
            fits, seen = {}, {}
            for v in VARIANTS:
                xt, dx = bv["pairs"][v]
                key = (xt.tobytes(), dx.tobytes()) if cache else None
                if cache and key in seen:
                    fits[v] = seen[key]
                    continue
                x0 = None if warm is None else warm.get((o, pname, v))
                fits[v] = estimate(xt, dx, x0=x0)
                if cache:
                    seen[key] = fits[v]
            res["panels"][(o, pname)] = {"fits": fits, "pairs": bv["pairs"] if collect else None,
                                         "decomposition": bv["decomposition"],
                                         "yearly": bv["yearly"], "k1": bv["k1"],
                                         "n_d1_dropped": p.n_d1_dropped}
    return res


def flat_observables(res: dict) -> dict:
    """{(outcome, panel, variant, convention, quantity): value}."""
    out = {}
    for (o, pn), d in res["panels"].items():
        for v, f in d["fits"].items():
            for c in CONVENTIONS:
                for q in QUANTS:
                    out[(o, pn, v, c, q)] = float(f[c][q])
    return out


CONTRASTS = {
    "V0B_minus_V1B": (("B", "V0"), ("B", "V1")),
    "V1B_minus_A": (("B", "V1"), ("A", "V0")),
    "V0B_minus_A": (("B", "V0"), ("A", "V0")),
    "V1pB_minus_A": (("B", "V1p"), ("A", "V0")),
    "V1mB_minus_A": (("B", "V1m"), ("A", "V0")),
    "V1B_minus_V1mB": (("B", "V1"), ("B", "V1m")),
}


def flat_contrasts(obs: dict, outcomes) -> dict:
    out = {}
    for o in outcomes:
        for name, ((p1, v1), (p2, v2)) in CONTRASTS.items():
            for c in CONVENTIONS:
                for q in ("xi_trap", "rho1"):
                    out[(o, name, c, q)] = obs[(o, p1, v1, c, q)] - obs[(o, p2, v2, c, q)]
    return out


CI_RULES = ("tol0", "tol5", "literal")
PRIMARY_CI = "tol0"
CI_TAIL = 0.025


def order_ci(v) -> tuple:
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    if n == 0:
        return np.nan, np.nan
    k = max(1, int(np.ceil(CI_TAIL * n - 1e-12)))
    return float(v[k - 1]), float(v[n - k])


def ci(values) -> dict:
    v = np.asarray(values, float)
    fin = v[np.isfinite(v)]
    n, nf = len(v), int(len(v) - len(fin))
    out = {"n_draws": n, "n_nonfinite": nf,
           "median": float(np.median(fin)) if len(fin) else np.nan}
    if n == 0 or len(fin) == 0:
        for r in CI_RULES:
            out |= {f"{r}_lo": np.nan, f"{r}_hi": np.nan, f"{r}_ok": False}
    else:
        q = order_ci(fin)
        ok5 = nf / n <= NONFINITE_MAX_FRAC
        out |= {"tol5_lo": q[0] if ok5 else np.nan, "tol5_hi": q[1] if ok5 else np.nan,
                "tol5_ok": bool(ok5),
                "tol0_lo": q[0] if nf == 0 else np.nan, "tol0_hi": q[1] if nf == 0 else np.nan,
                "tol0_ok": nf == 0}
        lo = order_ci(np.where(np.isfinite(v), v, -np.inf))[0]
        hi = order_ci(np.where(np.isfinite(v), v, np.inf))[1]
        out |= {"literal_lo": lo, "literal_hi": hi,
                "literal_ok": bool(np.isfinite(lo) and np.isfinite(hi))}
    out["ci_rule"] = PRIMARY_CI
    out["ci_lo"], out["ci_hi"] = out[f"{PRIMARY_CI}_lo"], out[f"{PRIMARY_CI}_hi"]
    out["ci_status"] = ("ok" if out[f"{PRIMARY_CI}_ok"] else
                        "not_evaluable_no_finite_draw" if len(fin) == 0 else
                        "not_evaluable_nonfinite")
    return out


def k1_guard(v1_pairs: tuple, rho1_v1: float) -> tuple | None:
    xt = np.asarray(v1_pairs[0], float)
    if (len(xt) < 20 or not np.all(np.isfinite(xt)) or not np.isfinite(rho1_v1)
            or rho1_v1 - 1.0 == 0.0):
        return None
    lo, hi = float(np.percentile(xt, 1)), float(np.percentile(xt, 99))
    if not lo < hi:
        return None
    return (lo, hi, -abs(float(rho1_v1) - 1.0))


def k1_prediction(k1rows: list, theta_v1: np.ndarray, xi_trap_v1: float,
                  rho1_v1: float = np.nan, guard: tuple | None = None) -> dict:
    """Predicted V0 - V1 difference of xi* and rho1 (D9 primary row) and the
    zeroth-order M_bar/|a1| row (report only).
    """
    from src import k1_theory as T
    from src.k1_contract import GUARD_KEYS, TRUE_PARAM_KEYS
    out = {"theory_version": T.THEORY_VERSION,
           "guarded": guard is not None,
           "guard_lo": np.nan if guard is None else float(guard[0]),
           "guard_hi": np.nan if guard is None else float(guard[1]),
           "guard_slope_floor": np.nan if guard is None else float(guard[2])}
    rows = [r for r in k1rows if r["n_stay"] > 0]
    if not rows or not np.all(np.isfinite(theta_v1)):
        return out | {"status": "prediction_undefined: no V1 fit or no transition",
                      "pred_dxi_star": np.nan, "pred_drho1": np.nan, "M_bar": np.nan,
                      "zeroth_order_dxi_star": np.nan}
    tp = dict(zip(TRUE_PARAM_KEYS, map(float, theta_v1))) | (
        dict.fromkeys(GUARD_KEYS) if guard is None else
        dict(zip(GUARD_KEYS, map(float, guard))))
    try:
        theta, guard = T.process_spec(tp)
        T.stationary_law(theta, guard)
        comp = [{"n_stay": float(r["n_stay"]), "mu_t": r["cp_mu_t"], "sigma_t": r["cp_sigma_t"],
                 "mu_t1": r["cp_mu_t1"], "sigma_t1": r["cp_sigma_t1"], "mech": 0.0,
                 "xbar": 0.0, "V": 1.0, "wH": 0.0,
                 "cp": (r["cp_mu_t"], r["cp_sigma_t"], r["cp_sigma_t1"])} for r in rows]
        open_ = [{"n_stay": float(r["n_stay"]), "mu_t": r["mu_x_t"], "sigma_t": r["sigma_x_t"],
                  "mu_t1": r["mu_x_t1"], "sigma_t1": r["sigma_x_t1"],
                  "mech": float(r["mech_term"]), "xbar": float(r["xi_bar_stay"]),
                  "V": float(r["var_stay"]), "wH": 0.0,
                  "cp": (r["cp_mu_t"], r["cp_sigma_t"], r["cp_sigma_t1"])} for r in rows]
        pc = T._pooled(theta, comp, guard)
        po = T._pooled(theta, open_, guard)
        out.update(status="ok",
                   pred_dxi_star=float(po["obs"]["xi_trap"] - pc["obs"]["xi_trap"]),
                   pred_drho1=float(po["obs"]["rho1"] - pc["obs"]["rho1"]),
                   xi_star_complete=float(pc["obs"]["xi_trap"]),
                   xi_star_open=float(po["obs"]["xi_trap"]))
    except Exception as e:
        out.update(status=f"prediction_undefined: {type(e).__name__}: {str(e)[:160]}",
                   pred_dxi_star=np.nan, pred_drho1=np.nan)
    w = np.array([r["n_stay"] for r in rows], float)
    M = np.array([r["mech_term"] for r in rows], float)
    out["M_bar"] = float(np.sum(w * M) / w.sum()) if w.sum() else np.nan
    a1 = float(rho1_v1) - 1.0
    out["a1_V1_ols"] = a1
    out["zeroth_order_dxi_star"] = (out["M_bar"] / abs(a1)
                                    if np.isfinite(a1) and a1 != 0 else np.nan)
    x = float(xi_trap_v1)
    s1 = (float(theta_v1[1] + 2 * theta_v1[2] * x + 3 * theta_v1[3] * x * x)
          if np.isfinite(x) and np.all(np.isfinite(theta_v1)) else np.nan)
    out["a1_drift_slope_at_trap_V1"] = s1
    out["zeroth_order_dxi_star_local_slope"] = (out["M_bar"] / abs(s1)
                                                if np.isfinite(s1) and s1 != 0 else np.nan)
    return out


def judge_h1(dec: pd.DataFrame) -> dict:
    if "family" not in dec:
        return {"H1": "fail", "H1_max_residual_registered": np.nan,
                "H1_max_residual_V1_V1p_V1m": np.nan}
    reg = dec[dec["family"].isin(["V0", "V0p"])]
    worst = float(reg["identity_residual"].abs().max()) if len(reg) else np.nan
    extra = dec[~dec["family"].isin(["V0", "V0p"])]
    worst_x = float(extra["identity_residual"].abs().max()) if len(extra) else 0.0
    ok = bool(np.isfinite(worst) and worst < H1_TOL and worst_x < H1_TOL)
    return {"H1": "pass" if ok else "fail", "H1_max_residual_registered": worst,
            "H1_max_residual_V1_V1p_V1m": worst_x}


def judge_h2(res: dict, o: str) -> dict:
    d = res["panels"][(o, "A")]
    dec = pd.DataFrame(d["decomposition"])
    if "family" not in dec:
        return {"H2": "fail", "H2_no_transitions": True}
    v0 = dec[dec["family"] == "V0"]
    ok_comp = bool(len(v0) and (v0["n_E"] == 0).all() and (v0["n_L"] == 0).all()
                   and (v0["M"] == 0).all())
    (x0, d0), (x1, d1) = d["pairs"]["V0"], d["pairs"]["V1"]
    same_len = len(x0) == len(x1)
    maxdiff = (float(max(np.abs(x0 - x1).max(initial=0), np.abs(d0 - d1).max(initial=0)))
               if same_len else np.inf)
    f0, f1 = d["fits"]["V0"], d["fits"]["V1"]
    obs_same = all((f0[c][q] == f1[c][q]) or (np.isnan(f0[c][q]) and np.isnan(f1[c][q]))
                   for c in CONVENTIONS for q in QUANTS)
    ok = ok_comp and same_len and maxdiff < H2_TOL and obs_same
    return {"H2": "pass" if ok else "fail", "H2_composition_zero": ok_comp,
            "H2_pairs_max_diff": maxdiff, "H2_observables_identical": bool(obs_same)}


def judge_h3a(dec_B: pd.DataFrame) -> dict:
    v0 = dec_B[(dec_B["family"] == "V0") & (dec_B["n_E"] > 0)]
    if v0.empty:
        return {"H3a": "no_entry_transition", "H3a_n_entry_transitions": 0}
    ok = bool((v0["M"].abs() > H3A_TOL).all())
    return {"H3a": "nonzero" if ok else "zero_on_some_transition",
            "H3a_n_entry_transitions": int(len(v0)),
            "H3a_min_abs_M": float(v0["M"].abs().min()),
            "H3a_M_over_bound_min": float(v0["M_over_bound"].min()),
            "H3a_M_over_bound_max": float(v0["M_over_bound"].max())}


def judge_h3b(obs_diff: float, pred: float, ci_v1: dict) -> dict:
    half = (ci_v1["ci_hi"] - ci_v1["ci_lo"]) / 2.0
    out = {"H3b_observed": obs_diff, "H3b_predicted": pred, "H3b_V1_halfwidth": half,
           "H3b_gate_threshold": GATE_FACTOR * half}
    if not np.isfinite(pred):
        return out | {"H3b": "not_evaluable_prediction_undefined"}
    if ci_v1["ci_status"] != "ok" or not np.isfinite(half):
        return out | {"H3b": "not_evaluable_bootstrap"}
    if abs(pred) < GATE_FACTOR * half:
        return out | {"H3b": "untestable"}
    if not np.isfinite(obs_diff):
        return out | {"H3b": "fail_nonfinite_observed"}
    ok = (np.sign(obs_diff) == np.sign(pred)) and abs(obs_diff - pred) <= H3B_REL_TOL * abs(pred)
    return out | {"H3b": "pass" if ok else "fail",
                  "H3b_rel_error": abs(obs_diff - pred) / abs(pred)}


def judge_h4(ci_xi: dict, ci_rho: dict) -> str:
    if ci_xi["ci_status"] != "ok" or ci_rho["ci_status"] != "ok":
        return "not_evaluable_bootstrap"
    ok = ci_xi["ci_lo"] <= 0 <= ci_xi["ci_hi"] and ci_rho["ci_lo"] <= 0 <= ci_rho["ci_hi"]
    return "consistent" if ok else "inconsistent"


def kill_beta(res: dict, cfg: Config, window: str) -> list:
    """Kill rule 4: PIB beta_t in [0.9, 1.3] every year, on every panel's
    full-set (V0) regression.  Returns the yearly table; raises on a miss."""
    out, bad = [], []
    for (o, pn), d in res["panels"].items():
        for y in d["yearly"]:
            r = {"window": window, "outcome": o, "panel": pn, **y}
            out.append(r)
            if o in cfg.beta_outcomes and not (cfg.beta_range[0] <= y["beta"] <= cfg.beta_range[1]):
                bad.append((o, pn, y["year"], y["beta"]))
    msg = f"kill rule 4 (beta_t outside {cfg.beta_range}): {bad[:8]}" if bad else None
    return out, msg


def kill_h1_h2(res: dict, cfg: Config, window: str):
    """Kill rule 1: every H1 / H2 check is computed and returned; the message
    names every failure (None when all pass)."""
    rows, bad = [], []
    for o in cfg.outcomes:
        for pn in PANELS:
            dec = pd.DataFrame(res["panels"][(o, pn)]["decomposition"])
            h1 = judge_h1(dec)
            rows.append({"window": window, "outcome": o, "panel": pn, **h1})
            if h1["H1"] != "pass":
                bad.append(f"H1 {window}/{o}/{pn}: {h1}")
        h2 = judge_h2(res, o)
        rows.append({"window": window, "outcome": o, "panel": "A", **h2})
        if h2["H2"] != "pass":
            bad.append(f"H2 {window}/{o}: {h2}")
    return rows, ("kill rule 1: " + "; ".join(bad)) if bad else None


def data_tables(res: dict, window: str) -> dict:
    """Decomposition and unit-count tables of the data pass (written even
    when a kill rule stops the run)."""
    dec_rows = [{"window": window, "outcome": o, "panel": pn, **r}
                for (o, pn), d in res["panels"].items() for r in d["decomposition"]]
    nunits = []
    for o, d in res["n_units"].items():
        kept = set(res["keep_years"][o].tolist())
        for j, y in enumerate(d["years"]):
            nunits.append({"window": window, "outcome": o, "year": int(y),
                           "n_B": int(d["B"][j]), "n_A": int(d["A"][j]), "n_A0": int(d["A0"][j]),
                           "kept_by_N_min": bool(int(y) in kept)})
    return {"decomposition": dec_rows, "units": nunits}


@dataclass
class BlockBase:
    m: Muni
    amc_ids: np.ndarray
    members: list
    local_edges: list
    block_of_amc: dict


def make_block_base(m: Muni, regions: dict) -> BlockBase:
    ids, inv = np.unique(m.amc, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(ids) + 1))
    members = [order[bounds[k]:bounds[k + 1]] for k in range(len(ids))]
    pos = np.empty(len(m.amc), int)
    for mem in members:
        pos[mem] = np.arange(len(mem))
    local = [[] for _ in ids]
    for a, b in m.edges:
        local[inv[a]].append((pos[a], pos[b]))
    local = [np.array(e, int).reshape(-1, 2) for e in local]
    blocks = {"amc": np.arange(len(ids))}
    pb = np.nan_to_num(m.pop_base, nan=-1.0)
    for s in ("immediate", "uf"):
        lab = np.asarray(regions[s])
        rep = np.array([lab[mem[np.argmax(pb[mem])]] for mem in members])
        _, blocks[s] = np.unique(rep, return_inverse=True)
    return BlockBase(m, ids, members, local, blocks)


def draw_muni(bb: BlockBase, scheme: str, rng: np.random.Generator) -> Muni:
    """One block draw with replacement: n_blocks blocks, every AMC of a drawn
    block with all its members and years, each copy a separate unit set."""
    blk = bb.block_of_amc[scheme]
    n_blocks = int(blk.max()) + 1
    drawn = rng.integers(0, n_blocks, size=n_blocks)
    amcs_in = [np.flatnonzero(blk == b) for b in range(n_blocks)] if scheme != "amc" else None
    amc_seq = (drawn if scheme == "amc"
               else np.concatenate([amcs_in[b] for b in drawn]))
    return muni_from_amc_seq(bb, amc_seq)


def muni_from_amc_seq(bb: BlockBase, amc_seq) -> Muni:
    """Municipal panel of a sequence of AMC copies (genealogy edges remapped
    inside each copy; the copy index is the AMC label)."""
    rows, edges, amc_lab, off = [], [], [], 0
    for k, a in enumerate(amc_seq):
        mem = bb.members[a]
        rows.append(mem)
        if len(bb.local_edges[a]):
            edges.append(bb.local_edges[a] + off)
        amc_lab.append(np.full(len(mem), k))
        off += len(mem)
    rows = np.concatenate(rows)
    m = bb.m
    return Muni(m.years, {o: y[rows] for o, y in m.Y.items()}, m.N[rows],
                np.concatenate(edges) if edges else np.zeros((0, 2), int),
                np.concatenate(amc_lab), m.code[rows], m.pop_base[rows])


_W = {}


def _draw_worker(args):
    w_idx, s_idx, b = args
    bb, cfg, keep, warm = _W["bb"], _W["cfg"], _W["keep"], _W["warm"]
    rng = np.random.default_rng([SEED, w_idx, s_idx, b])
    md = draw_muni(bb, SCHEMES[s_idx], rng)
    res = analyse(md, cfg, keep_years=keep, warm=warm, collect=False, cache=True)
    return b, flat_observables(res)


def bootstrap(bb: BlockBase, cfg: Config, keep: dict, warm: dict, w_idx: int,
              scheme: str, n_boot: int, workers: int = 1) -> list:
    s_idx = SCHEMES.index(scheme)
    _W.update(bb=bb, cfg=cfg, keep=keep, warm=warm)
    jobs = [(w_idx, s_idx, b) for b in range(n_boot)]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_draw_worker, jobs, chunksize=max(1, n_boot // (workers * 8)))
    else:
        out = [_draw_worker(j) for j in jobs]
    return [d for _, d in sorted(out, key=lambda t: t[0])]


PARTIAL_KEYS = ("yearly", "checks", "decomposition", "units")
OUTPUT_KEYS = ("observables", "contrasts", "decomposition", "yearly", "checks",
               "predictions", "verdicts", "units", "draws")


def data_pass(window: str, m: Muni, cfg: Config) -> dict:
    """Data pipeline of one window and its kill-rule checks (kill rule 5's
    year exclusion happens inside ``analyse``; its counts are in 'units').
    Returns the tables plus the kill messages; never raises on a kill."""
    t0 = time.time()
    res = analyse(m, cfg)
    yearly, msg4 = kill_beta(res, cfg, window)
    checks, msg1 = kill_h1_h2(res, cfg, window)
    return {"res": res, "yearly": yearly, "checks": checks, **data_tables(res, window),
            "kill": [f"[{window}] {x}" for x in (msg4, msg1) if x],
            "seconds": time.time() - t0}


def run_windows(inputs: dict, cfg: Config, n_boot: int, workers: int = 1,
                schemes=SCHEMES, log=print) -> dict:
    """inputs: {window: (muni, regions)} in registered order.  Phase 1 runs
    the data pass and kill rules on every window; any kill raises KillRule
    with the partial tables of ALL windows (no draw, no judgement made).
    Phase 2 (only if every window passed) bootstraps and judges each."""
    passes = {w: data_pass(w, m, cfg) for w, (m, _) in inputs.items()}
    kills = [k for dp in passes.values() for k in dp["kill"]]
    if kills:
        raise KillRule(" | ".join(kills),
                       partial={k: [r for dp in passes.values() for r in dp[k]]
                                for k in PARTIAL_KEYS})
    for w, dp in passes.items():
        log(f"[{w}] data pipeline, kill rules 1/4/5 passed ({dp['seconds']:.1f}s)")
    out = {k: [] for k in OUTPUT_KEYS}
    out["res"] = {}
    for w, (m, regions) in inputs.items():
        o = finish_window(w, passes[w], m, regions, cfg, n_boot, workers, schemes, log)
        for k in OUTPUT_KEYS:
            out[k] += o[k]
        out["res"][w] = o["res"]
    return out


def run_window(window: str, m: Muni, regions: dict, cfg: Config, n_boot: int,
               workers: int = 1, schemes=SCHEMES, log=print) -> dict:
    """One window (synthetic runs, tests): ``run_windows`` on one input."""
    out = run_windows({window: (m, regions)}, cfg, n_boot, workers, schemes, log)
    out["res"] = out["res"][window]
    return out


def finish_window(window: str, dp: dict, m: Muni, regions: dict, cfg: Config, n_boot: int,
                  workers: int = 1, schemes=SCHEMES, log=print) -> dict:
    """Bootstrap, judgements and tables of one window whose data pass (dp)
    passed every kill rule; called only by ``run_windows`` phase 2."""
    assert not dp["kill"], "finish_window on a killed data pass"
    w_idx = list(WINDOWS).index(window) if window in WINDOWS else 0
    res, yearly, checks = dp["res"], dp["yearly"], dp["checks"]
    obs = flat_observables(res)
    con = flat_contrasts(obs, cfg.outcomes)
    warm = {(o, pn, v): f["params"] for (o, pn), d in res["panels"].items()
            for v, f in d["fits"].items() if np.all(np.isfinite(f["params"]))}
    bb = make_block_base(m, regions)
    draws = {}
    for s in schemes:
        t1 = time.time()
        draws[s] = bootstrap(bb, cfg, res["keep_years"], warm, w_idx, s, n_boot, workers)
        log(f"[{window}] bootstrap {s}: {n_boot} draws ({time.time() - t1:.0f}s)")
    obs_rows = []
    for key, val in obs.items():
        o, pn, v, c, q = key
        r = {"window": window, "outcome": o, "panel": pn, "variant": v, "convention": c,
             "quantity": q, "estimate": val,
             "n_pairs": res["panels"][(o, pn)]["fits"][v]["n_pairs"],
             "id_status": res["panels"][(o, pn)]["fits"][v][c]["id_status"]}
        for s in schemes:
            cc = ci([d[key] for d in draws[s]])
            r |= {f"{s}_{k}": vv for k, vv in cc.items()}
        obs_rows.append(r)
    con_rows, con_ci = [], {}
    for key, val in con.items():
        o, name, c, q = key
        r = {"window": window, "outcome": o, "contrast": name, "convention": c,
             "quantity": q, "estimate": val}
        for s in schemes:
            vals = [d[(o,) + CONTRASTS[name][0] + (c, q)] - d[(o,) + CONTRASTS[name][1] + (c, q)]
                    for d in draws[s]]
            cc = ci(vals)
            con_ci[(s,) + key] = cc
            r |= {f"{s}_{k}": vv for k, vv in cc.items()}
        con_rows.append(r)
    pred_rows, verdicts = [], []
    dt = dp
    for o in cfg.outcomes:
        dB = pd.DataFrame(res["panels"][(o, "B")]["decomposition"])
        theta_v1 = res["panels"][(o, "B")]["fits"]["V1"]["params"]
        fv1 = res["panels"][(o, "B")]["fits"]["V1"]["limit"]
        guard = k1_guard(res["panels"][(o, "B")]["pairs"]["V1"], fv1["rho1"])
        args = (res["panels"][(o, "B")]["k1"], theta_v1, fv1["xi_trap"], fv1["rho1"])
        pred = k1_prediction(*args, guard=guard)
        if guard is None:
            pred = pred | {"status": "prediction_undefined: K1 guard not defined",
                           "pred_dxi_star": np.nan, "pred_drho1": np.nan}
        pu = k1_prediction(*args, guard=None)
        pred_rows.append({"window": window, "outcome": o, **pred,
                          "unguarded_status": pu["status"],
                          "unguarded_pred_dxi_star": pu["pred_dxi_star"],
                          "unguarded_pred_drho1": pu["pred_drho1"]})
        v = {"window": window, "outcome": o,
             "role": ("primary" if (window == PRIMARY_WINDOW and o == PRIMARY_OUTCOME)
                      else "secondary")}
        h1 = [ch for ch in checks if ch["outcome"] == o and "H1" in ch]
        v["H1"] = "pass" if all(ch["H1"] == "pass" for ch in h1) else "fail"
        v["H1_max_residual"] = max(max(ch["H1_max_residual_registered"],
                                       ch["H1_max_residual_V1_V1p_V1m"]) for ch in h1)
        h2 = [ch for ch in checks if ch["outcome"] == o and "H2" in ch][0]
        v |= {k: h2[k] for k in h2 if k.startswith("H2")}
        v |= judge_h3a(dB)
        key_v1 = (o, "B", "V1", "limit", "xi_trap")
        ci_v1 = ci([d[key_v1] for d in draws[PRIMARY_SCHEME]]) if PRIMARY_SCHEME in draws else \
            {"ci_lo": np.nan, "ci_hi": np.nan, "ci_status": "no_bootstrap"}
        v |= judge_h3b(con[(o, "V0B_minus_V1B", "limit", "xi_trap")],
                       pred["pred_dxi_star"], ci_v1)
        v["H3b_prediction_status"] = pred["status"]
        v["H3b_unguarded_predicted"] = pu["pred_dxi_star"]
        v["H3b_unguarded_status"] = pu["status"]
        v["H3b_zeroth_order_predicted"] = pred["zeroth_order_dxi_star"]
        h3c = con_ci.get((PRIMARY_SCHEME, o, "V0B_minus_V1B", "limit", "rho1"), {})
        v |= {"H3c_estimate": con[(o, "V0B_minus_V1B", "limit", "rho1")],
              "H3c_ci_lo": h3c.get("ci_lo"), "H3c_ci_hi": h3c.get("ci_hi"),
              "H3c_predicted_projection": pred["pred_drho1"]}
        for s in schemes:
            v[f"H4_{s}"] = judge_h4(con_ci[(s, o, "V1B_minus_A", "limit", "xi_trap")],
                                    con_ci[(s, o, "V1B_minus_A", "limit", "rho1")])
        if PRIMARY_SCHEME in schemes:
            v["H4"] = v[f"H4_{PRIMARY_SCHEME}"]
            v["H4_sensitivity_changes_verdict"] = bool(
                "immediate" in schemes and v["H4_immediate"] != v["H4"])
        verdicts.append(v)
    draw_rows = []
    for s in schemes:
        for b, d in enumerate(draws[s]):
            draw_rows.append({"window": window, "scheme": s, "draw": b,
                              **{"|".join(k): val for k, val in d.items()}})
    return {"observables": obs_rows, "contrasts": con_rows, "decomposition": dt["decomposition"],
            "yearly": yearly, "checks": checks, "predictions": pred_rows,
            "verdicts": verdicts, "units": dt["units"], "draws": draw_rows, "res": res}


def write_tables(out: dict, outdir: Path, prefix: str = "k2br") -> list:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("observables", "contrasts", "decomposition", "yearly", "checks",
                 "predictions", "verdicts", "units"):
        p = outdir / f"{prefix}_{name}.csv"
        pd.DataFrame(out[name]).to_csv(p, index=False)
        written.append(p)
    p = outdir / f"{prefix}_bootstrap_draws.csv.gz"
    pd.DataFrame(out["draws"]).to_csv(p, index=False, compression="gzip")
    written.append(p)
    return written


def synthetic_muni(n0: int = 400, years=range(2002, 2012), seed: int = 1,
                   splits: dict | None = None, exits: dict | None = None,
                   a1: float = -0.15, beta: float = 1.05, agri_nonpos: int = 3,
                   entrant_offset: float = 0.0, lag: int = 0,
                   return_truth: bool = False):
    """Toy municipal panel built from territorial PIECES, so the AMC sums are
    exact: every final municipality is a piece with its own latent OU x,
    log N and log Y = 0.5 + beta log N + x; before its entry year an
    entrant's piece is reported inside its parent(s) (split by fixed shares
    among several parents).

    splits : {entry_year: [(n_children, parents_per_child, chain)]} -- chain
        True makes the child's single parent a same-wave entrant.
    exits  : {year: n} municipalities without any record from that year on
        (artificial exits; panel B gets n_L > 0).
    entrant_offset : added to the entrants' latent x (selective entry).
    lag    : the first ``lag`` entrants (non-chained) get a population record
        one year before their first outcome record (the D3 lag of the IBGE
        estimates): in that year their population is listed separately while
        their output is still inside the parent.
    Returns (Muni, regions), plus truth = {"pieces_amc", "Y": {o: piece
    levels}, "N": piece levels, "lag_rows"} with return_truth."""
    rng = np.random.default_rng(seed)
    years = np.array(list(years))
    T = len(years)
    splits = splits if splits is not None else {years[3]: [(6, 1, False), (2, 2, False),
                                                           (1, 1, True)],
                                                years[6]: [(3, 1, False)]}
    exits = exits or {}
    parent_of, entry_of, shares = {}, {}, {}
    n = n0
    for y, specs in sorted(splits.items()):
        for nc, npar, chain in specs:
            for _ in range(nc):
                ps = list(rng.choice(n0, size=npar, replace=False))
                c = n
                n += 1
                if chain:
                    c2 = n
                    n += 1
                    parent_of[c2], entry_of[c2], shares[c2] = ps[:1], y, [1.0]
                    ps = [c2]
                parent_of[c], entry_of[c] = ps, y
                shares[c] = list(rng.dirichlet(np.ones(len(ps)))) if len(ps) > 1 else [1.0]
    x = np.zeros((n, T))
    x[:, 0] = rng.normal(0, 1, n)
    for j in range(1, T):
        x[:, j] = (1 + a1) * x[:, j - 1] + rng.normal(0, np.sqrt(1 - (1 + a1) ** 2), n)
    kids = np.array(sorted(entry_of), int)
    x[kids] += entrant_offset
    lnN = rng.normal(9.5, 1.0, n)[:, None] + np.cumsum(rng.normal(0.01, 0.005, (n, T)), axis=1)
    lnN[kids] -= 1.5
    N = np.exp(lnN)
    Yp = np.exp(0.5 + beta * lnN + 0.6 * x)
    Ya = np.exp(-1.0 + 0.8 * lnN + 0.9 * x + rng.normal(0, 0.3, (n, T)))
    repY, repA, repN = Yp.copy(), Ya.copy(), N.copy()
    present = np.ones((n, T), bool)
    for c in entry_of:
        present[c, years < entry_of[c]] = False

    def depth(c):
        return 1 + max((depth(p) for p in parent_of[c] if p in entry_of), default=0)
    order = sorted(entry_of, key=lambda c: -depth(c))
    for c in order:
        pre = years < entry_of[c]
        for p, s in zip(parent_of[c], shares[c]):
            repY[p, pre] += s * repY[c, pre]
            repA[p, pre] += s * repA[c, pre]
            repN[p, pre] += s * repN[c, pre]
    presN = present.copy()
    lag_rows = [c for c in sorted(entry_of)
                if all(p not in entry_of for p in parent_of[c])
                and c not in {q for ps in parent_of.values() for q in ps}][:lag]
    for c in lag_rows:
        jj = np.flatnonzero(years == entry_of[c] - 1)
        if len(jj):
            presN[c, jj] = True
            for p_, sh in zip(parent_of[c], shares[c]):
                repN[p_, jj] -= sh * N[c, jj]
    for y, k in exits.items():
        gone = rng.choice(n0, size=k, replace=False)
        present[np.ix_(gone, years >= y)] = False
        presN[np.ix_(gone, years >= y)] = False
    if agri_nonpos:
        z = rng.choice(n0, size=agri_nonpos, replace=False)
        repA[z, T // 2] = 0.0
    Y = {"pib": np.where(present, repY, np.nan), "vab_agro": np.where(present, repA, np.nan)}
    Nn = np.where(presN, repN, np.nan)
    edges = np.array([(c, p) for c, ps in parent_of.items() for p in ps], int).reshape(-1, 2)
    amc = _components(n, edges)
    m = Muni(years, Y, Nn, edges, amc, np.array([f"s{i:05d}" for i in range(n)]),
             np.where(present[:, 0], repN[:, 0], np.nan))
    reg = {"immediate": rng.integers(0, 25, n), "uf": rng.integers(0, 6, n)}
    reg["uf"] = reg["uf"][amc]
    if return_truth:
        return m, reg, {"pieces_amc": amc, "Y": {"pib": Yp, "vab_agro": Ya}, "N": N,
                        "lag_rows": lag_rows, "present": present, "agri_zero": agri_nonpos}
    return m, reg


def _coverage_module():
    spec = importlib.util.spec_from_file_location("k2br_coverage",
                                                  ROOT / "scripts" / "k2br_coverage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_structure(data: Path = DATA) -> dict:
    """Population (D4: 2023 = log-linear interpolation of the 2022 census and
    the 2024 estimate), genealogy parents (coverage-script rules, incl.
    POST2010_PARENTS) and DTB 2025 blocks.  No outcome is read here."""
    cov = _coverage_module()
    pop = cov.load_population(data)
    names = cov.load_names(data)
    gen = cov.parse_genealogy(data)
    ever = set().union(*[set(s.index) for s in pop.values()])
    par, _ = cov.resolve_parents(gen, ever)
    by_name = {cov.norm(nm): c for c, nm in names.items()}
    for en, pnames in cov.POST2010_PARENTS.items():
        c = by_name[cov.norm(en)]
        uf_s = en.rsplit(" - ", 1)[1]
        par[c] = {"parents": [by_name[cov.norm(f"{pn} - {uf_s}")] for pn in pnames]}
    common = pop[2022].index.intersection(pop[2024].index)
    pop[2023] = np.exp(0.5 * (np.log(pop[2022].loc[common]) + np.log(pop[2024].loc[common])))
    import zipfile
    with zipfile.ZipFile(data / "raw/dtb/DTB_2025.zip") as z:
        x = pd.read_excel(z.open("RELATORIO_DTB_BRASIL_2025_MUNICIPIOS.xls"), header=6, dtype=str)
    x = x[x["Código Município Completo"].fillna("").str.fullmatch(r"\d{7}")]
    x["code6"] = x["Código Município Completo"].str[:6]
    regions = {"immediate": dict(zip(x.code6, x["Região Geográfica Imediata"])),
               "uf": dict(zip(x.code6, x["UF"]))}
    return {"pop": pop, "parents": {c: v["parents"] for c, v in par.items()},
            "regions": regions}


def build_muni(struct: dict, window: str, outcomes: pd.DataFrame | None,
               cfg: Config = Config()) -> tuple[Muni, dict]:
    """Municipal panel of one window.  outcomes: long table (series, code6,
    year, pib, vab_agro) or None for a structure-only panel whose outcome
    records are the population records (tests; values are placeholders)."""
    y0, y1, series = cfg.windows[window]
    years = np.arange(y0, y1 + 1)
    pop = struct["pop"]
    codes = sorted(set().union(*[set(pop[y].index) for y in years if y in pop]))
    idx = {c: i for i, c in enumerate(codes)}
    n, T = len(codes), len(years)
    N = np.full((n, T), np.nan)
    for j, y in enumerate(years):
        s = pop[y]
        N[[idx[c] for c in s.index], j] = s.values
    Y = {}
    if outcomes is None:
        for o in cfg.outcomes:
            Y[o] = np.where(np.isfinite(N), 1.0, np.nan)
    else:
        sub = outcomes[(outcomes["series"] == series) & outcomes["year"].between(y0, y1)]
        sub = sub[sub["code6"].isin(idx)]
        r = sub["code6"].map(idx).to_numpy()
        cidx = (sub["year"] - y0).to_numpy()
        for o in cfg.outcomes:
            Y[o] = np.full((n, T), np.nan)
            Y[o][r, cidx] = sub[o].to_numpy(float)
    rec0 = np.isfinite(N[:, 0])
    for o in cfg.outcomes:
        rec0 &= np.isfinite(Y[o][:, 0])
    edges = []
    for c in codes:
        if rec0[idx[c]]:
            continue
        for p in struct["parents"].get(c, []):
            if p in idx:
                edges.append((idx[c], idx[p]))
    edges = np.array(edges, int).reshape(-1, 2)
    amc = _components(n, edges)
    reg = {s: np.array([struct["regions"][s].get(c, "?") for c in codes]) for s in ("immediate", "uf")}
    return Muni(years, Y, N, edges, amc, np.array(codes), N[:, 0].copy()), reg


def _pib_module():
    path = ROOT / "scripts" / "k2br_pib.py"
    if not path.exists():
        raise KillRule("scripts/k2br_pib.py (order step 3) is not committed")
    spec = importlib.util.spec_from_file_location("k2br_pib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_outcome_frame(df: pd.DataFrame) -> pd.DataFrame:
    need = {"series", "code6", "year", "pib", "vab_agro"}
    if not need <= set(df.columns):
        raise KillRule(f"outcome table lacks {need - set(df.columns)}")
    df = df.astype({"series": str, "code6": str, "year": int})
    if df.duplicated(["series", "code6", "year"]).any():
        raise KillRule("outcome table repeats (series, code6, year)")
    if not set(df["series"]) <= {w[2] for w in WINDOWS.values()}:
        raise KillRule(f"unknown PIB series {sorted(set(df['series']))}")
    return df


def _git(*a) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_registration(reg: dict, head: str, changed: list[str]) -> None:
    """Pure check: the registration names the code commit, seed and B of this
    script, and HEAD differs from the code commit only in ALLOWED_AFTER_CODE."""
    for k in ("code_commit", "seed", "n_boot", "schemes"):
        if k not in reg:
            raise KillRule(f"registration lacks {k!r}")
    if reg["seed"] != SEED or reg["n_boot"] != N_BOOT or list(reg["schemes"]) != list(SCHEMES):
        raise KillRule("registration seed / B / schemes differ from the script constants")
    extra = sorted(set(changed) - ALLOWED_AFTER_CODE)
    if extra:
        raise KillRule(f"HEAD differs from the registered code commit {reg['code_commit']} in "
                       f"{extra} (only {sorted(ALLOWED_AFTER_CODE)} may change)")


def check_attempts(lines: list[dict], acknowledge: int, code_commit: str) -> None:
    """Refuse after a completed run (any registration: the single run
    happened) and after a killed or crashed run under the same registered
    code commit (kill rules 1/2, or a code error: fix, re-register, then
    run); an interrupted earlier
    attempt must be acknowledged by number (it stays in the log and in the
    report)."""
    if any(r.get("status") == "completed" for r in lines):
        raise KillRule("a completed real run is already logged: the single run happened")
    for bad in ("killed", "crashed"):
        if any(r.get("status") == bad and r.get("code_commit") == code_commit for r in lines):
            raise KillRule(f"a run under code commit {code_commit} {bad}: fix and re-register")
    started = [r for r in lines if r.get("status") == "started"]
    if len(started) != acknowledge:
        raise KillRule(f"{len(started)} earlier attempt(s) in {ATTEMPT_LOG.name}; pass "
                       f"--acknowledge-prior-attempts {len(started)} to run again (logged)")


def kill_coverage_reproduces(data: Path = DATA) -> dict:
    """Kill rule 2: the coverage script rerun must reproduce the committed
    coverage tables byte for byte (meta compared without its git hash)."""
    cov = _coverage_module()
    tab = ROOT / "results" / "tables"
    with tempfile.TemporaryDirectory() as tmp:
        cov.OUT = Path(tmp)
        argv = sys.argv
        sys.argv = ["k2br_coverage.py", "--data", str(data)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cov.main()
        finally:
            sys.argv = argv
        diffs = []
        for f in sorted(Path(tmp).iterdir()):
            want = tab / f.name
            if f.suffix == ".json":
                a, b = json.loads(f.read_text()), json.loads(want.read_text())
                a.pop("git_hash", None)
                b.pop("git_hash", None)
                same = a == b
            else:
                same = want.exists() and f.read_bytes() == want.read_bytes()
            if not same:
                diffs.append(f.name)
    if diffs:
        raise KillRule(f"kill rule 2: coverage tables not reproduced: {diffs}")
    return {"coverage_reproduced": True}


def kill_existence(outcomes: pd.DataFrame) -> dict:
    """Kill rule 3 (re-verified at run time): the outcome records equal the
    committed existence table (written before any value was read)."""
    if not EXISTENCE_TABLE.exists():
        raise KillRule(f"kill rule 3: {EXISTENCE_TABLE} not committed")
    ex = pd.read_csv(EXISTENCE_TABLE, dtype={"code6": str, "series": str})
    got = set(map(tuple, outcomes[["series", "code6", "year"]].astype(str).values))
    want = set(map(tuple, ex[["series", "code6", "year"]].astype(str).values))
    if got != want:
        raise KillRule(f"kill rule 3: outcome records differ from the existence table "
                       f"({len(got - want)} extra, {len(want - got)} missing)")
    return {"existence_records": len(want)}


def execute(body, allout: dict) -> tuple[str, str, dict]:
    try:
        return "completed", "completed", dict(body() or {})
    except KillRule as e:
        for k, v in e.partial.items():
            allout.setdefault(k, [])
            allout[k] += v
        return f"killed: {e}", "killed", {}
    except Exception as e:
        return (f"crashed: {type(e).__name__}: {str(e)[:500]}", "crashed",
                {"exception_type": type(e).__name__,
                 "traceback": traceback.format_exc()[-4000:]})


def real_run(args) -> None:
    from src import k1_theory as T
    if _git("status", "--porcelain"):
        raise KillRule("working tree not clean: refuse (meta must name a committed state)")
    if not public_guard_skipped(True):
        raise KillRule(f"no registration {REG_FILE}: register seed and code commit first")
    reg = public_guard_skipped({'code_commit': _git('rev-parse', 'HEAD'), 'seed': SEED, 'n_boot': N_BOOT, 'schemes': list(SCHEMES)})
    head = _git("rev-parse", "HEAD")
    changed = _git("diff", "--name-only", reg.get("code_commit", "HEAD"), "HEAD").split()
    check_registration(reg, head, changed)
    lines = ([json.loads(l) for l in ATTEMPT_LOG.read_text().splitlines() if l.strip()]
             if ATTEMPT_LOG.exists() else [])
    check_attempts(lines, args.acknowledge_prior_attempts, reg["code_commit"])
    frozen = public_guard_skipped({'frozen_commit': None, 'theory_version': T.THEORY_VERSION, 'environment_at_run': T.environment(), 'environment_differs': None})
    integrity = kill_coverage_reproduces()
    pib = _pib_module()
    raw_sha = {f: sha256(DATA / f) for f in pib.RAW_FILES}
    ATTEMPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(ATTEMPT_LOG, "a") as f:
        f.write(json.dumps({"status": "started", "head": head, "time": stamp,
                            "code_commit": reg["code_commit"]}) + "\n")
    t0 = time.time()
    cfg = Config()
    allout = {k: [] for k in ("observables", "contrasts", "decomposition", "yearly", "checks",
                              "predictions", "verdicts", "units", "draws")}

    def body() -> dict:
        outcomes = check_outcome_frame(pib.outcome_values(DATA))
        existence = kill_existence(outcomes)
        struct = load_structure()
        inputs = {w: build_muni(struct, w, outcomes, cfg) for w in WINDOWS}
        o = run_windows(inputs, cfg, N_BOOT, workers=args.workers)
        for k in allout:
            allout[k] += o[k]
        return existence
    status, attempt_status, existence = execute(body, allout)
    print(status)
    tab = ROOT / "results" / "tables"
    written = []
    for step in ("write_tables", "write_meta"):
        try:
            if step == "write_tables":
                written = write_tables(allout, tab)
            else:
                write_meta(tab, head, reg, status, frozen, raw_sha, integrity, existence,
                           t0, stamp, written)
        except Exception as e:
            print(f"{step} failed: {type(e).__name__}: {e}")
            if attempt_status == "completed":
                status, attempt_status = f"crashed: {type(e).__name__} in {step}: {e}", "crashed"
                existence = existence | {"exception_type": type(e).__name__}
    with open(ATTEMPT_LOG, "a") as f:
        f.write(json.dumps({"status": attempt_status,
                            "head": head, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "code_commit": reg["code_commit"], "detail": status,
                            "exception_type": existence.get("exception_type")}) + "\n")


def write_meta(tab, head, reg, status, frozen, raw_sha, integrity, existence, t0, stamp,
               written) -> None:
    meta = {"script": "scripts/run_k2br.py", "git": head, "registration": reg,
            "seed": SEED, "n_boot": N_BOOT, "schemes": SCHEMES, "status": status,
            "k1_freeze": {"frozen_commit": frozen["frozen_commit"],
                          "theory_version": frozen["theory_version"],
                          "environment_at_run": frozen["environment_at_run"]},
            "pib_raw_sha256": raw_sha, **integrity, **existence,
            "constants": {"N_MIN": N_MIN, "BETA_RANGE": BETA_RANGE, "DDOF": DDOF,
                          "H1_TOL": H1_TOL, "H2_TOL": H2_TOL, "H3A_TOL": H3A_TOL,
                          "GATE_FACTOR": GATE_FACTOR, "H3B_REL_TOL": H3B_REL_TOL,
                          "NONFINITE_MAX_FRAC": NONFINITE_MAX_FRAC, "WINDOWS": WINDOWS},
            "runtime_s": time.time() - t0, "started": stamp,
            "outputs": [str(p.relative_to(ROOT)) for p in written]}
    (tab / "k2br_meta.json").write_text(json.dumps(meta, indent=1, default=str))


def synthetic_run(args) -> None:
    out = Path(args.out)
    cfg = Config(windows={"toy": (2002, 2011, "toy")}, n_min=100)
    m, regions = synthetic_muni(seed=args.seed)
    o = run_window("toy", m, regions, cfg, args.n_boot, workers=args.workers)
    written = write_tables(o, out, prefix="k2br_synthetic")
    print(pd.DataFrame(o["verdicts"]).T.to_string())
    print("wrote", *map(str, written), sep="\n  ")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--real", action="store_true")
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1, help="synthetic panel seed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--acknowledge-prior-attempts", type=int, default=0)
    args = ap.parse_args()
    if args.real:
        real_run(args)
    else:
        if args.out is None:
            ap.error("--synthetic needs --out (synthetic outputs never go to results/tables)")
        synthetic_run(args)


if __name__ == "__main__":
    main()
