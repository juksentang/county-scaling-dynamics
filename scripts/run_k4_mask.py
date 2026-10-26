"""Controlled masking on complete real panels: method validation on real data.

Two complete (balanced) panels: the US metropolitan areas of Panel A (built
by scripts/run_k2.py's own functions) and the Brazilian fixed-geography
(minimum comparable area) panel of PIB, window ref2010 (built by
scripts/run_k2br.py's own functions). The observed values are never
changed: a mask only changes which units are visible in which years (whole
blocks stop, or stop and restart, reporting). On the masked panel the
pipeline of each system is run unchanged (yearly regression and
standardisation on the visible set, V0 and V1 pairs, cubic likelihood fit).

Per mask: xi*_F (complete-panel V0 = V1 cubic attractor, the reference),
xi*_0 and xi*_1 (V0 and V1 on the masked panel), the composition dose, and
the guarded pooled K1 projection of xi*_0 - xi*_1 (src/k1_theory.py),
computed only from a V1 fit and the composition (type-checked API).
Tests: T1 prediction xi*_0 - xi*_1 against the projection; T2 recovery
xi*_1 - xi*_F; T3 uncorrected shift xi*_0 - xi*_F. Grid per system: three
fraction levels x three positions (low, high, mid) x two organisations
(random units, geographic blocks) x two timings (exit, gap), 36 cells x
N_REP replicates; the pilot stage runs 18 cells of the US panel once.
Bootstrap: one draw plan per system and stage, shared by all masks, B =
N_BOOT; each draw re-runs the whole pipeline.

Stopping rules (before any mask is evaluated): raw-file sha256 equal to the
shipped records; the complete-panel estimate reproduces the shipped Panel A
row (floats within 1e-12 max(1, |v|)); the panel is balanced; the no-mask
V0 and V1 pairs are equal and equal to those of the pipeline of the system;
during the run, Lemma 1 holds on every masked panel.

    python3 scripts/run_k4_mask.py --dev --out DIR          (synthetic panels)
    python3 scripts/run_k4_mask.py --preflight pilot        (stopping rules only)
    python3 scripts/run_k4_mask.py --pilot [--workers N]
    python3 scripts/run_k4_mask.py --preflight confirmatory --br-data data/brazil
    python3 scripts/run_k4_mask.py --confirmatory --br-data data/brazil [--workers N]

A registered stage needs a passing preflight receipt of the same stage at
the same commit in the same environment ($K4_BUILD_DIR/preflight, default
~/econp_mask_build/preflight). The confirmatory seeds use the entropy and
the frozen guards and criteria recorded in the shipped
results/tables/k4_mask/confirmatory/k4_meta.json; the confirmatory masks
identical to a shipped pilot mask are redrawn. Outputs:
results/tables/k4_mask/<stage>/.
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
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

import k1_guard_sensitivity as GS
from src.km_mle import _A
import run_k2 as K2
import run_k2br as KBR

LABEL = 'controlled masking on complete real panels'
SCRIPT_REL = "scripts/run_k4_mask.py"
SYSTEMS = ("US", "BR")
SYSTEM_INDEX = {"US": 0, "BR": 1}
STAGE_INDEX = {"dev": 0, "pilot": 1, "confirmatory": 2}
PILOT_ENTROPY = 20261026
DEV_ENTROPY = 0
FRACTIONS = {"us_like": 1.0 / 21.0, "intermediate": 3.0 / 13.0, "china_like": 0.5}
DELTA_EVENT_TARGET = {"us_like": 0.05, "intermediate": 0.30, "china_like": 1.0}
POSITIONS = ("low", "high", "mid")
ORGS = ("random", "geo")
TIMINGS = ("exit", "gap")
N_REP = 10
N_BOOT = 200
CAL_TOL = 0.20
MAX_ATTEMPTS = 50
POOL_MIN, POOL_MAX = 0.25, 0.75
GUARD_SET = tuple(GS.guard_grid())
REGISTERED_GUARD = GS.REGISTERED
CLUSTER_R = 2000
IDENTITY_TOL = K2.H1_TOL
EQUIV_TOL = K2.H2_TOL
REPRO_RTOL = 1e-12
CONV = "limit"
IDENTIFIED = "stable_zero"
CRITERIA = {
    "point_availability_min": 0.80,
    "slope_band": (0.70, 1.30),
    "intercept_band_rel": 0.25,
    "sign_agreement_min": 0.90,
    "level_abs_error_rel": 0.30,
    "level_min_masks": 10,
    "range_availability_min": 0.80,
    "range_coverage_min": 0.80,
    "range_width_rel_max": 1.00,
    "recovery_share_min": 0.80,
    "recovery_share_ci_lo_min": 0.50,
    "recovery_median_ratio_max": 0.50,
    "gate_factor": 0.5,
    "power_factor": 2.0,
    "gate_availability_min": 0.80,
    "null_rel_max": 0.25,
    "comparator_share_ci_lo_min": 0.50,
}
CRITERIA_DIRECTION = {
    "point_availability_min": "up", "slope_band": "narrow", "intercept_band_rel": "down",
    "sign_agreement_min": "up", "level_abs_error_rel": "down", "level_min_masks": "fixed",
    "range_availability_min": "up", "range_coverage_min": "up", "range_width_rel_max": "down",
    "recovery_share_min": "up", "recovery_share_ci_lo_min": "up",
    "recovery_median_ratio_max": "down", "gate_factor": "fixed", "power_factor": "fixed",
    "gate_availability_min": "up", "null_rel_max": "down", "comparator_share_ci_lo_min": "up"}
STRATA = ("active", "balanced", "null")
PRIMARY_STRATUM = "active"
EVAL_SCOPES = STRATA + ("pooled",)
US_KEY_A = ("metro", "pi", "main", "A", "none")
BR_WINDOW, BR_OUTCOME = "ref2010", "pib"
TAB = ROOT / "results" / "tables"
OUT = TAB / "k4_mask"
AUDIT = ROOT / 'results' / 'records'
ATTEMPT_LOG = AUDIT / "k4_mask_attempts.jsonl"
FREEZE = AUDIT / "k4_mask_freeze.json"
FREEZE_REL = "results/records/k4_mask_freeze.json"
BUILD_DIR = Path(os.environ.get("K4_BUILD_DIR", str(Path.home() / "econp_mask_build")))
PREFLIGHT_DIR = BUILD_DIR / "preflight"
PLAN = ROOT / 'results' / 'plan.md'
SECTION_HEADING = '## 16.'
SIGNOFF_TOKEN = '**Confirmatory stage signed by the author:** signed'
DEPENDENCIES = ("scripts/run_k4_mask.py", "tests/test_k4_mask.py", "scripts/run_k2.py",
                "scripts/run_k2br.py", "scripts/k2br_pib.py", "scripts/k2_coverage.py",
                "scripts/k2br_coverage.py", "scripts/k1_guard_sensitivity.py",
                "src/km_mle.py", "src/k1_theory.py", "src/k1_contract.py", "src/config.py")


class KillRule(RuntimeError):
    def __init__(self, rule: int, msg: str):
        super().__init__(f"kill rule {rule}: {msg}")
        self.rule = rule


def stratum_of(position: str, timing: str) -> str:
    """active = low/high x exit (net forcing); balanced = low/high x gap (exit
    and re-entry cancel in M_bar); null = mid (M ~ 0 by construction)."""
    if position == "mid":
        return "null"
    return "active" if timing == "exit" else "balanced"


def cell_grid() -> list[dict]:
    """The 36 cells of one system, in a fixed order (cell_i enters the seeds)."""
    out = []
    for fl in FRACTIONS:
        for pos in POSITIONS:
            for org in ORGS:
                for tim in TIMINGS:
                    out.append({"cell_i": len(out), "fraction_level": fl,
                                "fraction": FRACTIONS[fl], "position": pos,
                                "organisation": org, "timing": tim,
                                "stratum": stratum_of(pos, tim),
                                "cell": f"{fl}|{pos}|{org}|{tim}"})
    return out


PILOT_CELLS = tuple(
    [f"{fl}|{pos}|{org}|{tim}" for fl in ("us_like", "china_like") for pos in ("low", "high")
     for org in ORGS for tim in TIMINGS]
    + [f"china_like|mid|{org}|exit" for org in ORGS])


def stage_cells(stage: str, n_rep: int | None = None, cells: tuple | None = None) -> list[dict]:
    g = cell_grid()
    if stage == "pilot":
        g = [c for c in g if c["cell"] in PILOT_CELLS]
        n_rep = 1 if n_rep is None else n_rep
    if cells is not None:
        g = [c for c in g if c["cell"] in cells]
    n_rep = N_REP if n_rep is None else n_rep
    return [c | {"rep": r} for c in g for r in range(n_rep)]


def stage_entropy(stage: str, freeze_commit: str | None = None) -> int:
    if stage == "pilot":
        return PILOT_ENTROPY
    if stage == "dev":
        return DEV_ENTROPY
    if stage == "confirmatory":
        if not (isinstance(freeze_commit, str) and len(freeze_commit) == 40
                and all(ch in "0123456789abcdef" for ch in freeze_commit)):
            raise ValueError("confirmatory seeds need the full 40-hex freeze commit hash")
        return int(freeze_commit, 16)
    raise ValueError(stage)


def mask_seedseq(entropy: int, system: str, stage: str, cell_i: int, rep: int,
                 attempt: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy, spawn_key=(SYSTEM_INDEX[system], STAGE_INDEX[stage],
                                                      0, cell_i, rep, attempt))


def boot_seedseq(entropy: int, system: str, stage: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy, spawn_key=(SYSTEM_INDEX[system], STAGE_INDEX[stage], 1))


def cluster_seedseq(entropy: int, system: str, stage: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy, spawn_key=(SYSTEM_INDEX[system], STAGE_INDEX[stage], 2))


def pool_fraction(f: float) -> float:
    """Share of units (ordered by position) from which a mask of fraction f
    is drawn: clip(2 f, POOL_MIN, POOL_MAX)."""
    return float(np.clip(2.0 * f, POOL_MIN, POOL_MAX))


def position_order(values: np.ndarray, position: str, centre: float) -> np.ndarray:
    """Indices ordered for a position: low = ascending, high = descending,
    mid = ascending |value - centre|; ties by index (stable)."""
    v = np.asarray(values, float)
    if position == "low":
        key = v
    elif position == "high":
        key = -v
    elif position == "mid":
        key = np.abs(v - centre)
    else:
        raise ValueError(position)
    return np.argsort(key, kind="stable")


def block_means(xi: np.ndarray, geo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, inv = np.unique(geo, return_inverse=True)
    size = np.bincount(inv)
    mean = np.bincount(inv, weights=xi) / size
    return ids, mean, size


def mask_random(xi: np.ndarray, f: float, position: str, rng) -> np.ndarray:
    """Random units: round(f n) units drawn without replacement from the
    first ceil(q n) units of the position order (q = pool_fraction(f))."""
    n = len(xi)
    k = int(round(f * n))
    pool = position_order(xi, position, float(np.mean(xi)))[:int(np.ceil(pool_fraction(f) * n))]
    m = np.zeros(n, bool)
    if k >= 1:
        m[rng.choice(pool, size=k, replace=False)] = True
    return m


def mask_geo(xi: np.ndarray, geo: np.ndarray, f: float, position: str, rng) -> np.ndarray:
    """Geographic blocks: blocks ordered by their mean base-year xi_F (low /
    high / near the unit mean); the pool is the leading blocks up to the
    first whose cumulative unit share reaches q; blocks of the pool in a
    random order are added while the total stays <= (1 + CAL_TOL) f n, until
    it reaches (1 - CAL_TOL) f n.  Whole blocks only."""
    n = len(xi)
    ids, mean, size = block_means(xi, geo)
    order = position_order(mean, position, float(np.mean(xi)))
    cum = np.cumsum(size[order])
    n_pool = int(np.searchsorted(cum, pool_fraction(f) * n) + 1)
    pool = order[:min(n_pool, len(order))]
    target = f * n
    chosen, tot = [], 0
    for b in rng.permutation(pool):
        if tot + size[b] <= (1.0 + CAL_TOL) * target:
            chosen.append(b)
            tot += size[b]
        if tot >= (1.0 - CAL_TOL) * target:
            break
    return np.isin(geo, ids[chosen])


def calibrated(mask: np.ndarray, f: float) -> bool:
    return bool(mask.any() and abs(mask.mean() - f) <= CAL_TOL * f)


def mask_hash(system: str, timing: str, labels: np.ndarray, mask: np.ndarray) -> str:
    s = "|".join([system, timing] + sorted(map(str, np.asarray(labels)[mask])))
    return hashlib.sha256(s.encode()).hexdigest()


def construct_mask(system: str, stage: str, entropy: int, cell: dict, xi: np.ndarray,
                   geo: np.ndarray, labels: np.ndarray, exclude: set | frozenset = frozenset()):
    """(mask, attempt, status).  Redraws (attempt + 1) when a geo mask misses
    the calibration band or the mask equals an excluded (pilot) mask."""
    for att in range(MAX_ATTEMPTS):
        rng = np.random.default_rng(mask_seedseq(entropy, system, stage, cell["cell_i"],
                                                 cell["rep"], att))
        if cell["organisation"] == "random":
            m = mask_random(xi, cell["fraction"], cell["position"], rng)
        else:
            m = mask_geo(xi, geo, cell["fraction"], cell["position"], rng)
        if not calibrated(m, cell["fraction"]):
            continue
        if mask_hash(system, cell["timing"], labels, m) in exclude:
            continue
        return m, att, "ok"
    return None, MAX_ATTEMPTS, "mask_infeasible"


def visibility(mask: np.ndarray, n_years: int, timing: str) -> np.ndarray:
    """Visible unit-years: exit = masked units absent from year index T//2 to
    the end; gap = absent T//3 .. 2T//3 - 1, visible again from 2T//3."""
    T = n_years
    vis = np.ones((len(mask), T), bool)
    if timing == "exit":
        vis[np.ix_(mask, np.arange(T // 2, T))] = False
    elif timing == "gap":
        vis[np.ix_(mask, np.arange(T // 3, 2 * T // 3))] = False
    else:
        raise ValueError(timing)
    return vis


def event_transitions(visible: np.ndarray) -> set:
    """Year indices j of the transitions j -> j+1 at which some unit's
    visibility changes (an exit or a re-entry event)."""
    v = np.asarray(visible, bool)
    return set(np.flatnonzero((v[:, :-1] != v[:, 1:]).any(axis=0)).tolist())


def never_masked(visible: np.ndarray) -> np.ndarray:
    """Visibility of the F_S reference: the balanced panel of the units that
    are visible in every year (never masked by this mask)."""
    keep = np.asarray(visible, bool).all(axis=1)
    return np.repeat(keep[:, None], np.asarray(visible).shape[1], axis=1)


US_COMP_KEYS = ("year_t", "n_S", "M_t", "xbar_stay_t", "var_stay_t", "xbar_stay_t1",
                "var_stay_t1")
BR_COMP_KEYS = ("year_t", "n_stay", "mu_x_t", "sigma_x_t", "mu_x_t1", "sigma_x_t1",
                "cp_mu_t", "cp_sigma_t", "cp_mu_t1", "cp_sigma_t1", "mech_term",
                "xi_bar_stay", "var_stay")
COMP_KEYS = {"US": US_COMP_KEYS, "BR": BR_COMP_KEYS}


@dataclass(frozen=True, eq=False)
class V1Fit:
    """The only fit the projection may see: the masked V1 cubic fit."""
    system: str
    theta: tuple
    rho1: float
    xi_trap: float
    xi_t: np.ndarray

    def __post_init__(self):
        if self.system not in SYSTEMS:
            raise ValueError(self.system)
        object.__setattr__(self, "theta", tuple(map(float, self.theta)))
        object.__setattr__(self, "xi_t", np.array(self.xi_t, float, copy=True))
        self.xi_t.setflags(write=False)


@dataclass(frozen=True, eq=False)
class Composition:
    """Per-transition composition quantities: counts, M_t and the stayers'
    design (US: mean / variance of the stayers' xi in the full-set frame;
    BR: run_k2br's K1 rows without xi_bar_stay_V0).  Exactly COMP_KEYS[system]
    per row; any other key (a V0 increment, a V0 fit, the V0 - V1 gap) is an
    error."""
    system: str
    rows: tuple

    def __post_init__(self):
        want = set(COMP_KEYS[self.system])
        rows = []
        for r in self.rows:
            r = dict(r)
            if set(r) != want:
                raise ValueError(f"composition keys {sorted(set(r) ^ want)} not allowed / missing")
            rows.append(tuple((k, float(r[k])) for k in COMP_KEYS[self.system]))
        object.__setattr__(self, "rows", tuple(rows))

    @classmethod
    def from_us_dec(cls, dec: pd.DataFrame) -> "Composition":
        return cls("US", tuple({k: r[k] for k in US_COMP_KEYS} for r in dec.to_dict("records")))

    @classmethod
    def from_br_k1rows(cls, rows: list) -> "Composition":
        return cls("BR", tuple({k: r[k] for k in BR_COMP_KEYS} for r in rows))

    def records(self) -> list[dict]:
        out = [dict(r) for r in self.rows]
        for r in out:
            r["year_t"] = int(r["year_t"])
            if "n_stay" in r:
                r["n_stay"] = int(r["n_stay"])
        return out

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records(), columns=list(COMP_KEYS[self.system]))

    def m_bar(self) -> float:
        r = self.frame()
        n, M = ("n_S", "M_t") if self.system == "US" else ("n_stay", "mech_term")
        r = r[r[n] > 0]
        return float((r[n] * r[M]).sum() / r[n].sum()) if len(r) else np.nan


def project(v1: V1Fit, comp: Composition, guards=GUARD_SET) -> list[dict]:
    """The frozen guarded pooled K1 projection of xi*_0 - xi*_1 under every
    guard of ``guards`` ((support percentiles), floor factor), from a V1Fit
    and a Composition only.  A guard that is undefined, and any exception of
    the theory, give a status (never a stop)."""
    if type(v1) is not V1Fit or type(comp) is not Composition:
        raise TypeError("project() takes exactly a V1Fit and a Composition")
    if v1.system != comp.system:
        raise ValueError("V1Fit and Composition belong to different systems")
    theta = np.asarray(v1.theta, float)
    out = []
    for sup, fac in guards:
        g = GS.alt_guard(v1.xi_t, v1.rho1, sup, fac)
        row = {"support_lo_pct": sup[0], "support_hi_pct": sup[1], "floor_factor": fac,
               "registered_guard": (tuple(sup), fac) == (tuple(REGISTERED_GUARD[0]),
                                                         REGISTERED_GUARD[1]),
               "guard_lo": np.nan if g is None else g[0],
               "guard_hi": np.nan if g is None else g[1],
               "slope_floor": np.nan if g is None else g[2]}
        if g is None:
            out.append(row | {"status": "prediction_undefined: K1 guard not defined",
                              "pred": np.nan, "pred_rho1": np.nan, "proj_open": np.nan,
                              "proj_complete": np.nan})
            continue
        if v1.system == "US":
            p = K2.k1_prediction(theta, comp.frame(), g)
            row |= {"status": p["status"], "pred": p["pred_xi_trap"], "pred_rho1": p["pred_rho1"],
                    "proj_open": p["proj_open_xi_trap"],
                    "proj_complete": p["proj_complete_xi_trap"]}
        else:
            p = KBR.k1_prediction(comp.records(), theta, v1.xi_trap, v1.rho1, guard=g)
            row |= {"status": p["status"], "pred": p["pred_dxi_star"],
                    "pred_rho1": p["pred_drho1"], "proj_open": p.get("xi_star_open", np.nan),
                    "proj_complete": p.get("xi_star_complete", np.nan)}
        out.append(row)
    return out


def zeroth_order(v1: V1Fit, comp: Composition) -> float:
    """Report only: M_bar / |a1|, a1 = rho1(V1) - 1."""
    a1 = v1.rho1 - 1.0
    return comp.m_bar() / abs(a1) if np.isfinite(a1) and a1 != 0 else np.nan


def root_status(params, xi_data, n_grid: int = 600) -> str:
    if params is None or not np.all(np.isfinite(params)) or len(xi_data) < 2:
        return "failed"
    lo, hi = np.percentile(xi_data, 1), np.percentile(xi_data, 99)
    A = _A(np.asarray(params, float), np.linspace(lo, hi, n_grid))
    i = np.where(np.diff(np.sign(A)) != 0)[0]
    return "stable_zero" if np.any((A[i] > 0) & (A[i + 1] < 0)) else "fallback_argminA"


def _norm(e: dict, n_pairs: int, xi_data) -> dict:
    c = e.get(CONV) or {}
    p = e.get("params")
    p = None if p is None else np.asarray(p, float)
    return {"xi_trap": float(c.get("xi_trap", np.nan)), "rho1": float(c.get("rho1", np.nan)),
            "id_status": root_status(p, np.asarray(xi_data, float)),
            "path_status": str(c.get("id_status", "failed")),
            "params": p, "n_pairs": int(n_pairs)}


def _dose(n_S, n_E, n_L, M) -> dict:
    n_S, n_E, n_L, M = (np.asarray(a, float) for a in (n_S, n_E, n_L, M))
    k = n_S > 0
    n_S, n_E, n_L, M = n_S[k], n_E[k], n_L[k], M[k]
    ev = (n_E + n_L) / n_S
    return {"delta_event": float(ev.max()) if len(ev) else np.nan,
            "delta_window": float((n_E + n_L).sum() / n_S.sum()) if len(ev) else np.nan,
            "M_bar": float((n_S * M).sum() / n_S.sum()) if len(ev) else np.nan,
            "n_event_transitions": int((ev > 0).sum()),
            "M_events": json.dumps([float(x) for x in M[ev > 0]]),
            "delta_events": json.dumps([float(x) for x in ev[ev > 0]])}


@dataclass
class USImpl:
    """US Panel A as arrays; masked skeletons are run_k2.Skeleton objects run
    by run_k2.run_spec (lineage multiplicity weights for draws)."""
    codes: list
    lin: np.ndarray
    years: list
    transitions: list
    logN: np.ndarray
    logY: np.ndarray
    n_lin: int
    key: tuple = US_KEY_A

    def skeleton(self, visible: np.ndarray | None) -> K2.Skeleton:
        n, T = self.logN.shape
        vis = np.ones((n, T), bool) if visible is None else np.asarray(visible, bool)
        recs, rows = {}, {}
        for j, y in enumerate(self.years):
            idx = np.flatnonzero(vis[:, j])
            cs = [self.codes[i] for i in idx]
            recs[y] = dict(codes=cs, logN=self.logN[idx, j], logY=self.logY[idx, j],
                           lin=self.lin[idx], vintage="A", pos={c: k for k, c in enumerate(cs)})
            rows[y] = idx
        trans = {}
        for t in self.transitions:
            j = self.years.index(t)
            a, b = rows[t], rows[t + 1]
            common = np.intersect1d(a, b)
            ia, ib = np.searchsorted(a, common), np.searchsorted(b, common)
            trans[t] = dict(
                ia=ia.astype(int), ib=ib.astype(int),
                lev=np.setdiff1d(np.arange(len(a)), ia), ent=np.setdiff1d(np.arange(len(b)), ib),
                changed=np.zeros(len(common), bool), vintage_t="A", vintage_t1="A",
                v1d=dict(logN_t=self.logN[common, j], logY_t=self.logY[common, j],
                         logN_t1=self.logN[common, j + 1], logY_t1=self.logY[common, j + 1],
                         lin=self.lin[common], stay=np.ones(len(common), bool)))
        return K2.Skeleton(self.key, list(self.years), list(self.transitions), recs, trans, [])

    def naive(self, visible, warm=None) -> dict:
        """Naive comparator (report only): V0 with the pairs of the event
        transitions dropped (run_k2.run_spec on the masked skeleton restricted
        to the transitions without entry or exit)."""
        ev = event_transitions(visible)
        sk = self.skeleton(visible)
        keep = [t for t in sk.transitions if self.years.index(t) not in ev]
        sk = K2.Skeleton(sk.key, sk.years, keep, sk.recs, sk.trans, [])
        x, dx, w = K2.run_spec(sk)["pairs"]["V0"]
        return _norm(K2.estimate(x, dx, w, warm), np.round(w).sum(), K2._expand(x, dx, w)[0])

    def run(self, visible, draw=None, warm: dict | None = None, need_v1: bool = True,
            collect: bool = True) -> dict:
        warm = warm or {}
        res = K2.run_spec(self.skeleton(visible), draw)
        x, dx, w = res["pairs"]["V0"]
        out = {"V0": _norm(K2.estimate(x, dx, w, warm.get("V0")), np.round(w).sum(),
                           K2._expand(x, dx, w)[0]),
               "pairs": res["pairs"]}
        dec = res["dec"]
        out["identity_max"] = (float(np.nanmax(np.abs(dec["identity_residual"])))
                               if len(dec) else np.nan)
        if collect:
            out["dose"] = _dose(dec["n_S"], dec["n_E"], dec["n_L"], dec["M_t"])
        if need_v1:
            x1, dx1, w1 = res["pairs"]["V1"]
            e1 = _norm(K2.estimate(x1, dx1, w1, warm.get("V1")), np.round(w1).sum(),
                       K2._expand(x1, dx1, w1)[0])
            out["V1"] = e1
            if collect:
                out["comp"] = Composition.from_us_dec(dec)
                out["v1fit"] = (None if e1["params"] is None else
                                V1Fit("US", e1["params"], e1["rho1"], e1["xi_trap"],
                                      K2._expand(x1, dx1, w1)[0]))
        return out

    def draw_plan(self, ss: np.random.SeedSequence, n_boot: int) -> list:
        rng = np.random.default_rng(ss)
        blocks = np.arange(self.n_lin)
        return [K2.draw_weights(rng, self.n_lin, blocks) for _ in range(n_boot)]


@dataclass
class BRImpl:
    """BR Panel A (run_k2br.Panel, after kill rule 5's year exclusion); a
    draw is a sequence of AMC rows (with replacement), the panel of a draw is
    the AMC rows indexed by it (equal to run_k2br.panel_A of the drawn
    municipal copies), masks follow their AMC."""
    pA: object
    cfg: object

    def panel(self, visible, rows=None):
        p = self.pA
        vis = np.ones(p.present.shape, bool) if visible is None else np.asarray(visible, bool)
        if rows is None:
            return replace(p, present=p.present & vis)
        r = np.asarray(rows, int)
        return replace(p, Y=p.Y[r], N=p.N[r], present=p.present[r] & vis[r])

    def naive(self, visible, warm=None) -> dict:
        """Naive comparator (report only): V0 pairs (yearly standardisation on
        the visible set, run_k2br.standardise) of the transitions without entry
        or exit only (same S >= min_reg_n rule as run_k2br.build_variants)."""
        p = self.panel(visible)
        logY, logN = KBR._logs(p)
        xi = np.full(p.Y.shape, np.nan)
        for j in range(len(p.years)):
            xi[:, j] = KBR.standardise(logY[:, j], logN[:, j], p.present[:, j],
                                       self.cfg.ddof, self.cfg.min_reg_n)[0]
        ev = event_transitions(p.present)
        xs, ds = [], []
        for j in KBR.transitions(p):
            S = p.present[:, j] & p.present[:, j + 1]
            if j in ev or S.sum() < self.cfg.min_reg_n:
                continue
            xs.append(xi[S, j])
            ds.append(xi[S, j + 1] - xi[S, j])
        xt = np.concatenate(xs) if xs else np.zeros(0)
        dx = np.concatenate(ds) if ds else np.zeros(0)
        try:
            e = KBR.estimate(xt, dx, x0=warm)
        except Exception:
            e = {"params": None}
        out = _norm(e, len(xt), xt)
        if out["params"] is not None and not np.all(np.isfinite(out["params"])):
            out["params"] = None
        return out

    def run(self, visible, draw=None, warm: dict | None = None, need_v1: bool = True,
            collect: bool = True) -> dict:
        warm = warm or {}
        bv = KBR.build_variants(self.panel(visible, draw), self.cfg, collect=collect)
        out = {"pairs": bv["pairs"]}
        for v in (("V0", "V1") if need_v1 else ("V0",)):
            xt, dx = bv["pairs"][v]
            try:
                e = KBR.estimate(xt, dx, x0=warm.get(v))
            except Exception:
                e = {"params": None}
            out[v] = _norm(e, len(xt), xt)
            if out[v]["params"] is not None and not np.all(np.isfinite(out[v]["params"])):
                out[v]["params"] = None
        if collect:
            d0 = [r for r in bv["decomposition"] if r["family"] == "V0"]
            out["identity_max"] = (float(np.nanmax([abs(r["identity_residual"]) for r in d0]))
                                   if d0 else np.nan)
            out["dose"] = _dose([r["n_S"] for r in d0], [r["n_E"] for r in d0],
                                [r["n_L"] for r in d0], [r["M"] for r in d0])
            if need_v1:
                out["comp"] = Composition.from_br_k1rows(bv["k1"])
                e1 = out["V1"]
                out["v1fit"] = (None if e1["params"] is None else
                                V1Fit("BR", e1["params"], e1["rho1"], e1["xi_trap"],
                                      bv["pairs"]["V1"][0]))
        else:
            out["identity_max"] = np.nan
        return out

    def draw_plan(self, ss: np.random.SeedSequence, n_boot: int) -> list:
        rng = np.random.default_rng(ss)
        n = self.pA.present.shape[0]
        return [rng.integers(0, n, size=n) for _ in range(n_boot)]


@dataclass
class System:
    name: str
    impl: object
    labels: np.ndarray
    geo: np.ndarray
    geo_names: np.ndarray
    xi_base: np.ndarray
    base_year: int
    n_years: int
    committed: dict = field(default_factory=dict)


def us_impl_from_skeleton(sk: K2.Skeleton, n_lin: int) -> USImpl:
    """Arrays of a Panel A skeleton (kill rule 3: same area set every year)."""
    y0 = sk.years[0]
    codes = list(sk.recs[y0]["codes"])
    for y in sk.years:
        if list(sk.recs[y]["codes"]) != codes:
            raise KillRule(3, f"US Panel A is not balanced in {y}")
    logN = np.column_stack([sk.recs[y]["logN"] for y in sk.years])
    logY = np.column_stack([sk.recs[y]["logY"] for y in sk.years])
    return USImpl(codes, np.asarray(sk.recs[y0]["lin"], int), list(sk.years),
                  list(sk.transitions), logN, logY, n_lin, sk.key)


def us_system_from_world(world, n_min: int = K2.N_MIN) -> tuple[System, K2.Skeleton]:
    """US Panel A through run_k2's own functions (as k1_guard_sensitivity.us_cell)."""
    unit_lin, area_lin = K2.lineages(world, "metro")
    aggs = {("metro", lab, oc): K2.aggregate(world, "metro", lab, oc)
            for (sc, lab) in world.areas if sc == "metro" for oc in world.outcomes}
    drops, _ = K2.min_n_drops(world, aggs, n_min)
    sk = K2.build_skeleton(world, aggs, US_KEY_A, area_lin, drops[(US_KEY_A[1], US_KEY_A[2])])
    n_lin = max(unit_lin.values()) + 1
    impl = us_impl_from_skeleton(sk, n_lin)
    states = K2.lineage_states(world, unit_lin)
    st = states[impl.lin]
    names, geo = np.unique(st, return_inverse=True)
    r0 = sk.recs[sk.years[0]]
    xi = K2.sami(r0["logN"], r0["logY"], np.ones(len(r0["logN"])))[0]
    return System("US", impl, np.array(impl.codes), geo, names, xi, sk.years[0],
                  len(sk.years)), sk


def br_system_from_muni(m, regions: dict, cfg) -> System:
    """BR Panel A (AMC) of one window through run_k2br's own functions:
    panel_B -> kill-rule-5 years -> panel_A -> apply_year_exclusion; geo
    block = immediate region of the AMC (make_block_base)."""
    B = KBR.panel_B(m, BR_OUTCOME)
    ky = B.years[B.present.sum(axis=0) >= cfg.n_min]
    A = KBR.apply_year_exclusion(KBR.panel_A(m, BR_OUTCOME), ky)
    keep = A.present.any(axis=1)
    A = replace(A, Y=A.Y[keep], N=A.N[keep], present=A.present[keep])
    if not A.present.all():
        raise KillRule(3, f"BR Panel A is not balanced ({int((~A.present.all(axis=1)).sum())} "
                          "AMCs miss a year)")
    if not np.all(np.diff(A.years) == 1):
        raise KillRule(3, "BR Panel A years are not consecutive")
    bb = KBR.make_block_base(m, regions)
    geo = np.asarray(bb.block_of_amc["immediate"], int)[keep]
    lab_imm = np.asarray(regions["immediate"])
    pb = np.nan_to_num(m.pop_base, nan=-1.0)
    names = np.array([lab_imm[mem[np.argmax(pb[mem])]] for mem in bb.members])
    logY, logN = KBR._logs(A)
    xi = KBR.standardise(logY[:, 0], logN[:, 0], A.present[:, 0], cfg.ddof, cfg.min_reg_n)[0]
    return System("BR", BRImpl(A, cfg), np.asarray(m.code[bb.amc_ids])[keep], geo, names, xi,
                  int(A.years[0]), len(A.years))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity_rows(system: str, data: Path, tab: Path = TAB) -> list[dict]:
    """Kill rule 1: raw-file sha256 vs the committed records."""
    if system == "US":
        src = pd.read_csv(tab / "k2_coverage_sources.csv")
        reg = public_guard_skipped({'data_sha256': json.loads((tab / 'k2' / 'k2_meta.json').read_text())['data_sha256']})
        want = {r.file: r.sha256 for r in src.itertuples()} | reg.get("data_sha256", {})
    else:
        src = pd.read_csv(tab / "k2br_coverage_sources.csv")
        want = {r.file: r.sha256 for r in src.itertuples()}
        want |= json.loads((tab / "k2br_meta.json").read_text())["pib_raw_sha256"]
    rows = []
    for f, w in sorted(want.items()):
        h = _sha256(data / f) if (data / f).exists() else "missing"
        rows.append({"system": system, "file": f, "committed": w, "now": h, "pass": h == w})
    return rows


def committed_full(system: str, tab: Path = TAB) -> dict:
    """The committed Panel A V0 'limit' row (xi_trap, rho1, n_pairs, id_status)."""
    if system == "US":
        o = pd.read_csv(tab / "k2" / "k2_observables.csv")
        s, oc, per, pan, link = US_KEY_A
        o = o[(o.scope == s) & (o.outcome == oc) & (o.period == per) & (o.panel == pan)
              & (o.link == link) & (o.variant == "V0") & (o.conv == CONV)]
        conv_col = "conv"
    else:
        o = pd.read_csv(tab / "k2br_observables.csv")
        o = o[(o.window == BR_WINDOW) & (o.outcome == BR_OUTCOME) & (o.panel == "A")
              & (o.variant == "V0") & (o.convention == CONV)]
        conv_col = "convention"
    del conv_col
    xi = o[o.quantity == "xi_trap"].iloc[0]
    rh = o[o.quantity == "rho1"].iloc[0]
    return {"xi_trap": float(xi["estimate"]), "rho1": float(rh["estimate"]),
            "n_pairs": int(xi["n_pairs"]), "id_status": str(xi["id_status"])}


def reproduction_rows(system: System, full: dict, committed: dict) -> list[dict]:
    rows = []
    for k in ("xi_trap", "rho1", "n_pairs", "id_status"):
        a = committed[k]
        b = full["path_status"] if k == "id_status" else full[k]
        if isinstance(a, str):
            ok, d = a == b, 0.0 if a == b else np.inf
        else:
            d = abs(float(a) - float(b)) if np.isfinite(b) else np.inf
            ok = d <= REPRO_RTOL * max(1.0, abs(float(a)))
        rows.append({"system": system.name, "quantity": k, "committed": a, "rebuilt": b,
                     "abs_diff": d, "pass": bool(ok)})
    return rows


def equivalence_rows(system: System, full_run: dict, reference_pairs: tuple | None) -> list[dict]:
    """Kill rule 4: no-mask run has V0 pairs == V1 pairs, and (when given) this
    script's pairs equal the registered pipeline's V0 pairs."""
    rows = []
    p0, p1 = full_run["pairs"]["V0"], full_run["pairs"]["V1"]
    d = _pair_diff(p0, p1)
    rows.append({"system": system.name, "check": "no-mask V0 pairs == V1 pairs", "max_diff": d,
                 "pass": d < EQUIV_TOL})
    if reference_pairs is not None:
        d2 = _pair_diff(p0, reference_pairs)
        rows.append({"system": system.name, "check": "pairs == registered pipeline's Panel A V0",
                     "max_diff": d2, "pass": d2 == 0.0})
    return rows


def _pair_diff(a: tuple, b: tuple) -> float:
    if len(a) == 3 and len(b) == 3:
        return K2.pair_multiset_diff(a, b)
    xa, da = a[0], a[1]
    xb, db = b[0], b[1]
    if len(xa) != len(xb):
        return np.inf
    if not len(xa):
        return 0.0
    return float(max(np.max(np.abs(xa - xb)), np.max(np.abs(da - db))))


def point_record(system: System, mask_row: dict, vis: np.ndarray, full: dict,
                 guards=GUARD_SET, point_guard=REGISTERED_GUARD) -> tuple[dict, list, dict]:
    """Point estimates, dose and projections of one mask; kill rule 5."""
    r = system.impl.run(vis, need_v1=True, collect=True)
    if not (r["identity_max"] < IDENTITY_TOL):
        raise KillRule(5, f"{system.name} mask {mask_row['mask_id']}: Lemma 1 residual "
                          f"{r['identity_max']:.3g}")
    e0, e1 = r["V0"], r["V1"]
    rec = dict(mask_row)
    rec |= {"xi_F": full["xi_trap"], "id_F": full["id_status"], "rho1_F": full["rho1"],
            "xi_0": e0["xi_trap"], "id_0": e0["id_status"], "rho1_0": e0["rho1"],
            "xi_1": e1["xi_trap"], "id_1": e1["id_status"], "rho1_1": e1["rho1"],
            "path_0": e0["path_status"], "path_1": e1["path_status"],
            "a1_V1": e1["rho1"] - 1.0, "n_pairs_0": e0["n_pairs"], "n_pairs_1": e1["n_pairs"],
            "identity_max": r["identity_max"], **r["dose"]}
    rec["T1_obs"] = rec["xi_0"] - rec["xi_1"]
    rec["T2"] = rec["xi_1"] - rec["xi_F"]
    rec["T3"] = rec["xi_0"] - rec["xi_F"]
    vS = never_masked(vis)
    fs = system.impl.run(vS, need_v1=False, collect=False)["V0"]
    nv = system.impl.naive(vis)
    rec |= {"n_units_FS": int(vS[:, 0].sum()), "xi_FS": fs["xi_trap"], "id_FS": fs["id_status"],
            "xi_naive": nv["xi_trap"], "id_naive": nv["id_status"],
            "n_pairs_naive": nv["n_pairs"]}
    rec["T3_selection"] = rec["xi_FS"] - rec["xi_F"]
    rec["T3_event"] = rec["xi_0"] - rec["xi_FS"]
    rec["T2_vs_FS"] = rec["xi_1"] - rec["xi_FS"]
    rec["T_naive"] = rec["xi_naive"] - rec["xi_F"]
    proj = []
    if r["v1fit"] is None:
        rec["projection_status"] = "no_V1_fit"
    else:
        proj = project(r["v1fit"], r["comp"], guards)
        rec["projection_status"] = "evaluated"
        rec["zeroth_order"] = zeroth_order(r["v1fit"], r["comp"])
    for p in proj:
        p["mask_id"] = mask_row["mask_id"]
        p["point_guard"] = ((p["support_lo_pct"], p["support_hi_pct"]), p["floor_factor"]) == (
            tuple(point_guard[0]), point_guard[1])
        p["error"] = rec["T1_obs"] - p["pred"] if np.isfinite(p["pred"]) else np.nan
        p["xi0_in_support"] = bool(np.isfinite(p["guard_lo"])
                                   and p["guard_lo"] <= rec["xi_0"] <= p["guard_hi"])
        p["open_in_support"] = bool(np.isfinite(p["proj_open"]) and np.isfinite(p["guard_lo"])
                                    and p["guard_lo"] <= p["proj_open"] <= p["guard_hi"])
    ok = [p for p in proj if p["status"] == "ok" and np.isfinite(p["pred"])]
    pt = [p for p in proj if p["point_guard"]]
    rec["pred_point"] = pt[0]["pred"] if pt and pt[0]["status"] == "ok" else np.nan
    rec["pred_point_status"] = pt[0]["status"] if pt else rec["projection_status"]
    rec["n_guards"] = len(proj)
    rec["n_guards_defined"] = len(ok)
    rec["range_lo"] = min(p["pred"] for p in ok) if ok else np.nan
    rec["range_hi"] = max(p["pred"] for p in ok) if ok else np.nan
    warm = {"V0": e0["params"], "V1": e1["params"]}
    return rec, proj, warm


_G: dict = {}


def _draw_worker(b: int) -> list:
    sysm, plan, vis, warm, warmF = _G["system"], _G["plan"], _G["vis"], _G["warm"], _G["warmF"]
    d = plan[b]
    f = sysm.impl.run(None, draw=d, warm={"V0": warmF}, need_v1=False, collect=False)["V0"]
    rows = []
    for mid, v in vis.items():
        r = sysm.impl.run(v, draw=d, warm=warm[mid], need_v1=True, collect=False)
        rows.append({"draw": b, "mask_id": mid, "xi_F": f["xi_trap"], "id_F": f["id_status"],
                     "xi_0": r["V0"]["xi_trap"], "id_0": r["V0"]["id_status"],
                     "xi_1": r["V1"]["xi_trap"], "id_1": r["V1"]["id_status"]})
    return rows


def bootstrap(system: System, plan: list, vis: dict, warm: dict, warmF, workers: int = 1
              ) -> pd.DataFrame:
    _G.update(system=system, plan=plan, vis=vis, warm=warm, warmF=warmF)
    idx = list(range(len(plan)))
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            parts = pool.map(_draw_worker, idx, chunksize=max(1, len(idx) // (workers * 4)))
    else:
        parts = [_draw_worker(b) for b in idx]
    return pd.DataFrame([r for p in parts for r in p])


def mask_intervals(draws: pd.DataFrame) -> pd.DataFrame:
    """Per-mask CIs (run_k2br.ci: tol0 primary, tol5 / literal reported)."""
    out = []
    for mid, d in draws.groupby("mask_id", sort=True):
        d = d.sort_values("draw")
        rec = {"mask_id": mid, "n_draws": len(d)}
        series = {"xi_0": d["xi_0"], "xi_1": d["xi_1"], "xi_F": d["xi_F"],
                  "T1_obs": d["xi_0"] - d["xi_1"], "T2": d["xi_1"] - d["xi_F"],
                  "T3": d["xi_0"] - d["xi_F"]}
        for k, v in series.items():
            c = KBR.ci(v.to_numpy(float))
            rec |= {f"{k}_ci_lo": c["ci_lo"], f"{k}_ci_hi": c["ci_hi"],
                    f"{k}_ci_status": c["ci_status"], f"{k}_n_nonfinite": c["n_nonfinite"],
                    f"{k}_tol5_lo": c["tol5_lo"], f"{k}_tol5_hi": c["tol5_hi"]}
        for k in ("id_F", "id_0", "id_1"):
            rec[f"{k}_unidentified_draws"] = int((d[k] != IDENTIFIED).sum())
            rec[f"{k}_unidentified_share"] = float((d[k] != IDENTIFIED).mean()) if len(d) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def _ols(pred: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    if len(pred) < 3 or np.var(pred) == 0:
        return np.nan, np.nan
    b = float(np.cov(pred, obs, ddof=0)[0, 1] / np.var(pred))
    return b, float(np.mean(obs) - b * np.mean(pred))


def _half(lo, hi):
    return (np.asarray(hi, float) - np.asarray(lo, float)) / 2.0


def eval_frame(pts: pd.DataFrame, crit: dict = CRITERIA) -> pd.DataFrame:
    """Per-mask evaluation columns (stratum, availability, gate state, gates,
    errors, coverage, references).  A gate needs a finite tol0 interval; a
    mask without one is 'gate_unavailable' (counted, never silently dropped)."""
    d = pts.copy()
    if "stratum" not in d:
        d["stratum"] = [stratum_of(p, t) for p, t in zip(d["position"], d["timing"])]
    g = crit["gate_factor"]
    ident = (d["id_0"] == IDENTIFIED) & (d["id_1"] == IDENTIFIED)
    d["avail_point"] = ident & np.isfinite(d["pred_point"]) & np.isfinite(d["T1_obs"])
    d["avail_range"] = ident & (d["n_guards_defined"] > 0) & np.isfinite(d["T1_obs"])
    d["gate_T1"] = g * _half(d["T1_obs_ci_lo"], d["T1_obs_ci_hi"])
    d["gate_T3"] = g * _half(d["T3_ci_lo"], d["T3_ci_hi"])
    u1, u3 = np.isfinite(d["gate_T1"]), np.isfinite(d["gate_T3"])
    d["gate_T1_state"] = np.where(u1, "usable", "gate_unavailable")
    d["gate_T3_state"] = np.where(u3, "usable", "gate_unavailable")
    d["gated_T1"] = d["avail_point"] & u1 & (np.abs(d["T1_obs"]) > d["gate_T1"])
    d["gated_T1_range"] = d["avail_range"] & u1 & (np.abs(d["T1_obs"]) > d["gate_T1"])
    d["abs_err_point"] = np.abs(d["T1_obs"] - d["pred_point"])
    d["sign_agree"] = np.sign(d["T1_obs"]) == np.sign(d["pred_point"])
    d["in_range"] = (d["T1_obs"] >= d["range_lo"]) & (d["T1_obs"] <= d["range_hi"])
    d["ci_meets_range"] = (d["T1_obs_ci_hi"] >= d["range_lo"]) & (d["T1_obs_ci_lo"] <= d["range_hi"])
    d["T1_ci_has_0"] = (d["T1_obs_ci_lo"] <= 0) & (d["T1_obs_ci_hi"] >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["range_width_rel"] = (d["range_hi"] - d["range_lo"]) / np.abs(d["T1_obs"])
    d["avail_recovery"] = ident & (d["id_F"] == IDENTIFIED) & np.isfinite(d["T2"]) & np.isfinite(d["T3"])
    d["gated_T3"] = d["avail_recovery"] & u3 & (np.abs(d["T3"]) > d["gate_T3"])
    d["v1_closer"] = np.abs(d["T2"]) < np.abs(d["T3"])
    d["T2_ci_has_0"] = (d["T2_ci_lo"] <= 0) & (d["T2_ci_hi"] >= 0)
    d["avail_naive"] = (d["id_naive"] == IDENTIFIED) & np.isfinite(d["T_naive"])
    d["v1_beats_naive"] = np.abs(d["T2"]) < np.abs(d["T_naive"])
    d["avail_FS"] = (d["id_FS"] == IDENTIFIED) & np.isfinite(d["xi_FS"])
    return d


def _med(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def _share(x) -> float:
    x = np.asarray(x, bool)
    return float(x.mean()) if len(x) else np.nan


def _metrics(d: pd.DataFrame, crit: dict | None = None) -> dict:
    del crit
    a = d[d["avail_point"]]
    s = _med(np.abs(a["T1_obs"]))
    b, c0 = _ols(a["pred_point"].to_numpy(float), a["T1_obs"].to_numpy(float)) if len(a) else (np.nan, np.nan)
    gt = d[d["gated_T1"]]
    ar = d[d["avail_range"]]
    gr = d[d["gated_T1_range"]]
    rv = d[d["avail_recovery"]]
    g3 = d[d["gated_T3"]]
    g3n = g3[g3["avail_naive"]]
    g3s = g3[g3["avail_FS"]]
    a_un1 = int((a["gate_T1_state"] != "usable").sum())
    r_un3 = int((rv["gate_T3_state"] != "usable").sum())
    m = {"n_masks": len(d), "n_point_available": len(a),
         "point_availability": _share(d["avail_point"]),
         "range_availability": _share(d["avail_range"]),
         "all_guards_defined_share": _share(d["n_guards_defined"] == d["n_guards"]),
         "calibration_slope": b, "calibration_intercept": c0, "median_abs_obs": s,
         "median_abs_pred_point": _med(np.abs(a["pred_point"])),
         "intercept_over_s": c0 / s if s and np.isfinite(s) else np.nan,
         "median_abs_err_point": _med(a["abs_err_point"]),
         "p90_abs_err_point": float(np.percentile(a["abs_err_point"], 90)) if len(a) else np.nan,
         "n_gate_T1_unavailable": a_un1,
         "gate_T1_availability": 1.0 - a_un1 / len(a) if len(a) else np.nan,
         "n_gated_T1": len(gt),
         "sign_agreement_gated": _share(gt["sign_agree"]) if len(gt) else np.nan,
         "sign_agreement_gate_unavailable_as_fail": (float(gt["sign_agree"].sum()) / (len(gt) + a_un1)
                                                     if len(gt) + a_un1 else np.nan),
         "sign_agreement_all": _share(a["sign_agree"]) if len(a) else np.nan,
         "T1_ci_contains_0_share": _share(a["T1_ci_has_0"]) if len(a) else np.nan,
         "range_coverage": _share(ar["in_range"]) if len(ar) else np.nan,
         "range_ci_meets": _share(ar["ci_meets_range"]) if len(ar) else np.nan,
         "range_width_rel_median_gated": _med(gr["range_width_rel"]),
         "n_recovery_available": len(rv),
         "n_gate_T3_unavailable": r_un3,
         "gate_T3_availability": 1.0 - r_un3 / len(rv) if len(rv) else np.nan,
         "n_gated_T3": len(g3),
         "recovery_share_v1_closer": _share(g3["v1_closer"]) if len(g3) else np.nan,
         "recovery_share_gate_unavailable_as_fail": (float(g3["v1_closer"].sum()) / (len(g3) + r_un3)
                                                     if len(g3) + r_un3 else np.nan),
         "median_abs_T2": _med(np.abs(g3["T2"])),
         "median_abs_T3": _med(np.abs(g3["T3"])),
         "n_gated_T3_naive": len(g3n),
         "recovery_share_v1_beats_naive": _share(g3n["v1_beats_naive"]) if len(g3n) else np.nan,
         "median_abs_T_naive": _med(np.abs(g3n["T_naive"])),
         "n_gated_T3_FS": len(g3s),
         "median_abs_T3_selection": _med(np.abs(g3s["T3_selection"])),
         "median_abs_T3_event": _med(np.abs(g3s["T3_event"])),
         "median_abs_T2_vs_FS": _med(np.abs(g3s["T2_vs_FS"])),
         "T2_ci_contains_0_share": _share(rv["T2_ci_has_0"]) if len(rv) else np.nan,
         "id_0_unidentified_share": _share(d["id_0"] != IDENTIFIED),
         "id_1_unidentified_share": _share(d["id_1"] != IDENTIFIED),
         "id_naive_unidentified_share": _share(d["id_naive"] != IDENTIFIED),
         "id_FS_unidentified_share": _share(d["id_FS"] != IDENTIFIED),
         "point_projection_undefined_share": _share(~np.isfinite(d["pred_point"]))}
    for k in ("F", "0", "1"):
        col = f"id_{k}_unidentified_share"
        if col in d:
            m[f"draws_{col}_median"] = _med(d[col])
            m[f"draws_{col}_max"] = float(np.nanmax(d[col])) if d[col].notna().any() else np.nan
    for k in ("T1_obs", "T3"):
        col = f"{k}_n_nonfinite"
        if col in d:
            m[f"draws_{k}_nonfinite_max"] = float(np.nanmax(d[col])) if d[col].notna().any() else np.nan
    m["recovery_median_ratio"] = (m["median_abs_T2"] / m["median_abs_T3"]
                                  if m["median_abs_T3"] else np.nan)
    for fl in FRACTIONS:
        al = a[a["fraction_level"] == fl]
        m[f"level_{fl}_n"] = len(al)
        m[f"level_{fl}_median_abs_err"] = _med(al["abs_err_point"])
        m[f"level_{fl}_median_abs_obs"] = _med(np.abs(al["T1_obs"]))
        dl = d[d["fraction_level"] == fl]
        m[f"level_{fl}_median_abs_T3"] = _med(np.abs(dl["T3"]))
        m[f"level_{fl}_median_halfwidth_T3"] = _med(_half(dl["T3_ci_lo"], dl["T3_ci_hi"]))
    for pos in ("low", "high"):
        pc = d[(d["fraction_level"] == "china_like") & (d["position"] == pos)
               & (d["timing"] == "exit")]
        m[f"pc_{pos}_n"] = len(pc)
        m[f"pc_{pos}_median_T3"] = _med(pc["T3"])
        m[f"pc_{pos}_median_halfwidth_T3"] = _med(_half(pc["T3_ci_lo"], pc["T3_ci_hi"]))
    return m


CI_METRICS = ("calibration_slope", "calibration_intercept", "intercept_over_s",
              "sign_agreement_gated", "range_coverage", "recovery_share_v1_closer",
              "recovery_median_ratio", "median_abs_err_point", "recovery_share_v1_beats_naive")


def cluster_ci(d: pd.DataFrame, crit: dict, ss: np.random.SeedSequence, R: int = CLUSTER_R) -> dict:
    """Percentile intervals (K1 symmetric order statistics) of CI_METRICS
    under resampling of mask CELLS with replacement (masks within a cell are
    not independent; neither are cells -- the panel is one system)."""
    rng = np.random.default_rng(ss)
    cells = np.array(sorted(d["cell"].unique()))
    groups = {c: d[d["cell"] == c] for c in cells}
    vals = {k: [] for k in CI_METRICS}
    for _ in range(R):
        pick = rng.choice(cells, size=len(cells), replace=True)
        sub = pd.concat([groups[c] for c in pick], ignore_index=True)
        m = _metrics(sub, crit)
        for k in CI_METRICS:
            vals[k].append(m[k])
    out = {}
    for k, v in vals.items():
        v = np.asarray(v, float)
        fin = v[np.isfinite(v)]
        lo, hi = K2.order_ci(fin) if len(fin) else (np.nan, np.nan)
        out[k] = (lo, hi, int(len(v) - len(fin)))
    return out


def _inside(ci: tuple, lo: float, hi: float) -> bool:
    return bool(np.isfinite(ci[0]) and np.isfinite(ci[1]) and ci[0] >= lo and ci[1] <= hi)


def verdicts(m: dict, ci: dict, crit: dict = CRITERIA, null_m: dict | None = None,
             controls: bool = True) -> dict:
    v = {}
    gate1 = bool(np.isfinite(m["gate_T1_availability"])
                 and m["gate_T1_availability"] >= crit["gate_availability_min"])
    gate3 = bool(np.isfinite(m["gate_T3_availability"])
                 and m["gate_T3_availability"] >= crit["gate_availability_min"])
    c = {"P0_availability": bool(np.isfinite(m["point_availability"])
                                 and m["point_availability"] >= crit["point_availability_min"]),
         "P1_slope": _inside(ci["calibration_slope"], *crit["slope_band"]),
         "P2_intercept": _inside(ci["intercept_over_s"], -crit["intercept_band_rel"],
                                 crit["intercept_band_rel"]),
         "P3_sign": bool(np.isfinite(m["sign_agreement_gated"])
                         and m["sign_agreement_gated"] >= crit["sign_agreement_min"])}
    lev = []
    for fl in FRACTIONS:
        if m[f"level_{fl}_n"] >= crit["level_min_masks"]:
            lev.append(m[f"level_{fl}_median_abs_err"]
                       <= crit["level_abs_error_rel"] * m[f"level_{fl}_median_abs_obs"])
    c["P4_levels"] = bool(lev) and all(lev)
    v |= {f"T1_point_{k}": bool(x) for k, x in c.items()}
    v["T1_gate_availability"] = gate1
    v["T1_point"] = ("not_evaluable_availability" if not c["P0_availability"] else
                     "not_evaluable_gate_availability" if not gate1 else
                     "pass" if all(c.values()) else "fail")
    r0 = bool(np.isfinite(m["range_availability"])
              and m["range_availability"] >= crit["range_availability_min"])
    r1 = bool(np.isfinite(m["range_coverage"]) and m["range_coverage"] >= crit["range_coverage_min"])
    r2 = bool(np.isfinite(m["range_width_rel_median_gated"])
              and m["range_width_rel_median_gated"] <= crit["range_width_rel_max"])
    v |= {"T1_range_R0_availability": r0, "T1_range_R1_coverage": r1,
          "T1_range_R2_width": r2}
    v["T1_range"] = ("not_evaluable_availability" if not r0 else
                     "not_evaluable_gate_availability" if not gate1 else
                     "pass" if (r1 and r2) else "covers_but_uninformative" if r1 else "fail")
    q1 = bool(np.isfinite(m["recovery_share_v1_closer"])
              and m["recovery_share_v1_closer"] >= crit["recovery_share_min"]
              and np.isfinite(ci["recovery_share_v1_closer"][0])
              and ci["recovery_share_v1_closer"][0] > crit["recovery_share_ci_lo_min"])
    q2 = bool(np.isfinite(m["recovery_median_ratio"])
              and m["recovery_median_ratio"] <= crit["recovery_median_ratio_max"])
    q3 = bool(m["n_gated_T3_naive"] > 0
              and np.isfinite(ci["recovery_share_v1_beats_naive"][0])
              and ci["recovery_share_v1_beats_naive"][0] > crit["comparator_share_ci_lo_min"])
    v |= {"T2_V1_share": q1, "T2_V2_ratio": q2, "T2_V3_beats_naive": q3,
          "T2_gate_availability": gate3}
    v["T2"] = ("not_evaluable_no_gated_mask" if m["n_gated_T3"] == 0 else
               "not_evaluable_gate_availability" if not gate3 else
               ("recovers" if q3 else "closer_than_V0_only") if (q1 and q2) else "fail")
    if not controls:
        return v
    pf = crit["power_factor"]
    pc = {}
    for pos, sgn in (("low", -1.0), ("high", 1.0)):
        t3, hw = m[f"pc_{pos}_median_T3"], m[f"pc_{pos}_median_halfwidth_T3"]
        pc[pos] = bool(np.isfinite(t3) and np.isfinite(hw) and sgn * t3 > pf * hw)
        v[f"T3_powered_{pos}"] = pc[pos]
    v["T3_powered"] = pc["low"] and pc["high"]
    if null_m is None or not null_m.get("n_point_available"):
        v["N_null_control"] = "not_evaluable"
    else:
        ref = crit["null_rel_max"] * m["median_abs_obs"]
        n1 = bool(np.isfinite(ref) and null_m["median_abs_obs"] <= ref)
        n2 = bool(np.isfinite(ref) and null_m["median_abs_pred_point"] <= ref)
        v |= {"N1_null_obs": n1, "N2_null_pred": n2}
        v["N_null_control"] = "pass" if (n1 and n2) else "fail"
    return v


def evaluate(pts: pd.DataFrame, entropy: int, stage: str, crit: dict = CRITERIA,
             R: int = CLUSTER_R, crit_sets: dict | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    crit_sets = crit_sets or {"provisional": crit}
    rows, cells = [], []
    for sname, d0 in pts.groupby("system", sort=False):
        d = eval_frame(d0, crit)
        ms, cis = {}, {}
        for sc in EVAL_SCOPES:
            dd = d if sc == "pooled" else d[d["stratum"] == sc]
            if not len(dd):
                continue
            ms[sc] = _metrics(dd, crit)
            cis[sc] = cluster_ci(dd, crit, cluster_seedseq(entropy, sname, stage), R)
        base = {"system": sname, "stage": stage}
        for sc, m in ms.items():
            role = "primary" if sc == PRIMARY_STRATUM else "report_only"
            for k, val in m.items():
                lo, hi, nf = cis[sc].get(k, (np.nan, np.nan, 0))
                rows.append(base | {"criteria_set": "", "scope": sc, "role": role, "kind": "metric",
                                    "name": k, "value": val, "cluster_ci_lo": lo,
                                    "cluster_ci_hi": hi, "cluster_nonfinite": nf})
        for label, cr in crit_sets.items():
            for sc, m in ms.items():
                prim = sc == PRIMARY_STRATUM
                v = verdicts(m, cis[sc], cr, ms.get("null"), controls=prim)
                for k, val in v.items():
                    rows.append(base | {"criteria_set": label, "scope": sc,
                                        "role": "primary" if prim else "report_only",
                                        "kind": "verdict", "name": k, "value": val,
                                        "cluster_ci_lo": np.nan, "cluster_ci_hi": np.nan,
                                        "cluster_nonfinite": 0})
        for c, dc in d.groupby("cell", sort=True):
            mc = _metrics(dc, crit)
            cells.append({"system": sname, "stage": stage, "cell": c,
                          "stratum": dc["stratum"].iloc[0],
                          **{k: mc[k] for k in ("n_masks", "point_availability", "median_abs_obs",
                                                "median_abs_err_point", "sign_agreement_all",
                                                "range_coverage", "recovery_share_v1_closer",
                                                "median_abs_T2", "median_abs_T3",
                                                "n_gate_T1_unavailable", "n_gate_T3_unavailable",
                                                "recovery_share_v1_beats_naive")},
                          "mean_T1_obs": float(dc["T1_obs"].mean()),
                          "mean_pred_point": float(dc["pred_point"].mean()),
                          "mean_T2": float(dc["T2"].mean()), "mean_T3": float(dc["T3"].mean()),
                          "mean_T3_selection": float(dc["T3_selection"].mean()),
                          "mean_T3_event": float(dc["T3_event"].mean()),
                          "mean_T_naive": float(dc["T_naive"].mean()),
                          "mean_M_bar": float(dc["M_bar"].mean()),
                          "mean_delta_event": float(dc["delta_event"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(cells)


def pilot_diagnostics(pts: pd.DataFrame, proj: pd.DataFrame, crit: dict = CRITERIA) -> pd.DataFrame:
    rel = crit["level_abs_error_rel"]
    out = []
    for r in pts.itertuples(index=False):
        pr = proj[proj["mask_id"] == r.mask_id]
        ok = pr[(pr["status"] == "ok") & np.isfinite(pr["pred"])]
        reg = pr[pr["registered_guard"]]
        tol = rel * abs(r.T1_obs) if np.isfinite(r.T1_obs) else np.nan
        pt_ok = bool(np.isfinite(r.pred_point) and abs(r.T1_obs - r.pred_point) <= tol)
        any_ok = bool(len(ok) and (np.abs(r.T1_obs - ok["pred"]) <= tol).any())
        same_side = bool(len(ok) and (np.all(ok["error"] > 0) or np.all(ok["error"] < 0)))
        xi0_in = bool(len(reg) and reg["xi0_in_support"].iloc[0])
        open_in = bool(len(reg) and reg["open_in_support"].iloc[0])
        if pt_ok:
            cls = "point_ok"
        elif not len(ok):
            cls = "undefined"
        elif any_ok or not xi0_in or not open_in:
            cls = "guard_or_tail"
        elif same_side:
            cls = "fundamental"
        else:
            cls = "mixed"
        out.append({"mask_id": r.mask_id, "cell": r.cell, "T1_obs": r.T1_obs,
                    "pred_point": r.pred_point, "pred_point_status": r.pred_point_status,
                    "range_lo": r.range_lo, "range_hi": r.range_hi,
                    "n_guards_defined": r.n_guards_defined,
                    "zeroth_order": getattr(r, "zeroth_order", np.nan),
                    "point_within_tol": pt_ok, "any_guard_within_tol": any_ok,
                    "all_defined_same_side": same_side, "xi0_in_registered_support": xi0_in,
                    "open_proj_in_registered_support": open_in, "class": cls})
    return pd.DataFrame(out)


def build_masks(system: System, stage: str, entropy: int, cells: list,
                exclude: set | frozenset = frozenset()) -> tuple[pd.DataFrame, dict]:
    rows, vis = [], {}
    for c in cells:
        m, att, status = construct_mask(system.name, stage, entropy, c, system.xi_base,
                                        system.geo, system.labels, exclude)
        mid = f"{system.name}|{stage}|{c['cell']}|r{c['rep']}"
        rec = {"system": system.name, "stage": stage, "mask_id": mid, **c, "attempt": att,
               "mask_status": status}
        if m is not None:
            rec |= {"n_units": len(m), "n_masked": int(m.sum()), "f_achieved": float(m.mean()),
                    "delta_event_target": c["fraction"] / (1 - c["fraction"]),
                    "mean_xi_base_masked": float(system.xi_base[m].mean()),
                    "n_geo_blocks_masked": int(len(np.unique(system.geo[m]))),
                    "mask_hash": mask_hash(system.name, c["timing"], system.labels, m),
                    "units": json.dumps(sorted(map(str, system.labels[m]))),
                    "geo_blocks": json.dumps(sorted(map(str, np.unique(system.geo_names[system.geo[m]]))))}
            vis[mid] = visibility(m, system.n_years, c["timing"])
        rows.append(rec)
    return pd.DataFrame(rows), vis


def preflight_system(system: System, reference_pairs=None) -> dict:
    """Kill rules 2-4 of one system (no mask, no seed): full-panel
    reproduction of the committed Panel A row and the no-mask equivalence."""
    out = {"reproduction": [], "equivalence": []}
    full_run = system.impl.run(None, need_v1=True, collect=True)
    full = full_run["V0"]
    if system.committed:
        out["reproduction"] = reproduction_rows(system, full, system.committed)
        if not all(r["pass"] for r in out["reproduction"]):
            out["killed"] = KillRule(2, f"{system.name} full panel does not reproduce the "
                                        "committed Panel A estimate")
            return out
    out["equivalence"] = equivalence_rows(system, full_run, reference_pairs)
    if not all(r["pass"] for r in out["equivalence"]):
        out["killed"] = KillRule(4, f"{system.name} no-mask equivalence failed")
        return out
    out["full_run"] = full_run
    return out


def run_system(system: System, stage: str, entropy: int, cells: list, n_boot: int,
               workers: int = 1, guards=GUARD_SET, point_guard=REGISTERED_GUARD,
               exclude: set | frozenset = frozenset(), reference_pairs=None,
               log=print) -> dict:
    """Kill rules 2-4 for one system, then masks, point records, bootstrap."""
    out = preflight_system(system, reference_pairs)
    if "killed" in out:
        return out
    full = out.pop("full_run")["V0"]
    masks, vis = build_masks(system, stage, entropy, cells, exclude)
    out["masks"] = masks
    log(f"[{system.name}] {len(vis)} masks of {len(masks)} constructed")
    pts, proj, warm = [], [], {}
    t0 = time.time()
    for rec in masks.to_dict("records"):
        if rec["mask_id"] not in vis:
            pts.append(rec)
            continue
        p, pr, w = point_record(system, rec, vis[rec["mask_id"]], full, guards, point_guard)
        pts.append(p)
        proj += pr
        warm[rec["mask_id"]] = w
    log(f"[{system.name}] point estimates and projections ({time.time() - t0:.0f}s)")
    plan = system.impl.draw_plan(boot_seedseq(entropy, system.name, stage), n_boot)
    t0 = time.time()
    draws = bootstrap(system, plan, vis, warm, full["params"], workers)
    log(f"[{system.name}] bootstrap {n_boot} draws ({time.time() - t0:.0f}s)")
    ints = mask_intervals(draws) if len(draws) else pd.DataFrame({"mask_id": []})
    pts = pd.DataFrame(pts).merge(ints, on="mask_id", how="left")
    out |= {"points": pts, "projections": pd.DataFrame(proj), "draws": draws, "full": full}
    return out


def write(tables: dict, out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        if df is None:
            continue
        fn = f"k4_{name}.csv.gz" if name == "draws" else f"k4_{name}.csv"
        df.to_csv(out / fn, index=False)
        written.append(fn)
    return written


def run_stage(systems: list, stage: str, entropy: int, out: Path, n_boot: int, workers: int,
              cells_fn=stage_cells, guards=GUARD_SET, point_guard=REGISTERED_GUARD,
              crit: dict = CRITERIA, exclude: dict | None = None, reference_pairs: dict | None = None,
              R: int = CLUSTER_R, log=print, crit_sets: dict | None = None) -> tuple[str, list]:
    tabs = {k: [] for k in ("reproduction", "equivalence", "masks", "points", "projections",
                            "draws")}
    status = "completed"
    for s in systems:
        res = run_system(s, stage, entropy, cells_fn(stage), n_boot, workers, guards,
                         point_guard, (exclude or {}).get(s.name, frozenset()),
                         (reference_pairs or {}).get(s.name), log)
        for k in tabs:
            v = res.get(k)
            if v is None:
                continue
            tabs[k].append(pd.DataFrame(v) if isinstance(v, list) else v)
        if "killed" in res:
            status = f"killed: {res['killed']}"
            break
    frames = {k: (pd.concat(v, ignore_index=True) if v else None) for k, v in tabs.items()}
    if status == "completed" and frames["points"] is not None:
        summ, cells = evaluate(frames["points"], entropy, stage, crit, R, crit_sets)
        frames["evaluation"], frames["cells"] = summ, cells
        if stage in ("pilot", "dev") and frames["projections"] is not None and len(frames["projections"]):
            frames["pilot_diagnostics"] = pilot_diagnostics(frames["points"], frames["projections"], crit)
    return status, write(frames, out)


def synthetic_arrays(n: int = 300, years=range(2003, 2025), a1: float = -0.05,
                     n_geo: int = 20, geo_sd: float = 0.8, seed: int = 0, beta: float = 1.07,
                     cubic: float = 0.0):
    """Balanced toy panel: x = geo effect + persistent AR(1) (optionally with
    a cubic drift term), log Y = 1 + beta log N + x."""
    rng = np.random.default_rng(seed)
    years = list(years)
    T = len(years)
    geo = rng.integers(0, n_geo, n)
    ge = rng.normal(0, geo_sd, n_geo)
    z = np.zeros((n, T))
    z[:, 0] = rng.normal(0, 1, n)
    s = np.sqrt(1 - (1 + a1) ** 2)
    for j in range(1, T):
        zp = z[:, j - 1]
        z[:, j] = zp + a1 * zp - cubic * zp ** 3 + rng.normal(0, s, n)
    x = ge[geo][:, None] + z
    lnN = rng.normal(12.0, 1.0, n)[:, None] + 0.01 * np.arange(T)[None, :]
    lnY = 1.0 + beta * lnN + 0.3 * x
    return {"years": years, "geo": geo, "logN": lnN, "logY": lnY,
            "codes": [f"u{i:04d}" for i in range(n)]}


def synthetic_us_system(lin_size: int = 1, **kw) -> System:
    a = synthetic_arrays(**kw)
    n = len(a["codes"])
    lin = np.arange(n) // lin_size
    impl = USImpl(a["codes"], lin, a["years"], a["years"][:-1], a["logN"], a["logY"],
                  int(lin.max()) + 1)
    xi = K2.sami(a["logN"][:, 0], a["logY"][:, 0], np.ones(n))[0]
    names, geo = np.unique(a["geo"], return_inverse=True)
    return System("US", impl, np.array(a["codes"]), geo, names.astype(str), xi, a["years"][0],
                  len(a["years"]))


def synthetic_br_system(**kw) -> System:
    a = synthetic_arrays(**({"years": range(2002, 2024)} | kw))
    n, T = a["logN"].shape
    years = np.array(a["years"])
    pA = KBR.Panel("A", years, np.exp(a["logY"]), np.exp(a["logN"]), np.ones((n, T), bool),
                   np.zeros((0, 2), int), d1_dropped_by_year=np.zeros(T, int),
                   lag_amc_years=np.zeros(T, int))
    cfg = KBR.Config(n_min=10)
    logY, logN = KBR._logs(pA)
    xi = KBR.standardise(logY[:, 0], logN[:, 0], pA.present[:, 0], cfg.ddof, cfg.min_reg_n)[0]
    names, geo = np.unique(a["geo"], return_inverse=True)
    return System("BR", BRImpl(pA, cfg), np.array(a["codes"]), geo, names.astype(str), xi,
                  int(years[0]), T)


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()


def plan_section(plan_path: Path = PLAN) -> str:
    plan = Path(plan_path).read_text(encoding="utf-8")
    i = plan.find(SECTION_HEADING)
    if i < 0:
        sys.exit(f"refusing: no '{SECTION_HEADING}' section in the registration record")
    j = plan.find("\n## ", i + len(SECTION_HEADING))
    return plan[i:] if j < 0 else plan[i:j]


def check_registration(plan_path: Path = PLAN, code: str | None = None) -> str:
    if code is None:
        code = _git("log", "-1", "--format=%H", "--", SCRIPT_REL)
    sec = plan_section(plan_path)
    if not code or code[:7] not in sec:
        sys.exit(f"refusing: the registration does not name the code commit {code[:7]}")
    changed = _git("diff", "--name-only", code, "HEAD", "--", *DEPENDENCIES)
    if changed:
        sys.exit(f"refusing: dependencies changed since the code commit: {changed.split()}")
    return code


def check_freeze(path: Path = FREEZE) -> tuple[dict, str]:
    if not path.exists():
        sys.exit("refusing: no S2 freeze file (results/records/k4_mask_freeze.json)")
    commits = _git("log", "--format=%H", "--", FREEZE_REL).split()
    if len(commits) != 1:
        sys.exit(f"refusing: the freeze file must be committed exactly once ({len(commits)})")
    if _git("status", "--porcelain", "--", FREEZE_REL):
        sys.exit("refusing: the freeze file differs from its commit")
    fz = json.loads(path.read_text())
    return validate_freeze(fz), commits[0]


def tighten_ok(key: str, new) -> None:
    old, how = CRITERIA[key], CRITERIA_DIRECTION[key]
    if how == "narrow":
        lo, hi = (float(x) for x in new)
        if not (np.isfinite(lo) and np.isfinite(hi) and old[0] <= lo < hi <= old[1]):
            raise ValueError(f"freeze may only narrow {key} {old}: got {tuple(new)}")
        return
    if isinstance(new, (bool, str, tuple, list)) or not np.isfinite(float(new)):
        raise ValueError(f"freeze {key} must be a finite number")
    new = float(new)
    if how == "fixed" and new != float(old):
        raise ValueError(f"{key} is fixed at {old} by the registration")
    if how == "up" and new < old:
        raise ValueError(f"freeze may only raise {key} (>= {old}): got {new}")
    if how == "down" and new > old:
        raise ValueError(f"freeze may only lower {key} (<= {old}): got {new}")


def validate_freeze(fz: dict) -> dict:
    grid = {(tuple(s), f) for s, f in GUARD_SET}
    gs = [(tuple(g[0]), float(g[1])) for g in fz["guard_set"]]
    if not gs or any(g not in grid for g in gs):
        raise ValueError("freeze guard_set must be a non-empty subset of the 12 registered guards")
    pg = (tuple(fz["point_guard"][0]), float(fz["point_guard"][1]))
    if pg not in gs:
        raise ValueError("freeze point_guard must belong to guard_set")
    crit = dict(CRITERIA) | {k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in fz.get("criteria", {}).items()}
    if set(crit) != set(CRITERIA):
        raise ValueError(f"unknown criteria keys {sorted(set(crit) - set(CRITERIA))}")
    for k, new in crit.items():
        tighten_ok(k, new)
    if int(fz.get("n_boot", N_BOOT)) != N_BOOT or int(fz.get("n_rep", N_REP)) != N_REP:
        raise ValueError('n_boot / n_rep are registered and cannot change at the freeze')
    return {"guard_set": tuple(gs), "point_guard": pg, "criteria": crit}


def check_signoff(plan_path: Path = PLAN, freeze_commit: str | None = None) -> None:
    lines = [ln for ln in plan_section(plan_path).splitlines() if SIGNOFF_TOKEN in ln]
    if not lines:
        sys.exit('refusing: the registration has no author sign-off for the confirmatory stage')
    if freeze_commit is not None and not any(freeze_commit[:12] in ln for ln in lines):
        sys.exit(f"refusing: the S2 sign-off does not name the freeze commit {freeze_commit[:12]}")


def environment_fingerprint() -> dict:
    """What must match between a preflight and the logged run (same machine /
    node and the same numerical stack)."""
    import platform
    import scipy
    return {"hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS")}


def preflight_receipt_path(stage: str, pdir: Path | None = None) -> Path:
    return (pdir or PREFLIGHT_DIR) / f"k4_preflight_{stage}.json"


def check_preflight(stage: str, head: str, pdir: Path | None = None) -> dict:
    """A registered stage needs a passing preflight of the same stage at the
    same HEAD in the same environment."""
    f = preflight_receipt_path(stage, pdir)
    if not f.exists():
        sys.exit(f"refusing: no preflight receipt {f} (run --preflight {stage} first)")
    r = json.loads(f.read_text())
    if r.get("status") != "passed":
        sys.exit(f"refusing: the last {stage} preflight did not pass ({r.get('status')})")
    if r.get("head") != head:
        sys.exit("refusing: the preflight ran at another commit")
    if r.get("environment") != environment_fingerprint():
        sys.exit("refusing: the preflight ran in another environment")
    return r


def check_single_run(stage: str, log: Path = ATTEMPT_LOG) -> None:
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip() and json.loads(line).get("stage") == stage:
                sys.exit(f"refusing: {log.name} already holds a {stage} attempt (single run)")


def _log_attempt(rec: dict, log: Path = ATTEMPT_LOG) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def pilot_exclusions(tab: Path = OUT / "pilot") -> dict:
    f = tab / "k4_masks.csv"
    if not f.exists():
        sys.exit("refusing: the confirmatory stage needs the committed pilot mask table")
    d = pd.read_csv(f)
    d = d[d["mask_hash"].notna()]
    return {s: frozenset(g["mask_hash"]) for s, g in d.groupby("system")}


def load_real_systems(stage: str, us_data: Path, br_data: Path) -> tuple[list, dict]:
    systems, ref = [], {}
    world = K2.load_world(us_data, with_outcomes=True)
    us, sk = us_system_from_world(world)
    us.committed = committed_full("US")
    ref["US"] = K2.run_spec(sk)["pairs"]["V0"]
    systems.append(us)
    if stage == "confirmatory":
        pib = KBR._pib_module()
        outcomes = KBR.check_outcome_frame(pib.outcome_values(br_data))
        struct = KBR.load_structure(br_data)
        cfg = KBR.Config()
        m, regions = KBR.build_muni(struct, BR_WINDOW, outcomes, cfg)
        br = br_system_from_muni(m, regions, cfg)
        br.committed = committed_full("BR")
        res = KBR.analyse(m, cfg)
        ref["BR"] = res["panels"][(BR_OUTCOME, "A")]["pairs"]["V0"]
        systems.append(br)
    return systems, ref


def preflight_main(a) -> int:
    """--preflight STAGE: kill rules 1-4 on the real data of the stage (US;
    plus BR for confirmatory) -- no masks, no seeds, no attempt record, no
    output under the repository.  Writes the check tables and a receipt to
    PREFLIGHT_DIR (or --preflight-dir); the logged run requires a passing
    receipt at the same HEAD in the same environment."""
    stage = a.preflight
    pdir = (a.preflight_dir or PREFLIGHT_DIR).resolve()
    if pdir == ROOT.resolve() or ROOT.resolve() in pdir.parents:
        sys.exit("--preflight output never goes inside the repository")
    if _git("status", "--porcelain"):
        sys.exit("refusing: a preflight needs a clean tree")
    code = public_guard_skipped(_git('rev-parse', 'HEAD'))
    head = _git("rev-parse", "HEAD")
    out = pdir / f"{stage}_{head[:12]}_{time.strftime('%Y%m%dT%H%M%S')}"
    t0 = time.time()
    tables, status, err = {}, "passed", None
    us_data = a.us_data or K2.COV.default_data_dir()
    try:
        integ = integrity_rows("US", us_data)
        if stage == "confirmatory":
            integ += integrity_rows("BR", a.br_data)
        tables["integrity"] = pd.DataFrame(integ)
        if not tables["integrity"]["pass"].all():
            status = "killed: kill rule 1 (raw sha256)"
        else:
            systems, ref = load_real_systems(stage, us_data, a.br_data)
            rep_, eq_ = [], []
            for s_ in systems:
                r = preflight_system(s_, ref.get(s_.name))
                rep_ += r["reproduction"]
                eq_ += r["equivalence"]
                if "killed" in r:
                    status = f"killed: {r['killed']}"
                    break
            tables["reproduction"], tables["equivalence"] = pd.DataFrame(rep_), pd.DataFrame(eq_)
    except KillRule as e:
        status = f"killed: {e}"
    except Exception as e:
        status = f"crashed: {type(e).__name__}: {e}"
        err = traceback.format_exc()
    written = write(tables, out)
    rec = {"label": LABEL, "mode": "preflight", "stage": stage, "status": status, "head": head,
           "code_commit": code, "environment": environment_fingerprint(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
           "out": str(out), "written": written, "traceback": err}
    (out / "k4_preflight.json").write_text(json.dumps(rec, indent=1, default=str))
    preflight_receipt_path(stage, pdir).write_text(json.dumps(rec, indent=1, default=str))
    print(status)
    return 0 if status == "passed" else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=LABEL)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--confirmatory", action="store_true")
    g.add_argument("--dev", action="store_true", help="synthetic systems only")
    g.add_argument("--preflight", choices=("pilot", "confirmatory"), default=None,
                   help="kill rules 1-4 only: no masks, no seeds, no attempt record")
    ap.add_argument("--preflight-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="--dev only")
    ap.add_argument("--n-boot", type=int, default=8, help="--dev only")
    ap.add_argument("--n-rep", type=int, default=1, help="--dev only")
    ap.add_argument("--n-units", type=int, default=200, help="--dev only")
    ap.add_argument("--us-data", type=Path, default=None)
    ap.add_argument("--br-data", type=Path, default=KBR.DATA)
    a = ap.parse_args(argv)
    if a.preflight:
        return preflight_main(a)
    stage = "pilot" if a.pilot else "confirmatory" if a.confirmatory else "dev"
    if stage == "dev":
        if a.out is None:
            sys.exit("--dev needs --out")
        out = a.out.resolve()
        if out == TAB.resolve() or TAB.resolve() in out.parents:
            sys.exit("--dev output never goes to results/tables")
        systems = [synthetic_us_system(n=a.n_units, seed=1), synthetic_br_system(n=a.n_units, seed=2)]
        t0 = time.time()
        status, written = run_stage(systems, "dev", DEV_ENTROPY, out, a.n_boot, a.workers,
                                    cells_fn=lambda s: stage_cells("pilot", n_rep=a.n_rep),
                                    R=200)
        (out / "k4_meta.json").write_text(json.dumps(
            {"label": LABEL, "stage": "dev", "status": status, "synthetic": True,
             "n_boot": a.n_boot, "written": written, "seconds": round(time.time() - t0, 1)},
            indent=1))
        print(status)
        return 0 if status == "completed" else 3
    if a.out is not None:
        sys.exit("the registered stages write to results/tables/k4_mask/<stage>")
    out = OUT / stage
    if _git("status", "--porcelain"):
        sys.exit("refusing: a registered stage needs a clean tree")
    code = public_guard_skipped(_git('rev-parse', 'HEAD'))
    guards, point_guard, crit, freeze_commit, exclude = GUARD_SET, REGISTERED_GUARD, CRITERIA, None, {}
    crit_sets = {"provisional": CRITERIA}
    freeze_time = None
    if stage == "confirmatory":
        fz, freeze_commit = public_guard_skipped(((lambda m: validate_freeze({'guard_set': m['guards'], 'point_guard': m['point_guard'], 'criteria': m['criteria'], 'n_boot': m['n_boot'], 'n_rep': m['n_rep']}))(json.loads((OUT / 'confirmatory' / 'k4_meta.json').read_text())), None))
        public_guard_skipped(None)
        guards, point_guard, crit = fz["guard_set"], fz["point_guard"], fz["criteria"]
        crit_sets = {"frozen": crit, "provisional": CRITERIA}
        exclude = pilot_exclusions()
        freeze_time = public_guard_skipped(None)
    check_single_run(stage)
    head = _git("rev-parse", "HEAD")
    preflight = check_preflight(stage, head, a.preflight_dir)
    entropy = public_guard_skipped({'pilot': PILOT_ENTROPY, 'confirmatory': int(json.loads((OUT / 'confirmatory' / 'k4_meta.json').read_text())['entropy'])}[stage])
    from src import k1_theory as T
    frozen = public_guard_skipped({'frozen_commit': None, 'environment_differs': None})
    t0 = time.time()
    _log_attempt({"stage": stage, "status": "started", "head": head, "code_commit": code,
                  "freeze_commit": freeze_commit, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    out.mkdir(parents=True, exist_ok=True)
    written, err = [], None
    us_data = a.us_data or K2.COV.default_data_dir()
    try:
        integ = integrity_rows("US", us_data)
        if stage == "confirmatory":
            integ += integrity_rows("BR", a.br_data)
        integ = pd.DataFrame(integ)
        written += write({"integrity": integ}, out)
        if not integ["pass"].all():
            status = "killed: kill rule 1 (raw sha256)"
        else:
            systems, ref = load_real_systems(stage, us_data, a.br_data)
            status, w = run_stage(systems, stage, entropy, out, N_BOOT, a.workers,
                                  guards=guards, point_guard=point_guard, crit=crit,
                                  exclude=exclude, reference_pairs=ref, crit_sets=crit_sets)
            written += w
    except KillRule as e:
        status = f"killed: {e}"
    except Exception as e:
        status = f"crashed: {type(e).__name__}: {e}"
        err = traceback.format_exc()
    meta = {"label": LABEL, "stage": stage, "status": status, "git": head, "code_commit": code,
            "freeze_commit": freeze_commit, "freeze_commit_time": freeze_time,
            "run_start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
            "preflight": {k: preflight.get(k) for k in ("time", "head", "environment")},
            "criteria_sets": list(crit_sets), "entropy": str(entropy), "n_boot": N_BOOT,
            "n_rep": 1 if stage == "pilot" else N_REP, "guards": [[list(s), f] for s, f in guards],
            "point_guard": [list(point_guard[0]), point_guard[1]], "criteria": crit,
            "cluster_R": CLUSTER_R, "workers": a.workers,
            "k1_freeze": {"frozen_commit": frozen.get("frozen_commit"),
                          "environment_differs": frozen.get("environment_differs")},
            "written": written, "traceback": err, "seconds": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "k4_meta.json").write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
    _log_attempt({"stage": stage, "status": status.split(":")[0], "detail": status, "head": head,
                  "code_commit": code, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(status)
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
