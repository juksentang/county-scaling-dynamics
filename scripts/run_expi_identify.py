"""POST-HOC, REPORT ONLY.  Run after every result was known; it changes no verdict, no registered interval and no committed
table.  Single run, every output reported.

Why: ``km_mle.derived_observables`` returns a finite ``xi_trap`` both when
the fitted drift has a stable zero on the data support and when it has none
(then the grid node with the smallest |A| is returned in its place); the
paired addendum (scripts/run_expi_paired.py) did not save which case each
draw was in.  This script rebuilds EXACTLY the addendum's 1000 draws (seed
20261023, ``default_rng([SEED, b])``, the supplement's window-stratified
province block, full pipeline per draw, warm start at the full-sample fit)
and records, per draw, version (V0, V1) and window (pre, during), whether
the attractor is identified.

Definition of "identified" (read off ``derived_observables``; replicated
exactly by ``root_status`` and checked against it on every fit):
    support  = [P1, P99] of the pairs' xi_t (np.percentile), grid of 600
               equally spaced nodes (n_grid default);
    A        = the fitted cubic drift on the grid;
    a stable zero = an index i with A[i] > 0 and A[i+1] < 0 (a downward
               crossing: a zero with negative slope strictly inside the
               support), located by linear interpolation;
    identified  <=> at least one stable zero ('stable_zero'); xi* is then
               the FIRST (lowest) stable zero;
    not identified <=> no stable zero ('fallback_argminA'); the returned
               xi* is the placeholder grid[argmin |A|] (both conventions,
               unless the path rule below makes the 'nan' value NaN).
A separate, path-level flag (not an identification failure): fewer than two
grid nodes on [min(xi*, 0), max(xi*, 0)] -- 'subgrid_path' under 'limit'
(value kept), 'degenerate_path' under 'nan' (all path observables NaN).
derived_observables overwrites the root status by that path status, which
is why the root status is recomputed here from the fitted parameters.

Outputs (results/tables/, or --out for --dev):
    expi_identify_draws.csv.gz     per draw: the addendum's columns (rebuilt)
                                   + root status, number of stable / unstable
                                   zeros, first stable zero, placeholder,
                                   path status per convention
    expi_identify_fractions.csv    per version x window: identified fraction
                                   and the other counts
    expi_identify_intervals.csv    primary intervals on all draws, on the
                                   identified draws only, on the others, with
                                   the share of excluded draws below / inside /
                                   above the all-draw interval
    expi_identify_conventions.csv  limit vs nan: interval and "contains the
                                   null value" under each rule, agreement flags
    expi_identify_consistency.csv  the kill-rule comparison (every column)
    expi_identify.meta.json

Kill rule (before anything else is written): the rebuilt per-draw values
must equal the committed results/tables/expi_paired_draws.csv.gz column by
column -- counts exactly, NaN pattern identical, floats in the "exact" tier
(<= 1e-9 max(1,|v|)) or the registered "optimiser" tier of the addendum
(<= 2e-3 max(1,|v|)) -- and the full-sample point values must equal the
committed expi_paired_estimands.csv point estimates within the same tiers.
Besides, on every fit the limit-convention xi* must equal the recomputed
root-status trap exactly (a failure means root_status does not replicate
derived_observables: a code error).  Any miss -> status 'killed': only the
consistency table and meta are written.

    python3 -P scripts/run_expi_identify.py --workers 8      (the single run)
    python3 -P scripts/run_expi_identify.py --dev --n-boot 8 --out /tmp/x
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
import contextlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import run_expi_paired as P
import run_expi_rerun_ci as R
from run_k2br import ci as k1_ci
from src.km_mle import _A

TAB = os.path.join(ROOT, "results", "tables")
AUDIT = os.path.join(ROOT, 'results', 'records')
ATTEMPT_LOG = os.path.join(AUDIT, "expi_identify_attempts.jsonl")
PLAN = os.path.join(ROOT, 'results', 'plan.md')
REG_HEADING = "post-hoc diagnostic (2026-10-25)"
SECTION_START = "## 11."
SECTION_END = "## 12."
SCRIPT_REL = "scripts/run_expi_identify.py"
LABEL = "China post-hoc identifiability diagnostic (report only)"
SEED, N_BOOT, PERIODS, VARIANTS, CONVENTIONS, OBS = (P.SEED, P.N_BOOT, P.PERIODS, P.VARIANTS,
                                                     P.CONVENTIONS, P.OBS)
V0, V1 = VARIANTS
SHORT = {V0: "V0", V1: "V1"}
N_GRID = 600
COMMITTED_DRAWS = "expi_paired_draws.csv.gz"
COMMITTED_ESTIMANDS = "expi_paired_estimands.csv"
EXACT_RTOL, OPT_ATOL = P.CONS_EXACT_RTOL, P.CONS_ATOL
COUNT_SUFFIX = "|n_pairs_"


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def root_status(params: np.ndarray, xi_data: np.ndarray, xi_target: float = 0.0,
                n_grid: int = N_GRID) -> dict:
    """The root search of km_mle.derived_observables, replicated line for
    line (default support = 1-99 percentiles of xi_data, n_grid nodes),
    plus the counts it does not return.  'trap' is the xi* that
    derived_observables uses before any path rule (first stable zero, else
    the argmin-|A| placeholder)."""
    lo_s, hi_s = np.percentile(xi_data, 1), np.percentile(xi_data, 99)
    grid = np.linspace(lo_s, hi_s, n_grid)
    A = _A(params, grid)
    sign_change = np.where(np.diff(np.sign(A)) != 0)[0]
    stable, n_unstable = [], 0
    for i in sign_change:
        if A[i] > 0 and A[i + 1] < 0:
            frac = A[i] / (A[i] - A[i + 1])
            stable.append(grid[i] + frac * (grid[i + 1] - grid[i]))
        elif A[i] < 0 and A[i + 1] > 0:
            n_unstable += 1
    placeholder = float(grid[np.argmin(np.abs(A))])
    trap = float(stable[0]) if stable else placeholder
    lo, hi = min(trap, xi_target), max(trap, xi_target)
    n_path = int(np.count_nonzero((grid >= lo) & (grid <= hi)))
    return {"root_status": "stable_zero" if stable else "fallback_argminA",
            "n_stable": len(stable), "n_unstable": n_unstable,
            "first_stable": float(stable[0]) if stable else np.nan,
            "placeholder": placeholder, "trap": trap, "n_path_nodes": n_path,
            "subgrid": n_path < 2, "support_lo": float(lo_s), "support_hi": float(hi_s)}


ID_FIELDS = ("root_status", "n_stable", "n_unstable", "first_stable", "placeholder",
             "subgrid", "support_lo", "support_hi")


@contextlib.contextmanager
def capture_fits():
    """Run run_expi_paired.evaluate UNCHANGED while recording, for every
    (variant, window) it fits, the pairs' xi_t, the fitted parameters and the
    observables (the supplement's PAIRS / observables wrapped; restored on
    exit).  Yields the dict {(variant, window): (xi_t, params, obs)}."""
    seen: dict = {}
    cur: list = [None]
    orig_pairs, orig_obs = R.PAIRS, R.observables

    def wrap(var, fn):
        def f(panel, period):
            cur[0] = (var, period)
            return fn(panel, period)
        return f

    def obs(pairs, warm=None):
        o, th = orig_obs(pairs, warm=warm)
        seen[cur[0]] = (pairs["xi_t"].to_numpy(), np.asarray(th, float), o)
        return o, th

    R.PAIRS = {v: wrap(v, fn) for v, fn in orig_pairs.items()}
    R.observables = obs
    try:
        yield seen
    finally:
        R.PAIRS, R.observables = orig_pairs, orig_obs


def evaluate_identified(panels: dict, warm: dict | None) -> tuple[dict, dict, dict]:
    """P.evaluate on the panels (record and params exactly as the addendum)
    plus the identification columns '{var}|{field}_{p}', the reported
    id_status of each convention '{var}|{conv}|id_status_{p}' and the
    trap check '{var}|trap_check_{p}' (limit xi* == recomputed trap)."""
    with capture_fits() as seen:
        rec, params = P.evaluate(panels, warm)
    idr = {}
    for var in VARIANTS:
        for p in PERIODS:
            if (var, p) not in seen:
                idr[f"{var}|root_status_{p}"] = "fit_failed"
                for k in ID_FIELDS[1:]:
                    idr[f"{var}|{k}_{p}"] = np.nan
                for c in CONVENTIONS:
                    idr[f"{var}|{c}|id_status_{p}"] = "fit_failed"
                idr[f"{var}|trap_check_{p}"] = True
                continue
            xi, th, o = seen[(var, p)]
            rs = root_status(th, xi)
            for k in ID_FIELDS:
                idr[f"{var}|{k}_{p}"] = rs[k]
            for c in CONVENTIONS:
                idr[f"{var}|{c}|id_status_{p}"] = o[c]["id_status"]
            lim = o["limit"]["xi_trap"]
            idr[f"{var}|trap_check_{p}"] = bool(lim == rs["trap"])
    return rec, params, idr


_STATE: dict = {}


def _init(windows, members, warm):
    _STATE.update(windows=windows, members=members, warm=warm)


def one_draw(b: int) -> dict:
    panels = P.draw_panels(b, _STATE["windows"], _STATE["members"])
    rec, _, idr = evaluate_identified(panels, _STATE["warm"])
    return {"replicate": int(b), **rec, **idr}


def _tier(a: float, b: float) -> tuple[str, float, float]:
    if pd.isna(a) and pd.isna(b):
        return "exact", 0.0, 0.0
    if pd.isna(a) or pd.isna(b):
        return "fail", np.inf, 0.0
    d = abs(float(a) - float(b))
    s = max(1.0, abs(float(a)))
    return ("exact" if d <= EXACT_RTOL * s else "optimiser" if d <= OPT_ATOL * s else "fail",
            d, OPT_ATOL * s)


def consistency(point: dict, draws: pd.DataFrame, committed_draws: pd.DataFrame,
                committed_est: pd.DataFrame) -> pd.DataFrame:
    """Every column of the committed addendum draws (restricted to the
    replicates rebuilt here) against the rebuilt one, one row per column;
    then every committed point estimate against P.estimand_values(point).
    Columns only on one side fail; counts must be equal."""
    rows = []
    cd = committed_draws[committed_draws["replicate"].isin(draws["replicate"])]
    cd = cd.sort_values("replicate").reset_index(drop=True)
    nd = draws.sort_values("replicate").reset_index(drop=True)
    if not np.array_equal(cd["replicate"].to_numpy(), nd["replicate"].to_numpy()):
        rows.append({"table": COMMITTED_DRAWS, "column": "replicate", "n_compared": 0,
                     "n_exact": 0, "n_optimiser": 0, "n_fail": 1, "max_abs_diff": np.inf,
                     "tier": "fail", "note": "replicate sets differ"})
        return pd.DataFrame(rows)
    rebuilt_cols = [c for c in nd.columns if c in point or "|fail_" in c
                    or "composition|fail" in c]
    for c in sorted(set(cd.columns) | set(rebuilt_cols)):
        if c == "replicate":
            continue
        if c not in cd.columns or c not in nd.columns:
            rows.append({"table": COMMITTED_DRAWS, "column": c, "n_compared": 0, "n_exact": 0,
                         "n_optimiser": 0, "n_fail": 1, "max_abs_diff": np.inf, "tier": "fail",
                         "note": "only in " + ("rebuilt" if c in nd.columns else "committed")})
            continue
        a, b = cd[c], nd[c]
        if a.dtype == object or b.dtype == object:
            ok = [(str(x) == str(y)) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)]
            nf = int(len(ok) - sum(ok))
            rows.append({"table": COMMITTED_DRAWS, "column": c, "n_compared": len(ok),
                         "n_exact": int(sum(ok)), "n_optimiser": 0, "n_fail": nf,
                         "max_abs_diff": 0.0 if nf == 0 else np.inf,
                         "tier": "exact" if nf == 0 else "fail", "note": "string"})
            continue
        count = COUNT_SUFFIX in c
        tiers, mx = [], 0.0
        for x, y in zip(a.to_numpy(float), b.to_numpy(float)):
            t, d, _ = _tier(x, y)
            if count and t == "optimiser":
                t = "fail"
            tiers.append(t)
            mx = max(mx, d)
        n_e, n_o, n_f = tiers.count("exact"), tiers.count("optimiser"), tiers.count("fail")
        rows.append({"table": COMMITTED_DRAWS, "column": c, "n_compared": len(tiers),
                     "n_exact": n_e, "n_optimiser": n_o, "n_fail": n_f, "max_abs_diff": mx,
                     "tier": "fail" if n_f else "optimiser" if n_o else "exact",
                     "note": "count" if count else ""})
    pv = P.estimand_values(point)
    ce = committed_est.copy()
    ce["convention"] = ce["convention"].astype(str)
    for r in ce.itertuples(index=False):
        key = (r.estimand, r.convention)
        if key not in pv:
            rows.append({"table": COMMITTED_ESTIMANDS, "column": f"{key[0]}|{key[1]}",
                         "n_compared": 0, "n_exact": 0, "n_optimiser": 0, "n_fail": 1,
                         "max_abs_diff": np.inf, "tier": "fail", "note": "estimand not rebuilt"})
            continue
        t, d, _ = _tier(float(r.point_estimate), float(pv[key]))
        rows.append({"table": COMMITTED_ESTIMANDS, "column": f"{key[0]}|{key[1]}",
                     "n_compared": 1, "n_exact": int(t == "exact"),
                     "n_optimiser": int(t == "optimiser"), "n_fail": int(t == "fail"),
                     "max_abs_diff": d, "tier": t, "note": "point_estimate"})
    return pd.DataFrame(rows)


def trap_check_rows(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in VARIANTS:
        for p in PERIODS:
            c = f"{var}|trap_check_{p}"
            ok = draws[c].astype(bool)
            rows.append({"table": "root_status_vs_derived_observables", "column": c,
                         "n_compared": len(ok), "n_exact": int(ok.sum()), "n_optimiser": 0,
                         "n_fail": int((~ok).sum()), "max_abs_diff": 0.0 if ok.all() else np.inf,
                         "tier": "exact" if ok.all() else "fail",
                         "note": "limit xi* == recomputed trap"})
    return pd.DataFrame(rows)


def identified_mask(d, var: str, p: str) -> np.ndarray:
    g = (lambda k: d[k]) if isinstance(d, pd.DataFrame) else (lambda k: pd.Series([d[k]]))
    return (g(f"{var}|root_status_{p}") == "stable_zero").to_numpy()


def fractions_table(point_id: dict, draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in VARIANTS:
        for p in PERIODS:
            st = draws[f"{var}|root_status_{p}"]
            ns = draws[f"{var}|n_stable_{p}"]
            ident = (st == "stable_zero").to_numpy()
            sub = draws[f"{var}|subgrid_{p}"].astype(object).eq(True).to_numpy()
            xl = draws[f"{var}|limit|xi_trap_{p}"].to_numpy(float)
            xn = draws[f"{var}|nan|xi_trap_{p}"].to_numpy(float)
            nb = len(draws)
            fb = st == "fallback_argminA"
            phx = np.abs(xl[fb.to_numpy()])
            rows.append({
                "version": SHORT[var], "window": p, "n_draws": nb,
                "n_identified": int(ident.sum()), "frac_identified": ident.mean() if nb else np.nan,
                "n_not_identified": int(fb.sum()), "frac_not_identified": fb.mean() if nb else np.nan,
                "n_fit_failed": int((st == "fit_failed").sum()),
                "n_multiple_stable_zeros": int((ns >= 2).sum()),
                "n_subgrid_path": int(sub.sum()),
                "n_subgrid_and_identified": int((sub & ident).sum()),
                "n_subgrid_and_not_identified": int((sub & ~ident).sum()),
                "n_nan_convention_nonfinite": int((~np.isfinite(xn)).sum()),
                "n_limit_convention_nonfinite": int((~np.isfinite(xl)).sum()),
                "placeholder_abs_xi_median": float(np.median(phx)) if len(phx) else np.nan,
                "placeholder_abs_xi_max": float(np.max(phx)) if len(phx) else np.nan,
                "point_root_status": point_id[f"{var}|root_status_{p}"],
                "point_n_stable_zeros": point_id[f"{var}|n_stable_{p}"],
                "point_subgrid_path": bool(point_id[f"{var}|subgrid_{p}"]),
            })
    return pd.DataFrame(rows)


def _level_and_shift(d) -> dict:
    """Per-cell xi* and the within-version during - pre shift, as
    {(name, conv): values} (the addendum's paired estimands come from
    P.estimand_values)."""
    g = (lambda k: np.asarray(d[k], float)) if isinstance(d, pd.DataFrame) else (
        lambda k: np.asarray(d.get(k, np.nan), float))
    out = {}
    for c in CONVENTIONS:
        for v in VARIANTS:
            for p in PERIODS:
                out[(f"xi_trap_{SHORT[v]}_{p}", c)] = g(f"{v}|{c}|xi_trap_{p}")
            out[(f"xi_trap_{SHORT[v]}_during_minus_pre", c)] = (
                g(f"{v}|{c}|xi_trap_during") - g(f"{v}|{c}|xi_trap_pre"))
    return out


DIAG_ESTIMANDS = {
    "xi_trap_V0_minus_V1_pre": ([(V0, "pre"), (V1, "pre")], 0.0),
    "xi_trap_V0_minus_V1_during": ([(V0, "during"), (V1, "during")], 0.0),
    "xi_shift_V0_minus_V1": ([(v, p) for v in VARIANTS for p in PERIODS], 0.0),
    "tau_ratio_V0_over_V1": ([(v, p) for v in VARIANTS for p in PERIODS], 1.0),
    **{f"xi_trap_{SHORT[v]}_{p}": ([(v, p)], 0.0) for v in VARIANTS for p in PERIODS},
    **{f"xi_trap_{SHORT[v]}_during_minus_pre": ([(v, "pre"), (v, "during")], 0.0)
       for v in VARIANTS},
}
SUBSETS = ("all", "identified", "not_identified", "identified_regular_path")


def all_values(d) -> dict:
    v = {k: val for k, val in P.estimand_values(d).items() if k[0] in DIAG_ESTIMANDS}
    return v | _level_and_shift(d)


def subset_masks(draws: pd.DataFrame, cells: list) -> dict:
    n = len(draws)
    ident = np.ones(n, bool)
    regular = np.ones(n, bool)
    for v, p in cells:
        ident &= identified_mask(draws, v, p)
        regular &= ~draws[f"{v}|subgrid_{p}"].astype(object).eq(True).to_numpy()
    return {"all": np.ones(n, bool), "identified": ident, "not_identified": ~ident,
            "identified_regular_path": ident & regular}


def _summ(vals: np.ndarray, null: float) -> dict:
    c = k1_ci(vals)
    qm, qlo, qhi, qn = P.supplement_quantile_ci(vals)
    fin = vals[np.isfinite(vals)]
    out = {"n": len(vals), "n_nonfinite": c["n_nonfinite"], "median": c["median"],
           "tol0_lo": c["tol0_lo"], "tol0_hi": c["tol0_hi"], "tol0_status": c["ci_status"],
           "tol5_lo": c["tol5_lo"], "tol5_hi": c["tol5_hi"],
           "literal_lo": c["literal_lo"], "literal_hi": c["literal_hi"],
           "supp_lo": qlo, "supp_hi": qhi, "supp_n": qn,
           "share_gt_null": float((fin > null).mean()) if len(fin) else np.nan}
    for r in ("tol0", "tol5", "supp"):
        lo, hi = out[f"{r}_lo"], out[f"{r}_hi"]
        out[f"{r}_contains_null"] = (bool(lo <= null <= hi) if np.isfinite(lo) and np.isfinite(hi)
                                     else None)
    return out


def intervals_table(point: dict, draws: pd.DataFrame) -> pd.DataFrame:
    pv, dv = all_values(point), all_values(draws)
    rows = []
    for (name, conv), vals in dv.items():
        cells, null = DIAG_ESTIMANDS[name]
        masks = subset_masks(draws, cells)
        ref = _summ(vals, null)
        for s in SUBSETS:
            m = masks[s]
            r = {"estimand": name, "convention": conv, "subset": s, "null_value": null,
                 "cells": ";".join(f"{SHORT[v]}_{p}" for v, p in cells),
                 "point_estimate": float(np.asarray(pv[(name, conv)], float)),
                 "n_subset": int(m.sum()), "share_of_draws": float(m.mean()),
                 **{k: v for k, v in _summ(vals[m], null).items() if k != "n"}}
            r["tol0_lo_minus_all"] = r["tol0_lo"] - ref["tol0_lo"]
            r["tol0_hi_minus_all"] = r["tol0_hi"] - ref["tol0_hi"]
            r["supp_lo_minus_all"] = r["supp_lo"] - ref["supp_lo"]
            r["supp_hi_minus_all"] = r["supp_hi"] - ref["supp_hi"]
            x = vals[m]
            fin = x[np.isfinite(x)]
            lo, hi = ref["supp_lo"], ref["supp_hi"]
            for tag, sel in (("below", fin < lo), ("inside", (fin >= lo) & (fin <= hi)),
                             ("above", fin > hi)):
                r[f"share_{tag}_all_supp_interval"] = (float(sel.mean()) if len(fin)
                                                       else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def conventions_table(intervals: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """limit vs nan per estimand and subset: the interval under each rule,
    'contains the null' per convention and whether the two agree (None when
    a rule is not evaluable in either convention), the median sign, and per
    draw how often the two conventions give the same value."""
    dv = all_values(draws)
    rows = []
    for name in DIAG_ESTIMANDS:
        cells, _ = DIAG_ESTIMANDS[name]
        masks = subset_masks(draws, cells)
        for s in ("all", "identified"):
            a = intervals[(intervals.estimand == name) & (intervals.convention == "limit")
                          & (intervals.subset == s)].iloc[0]
            b = intervals[(intervals.estimand == name) & (intervals.convention == "nan")
                          & (intervals.subset == s)].iloc[0]
            r = {"estimand": name, "subset": s, "n_subset": int(a["n_subset"])}
            for rule in ("tol0", "tol5", "supp"):
                cl, cn = a[f"{rule}_contains_null"], b[f"{rule}_contains_null"]
                r |= {f"limit_{rule}_lo": a[f"{rule}_lo"], f"limit_{rule}_hi": a[f"{rule}_hi"],
                      f"nan_{rule}_lo": b[f"{rule}_lo"], f"nan_{rule}_hi": b[f"{rule}_hi"],
                      f"limit_{rule}_contains_null": cl, f"nan_{rule}_contains_null": cn,
                      f"{rule}_agree": (None if cl is None or cn is None or pd.isna(cl)
                                        or pd.isna(cn) else bool(cl == cn))}
            r["median_sign_agree"] = bool(np.sign(a["median"]) == np.sign(b["median"]))
            m = masks[s]
            xl, xn = dv[(name, "limit")][m], dv[(name, "nan")][m]
            both = np.isfinite(xl) & np.isfinite(xn)
            r |= {"n_both_finite": int(both.sum()),
                  "n_equal_both_finite": int((both & (xl == xn)).sum()),
                  "n_nan_nonfinite_only": int((np.isfinite(xl) & ~np.isfinite(xn)).sum()),
                  "max_abs_diff_both_finite": (float(np.max(np.abs(xl[both] - xn[both])))
                                               if both.any() else np.nan)}
            rows.append(r)
    return pd.DataFrame(rows)


def check_registration(plan_path: str = PLAN, code: str | None = None) -> str:
    if code is None:
        code = _git("log", "-1", "--format=%H", "--", SCRIPT_REL)
    with open(plan_path, encoding="utf-8") as f:
        plan = f.read()
    a, b = plan.find(SECTION_START), plan.find(SECTION_END)
    sec = plan[a:b] if 0 <= a < b else ""
    i = sec.find(REG_HEADING)
    if i < 0:
        sys.exit(f"refusing: no registration '{REG_HEADING}' in the registration record")
    if not code or code[:7] not in sec[i:]:
        sys.exit(f"refusing: the registration does not name the code commit {code[:7]}")
    return code


def check_single_run(log: str = ATTEMPT_LOG) -> None:
    if os.path.exists(log):
        with open(log) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        if lines:
            sys.exit(f"refusing: {len(lines)} attempt line(s) already logged in {log}; "
                     "the post-hoc diagnostic is a single run")


def _log(rec: dict) -> None:
    os.makedirs(AUDIT, exist_ok=True)
    with open(ATTEMPT_LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def analyse(panel: pd.DataFrame, n_boot: int, workers: int, committed_tab: str,
            out: str) -> dict:
    """Rebuild the draws, run the kill rule, write the tables.  Returns the
    meta fragment (status, written files, consistency tiers)."""
    windows, members = P.window_setup(panel)
    point, warm, point_id = evaluate_identified(windows, None)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(windows, members, warm)) as ex:
        recs = list(ex.map(one_draw, range(n_boot), chunksize=4))
    secs = time.time() - t0
    draws = pd.DataFrame(recs).sort_values("replicate").reset_index(drop=True)
    cd = pd.read_csv(os.path.join(committed_tab, COMMITTED_DRAWS))
    ce = pd.read_csv(os.path.join(committed_tab, COMMITTED_ESTIMANDS), keep_default_na=False,
                     na_values=[""])
    cons = pd.concat([consistency(point, draws, cd, ce), trap_check_rows(draws),
                      trap_check_rows(pd.DataFrame([point_id]))
                      .assign(table="root_status_vs_derived_observables (full sample)")],
                     ignore_index=True)
    killed = bool((cons["tier"] == "fail").any())
    os.makedirs(out, exist_ok=True)
    cons.to_csv(os.path.join(out, "expi_identify_consistency.csv"), index=False)
    written = ["expi_identify_consistency.csv"]
    if not killed:
        fr = fractions_table(point_id, draws)
        iv = intervals_table(point, draws)
        cv = conventions_table(iv, draws)
        fr.to_csv(os.path.join(out, "expi_identify_fractions.csv"), index=False)
        iv.to_csv(os.path.join(out, "expi_identify_intervals.csv"), index=False)
        cv.to_csv(os.path.join(out, "expi_identify_conventions.csv"), index=False)
        draws.to_csv(os.path.join(out, "expi_identify_draws.csv.gz"), index=False)
        written += ["expi_identify_fractions.csv", "expi_identify_intervals.csv",
                    "expi_identify_conventions.csv", "expi_identify_draws.csv.gz"]
    return {"status": "killed: consistency check failed" if killed else "completed",
            "killed": killed, "written": written, "seconds": secs,
            "consistency": {"tiers": cons["tier"].value_counts().to_dict(),
                            "max_abs_diff": float(cons["max_abs_diff"].replace(np.inf, np.nan)
                                                  .max()),
                            "exact_rtol": EXACT_RTOL, "optimiser_atol": OPT_ATOL},
            "point_root_status": {k: v for k, v in point_id.items() if "root_status" in k},
            "provinces": {p: len(members[p]) for p in PERIODS}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--dev", action="store_true", help="dirty tree / small B; never results/tables")
    ap.add_argument("--out", default=TAB)
    a = ap.parse_args(argv)
    if a.dev and os.path.commonpath([os.path.realpath(a.out),
                                     os.path.realpath(TAB)]) == os.path.realpath(TAB):
        sys.exit("--dev output never goes to results/tables")
    if not a.dev and os.path.realpath(a.out) != os.path.realpath(TAB):
        sys.exit("the single run writes to results/tables")
    dirty = _git("status", "--porcelain")
    code = None
    if not a.dev:
        if dirty or a.n_boot != N_BOOT:
            sys.exit(f"refusing: the single run needs a clean tree and n_boot = {N_BOOT}; "
                     "use --dev for a throw-away run")
        code = public_guard_skipped(None)
        check_single_run()
    from src.data_merger import load_panel
    head = _git("rev-parse", "HEAD")
    if not a.dev:
        _log({"status": "started", "head": head, "code_commit": code,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    res = analyse(load_panel("panel_scaling"), a.n_boot, a.workers, TAB, a.out)
    import scipy
    meta = {"label": LABEL, "script": SCRIPT_REL, "git": head, "code_commit": code,
            "dirty": bool(dirty), "dev": bool(a.dev), "post_hoc": True, "report_only": True,
            "seed": SEED, "n_boot": a.n_boot, "scheme": R.SCHEME,
            "rebuilds": "scripts/run_expi_paired.py draws (expi_paired_draws.csv.gz)",
            "identified_definition": "a stable zero (A[i] > 0, A[i+1] < 0) of the fitted drift "
                                     "on the 600-node grid over the 1-99 percentile support "
                                     "of the pairs' xi_t (km_mle.derived_observables)",
            **res, "workers": a.workers,
            "panel_sha256": P._sha256(os.path.join(ROOT, "data_clean", "panel_scaling.parquet")),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "python": sys.version.split()[0]},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "expi_identify.meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str, ensure_ascii=False)
    if not a.dev:
        _log({"status": "killed" if res["killed"] else "completed", "head": head,
              "code_commit": code, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "detail": res["status"]})
    print(res["status"], f"({a.n_boot} draws in {res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
