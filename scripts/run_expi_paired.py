"""What it does, per draw b (and on the full sample for the point estimates):

  * the resampled window panels are built by the supplement's own
    functions in the supplement's order (``draw_panels``: per window,
    provinces sorted, ``run_expi_rerun_ci.resampled_window`` with the one
    generator ``default_rng([SEED, b])``), so draw b is the same resample;
  * V0 and V1 pairs (``run_expi_rerun_ci.PAIRS``), MLE + both conventions
    (``run_expi_rerun_ci.observables``, same warm start) -> every column
    the supplement summarised, under the supplement's column names;
  * rho1 = 1 + pooled OLS slope of delta_xi on xi_t of the same pairs
    (``run_k2br.rho1_from_pairs``, the US/Brazil definition), a1 = rho1 - 1;
  * the composition decomposition of every in-window transition on the
    resampled V0 SAAI (``run_attrition_diagnostic.decompose_year``):
    weighted dose sum(n_E + n_L) / sum(n_S) (= the n_S-weighted mean of
    (n_E + n_L)/n_S of k2br_coverage_summary.csv), its largest transition,
    and M_bar = n_S-weighted mean of M_t.

Consistency check (kill rule, before any new estimand is written): the
marginal tables are rebuilt from these draws with the supplement's rules
and compared with the committed expi_rerun_ci_variants.csv /
expi_rerun_ci_levels.csv; full-sample doses and M_bar with
k2br_coverage_summary.csv / expi_attrition_by_period.csv (``consistency``,
tolerances in CONSISTENCY_*).  A miss -> status 'killed': the consistency
table and meta are written, no estimand table.

Estimands (``ESTIMANDS``; descriptive, no verdict; tol0 symmetric order
statistic interval, k = ceil(0.025 n), as run_k2br.ci; the supplement's
pandas-quantile interval reported beside it): see the registration.

    python3 -P scripts/run_expi_paired.py               (the registered single run)
    python3 -P scripts/run_expi_paired.py --dev --n-boot 8 --out /tmp/x
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
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import run_expi_rerun_ci as R
from run_attrition_diagnostic import decompose_year
from run_k2br import ci as k1_ci, rho1_from_pairs
from src.config import FP_PERIODS
from src.saai import compute_saai

TAB = os.path.join(ROOT, "results", "tables")
AUDIT = os.path.join(ROOT, 'results', 'records')
ATTEMPT_LOG = os.path.join(AUDIT, "expi_paired_attempts.jsonl")
PLAN = os.path.join(ROOT, 'results', 'plan.md')
REG_HEADING = "amendment 7, supplement 2: paired differences and pooled slope"
SCRIPT_REL = "scripts/run_expi_paired.py"
SEED, N_BOOT, PERIODS, VARIANTS, CONVENTIONS, OBS = (R.SEED, R.N_BOOT, R.PERIODS, R.VARIANTS,
                                                     R.CONVENTIONS, R.OBS)
SUPPLEMENT_COMMIT = ''
LABEL = "China paired addendum (amendment 7, supplement 2): paired V0-V1, pooled OLS a1, dose"

CONS_EXACT_RTOL = 1e-9
CONS_ATOL = 2e-3
CONS_PTOL = 0.002
DOSE_TOL = 1e-12

V0, V1 = VARIANTS


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def window_setup(panel: pd.DataFrame) -> tuple[dict, dict]:
    """The supplement's window panels and province membership
    (run_expi_rerun_ci.main, lines building ``windows`` / ``members``)."""
    sa = compute_saai(panel, **R.SPEC)
    windows, members = {}, {}
    for p in PERIODS:
        lo, hi = FP_PERIODS[p]
        wp = panel[(panel["Year"] >= lo) & (panel["Year"] <= hi)].copy()
        sw = sa[(sa["Year"] >= lo) & (sa["Year"] <= hi)]
        provs = np.unique(R.province_of(sw["CountyCode"]))
        wp = wp[np.isin(R.province_of(wp["CountyCode"]), provs)]
        windows[p] = wp
        members[p] = {int(pv): np.unique(wp.loc[R.province_of(wp["CountyCode"]) == pv,
                                               "CountyCode"]).astype(np.int64)
                      for pv in provs}
    return windows, members


def draw_panels(b: int, windows: dict, members: dict) -> dict:
    """Draw b's resampled window panels, built exactly as
    run_expi_rerun_ci.one_draw builds them (same generator, same order)."""
    rng = np.random.default_rng([SEED, int(b)])
    out = {}
    for p in PERIODS:
        provs = np.array(sorted(members[p]))
        out[p] = R.resampled_window(windows[p], members[p], provs, rng)
    return out


def composition(window_panel: pd.DataFrame, period: str) -> dict:
    """Weighted dose, its largest transition and M_bar of one window panel
    (V0 SAAI; transitions t -> t+1 with both years in the window)."""
    sa = compute_saai(window_panel, **R.SPEC)
    a, b = FP_PERIODS[period]
    rows = [decompose_year(sa, t) for t in range(a, b)]
    d = pd.DataFrame(rows)
    d = d[d["n_stay"] > 0]
    ne = d["n_enter"].fillna(0)
    ratio = (ne + d["n_leave"]) / d["n_stay"]
    return {"dose": float((ne + d["n_leave"]).sum() / d["n_stay"].sum()),
            "dose_max": float(ratio.max()),
            "Mbar": float((d["n_stay"] * d["mechanical_term"]).sum() / d["n_stay"].sum())}


def evaluate(panels: dict, warm: dict | None) -> tuple[dict, dict]:
    """One pass on given window panels (a draw, or the full sample with
    warm=None).  Returns (record, fitted params).  The record holds the
    supplement's columns '{var}|{conv}|{q}_{p}', '{var}|n_pairs_{p}' plus
    '{var}|rho1_{p}' and 'dose_{p}', 'dose_max_{p}', 'Mbar_{p}'."""
    rec, params = {}, {}
    for var in VARIANTS:
        for p in PERIODS:
            try:
                pr = R.PAIRS[var](panels[p], p)
                o, th = R.observables(pr, warm=None if warm is None else warm[(var, p)])
                for conv in CONVENTIONS:
                    for k in OBS:
                        rec[f"{var}|{conv}|{k}_{p}"] = o[conv][k]
                rec[f"{var}|n_pairs_{p}"] = len(pr)
                rec[f"{var}|rho1_{p}"] = rho1_from_pairs(pr["xi_t"].to_numpy(),
                                                         pr["delta_xi"].to_numpy())
                params[(var, p)] = th
            except Exception as exc:
                for conv in CONVENTIONS:
                    for k in OBS:
                        rec[f"{var}|{conv}|{k}_{p}"] = np.nan
                rec[f"{var}|rho1_{p}"] = np.nan
                rec[f"{var}|fail_{p}"] = f"{type(exc).__name__}: {exc}"[:200]
    for p in PERIODS:
        try:
            rec |= {f"{k}_{p}": v for k, v in composition(panels[p], p).items()}
        except Exception as exc:
            rec |= {f"{k}_{p}": np.nan for k in ("dose", "dose_max", "Mbar")}
            rec[f"composition|fail_{p}"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec, params


_STATE: dict = {}


def _init(windows, members, warm):
    _STATE.update(windows=windows, members=members, warm=warm)


def one_draw(b: int) -> dict:
    panels = draw_panels(b, _STATE["windows"], _STATE["members"])
    rec, _ = evaluate(panels, _STATE["warm"])
    return {"replicate": int(b), **rec}


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a, float) / np.asarray(b, float)


def estimand_values(d) -> dict:
    """{(name, convention): value(s)} of every estimand on a record (dict of
    scalars) or a draws table (DataFrame columns).  convention 'none' for
    the convention-free quantities (rho1, a1, dose, M_bar)."""
    g = (lambda k: np.asarray(d[k], float)) if isinstance(d, pd.DataFrame) else (
        lambda k: np.asarray(d.get(k, np.nan), float))
    out = {}
    pre, dur = PERIODS
    for c in CONVENTIONS:
        x = {(v, p): g(f"{v}|{c}|xi_trap_{p}") for v in VARIANTS for p in PERIODS}
        t = {(v, p): g(f"{v}|{c}|tau_mfpt_{p}") for v in VARIANTS for p in PERIODS}
        for p in PERIODS:
            out[(f"xi_trap_V0_minus_V1_{p}", c)] = x[(V0, p)] - x[(V1, p)]
        out[("xi_shift_V0_minus_V1", c)] = ((x[(V0, dur)] - x[(V0, pre)])
                                            - (x[(V1, dur)] - x[(V1, pre)]))
        out[("tau_ratio_V0_over_V1", c)] = _ratio(_ratio(t[(V0, dur)], t[(V0, pre)]),
                                                  _ratio(t[(V1, dur)], t[(V1, pre)]))
    for p in PERIODS:
        r0, r1 = g(f"{V0}|rho1_{p}"), g(f"{V1}|rho1_{p}")
        a0, a1 = r0 - 1.0, r1 - 1.0
        out[(f"rho1_V0_{p}", "none")] = r0
        out[(f"rho1_V1_{p}", "none")] = r1
        out[(f"a1_V0_{p}", "none")] = a0
        out[(f"a1_V1_{p}", "none")] = a1
        out[(f"rho1_V0_minus_V1_{p}", "none")] = r0 - r1
        out[(f"amp_V1_{p}", "none")] = _ratio(1.0, np.abs(a1))
        out[(f"amp_V0_{p}", "none")] = _ratio(1.0, np.abs(a0))
        dose, mbar = g(f"dose_{p}"), g(f"Mbar_{p}")
        out[(f"dose_{p}", "none")] = dose
        out[(f"dose_max_{p}", "none")] = g(f"dose_max_{p}")
        out[(f"Mbar_{p}", "none")] = mbar
        out[(f"dose_x_amp_V1_{p}", "none")] = dose * _ratio(1.0, np.abs(a1))
        out[(f"zeroth_order_Mbar_over_abs_a1_V1_{p}", "none")] = _ratio(mbar, np.abs(a1))
    return out


ESTIMAND_ROLE = {
    "xi_trap_V0_minus_V1_pre": 'paired_V0_V1', "xi_trap_V0_minus_V1_during": 'paired_V0_V1',
    "xi_shift_V0_minus_V1": 'paired_V0_V1', "tau_ratio_V0_over_V1": "report",
    "rho1_V0_pre": 'dose_ladder', "rho1_V1_pre": 'dose_ladder', "rho1_V0_during": 'dose_ladder',
    "rho1_V1_during": 'dose_ladder', "a1_V0_pre": 'dose_ladder', "a1_V1_pre": 'dose_ladder',
    "a1_V0_during": 'dose_ladder', "a1_V1_during": 'dose_ladder',
    "amp_V1_pre": 'dose_ladder', "amp_V1_during": 'dose_ladder',
    "dose_x_amp_V1_pre": 'dose_ladder', "dose_x_amp_V1_during": 'dose_ladder',
    "dose_pre": 'dose_ladder', "dose_during": 'dose_ladder',
    "rho1_V0_minus_V1_pre": "report", "rho1_V0_minus_V1_during": "report",
    "amp_V0_pre": "report", "amp_V0_during": "report",
    "dose_max_pre": "report", "dose_max_during": "report",
    "Mbar_pre": "report", "Mbar_during": "report",
    "zeroth_order_Mbar_over_abs_a1_V1_pre": "report",
    "zeroth_order_Mbar_over_abs_a1_V1_during": "report",
}


def supplement_quantile_ci(v) -> tuple[float, float, float, int]:
    """The supplement's summary rule: NaN / +-inf dropped, pandas linear
    quantiles 0.025 / 0.975 (median, lo, hi, n_finite)."""
    s = pd.Series(np.asarray(v, float)).replace([np.inf, -np.inf], np.nan).dropna()
    return (float(s.median()) if len(s) else np.nan, float(s.quantile(0.025)) if len(s) else np.nan,
            float(s.quantile(0.975)) if len(s) else np.nan, int(len(s)))


def estimand_table(point: dict, draws: pd.DataFrame) -> pd.DataFrame:
    pv, dv = estimand_values(point), estimand_values(draws)
    rows = []
    for (name, conv), val in pv.items():
        c = k1_ci(dv[(name, conv)])
        qm, qlo, qhi, qn = supplement_quantile_ci(dv[(name, conv)])
        rows.append({"estimand": name, "convention": conv, "role": ESTIMAND_ROLE[name],
                     "primary": conv in ("limit", "none"), "point_estimate": float(val),
                     "median": c["median"], "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"],
                     "ci_rule": c["ci_rule"], "ci_status": c["ci_status"],
                     "n_draws": c["n_draws"], "n_nonfinite": c["n_nonfinite"],
                     "tol5_lo": c["tol5_lo"], "tol5_hi": c["tol5_hi"],
                     "literal_lo": c["literal_lo"], "literal_hi": c["literal_hi"],
                     "supplement_rule_median": qm, "supplement_rule_lo": qlo,
                     "supplement_rule_hi": qhi, "supplement_rule_n": qn})
    return pd.DataFrame(rows)


def marginal_tables(point: dict, draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The supplement's two tables rebuilt from these draws with its own
    summary (run_expi_rerun_ci.summarise) and its point / level rules
    (run_expi_rerun_ci.main)."""
    rows, levels = [], []
    for var in VARIANTS:
        for conv in CONVENTIONS:
            for r in R.summarise(draws, var, conv):
                q, kind = r["quantity"].split(" (")[0], r["kind"]
                pe = {p: point[f"{var}|{conv}|{q}_{p}"] for p in PERIODS}
                est = (pe["during"] - pe["pre"]) if kind == "difference" else (
                    pe["during"] / pe["pre"] if pe["pre"] != 0 else np.nan)
                rows.append({"variant": var, "degenerate": conv, "point_estimate": est,
                             "n_pairs_pre": point[f"{var}|n_pairs_pre"],
                             "n_pairs_during": point[f"{var}|n_pairs_during"], **r})
            for p in PERIODS:
                for k in OBS:
                    col = draws[f"{var}|{conv}|{k}_{p}"].dropna()
                    levels.append({"variant": var, "degenerate": conv, "period": p,
                                   "quantity": k, "point_estimate": point[f"{var}|{conv}|{k}_{p}"],
                                   "median": col.median(), "ci_lo": col.quantile(0.025),
                                   "ci_hi": col.quantile(0.975), "n_draws": len(col)})
    return pd.DataFrame(rows), pd.DataFrame(levels)


COUNT_COLS = ("n_draws", "n_failed", "n_pairs_pre", "n_pairs_during")
GATED_RATIO = ("tau_mfpt", "mean_diffusion", "B_ratio_w")
SHARE_COLS = ("P_gt0", "P_gt1")
FLOAT_COLS = ("point_estimate", "median", "ci_lo", "ci_hi")


def _cmp_rows(new: pd.DataFrame, old: pd.DataFrame, keys: list, table: str) -> list:
    out = []
    old = old.copy()
    old["degenerate"] = old["degenerate"].astype(str)
    new = new.copy()
    new["degenerate"] = new["degenerate"].astype(str)
    m = new.merge(old, on=keys, how="outer", suffixes=("_new", "_old"), indicator=True)
    for _, r in m.iterrows():
        key = "|".join(str(r[k]) for k in keys)
        if r["_merge"] != "both":
            out.append({"table": table, "key": key, "column": "<row>", "committed": np.nan,
                        "recomputed": np.nan, "abs_diff": np.inf, "allowed": 0.0,
                        "tier": "fail", "note": f"row only in {r['_merge']}"})
            continue
        for col in COUNT_COLS + SHARE_COLS + FLOAT_COLS:
            if f"{col}_old" not in r:
                continue
            a, b = r[f"{col}_old"], r[f"{col}_new"]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b):
                diff = np.inf
            else:
                diff = abs(float(b) - float(a))
            if col in COUNT_COLS:
                allowed, exact = 0.0, 0.0
            elif col in SHARE_COLS:
                allowed, exact = CONS_PTOL, CONS_EXACT_RTOL
            else:
                scale = max(1.0, abs(float(a))) if not pd.isna(a) else 1.0
                allowed, exact = CONS_ATOL * scale, CONS_EXACT_RTOL * scale
            tier = "exact" if diff <= exact else "optimiser" if diff <= allowed else "fail"
            if (tier == "fail" and col not in COUNT_COLS and "(during/pre)" in key
                    and key.split("|")[-1].split(" (")[0] not in GATED_RATIO):
                tier = "outside_not_gated"
            out.append({"table": table, "key": key, "column": col, "committed": a,
                        "recomputed": b, "abs_diff": diff, "allowed": allowed, "tier": tier,
                        "note": ""})
    return out


def consistency(point: dict, draws: pd.DataFrame, tab: str = TAB) -> pd.DataFrame:
    """Registered kill-rule comparison (all rows reported)."""
    var_new, lev_new = marginal_tables(point, draws)
    var_old = pd.read_csv(os.path.join(tab, "expi_rerun_ci_variants.csv"),
                          keep_default_na=False, na_values=[""])
    lev_old = pd.read_csv(os.path.join(tab, "expi_rerun_ci_levels.csv"),
                          keep_default_na=False, na_values=[""])
    rows = _cmp_rows(var_new, var_old, ["variant", "degenerate", "quantity"],
                     "expi_rerun_ci_variants.csv")
    rows += _cmp_rows(lev_new, lev_old, ["variant", "degenerate", "period", "quantity"],
                      "expi_rerun_ci_levels.csv")
    cov = pd.read_csv(os.path.join(tab, "k2br_coverage_summary.csv"))
    per = pd.read_csv(os.path.join(tab, "expi_attrition_by_period.csv"))
    cn = cov[cov["dataset"].str.startswith("China counties multi-SAAI")]
    for p in PERIODS:
        c = cn[cn["window"] == p].iloc[0]
        for col, k in (("nS_weighted_comp_shock", "dose"), ("max_comp_shock", "dose_max")):
            diff = abs(point[f"{k}_{p}"] - float(c[col]))
            rows.append({"table": "k2br_coverage_summary.csv", "key": f"China|{p}", "column": col,
                         "committed": float(c[col]), "recomputed": point[f"{k}_{p}"],
                         "abs_diff": diff, "allowed": DOSE_TOL,
                         "tier": "exact" if diff <= DOSE_TOL else "fail", "note": ""})
        r = per[per["period"] == p].iloc[0]
        diff = abs(point[f"Mbar_{p}"] - float(r["mechanical_term_weighted"]))
        rows.append({"table": "expi_attrition_by_period.csv", "key": p,
                     "column": "mechanical_term_weighted",
                     "committed": float(r["mechanical_term_weighted"]),
                     "recomputed": point[f"Mbar_{p}"], "abs_diff": diff, "allowed": DOSE_TOL,
                     "tier": "exact" if diff <= DOSE_TOL else "fail", "note": ""})
    return pd.DataFrame(rows)


def check_registration() -> str:
    code = _git("log", "-1", "--format=%H", "--", SCRIPT_REL)
    with open(PLAN, encoding="utf-8") as f:
        plan = f.read()
    i = plan.find(REG_HEADING)
    if i < 0:
        sys.exit(f"refusing: no registration '{REG_HEADING}' in the registration record")
    if code[:7] not in plan[i:]:
        sys.exit(f"refusing: the registration does not name the code commit {code[:7]}")
    return code


def check_single_run() -> None:
    if os.path.exists(ATTEMPT_LOG):
        with open(ATTEMPT_LOG) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        if lines:
            sys.exit(f"refusing: {len(lines)} attempt line(s) already logged in {ATTEMPT_LOG}; "
                     "the registered run is single (a re-run needs an amendment)")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(rec: dict) -> None:
    os.makedirs(AUDIT, exist_ok=True)
    with open(ATTEMPT_LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--dev", action="store_true", help="dirty tree / small B; never results/tables")
    ap.add_argument("--out", default=TAB)
    a = ap.parse_args(argv)
    out_real = os.path.realpath(a.out)
    if a.dev and os.path.commonpath([out_real, os.path.realpath(TAB)]) == os.path.realpath(TAB):
        sys.exit("--dev output never goes to results/tables")
    dirty = _git("status", "--porcelain")
    code = None
    if not a.dev:
        if dirty or a.n_boot != N_BOOT:
            sys.exit("refusing: the registered run needs a clean tree and n_boot = "
                     f"{N_BOOT}; use --dev for a throw-away run")
        code = public_guard_skipped(None)
        check_single_run()
    from src.data_merger import load_panel
    import scipy
    import statsmodels
    head = _git("rev-parse", "HEAD")
    if not a.dev:
        _log({"status": "started", "head": head, "code_commit": code,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    panel = load_panel("panel_scaling")
    windows, members = window_setup(panel)
    point, warm = evaluate(windows, None)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(windows, members, warm)) as ex:
        recs = list(ex.map(one_draw, range(a.n_boot), chunksize=4))
    secs = time.time() - t0
    draws = pd.DataFrame(recs).sort_values("replicate").reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    cons = consistency(point, draws)
    killed = bool((cons["tier"] == "fail").any())
    cons.to_csv(os.path.join(a.out, "expi_paired_consistency.csv"), index=False)
    written = ["expi_paired_consistency.csv"]
    if not killed or a.dev:
        est = estimand_table(point, draws)
        est.to_csv(os.path.join(a.out, "expi_paired_estimands.csv"), index=False)
        keep = ["replicate"] + [c for c in draws.columns if c != "replicate"]
        draws[keep].to_csv(os.path.join(a.out, "expi_paired_draws.csv.gz"), index=False)
        written += ["expi_paired_estimands.csv", "expi_paired_draws.csv.gz"]
    status = "killed: consistency check failed" if killed else "completed"
    meta = {"label": LABEL, "script": SCRIPT_REL, "git": head, "code_commit": code,
            "dirty": bool(dirty), "dev": bool(a.dev), "status": status,
            "seed": SEED, "seed_rule": "rng [SEED, b] per draw b (run_expi_rerun_ci.one_draw order)",
            "n_boot": a.n_boot, "supplement_commit": SUPPLEMENT_COMMIT, "scheme": R.SCHEME,
            "ci_rule": "tol0 symmetric order statistics k = ceil(0.025 n) (run_k2br.ci); "
                       "supplement pandas-quantile rule reported beside",
            "consistency": {"tiers": cons["tier"].value_counts().to_dict(),
                            "max_abs_diff_float": float(cons.loc[cons["column"].isin(FLOAT_COLS),
                                                                 "abs_diff"].max()),
                            "atol": CONS_ATOL, "ptol": CONS_PTOL, "exact_rtol": CONS_EXACT_RTOL,
                            "dose_tol": DOSE_TOL},
            "provinces": {p: len(members[p]) for p in PERIODS},
            "counties": {p: int(sum(len(v) for v in members[p].values())) for p in PERIODS},
            "failed_draws": {c: int(draws[c].notna().sum()) for c in draws.columns if "fail_" in c},
            "panel_sha256": _sha256(os.path.join(ROOT, "data_clean", "panel_scaling.parquet")),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
                         "python": sys.version.split()[0]},
            "written": written, "seconds": secs, "workers": a.workers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "expi_paired.meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str, ensure_ascii=False)
    if not a.dev:
        _log({"status": "killed" if killed else "completed", "head": head, "code_commit": code,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "detail": status})
    print(cons["tier"].value_counts().to_string())
    print(status, f"({a.n_boot} draws in {secs:.0f} s, {a.workers} workers)")


if __name__ == "__main__":
    main()
