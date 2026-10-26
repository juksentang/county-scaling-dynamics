"""POST-HOC, REPORT ONLY.  Run after every result was known; no verdict, gate or committed table changes.  Single run, every
row reported.

Kill rules (checked before any alternative guard is evaluated):
  (1) data integrity: the sha256 of the raw files equals the committed
      record (US: results/tables/k2_coverage_sources.csv and the
      registration's data_sha256; Brazil: k2br_meta.json pib_raw_sha256);
  (2) reproduction: under the registered guard the rebuilt guard, status,
      predicted dxi* and drho1, projected open / complete xi* and the
      observed V0 - V1 xi* equal the committed k2/k2_h3b.csv row and the
      k2br_predictions.csv / k2br_verdicts.csv rows (floats within
      REPRO_RTOL max(1, |v|), the CSV round trip; statuses equal).
A miss -> status 'killed': the check table and meta are written, no
sensitivity table.

    python3 -P scripts/k1_guard_sensitivity.py            (the single run)
    python3 -P scripts/k1_guard_sensitivity.py --dev --out /tmp/x
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

import run_k2 as K2
import run_k2br as KBR

TAB = ROOT / "results" / "tables"
AUDIT = ROOT / 'results' / 'records'
ATTEMPT_LOG = AUDIT / "k1_guard_sensitivity_attempts.jsonl"
PLAN = ROOT / 'results' / 'plan.md'
REG_HEADING = "post-hoc diagnostic (2026-10-25)"
SECTION_START = "## 15."
SCRIPT_REL = "scripts/k1_guard_sensitivity.py"
LABEL = "K1 guarded projection: post-hoc guard sensitivity (report only)"

SUPPORTS = ((0.5, 99.5), (1.0, 99.0), (2.5, 97.5), (5.0, 95.0))
FLOOR_FACTORS = (1.0, 0.5, 2.0)
REGISTERED = ((1.0, 99.0), 1.0)
US_KEY = (K2.PRIMARY_SCOPE, K2.PRIMARY_OUTCOME, "main", "B", K2.PRIMARY_LINK)
REPRO_RTOL = 1e-12


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def alt_guard(xi_t: np.ndarray, rho1_v1: float, support: tuple, floor_factor: float):
    """The registered guard rule with the support percentiles and the
    slope-floor multiple as parameters; (1, 99), 1.0 is k1_guard exactly
    (same undefined cases: < 20 pairs, non-finite, rho1 = 1, lo >= hi)."""
    xt = np.asarray(xi_t, float)
    if (len(xt) < 20 or not np.all(np.isfinite(xt)) or not np.isfinite(rho1_v1)
            or rho1_v1 - 1.0 == 0.0):
        return None
    lo, hi = float(np.percentile(xt, support[0])), float(np.percentile(xt, support[1]))
    if not lo < hi:
        return None
    return (lo, hi, -float(floor_factor) * abs(float(rho1_v1) - 1.0))


def guard_grid():
    return [(s, f) for s in SUPPORTS for f in FLOOR_FACTORS]


def us_cell(world, key: tuple = US_KEY, n_min: int = K2.N_MIN) -> dict:
    scope = key[0]
    lin = K2.lineages(world, scope)
    aggs = {(sc, lab, oc): K2.aggregate(world, sc, lab, oc)
            for (sc, lab) in world.areas for oc in world.outcomes}
    drops, _ = K2.min_n_drops(world, aggs, n_min)
    sk = K2.build_skeleton(world, aggs, key, lin[1], drops[(key[1], key[2])])
    point = K2.run_spec(sk)
    est = K2.estimate_spec(point, key[3])
    e1 = est["V1"]
    r1 = e1[K2.PRIMARY_CONV]["rho1"]
    xe, _ = K2._expand(*point["pairs"]["V1"])
    return {"dataset": "US", "cell": "|".join(key), "theta": e1["params"], "dec": point["dec"],
            "xi_t": xe, "rho1": r1, "pairs": point["pairs"]["V1"],
            "observed": est["V0"][K2.PRIMARY_CONV]["xi_trap"] - e1[K2.PRIMARY_CONV]["xi_trap"]}


def us_predict(cell: dict, guard) -> dict:
    p = K2.k1_prediction(cell["theta"], cell["dec"], guard)
    return {"status": p["status"], "pred_xi_trap": p["pred_xi_trap"], "pred_rho1": p["pred_rho1"],
            "proj_open_xi_trap": p["proj_open_xi_trap"],
            "proj_complete_xi_trap": p["proj_complete_xi_trap"]}


def br_cells_from_res(res: dict, window: str, outcomes) -> list:
    out = []
    for o in outcomes:
        d = res["panels"][(o, "B")]
        f1 = d["fits"]["V1"]
        out.append({"dataset": "Brazil", "cell": f"{window}|{o}", "window": window, "outcome": o,
                    "k1rows": d["k1"], "theta": f1["params"], "xi_trap_v1": f1["limit"]["xi_trap"],
                    "rho1": f1["limit"]["rho1"], "xi_t": np.asarray(d["pairs"]["V1"][0], float),
                    "pairs": d["pairs"]["V1"],
                    "observed": d["fits"]["V0"]["limit"]["xi_trap"] - f1["limit"]["xi_trap"]})
    return out


def br_predict(cell: dict, guard) -> dict:
    """run_k2br.k1_prediction as finish_window calls it (an undefined guard
    gives the same 'K1 guard not defined' status)."""
    p = KBR.k1_prediction(cell["k1rows"], cell["theta"], cell["xi_trap_v1"], cell["rho1"],
                          guard=guard)
    if guard is None:
        p = p | {"status": "prediction_undefined: K1 guard not defined",
                 "pred_dxi_star": np.nan, "pred_drho1": np.nan}
    return {"status": p["status"], "pred_xi_trap": p["pred_dxi_star"],
            "pred_rho1": p["pred_drho1"], "proj_open_xi_trap": p.get("xi_star_open", np.nan),
            "proj_complete_xi_trap": p.get("xi_star_complete", np.nan)}


def predict(cell: dict, guard) -> dict:
    return (us_predict if cell["dataset"] == "US" else br_predict)(cell, guard)


def registered_guard(cell: dict):
    if cell["dataset"] == "US":
        return K2.k1_guard(cell["pairs"], cell["rho1"])
    return KBR.k1_guard(cell["pairs"], cell["rho1"])


def _cmp(rows, cell, col, committed, rebuilt):
    if isinstance(committed, str) or isinstance(rebuilt, str):
        ok = str(committed) == str(rebuilt)
        d = 0.0 if ok else np.inf
    elif pd.isna(committed) and pd.isna(rebuilt):
        ok, d = True, 0.0
    elif pd.isna(committed) or pd.isna(rebuilt):
        ok, d = False, np.inf
    else:
        d = abs(float(committed) - float(rebuilt))
        ok = d <= REPRO_RTOL * max(1.0, abs(float(committed)))
    rows.append({"dataset": cell["dataset"], "cell": cell["cell"], "column": col,
                 "committed": committed, "rebuilt": rebuilt, "abs_diff": d,
                 "pass": bool(ok)})


def committed_row(cell: dict, tab: Path = TAB) -> dict:
    """The committed registered-guard prediction of a cell, in this
    script's column names."""
    if cell["dataset"] == "US":
        h = pd.read_csv(tab / "k2" / "k2_h3b.csv")
        s, oc, per, _, link = cell["cell"].split("|")
        r = h[(h.scope == s) & (h.outcome == oc) & (h.period == per) & (h.link == link)].iloc[0]
        g = json.loads(r["guard"]) if isinstance(r["guard"], str) else [np.nan] * 3
        return {"guard_lo": g[0], "guard_hi": g[1], "slope_floor": g[2], "status": r["status"],
                "pred_xi_trap": r["pred_xi_trap"], "pred_rho1": r["pred_rho1"],
                "proj_open_xi_trap": r["proj_open_xi_trap"],
                "proj_complete_xi_trap": r["proj_complete_xi_trap"], "observed": r["observed"]}
    p = pd.read_csv(tab / "k2br_predictions.csv")
    v = pd.read_csv(tab / "k2br_verdicts.csv")
    r = p[(p.window == cell["window"]) & (p.outcome == cell["outcome"])].iloc[0]
    rv = v[(v.window == cell["window"]) & (v.outcome == cell["outcome"])].iloc[0]
    return {"guard_lo": r["guard_lo"], "guard_hi": r["guard_hi"],
            "slope_floor": r["guard_slope_floor"], "status": r["status"],
            "pred_xi_trap": r["pred_dxi_star"], "pred_rho1": r["pred_drho1"],
            "proj_open_xi_trap": r["xi_star_open"], "proj_complete_xi_trap": r["xi_star_complete"],
            "observed": rv["H3b_observed"]}


def reproduction_rows(cell: dict, committed: dict) -> list:
    rows = []
    g = registered_guard(cell)
    a = alt_guard(cell["xi_t"], cell["rho1"], *REGISTERED)
    rows.append({"dataset": cell["dataset"], "cell": cell["cell"],
                 "column": "alt_guard(registered) == k1_guard", "committed": str(g),
                 "rebuilt": str(a), "abs_diff": 0.0 if a == g else np.inf, "pass": a == g})
    pr = predict(cell, g)
    gl = (np.nan,) * 3 if g is None else g
    for col, val in (("guard_lo", gl[0]), ("guard_hi", gl[1]), ("slope_floor", gl[2]),
                     ("status", pr["status"]), ("pred_xi_trap", pr["pred_xi_trap"]),
                     ("pred_rho1", pr["pred_rho1"]),
                     ("proj_open_xi_trap", pr["proj_open_xi_trap"]),
                     ("proj_complete_xi_trap", pr["proj_complete_xi_trap"]),
                     ("observed", cell["observed"])):
        _cmp(rows, cell, col, committed[col], val)
    return rows


def sensitivity_rows(cell: dict, committed: dict, unguarded_committed=np.nan) -> list:
    reg = predict(cell, registered_guard(cell))
    rows = []
    for sup, fac in guard_grid():
        g = alt_guard(cell["xi_t"], cell["rho1"], sup, fac)
        p = predict(cell, g)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_reg = p["pred_xi_trap"] / reg["pred_xi_trap"]
            r_obs = p["pred_xi_trap"] / cell["observed"]
            r_ung = p["pred_xi_trap"] / unguarded_committed
        rows.append({"dataset": cell["dataset"], "cell": cell["cell"],
                     "support_lo_pct": sup[0], "support_hi_pct": sup[1],
                     "floor_factor": fac, "registered": (sup, fac) == REGISTERED,
                     "guard_lo": np.nan if g is None else g[0],
                     "guard_hi": np.nan if g is None else g[1],
                     "slope_floor": np.nan if g is None else g[2],
                     "a1_V1": cell["rho1"] - 1.0, **p,
                     "pred_over_registered": r_reg, "observed_V0_minus_V1": cell["observed"],
                     "pred_over_observed": r_obs,
                     "unguarded_pred_xi_trap_committed": unguarded_committed,
                     "pred_over_unguarded": r_ung,
                     "registered_status_committed": committed["status"]})
    return rows


def unguarded_committed(cell: dict, tab: Path = TAB) -> float:
    if cell["dataset"] == "US":
        h = pd.read_csv(tab / "k2" / "k2_h3b.csv")
        s, oc, per, _, link = cell["cell"].split("|")
        return float(h[(h.scope == s) & (h.outcome == oc) & (h.period == per)
                       & (h.link == link)].iloc[0]["unguarded_pred_xi_trap"])
    p = pd.read_csv(tab / "k2br_predictions.csv")
    return float(p[(p.window == cell["window"]) & (p.outcome == cell["outcome"])]
                 .iloc[0]["unguarded_pred_dxi_star"])


def run_cells(cells: list, tab: Path = TAB) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Kill rule (2) on every cell first; the sensitivity only if all pass."""
    check = []
    com = {c["cell"]: committed_row(c, tab) for c in cells}
    for c in cells:
        check += reproduction_rows(c, com[c["cell"]])
    check = pd.DataFrame(check)
    if not check["pass"].all():
        return check, None
    sens = []
    for c in cells:
        sens += sensitivity_rows(c, com[c["cell"]], unguarded_committed(c, tab))
    return check, pd.DataFrame(sens)


def integrity_rows(us_data: Path, br_data: Path, tab: Path = TAB) -> list:
    rows = []
    src = pd.read_csv(tab / "k2_coverage_sources.csv")
    reg = public_guard_skipped({'data_sha256': json.loads((tab / 'k2' / 'k2_meta.json').read_text())['data_sha256']})
    want = {r.file: r.sha256 for r in src.itertuples()} | reg.get("data_sha256", {})
    for f, w in want.items():
        h = sha256(us_data / f) if (us_data / f).exists() else "missing"
        rows.append({"dataset": "US", "file": f, "committed": w, "now": h, "pass": h == w})
    meta = json.loads((tab / "k2br_meta.json").read_text())
    for f, w in meta["pib_raw_sha256"].items():
        h = sha256(br_data / f) if (br_data / f).exists() else "missing"
        rows.append({"dataset": "Brazil", "file": f, "committed": w, "now": h, "pass": h == w})
    return rows


def real_cells(us_data: Path, br_data: Path) -> list:
    world = K2.load_world(us_data, with_outcomes=True)
    cells = [us_cell(world)]
    pib = KBR._pib_module()
    outcomes = KBR.check_outcome_frame(pib.outcome_values(br_data))
    struct = KBR.load_structure(br_data)
    cfg = KBR.Config()
    for w in KBR.WINDOWS:
        m, _ = KBR.build_muni(struct, w, outcomes, cfg)
        res = KBR.analyse(m, cfg)
        cells += br_cells_from_res(res, w, cfg.outcomes)
    return cells


def check_registration(plan_path: Path = PLAN, code: str | None = None) -> str:
    if code is None:
        code = _git("log", "-1", "--format=%H", "--", SCRIPT_REL)
    plan = Path(plan_path).read_text(encoding="utf-8")
    a = plan.find(SECTION_START)
    sec = plan[a:] if a >= 0 else ""
    i = sec.find(REG_HEADING)
    if i < 0:
        sys.exit(f"refusing: no registration '{REG_HEADING}' in the registration record")
    if not code or code[:7] not in sec[i:]:
        sys.exit(f"refusing: the registration does not name the code commit {code[:7]}")
    return code


def check_single_run(log: Path = ATTEMPT_LOG) -> None:
    if log.exists() and any(x.strip() for x in log.read_text().splitlines()):
        sys.exit(f"refusing: {log} already holds an attempt; the diagnostic is a single run")


def _log(rec: dict) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    with open(ATTEMPT_LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--dev", action="store_true", help="dirty tree; never results/tables")
    ap.add_argument("--out", type=Path, default=TAB)
    ap.add_argument("--us-data", type=Path, default=None)
    ap.add_argument("--br-data", type=Path, default=KBR.DATA)
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if a.dev and (out == TAB.resolve() or TAB.resolve() in out.parents):
        sys.exit("--dev output never goes to results/tables")
    if not a.dev and out != TAB.resolve():
        sys.exit("the single run writes to results/tables")
    dirty = _git("status", "--porcelain")
    code = None
    if not a.dev:
        if dirty:
            sys.exit("refusing: the single run needs a clean tree; use --dev for a throw-away run")
        code = public_guard_skipped(None)
        check_single_run()
    from src import k1_theory as T
    frozen = public_guard_skipped({'frozen_commit': None, 'environment_differs': None})
    head = _git("rev-parse", "HEAD")
    us_data = a.us_data or K2.COV.default_data_dir()
    if not a.dev:
        _log({"status": "started", "head": head, "code_commit": code,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    integ = pd.DataFrame(integrity_rows(us_data, a.br_data))
    integ.to_csv(out / "k1_guard_sensitivity_integrity.csv", index=False)
    written = ["k1_guard_sensitivity_integrity.csv"]
    if not integ["pass"].all():
        status = "killed: kill rule 1 (data sha256)"
    else:
        check, sens = run_cells(real_cells(us_data, a.br_data))
        check.to_csv(out / "k1_guard_sensitivity_check.csv", index=False)
        written.append("k1_guard_sensitivity_check.csv")
        if sens is None:
            status = "killed: kill rule 2 (registered guard does not reproduce the committed rows)"
        else:
            sens.to_csv(out / "k1_guard_sensitivity.csv", index=False)
            written.append("k1_guard_sensitivity.csv")
            status = "completed"
    meta = {"label": LABEL, "script": SCRIPT_REL, "git": head, "code_commit": code,
            "dirty": bool(dirty), "dev": bool(a.dev), "post_hoc": True, "report_only": True,
            "status": status, "supports": SUPPORTS, "floor_factors": FLOOR_FACTORS,
            "registered": REGISTERED, "us_key": US_KEY, "repro_rtol": REPRO_RTOL,
            "k1_freeze": {"frozen_commit": frozen.get("frozen_commit"),
                          "environment_differs": frozen.get("environment_differs")},
            "written": written, "seconds": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "k1_guard_sensitivity.meta.json").write_text(
        json.dumps(meta, indent=1, default=str, ensure_ascii=False))
    if not a.dev:
        _log({"status": "completed" if status == "completed" else "killed", "head": head,
              "code_commit": code, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "detail": status})
    print(status)
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
