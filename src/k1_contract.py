"""Units: latent x is the county-level log residual in the simulator's own
units; ξ is the within-year standardised index the real pipeline sees.  One
transition is year t → t+1, and Δt = 1 year everywhere.
"""
def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import ast
import json
import math
import re
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 11
REPO = Path(__file__).resolve().parents[1]

X_UNITS = "latent_x"

RE_PLACEMENTS = {
    "shock": True,
    "persistent": False,
    "none": True,
    "initial_condition": True,
}
GATED_RE_PLACEMENT = "shock"
CONTROL_RE_PLACEMENT = "persistent"
EXIT_DATINGS = {
    "none": True,
    "exogenous": True,
    "info_t": True,
    "info_t1": False,
}

COMPOSITION_COLUMNS = {
    "replicate": "int, replicate index within the cell",
    "period": "'pre' | 'during'",
    "year_t": "int, first year of the transition",
    "n_stay": "counties present in t and t+1 (S_t)",
    "n_leave": "present in t only",
    "n_enter": "present in t+1 only",
    "mu_x_t": "mean of the LATENT x over C_t (ddof 0), x units",
    "sigma_x_t": "s.d. of the LATENT x over C_t (ddof 0), x units",
    "mu_x_t1": "same over C_{t+1}",
    "sigma_x_t1": "same over C_{t+1}",
    "x_units": f"must equal {X_UNITS!r}",
    "mech_term": "(n_L·m_L − n_E·m_E)/n_S in ξ units (Lemma 1); must equal the stayers' mean Δξ",
    "xi_bar_stay": "mean of ξ_t over S_t, ξ units, from the pipeline's own ξ",
    "var_stay": "variance (ddof 0) of ξ_t over S_t, ξ units, from the pipeline's own ξ",
    "pipeline": "'saai' (real compute_saai on generated outputs) | 'direct' (standardise x directly)",
    "re_placement": f"one of {tuple(RE_PLACEMENTS)} (constant per cell; also in the meta)",
    "exit_dating": f"one of {tuple(EXIT_DATINGS)} (constant per cell; also in the meta)",
}

STANDARDISATION_COLUMNS = {
    "year": "int",
    "n": "counties in the complete panel that year (the window's size; the "
         "window sizes sum to the cell's N)",
    "mu_x": "mean of the latent x over the complete panel (ddof 0)",
    "sigma_x": "s.d. of the latent x over the complete panel (ddof 0)",
    "x_units": f"must equal {X_UNITS!r}",
    "common_shock_wH": "w·H of the window (amendment 4 D2): the province-common "
                       "innovation share w of the simulated chain times the "
                       "provincial Herfindahl H = Σ_p s_p² of the complete "
                       "panel's counties; 0 without province-common shocks "
                       "(re_placement 'persistent', w = 0)",
}
SHOCK_SHARE_COLUMN = "common_shock_wH"

TRUE_PARAM_KEYS = ("a0", "a1", "a2", "a3", "c0", "c1", "c2")
GUARD_KEYS = ("guard_lo", "guard_hi", "slope_floor")
TRUE_PROCESS_KEYS = TRUE_PARAM_KEYS + GUARD_KEYS
LOGB_CLIP = 20.0
EDGE_SD = 10.0
EDGE_MASS_TOL = 1e-6


def check_true_params(true_params: dict) -> tuple:
    """(theta, guard): theta the 7-vector of TRUE_PARAM_KEYS, guard None or
    (lo, hi, slope_floor).  Refuses a dict without the GUARD_KEYS, a partial
    guard, lo >= hi, slope_floor >= 0 and non-finite values."""
    missing = [k for k in TRUE_PROCESS_KEYS if k not in true_params]
    if missing:
        raise ValueError(f"true_params lacks {missing}: every process declares "
                         f"{GUARD_KEYS} (None for an unguarded process; contract "
                         f"version {CONTRACT_VERSION})")
    theta = np.array([float(true_params[k]) for k in TRUE_PARAM_KEYS])
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"non-finite true parameters {theta}")
    g = [true_params[k] for k in GUARD_KEYS]
    if all(v is None for v in g):
        return theta, None
    if any(v is None for v in g):
        raise ValueError(f"partial guard {dict(zip(GUARD_KEYS, g))}: all three or none")
    lo, hi, fl = (float(v) for v in g)
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(fl)):
        raise ValueError("guard values must be finite")
    if not lo < hi:
        raise ValueError(f"guard_lo {lo} must be < guard_hi {hi}")
    if not fl < 0:
        raise ValueError(f"slope_floor {fl} must be < 0 (the value used, -|a1|)")
    return theta, (lo, hi, fl)

CALIBRATION_FILE = "src/k1_sim_calibration.json"
DESIGN_PROCESS_FILE = "src/k1_design_processes.json"
REGISTERED_FITS = {"ou": ("ou", False), "cubic": ("cubic_constB", True),
                   "cubic_mult": ("full", True)}
DESIGN_MEMBERS = {"cubic_bistable_0": ("pre", "during"),
                  "cubic_bistable_1": ("pre", "during"),
                  "cubic_bistable_2": ("pre", "during"),
                  "cubic_bistable_4": ("pre", "during"),
                  "cubic_shift": ("during",)}


_CAL = {}


def _calibration() -> dict:
    """The committed calibration next to this file (read once per process;
    the theory freeze fingerprints the file, k1_theory.FROZEN_FILES)."""
    if "cal" not in _CAL:
        _CAL["cal"] = json.loads((Path(__file__).resolve().parents[1]
                                  / CALIBRATION_FILE).read_text())
    return _CAL["cal"]


def _design_processes() -> dict:
    if "design" not in _CAL:
        p = Path(__file__).resolve().parents[1] / DESIGN_PROCESS_FILE
        d = json.loads(p.read_text()) if p.exists() else {"members": {}}
        mem = d.get("members")
        if not isinstance(mem, dict):
            raise ValueError(f"{DESIGN_PROCESS_FILE}: 'members' must be a dict")
        unknown = sorted(set(mem) - set(DESIGN_MEMBERS))
        if unknown:
            raise ValueError(f"{DESIGN_PROCESS_FILE}: unknown members {unknown} "
                             f"(registered: {sorted(DESIGN_MEMBERS)})")
        _CAL["design"] = mem
    return _CAL["design"]


def _check_member(process: str) -> None:
    if process in PROCESS_FAMILIES:
        raise ValueError(f"{process!r} is the registered FAMILY {PROCESS_FAMILIES[process][0]}, "
                         f"not one process: use one of its members "
                         f"{list(PROCESS_FAMILIES[process][1])}")
    if process not in PROCESSES:
        raise ValueError(f"unknown process {process!r}")


def registered_true_params(process: str, period: str) -> dict:
    """The registered true_params dict (TRUE_PROCESS_KEYS) of process (a
    MEMBER key of PROCESSES) and period: (i)–(iii) and (v) 'pre' from
    CALIBRATION_FILE, the (iv) members and (v) 'during' from
    DESIGN_PROCESS_FILE.  Raises for an unknown period, for a family key
    ('cubic_bistable') and for a member whose constants are not committed."""
    _check_member(process)
    if period not in ("pre", "during"):
        raise ValueError(f"unknown period {period!r}")
    if process == "cubic_shift" and period == "pre":
        return registered_true_params("cubic", "pre")
    if process in DESIGN_MEMBERS:
        ent = (_design_processes().get(process) or {}).get(period)
        if ent is None:
            raise ValueError(f"process {process!r} {PROCESSES[process]} has no registered "
                             f"constants for {period!r} in {DESIGN_PROCESS_FILE}: k1_sim "
                             "commits its "
                             "calibrated design constants there (reviewed commit, before "
                             "the theory freeze) before any prediction, truth or S0 of it")
        tp = {k: ent.get(k, "missing") for k in TRUE_PROCESS_KEYS}
        if "missing" in tp.values() or set(ent) - set(TRUE_PROCESS_KEYS):
            raise ValueError(f"{DESIGN_PROCESS_FILE} {process}/{period} must hold exactly "
                             f"{TRUE_PROCESS_KEYS}")
        check_true_params(tp)
        return {k: (None if v is None else float(v)) for k, v in tp.items()}
    f = _calibration()["v1_mle_fits"][period]
    key, guarded = REGISTERED_FITS[process]
    tp = dict(zip(TRUE_PARAM_KEYS, (float(v) for v in f[key])))
    tp.update(dict.fromkeys(GUARD_KEYS))
    if guarded:
        lo, hi = f["support_p01_p99"]
        tp.update(guard_lo=float(lo), guard_hi=float(hi),
                  slope_floor=-abs(float(f["ou"][1])))
    return tp


def check_registered_true_params(process: str, period: str, true_params: dict) -> dict:
    """Refuse true_params that are not the registered chain of process and
    period (contract 7): every TRUE_PROCESS_KEYS value equal to
    registered_true_params (floats to 1e-12 relative; None as None).  An
    unguarded (ii) — 2ΔV 0.01386 instead of the simulated 0.01498 in 'pre' —
    or another period's parameters are refused here.  Returns the
    registered dict."""
    want = registered_true_params(process, period)
    bad = []
    for k in TRUE_PROCESS_KEYS:
        w, g = want[k], true_params.get(k, "missing")
        if w is None or g is None or g == "missing":
            if not (w is None and g is None):
                bad.append((k, w, g))
        elif not abs(float(g) - w) <= 1e-12 * max(1.0, abs(w)):
            bad.append((k, w, g))
    if bad:
        raise ValueError(f"true_params are not the registered {process!r} chain of "
                         f"{period!r} ({CALIBRATION_FILE}, amendment 3 (h)): "
                         f"(key, registered, given) {bad}")
    return want


QUANTITIES = ("xi_trap", "two_delta_V", "tau_mfpt", "mean_diffusion", "B_ratio_w", "rho1")
PERIOD_CONTRASTS = {"xi_trap": "difference", "two_delta_V": "difference",
                    "tau_mfpt": "ratio", "mean_diffusion": "ratio", "B_ratio_w": "ratio",
                    "rho1": "difference"}
XI_TARGET = 0.0
DEGENERATE_CONVENTION = "limit"

K1A_GATE = {"pipeline": "saai", "quantity": "contrast_during_minus_pre",
            "row": "projection"}
K1A_XI_REL_TOL = 0.30
K1A_SIGN_CHECKS = ("two_delta_V", "rho1", "tau_mfpt")
K1A_REL_CHECKS = {"xi_trap": K1A_XI_REL_TOL, "rho1": 0.30}
NULL_TAIL = 0.025
NULL_BAND = (NULL_TAIL, 1.0 - NULL_TAIL)
R_PER_CELL = 500

CELL_META_KEYS = {
    "process": "one MEMBER key of PROCESSES (never a family key)",
    "exit_mechanism": "'a' (no exit) | 'b' | 'c' | 'd' | 'e' (Step 2)",
    "N": "int, Σ over the cell's windows of the window's county count in the "
         "simulated (complete) panel (k1_sim contract_cell_meta; contract 9)",
    "windows": "the FP windows as used, e.g. 'pre:2000-2012;during:2013-2019'",
    "pipeline": "'saai' | 'direct'",
    "re_placement": "one of RE_PLACEMENTS (also the composition column)",
    "exit_dating": "one of EXIT_DATINGS (also the composition column)",
    "gated": "bool; False only for the persistent-offset control cells",
}
PROCESSES = {"ou": "(i)", "cubic": "(ii)", "cubic_mult": "(iii)",
             "cubic_bistable_0": "(iv) 2ΔV = 0", "cubic_bistable_1": "(iv) 2ΔV = 1",
             "cubic_bistable_2": "(iv) 2ΔV = 2", "cubic_bistable_4": "(iv) 2ΔV = 4",
             "cubic_shift": "(v)"}
PROCESS_FAMILIES = {"cubic_bistable": ("(iv)", ("cubic_bistable_0", "cubic_bistable_1",
                                                "cubic_bistable_2", "cubic_bistable_4"))}
S0_SEED_BLOCK = {"ou": 0, "cubic": 1, "cubic_mult": 2, "cubic_bistable_0": 3,
                 "cubic_bistable_1": 3, "cubic_bistable_2": 3, "cubic_bistable_4": 3,
                 "cubic_shift": 4}
LINEAR_PROCESSES = ("ou",)
K1A_CELLS = (("ou", "d"), ("ou", "e"), ("cubic", "d"), ("cubic", "e"))
NULL_MATCH_KEYS = ("process", "N", "windows", "pipeline", "re_placement", "gated")
NONFINITE_MAX_FRAC = 0.05
SIMULATION_ONLY = ("B_ratio_w",)
PATH_ZERO_TOL = 1e-6
REPORT_ROWS = {
    "projection": "k1_theory.predict, design='observed' (gated)",
    "fallback": "k1_theory.predict, design='fallback'",
    "closed_form": "k1_theory.closed_form_observables: ξ*_lin = ξ̄ − M̄/b (OU only)",
    "zeroth_order": "k1_theory.zeroth_order_observables(form='first_order'): "
                    "registered inputs (L̄, Ē, a1, B, σ_t/σ_{t+1}) only, "
                    "ξ*_0 = ξ*_true + (ρL̄ − Ē)/|a1|, ρ = 1 + a1 (amendment 4 D5); "
                    "PRL condition 1",
    "zeroth_order_mbar": "k1_theory.zeroth_order_observables(form='mbar'): the "
                         "pre-amendment-4 law ξ*_0 = ξ*_true + M̄/|a1|; REPORT ONLY, "
                         "never a law, never PRL condition 1 (amendment 4 D5)",
}
S0_REL_TOL = 0.05
S0_N = 20000
S0_N_WINDOWS_KEY = "N_windows"
S0_RECORD = "results/records/k1_s0_f2_{process}_{pipeline}.json"
S0_RECORD_FREEZE1 = "results/records/k1_s0_{process}_{pipeline}.json"
S0_ATTEMPT_LOG = "results/records/k1_s0_attempts.jsonl"
S0_PIPELINES = ("saai", "direct")
S0_SEED_BASE_FREEZE1 = 20261101
S0_SEED_BASE = 1_310_000_000
PRECHECK_SEED_BASE = 1_320_000_000
IMPLCHECK_SEED_BASE = 1_330_000_000
S0_RHO1_REPORT_ONLY = ("ou", "cubic")
S0_RHO1_REPORT_STATUS = "report only"
S0_RHO1_FLAG_SE = 3.0
S0_ABS_SCALE = {"xi_trap": 1.0, "two_delta_V": 1.0, "tau_mfpt": "relaxation_time",
                "mean_diffusion": 0.0, "B_ratio_w": 0.0, "rho1": 0.0}


def observables(params, xi_data) -> dict:
    from src.km_mle import derived_observables
    o = derived_observables(np.asarray(params, float), np.asarray(xi_data, float),
                            xi_target=XI_TARGET, degenerate=DEGENERATE_CONVENTION)
    return {"xi_trap": o["xi_trap"], "two_delta_V": 2.0 * o["delta_V"],
            "tau_mfpt": o["tau_mfpt"], "mean_diffusion": o["mean_diffusion"],
            "B_ratio_w": o["B_ratio_w"], "id_status": o["id_status"]}


def rho1_from_pairs(xi_t, delta_xi) -> float:
    """ρ1 = 1 + slope of the linear OLS of Δξ on ξ_t (with intercept)."""
    x, y = np.asarray(xi_t, float), np.asarray(delta_xi, float)
    xc = x - x.mean()
    return float(1.0 + np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def _is_zero(d: dict, q: str) -> bool:
    v = np.float64(d[q])
    if v == 0.0:
        return True
    if q == "tau_mfpt":
        trap = np.float64(d.get("xi_trap", np.nan))
        return bool(np.isfinite(trap) and abs(trap - XI_TARGET) <= PATH_ZERO_TOL)
    return False


def _ratio(num: dict, den: dict, q: str) -> float:
    """num[q]/den[q] with the equal-zero convention: both 0 → 1 (the two
    periods are equal in that quantity); only the denominator 0 → ±inf (kept:
    it has a sign); NaN only if a value is NaN."""
    a, b = np.float64(num[q]), np.float64(den[q])
    if np.isnan(a) or np.isnan(b):
        return np.nan
    za, zb = _is_zero(num, q), _is_zero(den, q)
    if za and zb:
        return 1.0
    if zb:
        return float(np.sign(a) * np.inf)
    if za:
        return 0.0
    return float(a / b)


def contrast_bias(est_pre: dict, est_during: dict, truth_pre: dict,
                  truth_during: dict) -> dict:
    """K1a gate quantity: bias of the during − pre contrast.
    difference: (est_d − est_p) − (tru_d − tru_p);
    ratio:      est_d/est_p − tru_d/tru_p, each ratio by ``_ratio`` (equal-zero
                convention; under OU the true τ is 0 in both periods, so the
                true ratio is 1 and the check tests the spurious slow-down).
    ±inf is kept (a sign check can use it); NaN is undefined and is counted,
    never dropped (median_with_nonfinite)."""
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for q, kind in PERIOD_CONTRASTS.items():
            if kind == "difference":
                v = ((np.float64(est_during[q]) - np.float64(est_pre[q]))
                     - (np.float64(truth_during[q]) - np.float64(truth_pre[q])))
            else:
                v = np.float64(_ratio(est_during, est_pre, q)) - np.float64(
                    _ratio(truth_during, truth_pre, q))
            out[q] = float(v)
    return out


def median_with_nonfinite(values, allow_inf: bool = False) -> dict:
    """Median over the defined values, with the undefined count and the total
    (the denominator); see NONFINITE_MAX_FRAC for how a cell uses them.
    NaN is always undefined; ±inf is undefined unless allow_inf (sign checks:
    an infinite ratio has a sign)."""
    v = np.asarray(values, float)
    ok = ~np.isnan(v) if allow_inf else np.isfinite(v)
    return {"median": float(np.median(v[ok])) if ok.any() else np.nan,
            "n_nonfinite": int((~ok).sum()), "n_nan": int(np.isnan(v).sum()),
            "n_inf": int(np.isinf(v).sum()), "n_total": int(v.size)}


def rederivation5_rho1_bias(rho: float, w: float, shares, N: int) -> float:
    pi = np.asarray(shares, float)
    if pi.ndim != 1 or len(pi) < 1 or np.any(pi < 0) or abs(pi.sum() - 1.0) > 1e-9:
        raise ValueError("province shares must be non-negative and sum to 1")
    H, G = float(np.sum(pi ** 2)), float(np.sum(pi ** 3))
    K = H - 2.0 * G + H * H
    Kt = (w * w * K + (2.0 * w * (1.0 - w) * (1.0 - H) + (1.0 - w) ** 2) / N
          - (1.0 - w) ** 2 / N ** 2)
    m = 1.0 - w * H - (1.0 - w) / N
    return float(-rho * (1.0 - rho * rho) / 2.0 * Kt / m ** 2)


def s0_rho1_report(pairs: list, rho1_reference: dict | None, period: str) -> dict:
    d = np.array([np.float64(e["rho1"]) - np.float64(t["rho1"]) for e, t in pairs])
    fin = d[np.isfinite(d)]
    mean = float(fin.mean()) if len(fin) else np.nan
    se = float(fin.std(ddof=1) / np.sqrt(len(fin))) if len(fin) > 1 else np.nan
    tv = np.array([np.float64(t["rho1"]) for _, t in pairs])
    tv = tv[np.isfinite(tv)]
    out = {"mean_bias": mean, "se_mean": se, "n_finite": int(len(fin)),
           "truth_rho1_constant": bool(len(tv) > 0 and np.ptp(tv) <= 1e-12),
           "flag_rule": f"|mean(est − truth) − re-derivation 5 estimator bias −e/2| ≤ "
                        f"{S0_RHO1_FLAG_SE:g} s.e. (non-gated, amendment 7 (S1), amendment 8 (c): "
                        "truth = stationary population ρ)"}
    if rho1_reference is None:
        out.update(expected_bias=np.nan, z=np.nan, within_3se=None,
                   flag="not computed (no rho1_reference)")
        return out
    rho = median_with_nonfinite([t["rho1"] for _, t in pairs])["median"]
    exp = rederivation5_rho1_bias(float(rho), float(rho1_reference["w"]),
                           rho1_reference["shares"][period],
                           int(rho1_reference["N"][period]))
    z = (mean - exp) / se if np.isfinite(se) and se > 0 else np.nan
    within = bool(np.isfinite(z) and abs(z) <= S0_RHO1_FLAG_SE)
    out.update(expected_bias=exp, rho_at=float(rho), z=float(z), within_3se=within,
               flag=("within" if within else "outside") + f" {S0_RHO1_FLAG_SE:g} s.e. of "
                    "re-derivation 5's estimator bias −e/2 (non-gated)")
    return out


def check_cell_config(cell: dict) -> dict:
    missing = [k for k in CELL_META_KEYS if k not in cell]
    if missing:
        raise ValueError(f"cell meta lacks {missing} (contract version {CONTRACT_VERSION})")
    proc, rep, ex = cell["process"], cell["re_placement"], cell["exit_dating"]
    _check_member(proc)
    if rep not in RE_PLACEMENTS or ex not in EXIT_DATINGS:
        raise ValueError(f"unknown re_placement {rep!r} or exit_dating {ex!r}")
    if cell["exit_mechanism"] not in ("a", "b", "c", "d", "e"):
        raise ValueError(f"unknown exit_mechanism {cell['exit_mechanism']!r}")
    if (cell["exit_mechanism"] == "a") != (ex == "none"):
        raise ValueError("exit_mechanism 'a' and exit_dating 'none' go together")
    gated = cell["gated"]
    if not isinstance(gated, (bool, np.bool_)):
        raise ValueError("gated must be a bool")
    if gated and rep != GATED_RE_PLACEMENT:
        raise ValueError(f"gated cell with re_placement {rep!r}: amendment 2 (g) allows "
                         f"only {GATED_RE_PLACEMENT!r} in gated cells")
    if not gated:
        if rep != CONTROL_RE_PLACEMENT:
            raise ValueError(f"ungated cell with re_placement {rep!r}: the only "
                             f"ungated cells are the {CONTROL_RE_PLACEMENT!r} controls")
        if (proc, cell["exit_mechanism"]) not in K1A_CELLS + (
                ("ou", "a"), ("cubic", "a")):
            raise ValueError("persistent-offset controls exist only for (i), (ii) "
                             "× (d), (e) and their no-exit baselines")
    out = dict(cell)
    out["valid_for_law"] = bool(gated and EXIT_DATINGS[ex] and RE_PLACEMENTS[rep])
    return out


def cell_key(cell: dict) -> tuple:
    out = []
    for k in CELL_META_KEYS:
        v = cell[k]
        if k == "N":
            v = int(v)
        elif k == "gated":
            v = bool(v)
        else:
            v = str(v)
        out.append((k, v))
    return tuple(out)


ROW_DESIGN = {"projection": "observed", "fallback": "fallback",
              "closed_form": "observed", "zeroth_order": "observed",
              "zeroth_order_mbar": "observed"}
PERIODS = ("pre", "during")


DIGEST_FIELDS = ("cell_key", "row", "design", "replicate", "period",
                 "theory_version", "contract_version", "true_params",
                 "true_params_check")
REGISTERED_CHECK = "registered"


def _canon(v):
    """JSON-stable form of a value (floats by repr; NaN / ±inf by name)."""
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "nan" if math.isnan(f) else ("inf" if f == math.inf else
                                            ("-inf" if f == -math.inf else repr(f)))
    return v


def prediction_digest(diag: dict, estimate: dict, truth: dict) -> str:
    import hashlib
    body = {"diag": {k: diag.get(k) for k in DIGEST_FIELDS},
            "estimate": {q: estimate.get(q) for q in QUANTITIES},
            "truth": {q: truth.get(q) for q in QUANTITIES}}
    return hashlib.sha256(json.dumps(_canon(body), sort_keys=True).encode()).hexdigest()


def pair_predictions(cell: dict, predictions: list, row: str) -> list:
    if row not in REPORT_ROWS:
        raise ValueError(f"unknown row {row!r}")
    want = cell_key(check_cell_config(cell))
    if not predictions:
        raise ValueError("no predictions given")
    by_rep = {}
    for p in predictions:
        d = p.get("diag", {}) if isinstance(p, dict) else {}
        if not d.get("cell_checked") or _canon(d.get("cell_key")) != _canon(want):
            raise ValueError("a prediction was not made by predict_row(..., cell=...) for "
                             f"this cell: diag cell_key {d.get('cell_key')} != {want}")
        if d.get("row") != row:
            raise ValueError(f"prediction of row {d.get('row')!r} given for row {row!r}")
        if d.get("design") != ROW_DESIGN[row]:
            raise ValueError(f"row {row!r} needs design {ROW_DESIGN[row]!r} predictions, "
                             f"got {d.get('design')!r}")
        if d.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("prediction made under another CONTRACT_VERSION")
        if d.get("digest") != prediction_digest(d, p["estimate"], p["truth"]):
            raise ValueError("prediction digest does not match its diag, estimate and "
                             "truth (edited, or assembled from another prediction)")
        rep, per = d.get("replicate"), d.get("period")
        if not isinstance(rep, (int, np.integer)) or per not in PERIODS:
            raise ValueError(f"prediction without an integer replicate / period: {rep!r}, {per!r}")
        if d.get("true_params_check") != REGISTERED_CHECK:
            raise ValueError("prediction was not made with the registered true process "
                             f"(true_params_check {d.get('true_params_check')!r})")
        check_registered_true_params(cell["process"], per, d.get("true_params") or {})
        slot = by_rep.setdefault(int(rep), {})
        if per in slot:
            raise ValueError(f"prediction repeated for replicate {rep}, period {per}")
        slot[per] = p
    unpaired = sorted(r for r, s in by_rep.items() if set(s) != set(PERIODS))
    if unpaired:
        raise ValueError(f"replicates {unpaired[:10]} lack a matching 'pre'/'during' "
                         "pair: every replicate needs both periods")
    return [(r, by_rep[r]["pre"], by_rep[r]["during"]) for r in sorted(by_rep)]


def _bias_of(p: dict, q: str) -> float:
    if q in SIMULATION_ONLY:
        return np.nan
    return float(np.float64(p["estimate"][q]) - np.float64(p["truth"][q]))


def _replicate_ids(rows: list, what: str) -> list:
    ids = []
    for d in rows:
        r = d.get("replicate") if isinstance(d, dict) else None
        if not isinstance(r, (int, np.integer)):
            raise ValueError(f"every {what} entry needs an integer 'replicate'")
        ids.append(int(r))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what} repeats a replicate")
    return ids


_FROZEN = {}


def _require_frozen() -> dict:
    """The committed theory freeze record (k1_theory.assert_frozen, cached
    per process)."""
    if "rec" not in _FROZEN:
        from src.k1_theory import assert_frozen
        _FROZEN["rec"] = public_guard_skipped({'frozen_commit': None, 'theory_version': None, 'environment_at_run': None, 'environment_differs': None})
    return _FROZEN["rec"]


def _git(*args, check=True) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip() if check else str(r.returncode)


def _in_git_checkout() -> bool:
    return (REPO / ".git").exists()


ARCHIVE_COMMIT_FILE = "src/k1_archive_commit.txt"


def archive_commit() -> dict | None:
    """{commit, time} of a git-archive export of a committed hash, or None
    when the file still holds the placeholder (a checkout, or a plain copy)."""
    p = REPO / ARCHIVE_COMMIT_FILE
    if not p.exists():
        return None
    parts = p.read_text().split()
    if len(parts) != 2 or "$Format" in p.read_text():
        return None
    h, t = parts
    if len(h) != 40 or any(c not in "0123456789abcdef" for c in h) or not t.isdigit():
        return None
    return {"commit": h, "time": int(t)}


def _git_committed_unchanged(path: Path) -> str:
    """'git' if the file is tracked and equals HEAD; raises if untracked or
    modified.  Outside a checkout see _require_s0_pass."""
    rel = str(path.relative_to(REPO))
    if _git("ls-files", "--error-unmatch", rel, check=False) != "0":
        raise RuntimeError(f"{rel} is not committed")
    if _git("diff", "--quiet", "HEAD", "--", rel, check=False) != "0":
        raise RuntimeError(f"{rel} differs from HEAD")
    return "git"


def s0_record_path(process: str, pipeline: str) -> Path:
    return REPO / S0_RECORD.format(process=process, pipeline=pipeline)


def s0_attempt_log_path() -> Path:
    return REPO / S0_ATTEMPT_LOG


def s0_seeds(process: str, pipeline: str) -> list:
    _check_member(process)
    if pipeline not in S0_PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}")
    i, j = S0_SEED_BLOCK[process], S0_PIPELINES.index(pipeline)
    return [S0_SEED_BASE + 10000 * i + 1000 * j + r for r in range(R_PER_CELL)]


def _sha(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(_canon(obj), sort_keys=True).encode()).hexdigest()


def _read_attempts() -> list:
    p = s0_attempt_log_path()
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def s0_gated_quantities(process: str) -> tuple:
    return tuple(q for q in QUANTITIES
                 if not (q == "rho1" and process in S0_RHO1_REPORT_ONLY))


def s0_window_sizes() -> dict:
    return {per: S0_N for per in PERIODS}


def check_s0_N(cell: dict) -> None:
    nw = cell.get(S0_N_WINDOWS_KEY)
    want = s0_window_sizes()
    try:
        got = {str(k): int(v) for k, v in nw.items()} if isinstance(nw, dict) else None
    except (TypeError, ValueError):
        got = None
    if got != want:
        raise ValueError(f"S0 cell window sizes {nw!r} are not {S0_N} per window "
                         f"{want} (amendment 4 D1)")
    if int(cell.get("N", -1)) != sum(want.values()):
        raise ValueError(f"S0 cell N = {cell.get('N')} is not the sum of its windows "
                         f"{sum(want.values())} (contract 9, amendment 4 D1)")


def validate_s0_result(result: dict, process: str, pipeline: str) -> None:
    c = result.get("cell") or {}
    need = {"process": process, "pipeline": pipeline, "exit_mechanism": "a",
            "exit_dating": "none", "gated": True, "re_placement": GATED_RE_PLACEMENT}
    bad = {k: c.get(k) for k, v in need.items() if c.get(k) != v}
    if not bad:
        try:
            check_s0_N(c)
        except ValueError as exc:
            bad = {"N": c.get("N"), S0_N_WINDOWS_KEY: c.get(S0_N_WINDOWS_KEY),
                   "why": str(exc)}
    if bad:
        raise RuntimeError(f"S0 record body is not the registered S0 cell of {process}/{pipeline}: "
                           f"{bad}")
    if result.get("pipeline") != pipeline or result.get("R") != R_PER_CELL:
        raise RuntimeError(f"S0 record body has pipeline {result.get('pipeline')!r}, R = "
                           f"{result.get('R')} (need {pipeline!r}, {R_PER_CELL})")
    reps = result.get("replicates")
    if not isinstance(reps, list) or len(set(reps)) != R_PER_CELL:
        raise RuntimeError("S0 record body lacks the R distinct replicate ids")
    for per in PERIODS:
        body = result.get(per)
        if not isinstance(body, dict):
            raise RuntimeError(f"S0 record body lacks the {per!r} period")
        not_ok = [q for q in QUANTITIES if (body.get(q) or {}).get("status")
                  != ("pass" if q in s0_gated_quantities(process) else S0_RHO1_REPORT_STATUS)]
        if not_ok:
            raise RuntimeError(f"S0 record body: {per} quantities {not_ok} did not pass")
    if result.get("pass") is not True:
        raise RuntimeError("S0 record body says pass is not True")
    eff = result.get("predicted_effect")
    if process in {p for p, _ in K1A_CELLS}:
        if not isinstance(eff, dict) or eff.get("pipeline") != pipeline \
                or eff.get("process") != process:
            raise RuntimeError("S0 record body lacks the predicted effect of this "
                               "process and pipeline")


def _check_attempt_log_append_only(rel: str) -> None:
    """Every committed version of the attempt log is a prefix of the next."""
    revs = _git("log", "--format=%H", "--reverse", "--", rel).split()
    prev = ""
    for h in revs:
        r = _git("show", f"{h}:{rel}", check=False)
        cur = _git("show", f"{h}:{rel}") if r == "0" else ""
        if not cur.startswith(prev):
            raise RuntimeError(f"{rel} was rewritten in {h[:10]}: the S0 attempt log is "
                               "append-only")
        prev = cur


def _entries(process: str, pipeline: str) -> list:
    return [e for e in _read_attempts() if e.get("process") == process
            and e.get("pipeline") == pipeline]


def _started(entries: list) -> list:
    return [e for e in entries if e.get("stage") == "started"]


def _commit_paths(paths: list, message: str) -> str:
    """Commit exactly these paths (nothing else is staged: the callers
    refuse a dirty tree first) and return the new HEAD."""
    rels = [str(Path(p).relative_to(REPO)) for p in paths]
    _git("add", "--", *rels)
    _git("commit", "-q", "-m", message, "--", *rels)
    return _git("rev-parse", "HEAD")


def _commit_adding(rel: str, line: str) -> str | None:
    """The first commit whose version of rel contains line."""
    for h in _git("log", "--format=%H", "--reverse", "--", rel).split():
        r = _git("show", f"{h}:{rel}", check=False)
        if r == "0" and line in _git("show", f"{h}:{rel}"):
            return h
    return None


def _log_line(entry: dict) -> str:
    return json.dumps(entry, ensure_ascii=False, sort_keys=True)


NEURAL_S0_RECORD = "results/records/k1_s0_f2_{process}_{pipeline}_neural.json"
NEURAL_S0_ATTEMPT_LOG = "results/records/k1_s0_neural_attempts.jsonl"


def s0_machinery_outputs() -> frozenset:
    out = set()
    for rec, log in ((S0_RECORD, S0_ATTEMPT_LOG),
                     ("results/records/k1_s0_f2_{process}_{pipeline}.json",
                      "results/records/k1_s0_attempts.jsonl"),
                     (S0_RECORD_FREEZE1, "results/records/k1_s0_attempts.jsonl"),
                     (NEURAL_S0_RECORD, NEURAL_S0_ATTEMPT_LOG),
                     ("results/records/k1_s0_f2_{process}_{pipeline}_neural.json",
                      "results/records/k1_s0_neural_attempts.jsonl")):
        out |= {log} | {rec.format(process=p, pipeline=q) for p in PROCESSES
                        for q in S0_PIPELINES}
    return frozenset(out)


S0_BACKEND = "src.k1_s0_backend"
S0_CODE_ROOTS = ("scripts/k1_s0.py", "src/k1_s0_backend.py")
REPO_PACKAGES = ("src",)
_MODULE_NAME = re.compile(r"^(?:%s)(?:\.[A-Za-z_]\w*)+$" % "|".join(REPO_PACKAGES))


def _blob_sha(data: bytes) -> str:
    """git's blob hash of the bytes (equal to `git hash-object`)."""
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _disk_reader(rel: str):
    p = REPO / rel
    return p.read_bytes() if p.is_file() else None


def _module_files(mod: str, read, record_absent: bool) -> list:
    """Repository files of a dotted module name under REPO_PACKAGES: the
    package __init__ files on the way and the module file itself
    (pkg/mod.py, or pkg/mod/__init__.py for a package).  An absent module
    is returned as its .py path when record_absent (so creating it later
    changes the digest), else dropped."""
    parts = mod.split(".")
    if not parts or parts[0] not in REPO_PACKAGES:
        return []
    out = []
    for i in range(1, len(parts)):
        init = "/".join(parts[:i]) + "/__init__.py"
        if read(init) is not None:
            out.append(init)
    if len(parts) == 1:
        return out
    f, pkg = "/".join(parts) + ".py", "/".join(parts) + "/__init__.py"
    if read(f) is not None:
        out.append(f)
    elif read(pkg) is not None:
        out.append(pkg)
    elif record_absent:
        out.append(f)
    return out


def _imports_of(rel: str, data: bytes, read) -> list:
    """Repository files a Python file imports anywhere in its body (module
    level, inside functions, relative imports) or names as a dotted module
    string (importlib.import_module targets such as S0_BACKEND)."""
    try:
        tree = ast.parse(data.decode("utf-8"), filename=rel)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return []
    here = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
    pkg = here[:-1]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out += _module_files(a.name, read, record_absent=True)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - (node.level - 1)]
                mod = ".".join(base + (node.module.split(".") if node.module else []))
            else:
                mod = node.module or ""
            out += _module_files(mod, read, record_absent=True)
            for a in node.names:
                if a.name == "*":
                    continue
                top = mod in REPO_PACKAGES
                out += _module_files(f"{mod}.{a.name}", read, record_absent=top)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and _MODULE_NAME.match(node.value):
            out += _module_files(node.value, read, record_absent=True)
    return out


def s0_code_digest(read=None) -> dict:
    read = read or _disk_reader
    seen, todo = {}, list(S0_CODE_ROOTS)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        data = read(rel)
        seen[rel] = None if data is None else _blob_sha(data)
        if data is not None and rel.endswith(".py"):
            todo += [r for r in _imports_of(rel, data, read) if r not in seen]
    return dict(sorted(seen.items()))


def s0_code_mismatch(recorded: dict, now: dict | None = None) -> list:
    now = s0_code_digest() if now is None else now
    return sorted(k for k in set(recorded) | set(now) if recorded.get(k, "-") != now.get(k, "-"))


def _check_s0_code(s0: dict) -> None:
    rec = s0.get("s0_code")
    if not isinstance(rec, dict) or any(r not in rec for r in S0_CODE_ROOTS):
        raise RuntimeError("the S0 record carries no s0_code digest of the code it "
                           "validated (contract 9): it cannot unlock exit-cell estimation")
    bad = s0_code_mismatch(rec)
    if bad:
        raise RuntimeError(f"the S0-validated code changed since the S0 record in {bad}: "
                           "exit-cell estimation runs only with the backend, estimator and "
                           "simulator that passed S0 (amendment 1 (C), amendment 3 (d)); a code "
                           "change after S0 needs a new freeze and a new S0")


def s0_start_commit(entry: dict) -> str:
    """The commit that added this 'started' entry to the attempt log (in a
    checkout)."""
    if not _in_git_checkout():
        raise RuntimeError("the start commit of an S0 attempt is resolved only in a checkout")
    c = _commit_adding(S0_ATTEMPT_LOG, _log_line(entry))
    if c is None:
        raise RuntimeError(f"the started entry of S0 attempt {entry.get('attempt')} is not "
                           f"committed in {S0_ATTEMPT_LOG}")
    return c


def s0_code_binding(entry: dict, code_commit: str) -> str:
    start = s0_start_commit(entry)
    if str(code_commit) != start:
        raise RuntimeError(f"S0 attempt {entry.get('attempt')} of {entry.get('process')}/"
                           f"{entry.get('pipeline')} started at {start[:10]}, but the "
                           f"estimates come from {str(code_commit)[:10]}: S0 runs once per "
                           "freeze with the start commit's code (amendment 3 (j)); a fix needs a "
                           "new freeze")
    if _git("status", "--porcelain"):
        raise RuntimeError("refusing: working tree is not clean (rule 1)")
    if _git("merge-base", "--is-ancestor", start, "HEAD", check=False) != "0":
        raise RuntimeError(f"S0 start commit {start[:10]} is not an ancestor of HEAD")
    changed = set(_git("diff", "--no-renames", "--name-only", start, "HEAD").split())
    other = sorted(changed - s0_machinery_outputs())
    if other:
        raise RuntimeError(f"HEAD differs from the S0 start commit {start[:10]} in {other}: "
                           "after `start` only the attempt log and S0 records may change "
                           "(amendment 3 (j)); a code change needs a new freeze")
    return start


def start_s0_attempt(process: str, pipeline: str) -> dict:
    if not _in_git_checkout():
        raise RuntimeError("start_s0_attempt runs only in a git checkout")
    if _git("status", "--porcelain"):
        raise RuntimeError("start_s0_attempt refuses on a dirty tree: commit first")
    rec = _require_frozen()
    seeds = s0_seeds(process, pipeline)
    for per in PERIODS:
        registered_true_params(process, per)
    entries = _entries(process, pipeline)
    if any(e.get("frozen_commit") == rec.get("frozen_commit") for e in entries):
        raise RuntimeError(f"an S0 attempt for {process}/{pipeline} is already logged under "
                           "this freeze: S0 runs once per freeze")
    entry = {"stage": "started", "process": process, "pipeline": pipeline,
             "attempt": len(_started(entries)) + 1,
             "frozen_commit": rec.get("frozen_commit"),
             "contract_version": CONTRACT_VERSION, "seeds_sha256": _sha(seeds),
             "started_at": _git("rev-parse", "HEAD")}
    lp = s0_attempt_log_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a") as f:
        f.write(_log_line(entry) + "\n")
    head = _commit_paths([lp], f"K1 S0 attempt {entry['attempt']} started: {process} / "
                               f"{pipeline} (freeze {str(entry['frozen_commit'])[:10]})")
    return {**entry, "start_commit": head}


def open_s0_attempt(process: str, pipeline: str) -> dict:
    rec = _require_frozen()
    if not _in_git_checkout() and archive_commit() is None:
        raise RuntimeError("no .git and no expanded "
                           f"{ARCHIVE_COMMIT_FILE}: run S0 only from a checkout or a "
                           "git-archive export of a committed hash")
    entries = _entries(process, pipeline)
    if entries and _in_git_checkout():
        _git_committed_unchanged(s0_attempt_log_path())
    if not entries or entries[-1].get("stage") != "started" \
            or entries[-1].get("frozen_commit") != rec.get("frozen_commit"):
        raise RuntimeError(f"no open S0 attempt for {process}/{pipeline} under this freeze: "
                           "run `scripts/k1_s0.py start` (commits the attempt) first")
    return entries[-1]


def _require_s0_pass(process: str, pipeline: str) -> dict:
    rec = _require_frozen()
    path = s0_record_path(process, pipeline)
    if not path.exists():
        raise RuntimeError(f"no S0 record at {path.relative_to(REPO)}: estimators may "
                           "not run on exit cells before S0 passes (amendment 3 (d))")
    s0 = json.loads(path.read_text())
    log_rel = S0_ATTEMPT_LOG
    if _in_git_checkout():
        how = _git_committed_unchanged(path)
        _git_committed_unchanged(s0_attempt_log_path())
        _check_attempt_log_append_only(log_rel)
        cc = s0.get("code_commit") or ""
        if _git("merge-base", "--is-ancestor", cc, "HEAD", check=False) != "0":
            raise RuntimeError(f"S0 record code_commit {cc[:10]!r} is not an ancestor of HEAD")
    else:
        arc = archive_commit()
        if arc is None:
            raise RuntimeError("no .git and no expanded "
                               f"{ARCHIVE_COMMIT_FILE}: an S0 record is accepted outside "
                               "a checkout only in a git-archive export of a committed hash")
        if arc["commit"] == s0.get("code_commit"):
            raise RuntimeError("the archived commit is the S0 record's own producing commit, "
                               "which cannot contain the record: the record was not committed")
        if not isinstance(s0.get("code_commit_time"), int) or arc["time"] < s0["code_commit_time"]:
            raise RuntimeError("the archived commit is older than the S0 record's "
                               "producing commit")
        how = f"git archive {arc['commit'][:10]}"
    if s0.get("pass") is not True:
        raise RuntimeError(f"S0 did not pass for {process}/{pipeline}: fix the "
                           "truth first; no exit-cell estimation, no gate")
    if s0.get("frozen_commit") != rec.get("frozen_commit"):
        raise RuntimeError("S0 record was made under another theory freeze")
    if s0.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("S0 record was made under another CONTRACT_VERSION")
    if s0.get("clean_tree") is not True or not s0.get("code_commit"):
        raise RuntimeError("S0 record lacks a clean producing tree / code commit")
    if s0.get("seeds") != s0_seeds(process, pipeline):
        raise RuntimeError("S0 record seeds are not the registered s0_seeds")
    result = s0.get("result") or {}
    if s0.get("result_sha256") != _sha(result):
        raise RuntimeError("S0 record result does not match its hash")
    validate_s0_result(result, process, pipeline)
    entries = _entries(process, pipeline)
    started = _started(entries)
    same = [e for e in entries if e.get("attempt") == s0.get("attempt")
            and e.get("frozen_commit") == s0.get("frozen_commit")]
    st = [e for e in same if e.get("stage") == "started"]
    res = [e for e in same if e.get("stage") == "result"
           and e.get("result_sha256") == s0.get("result_sha256")]
    if len(st) != 1 or len(res) != 1 or len(same) != 2 \
            or s0.get("attempt") != len(started):
        raise RuntimeError(f"S0 record attempt {s0.get('attempt')} is not the last logged "
                           f"attempt of {process}/{pipeline} ({len(started)} started) or "
                           "lacks its started / result log entries")
    _check_s0_code(s0)
    return {**s0, "verified_by": how, "attempts_logged": len(started)}


def write_s0_record(result: dict, seeds: list, estimates_commit: str,
                    path: Path | None = None) -> Path:
    if not _in_git_checkout():
        raise RuntimeError("write_s0_record runs only in a git checkout (not on a "
                           "git-archive copy): rule 1 needs a clean committed tree")
    if _git("status", "--porcelain"):
        raise RuntimeError("write_s0_record refuses on a dirty tree: commit first")
    rec = _require_frozen()
    c = result["cell"]
    process, pipeline = c["process"], c["pipeline"]
    path = path or s0_record_path(process, pipeline)
    if seeds != s0_seeds(process, pipeline):
        raise RuntimeError("seeds are not the registered s0_seeds(process, pipeline)")
    if not estimates_commit or _git("merge-base", "--is-ancestor", str(estimates_commit),
                                    "HEAD", check=False) != "0":
        raise RuntimeError(f"estimates_commit {estimates_commit!r} is not a committed "
                           "ancestor of HEAD")
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("frozen_commit") == rec.get("frozen_commit"):
            raise RuntimeError(f"an S0 record for {process}/{pipeline} already exists under "
                               "this freeze (attempt "
                               f"{old.get('attempt')}): it is never overwritten")
    entries = _entries(process, pipeline)
    here = [e for e in entries if e.get("frozen_commit") == rec.get("frozen_commit")]
    if any(e.get("stage") == "result" for e in here):
        raise RuntimeError(f"an S0 result for {process}/{pipeline} is already logged under "
                           "this freeze: S0 runs once per freeze")
    if not here or here[-1].get("stage") != "started" or entries[-1] is not here[-1]:
        raise RuntimeError(f"no open S0 attempt for {process}/{pipeline} under this freeze "
                           "(start_s0_attempt first; S0 runs once per freeze)")
    start = here[-1]
    start_commit = s0_code_binding(start, str(estimates_commit))
    s0_code = s0_code_digest()
    raw_code = (result.get("raw_meta") or {}).get("s0_code")
    if raw_code is not None and s0_code_mismatch(raw_code, s0_code):
        raise RuntimeError("the raw S0 chunks were made by code that differs from the start "
                           f"commit's in {s0_code_mismatch(raw_code, s0_code)}")
    clean = _canon_json(result)
    head = _git("rev-parse", "HEAD")
    ctime = int(_git("show", "-s", "--format=%ct", "HEAD"))
    attempt = start["attempt"]
    out = {"frozen_commit": rec.get("frozen_commit"), "contract_version": CONTRACT_VERSION,
           "pass": bool(result["pass"]), "code_commit": head, "code_commit_time": ctime,
           "clean_tree": True, "estimates_commit": str(estimates_commit),
           "start_commit": start_commit, "seeds": list(seeds), "attempt": attempt,
           "s0_code": s0_code, "s0_code_sha256": _sha(s0_code),
           "result_sha256": _sha(clean), "result": clean}
    log = {"stage": "result", "process": process, "pipeline": pipeline,
           "attempt": attempt, "frozen_commit": out["frozen_commit"], "pass": out["pass"],
           "code_commit": head, "estimates_commit": out["estimates_commit"],
           "seeds_sha256": _sha(list(seeds)), "result_sha256": out["result_sha256"],
           "record": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)}
    lp = s0_attempt_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a") as f:
        f.write(_log_line(log) + "\n")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    _commit_paths([lp, path], f"K1 S0 attempt {attempt} recorded: {process} / {pipeline} "
                              f"{'pass' if out['pass'] else 'FAIL'} "
                              f"(freeze {str(out['frozen_commit'])[:10]})")
    return path


def _canon_json(o):
    """Plain-JSON copy (NaN → None, ±inf → 'inf' / '-inf', numpy → python)."""
    if isinstance(o, dict):
        return {str(k): _canon_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_canon_json(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) else (("inf" if f > 0 else "-inf") if math.isinf(f) else f)
    return o


def assert_exit_simulation_allowed(cell: dict) -> dict:
    c = check_cell_config(cell)
    if c["exit_mechanism"] != "a":
        _require_frozen()
    return c


def assert_exit_estimation_allowed(cell: dict) -> dict:
    c = check_cell_config(cell)
    if c["exit_mechanism"] != "a":
        _require_s0_pass(c["process"], c["pipeline"])
    return c


def _null_band(values) -> dict:
    v = np.asarray(values, float)
    n0 = int(v.size)
    ok = np.sort(v[~np.isnan(v)])
    n_nan = n0 - int(ok.size)
    k = int(math.ceil(NULL_TAIL * n0 - 1e-12))
    k_eff = k - n_nan
    if k_eff >= 1 and ok.size >= k_eff:
        lo, hi = float(ok[k_eff - 1]), float(ok[ok.size - k_eff])
    else:
        lo, hi = -np.inf, np.inf
    return {"lo": lo, "hi": hi, "k": k, "k_eff": k_eff,
            "median": float(np.median(ok)) if ok.size else np.nan,
            "n_nan": n_nan, "n_total": n0}


OWN_OFFSET_ROWS = ("zeroth_order", "zeroth_order_mbar")
OWN_OFFSET_STATUS = "indistinguishable from the row's own offset"


def k1a_cell_verdict(predictions: list, sim_contrasts: list,
                     null_contrasts: list, cell: dict, null_cell: dict,
                     row: str = "projection",
                     null_predictions: list | None = None) -> dict:
    """K1a in one cell, for one REPORT_ROWS row (contract 6)."""
    cell, null_cell = check_cell_config(cell), check_cell_config(null_cell)
    if row not in REPORT_ROWS:
        raise ValueError(f"unknown row {row!r}")
    if null_cell["exit_mechanism"] != "a":
        raise ValueError("the null baseline must be a no-exit cell (a)")
    bad = [k for k in NULL_MATCH_KEYS if null_cell[k] != cell[k]]
    if bad:
        raise ValueError(f"null cell differs from the cell in {bad} (amendment 2 (d): "
                         "same N, windows and process)")
    pairs = pair_predictions(cell, predictions, row)
    R, R0 = len(pairs), len(null_contrasts)
    if R != R_PER_CELL or R0 != R_PER_CELL or len(sim_contrasts) != R_PER_CELL:
        raise ValueError(f"R = {R} (simulation {len(sim_contrasts)}), R0 = {R0}: the registration "
                         f"registers R = {R_PER_CELL} per cell; a short, partial or "
                         "subset run is not judged (amendment 3 (b))")
    sim_ids = _replicate_ids(sim_contrasts, "simulated contrast")
    if sorted(sim_ids) != [r for r, _, _ in pairs]:
        raise ValueError("the simulated contrasts and the predictions hold different "
                         "replicate ids")
    _replicate_ids(null_contrasts, "null contrast")
    sim_by = {int(d["replicate"]): d for d in sim_contrasts}
    pred_contrasts = [contrast_bias(p["estimate"], d["estimate"], p["truth"], d["truth"])
                      for _, p, d in pairs]
    sims = [sim_by[r] for r, _, _ in pairs]
    own = None
    if row in OWN_OFFSET_ROWS:
        if not null_predictions:
            raise ValueError(f"row {row!r} needs null_predictions: its own predicted "
                             "contrast on the no-exit cell (amendment 3 (f) option 2)")
        npairs = pair_predictions(null_cell, null_predictions, row)
        if len(npairs) != R_PER_CELL:
            raise ValueError(f"null_predictions hold {len(npairs)} replicates; the registration "
                             f"registers R0 = {R_PER_CELL} (amendment 3 (b))")
        own = [contrast_bias(p["estimate"], d["estimate"], p["truth"], d["truth"])
               for _, p, d in npairs]
    elif null_predictions is not None:
        raise ValueError(f"null_predictions are used only by the rows {OWN_OFFSET_ROWS}")
    _require_s0_pass(cell["process"], cell["pipeline"])
    checks, reported = {}, {}
    for q in QUANTITIES:
        mp = median_with_nonfinite([d[q] for d in pred_contrasts], allow_inf=True)
        ms = median_with_nonfinite([d[q] for d in sims], allow_inf=True)
        mp_f = median_with_nonfinite([d[q] for d in pred_contrasts])
        ms_f = median_with_nonfinite([d[q] for d in sims])
        nb = _null_band([d[q] for d in null_contrasts])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.float64(abs(mp_f["median"] - ms_f["median"])) / abs(ms_f["median"])
        reported[q] = {"pred": mp, "sim": ms, "pred_finite": mp_f, "sim_finite": ms_f,
                       "rel_error": float(rel), "null": nb}
        mo = None
        if own is not None:
            mo = median_with_nonfinite([d[q] for d in own], allow_inf=True)
            reported[q]["own_offset"] = mo
        if q in K1A_REL_CHECKS:
            if max(mp_f["n_nonfinite"], ms_f["n_nonfinite"]) > NONFINITE_MAX_FRAC * R:
                st = "fail (non-finite)"
            elif ms_f["median"] == 0.0:
                st = "not evaluable"
            else:
                st = "pass" if rel <= K1A_REL_CHECKS[q] else "fail"
            checks[f"{q}_rel"] = st
        if q in K1A_SIGN_CHECKS:
            if (max(mp["n_nonfinite"], ms["n_nonfinite"]) > NONFINITE_MAX_FRAC * R
                    or nb["n_nan"] > NONFINITE_MAX_FRAC * R0
                    or (mo is not None and mo["n_nonfinite"] > NONFINITE_MAX_FRAC * R0)):
                st = "fail (non-finite)"
            elif ms["median"] == 0.0:
                st = "not evaluable"
            else:
                sp_, ss = np.sign(mp["median"]), np.sign(ms["median"])
                if not (sp_ == ss and sp_ != 0):
                    st = "fail"
                elif not ((sp_ > 0 and ms["median"] > nb["hi"]) or (
                        sp_ < 0 and ms["median"] < nb["lo"])):
                    st = "indistinguishable from null"
                elif mo is not None and not abs(mp["median"]) > abs(mo["median"]):
                    st = OWN_OFFSET_STATUS
                else:
                    st = "pass"
            checks[f"{q}_sign"] = st
    passed = all(v == "pass" for v in checks.values())
    judged = row != "closed_form" or cell["process"] in LINEAR_PROCESSES
    if not judged:
        law = "not judged (closed_form is judged on OU cells only)"
    elif not cell["valid_for_law"]:
        law = "does not test the law"
    else:
        law = "pass" if passed else "fail"
    label = (f"{cell['process']}-{cell['exit_mechanism']} {cell['pipeline']} {row} "
             f"R={R} R0={R0} N={cell['N']} {cell['re_placement']}"
             f"{'' if cell['gated'] else ' (control, not gated)'}: "
             f"{'pass' if passed else 'fail'}; law_status {law}")
    return {"checks": checks, "pass": passed, "judged": judged,
            "valid_for_law": cell["valid_for_law"], "law_status": law,
            "row": row, "cell": cell, "pipeline": cell["pipeline"],
            "reported": reported, "R": R, "R_null": R0, "label": label,
            "replicates": [r for r, _, _ in pairs]}


def law_claims(verdicts: list) -> dict:
    by_row = {}
    for v in verdicts:
        c = check_cell_config(v["cell"])
        if c["gated"] is not True or c["re_placement"] != GATED_RE_PLACEMENT:
            raise ValueError(f"verdict of a non-gated / non-{GATED_RE_PLACEMENT!r} cell "
                             f"({c['process']}-{c['exit_mechanism']}, {c['re_placement']}): "
                             "law_claims takes the gated cells only")
        if c["pipeline"] != K1A_GATE["pipeline"]:
            raise ValueError("law_claims uses the saai pipeline only")
        key = (c["process"], c["exit_mechanism"])
        if key not in K1A_CELLS:
            raise ValueError(f"{key} is not a K1a cell")
        if v.get("R") != R_PER_CELL or v.get("R_null") != R_PER_CELL:
            raise ValueError(f"verdict for {key} has R = {v.get('R')}, R0 = "
                             f"{v.get('R_null')} (need {R_PER_CELL})")
        by_row.setdefault(v["row"], {})
        if key in by_row[v["row"]]:
            raise ValueError(f"duplicate verdict for {v['row']} {key}")
        by_row[v["row"]][key] = v["law_status"] == "pass"
    for r in REPORT_ROWS:
        if set(by_row.get(r, {})) != set(K1A_CELLS):
            raise ValueError(f"row {r!r} lacks K1a cells "
                             f"{sorted(set(K1A_CELLS) - set(by_row.get(r, {})))}")
    proj, cf, zo = by_row["projection"], by_row["closed_form"], by_row["zeroth_order"]
    ou = [k for k in K1A_CELLS if k[0] in LINEAR_PROCESSES]
    return {
        "projection": "prediction confirmed" if all(proj.values()) else "not confirmed",
        "fallback": "reported only",
        "zeroth_order_mbar": "reported only",
        "closed_form": "law under linear drift" if all(cf[k] for k in ou) else "not a law",
        "zeroth_order": "law" if all(zo.values()) else "not a law",
        "prl_condition_1": bool(all(zo.values())),
        "cells": {r: {f"{p}-{e}": ok for (p, e), ok in d.items()}
                  for r, d in by_row.items()},
    }


def predicted_effect_sizes(process: str, pipeline: str, entries: list) -> dict:
    """entries: exactly one dict per K1a cell of the process, i.e. exactly
    (process, 'd') and (process, 'e') on this pipeline, gated 'shock', each
        {"cell": cell meta, "predictions": [2R k1_theory.predict_row(
            "projection", ..., cell=cell) outputs]}
    with R = R_PER_CELL (pair_predictions: same cell_key, row, design,
    digest, one 'pre' and one 'during' per replicate).  Per cell and
    quantity: the median over the R replicates of bias_during − bias_pre,
    each bias = estimate − truth computed HERE (the predicted contrast of the
    estimate in the quantity's own units, also for the ratio quantities — τ
    in years — whose gated form is dimensionless), with the non-finite
    replicates COUNTED: more than NONFINITE_MAX_FRAC of them in any cell
    refuses (the cap must not loosen because predictions are missing).  The
    effect is the smaller |median| of the two cells.  SIMULATION_ONLY
    quantities get NaN with status 'simulation only'; an exactly-0 effect
    gets status 'zero'.  Requires the theory freeze.
    """
    if process not in {p for p, _ in K1A_CELLS}:
        raise ValueError(f"{process!r} has no K1a cells; its S0 has no predicted contrast")
    if pipeline not in S0_PIPELINES:
        raise ValueError(f"pipeline must be one of {S0_PIPELINES}, got {pipeline!r}")
    if not entries:
        raise ValueError("predicted_effect_sizes needs the process's two K1a cells")
    want = sorted(k for k in K1A_CELLS if k[0] == process)
    got, paired = [], []
    for e in entries:
        c = check_cell_config(e["cell"])
        got.append((c["process"], c["exit_mechanism"]))
        if not c["gated"] or c["re_placement"] != GATED_RE_PLACEMENT:
            raise ValueError("the predicted contrasts come from the gated 'shock' cells")
        if c["pipeline"] != pipeline:
            raise ValueError(f"cell pipeline {c['pipeline']!r} != {pipeline!r}: the S0 cap "
                             "of a pipeline uses that pipeline's predictions")
        pr = pair_predictions(c, e.get("predictions") or [], "projection")
        if len(pr) != R_PER_CELL:
            raise ValueError(f"need R = {R_PER_CELL} replicates with a pre / during "
                             f"prediction, got {len(pr)}")
        paired.append((f"{c['process']}-{c['exit_mechanism']}", pr))
    if sorted(got) != want:
        raise ValueError(f"need exactly the K1a cells {want}, got {sorted(got)}")
    rec = _require_frozen()
    effect, status, per_cell = {}, {}, {}
    for q in QUANTITIES:
        if q in SIMULATION_ONLY:
            effect[q], status[q] = np.nan, "simulation only"
            continue
        vals = []
        for key, pr in paired:
            m = median_with_nonfinite([_bias_of(d, q) - _bias_of(p, q) for _, p, d in pr])
            per_cell.setdefault(key, {})[q] = m
            if m["n_nonfinite"] > NONFINITE_MAX_FRAC * m["n_total"]:
                raise ValueError(f"predicted {q} contrast undefined in {m['n_nonfinite']} "
                                 f"of {m['n_total']} replicates of {key} (> "
                                 f"{NONFINITE_MAX_FRAC:.0%}): S0 cannot be capped; refused")
            vals.append(abs(m["median"]))
        effect[q] = float(min(vals))
        status[q] = "zero" if effect[q] == 0.0 else "predicted"
    return {"process": process, "pipeline": pipeline, "effect": effect,
            "status": status, "per_cell": per_cell, "R": R_PER_CELL,
            "frozen_commit": rec.get("frozen_commit")}


def s0_check(cell: dict, estimates: dict, truths: dict,
             k1a_entries: list | None, rho1_reference: dict | None = None) -> dict:
    c = check_cell_config(cell)
    need = {"exit_mechanism": "a", "exit_dating": "none", "gated": True,
            "re_placement": GATED_RE_PLACEMENT}
    bad = {k: c[k] for k, v in need.items() if c[k] != v}
    if bad:
        raise ValueError(f"S0 runs on cell (a), no exit, gated 'shock', N = {S0_N} per "
                         f"window; got {bad}")
    try:
        check_s0_N(c)
    except ValueError as exc:
        raise ValueError(f"S0 runs on cell (a) with N = {S0_N} per window: {exc}") from exc
    if c["pipeline"] not in S0_PIPELINES:
        raise ValueError(f"S0 pipeline must be labelled 'saai' or 'direct', got "
                         f"{c['pipeline']!r}")
    for name, d in (("estimates", estimates), ("truths", truths)):
        if not isinstance(d, dict) or set(d) != set(PERIODS):
            raise ValueError(f"S0 {name} must be {{'pre': [...], 'during': [...]}}: both "
                             "periods are checked (one period never unlocks a process)")
    ids = None
    for per in PERIODS:
        e_ids = _replicate_ids(estimates[per], f"{per} estimate")
        t_ids = _replicate_ids(truths[per], f"{per} truth")
        if len(e_ids) != R_PER_CELL or len(t_ids) != R_PER_CELL:
            raise ValueError(f"S0 {per} has R = {len(e_ids)} / {len(t_ids)}: the registration registers "
                             f"R = {R_PER_CELL} (amendment 3 (b))")
        if sorted(e_ids) != sorted(t_ids):
            raise ValueError(f"S0 {per}: estimates and truths hold different replicates")
        if ids is not None and sorted(e_ids) != ids:
            raise ValueError("S0: the pre and during replicates differ")
        ids = sorted(e_ids)
    if ids != list(range(R_PER_CELL)):
        raise ValueError(f"S0 replicate ids must be exactly 0 … {R_PER_CELL - 1} (the "
                         "registered s0_seeds index them)")
    seeds = s0_seeds(c["process"], c["pipeline"])
    for per in PERIODS:
        off = [int(e["replicate"]) for e in estimates[per]
               if e.get("seed") != seeds[int(e["replicate"])]]
        if off:
            raise ValueError(f"S0 {per} estimates of replicates {off[:5]} do not carry the "
                             "registered seed s0_seeds(process, pipeline)[replicate]")
        for t in truths[per]:
            check_registered_true_params(c["process"], per, t.get("true_params") or {})
    R = R_PER_CELL
    rec = _require_frozen()
    k1a_process = c["process"] in {p for p, _ in K1A_CELLS}
    if k1a_process:
        if not k1a_entries:
            raise ValueError("s0_check needs the K1a entries of this process and pipeline "
                             "(it computes predicted_effect_sizes itself)")
        predicted_effect = predicted_effect_sizes(c["process"], c["pipeline"], k1a_entries)
    elif k1a_entries is not None:
        raise ValueError(f"{c['process']!r} has no K1a cells: pass k1a_entries=None")
    else:
        predicted_effect = None
    tol_eff, basis = {}, {}
    for q in QUANTITIES:
        if not k1a_process:
            basis[q], tol_eff[q] = "physical only (no K1a cells for this process)", np.inf
            continue
        st = predicted_effect["status"][q]
        eff = np.float64(predicted_effect["effect"][q])
        if st == "simulation only" and q in SIMULATION_ONLY:
            basis[q], tol_eff[q] = "physical only (simulation-only quantity)", np.inf
        elif st == "zero" and eff == 0.0:
            basis[q], tol_eff[q] = "physical only (frozen prediction exactly 0)", np.inf
        elif st == "predicted" and np.isfinite(eff) and eff > 0:
            basis[q], tol_eff[q] = "stricter of physical and effect", S0_REL_TOL * eff
        else:
            raise ValueError(f"predicted effect of {q} is {eff} with status {st!r}: "
                             "a non-finite prediction never loosens S0")
    out = {}
    for per in PERIODS:
        t_by = {int(t["replicate"]): t for t in truths[per]}
        pairs = [(e, t_by[int(e["replicate"])]) for e in
                 sorted(estimates[per], key=lambda d: int(d["replicate"]))]
        rho1_t = median_with_nonfinite([t["rho1"] for _, t in pairs])["median"]
        res = {}
        for q in QUANTITIES:
            bias = median_with_nonfinite([np.float64(e[q]) - np.float64(t[q])
                                          for e, t in pairs])
            tru = median_with_nonfinite([t[q] for _, t in pairs])["median"]
            sc = S0_ABS_SCALE[q]
            scale = 1.0 / abs(1.0 - rho1_t) if sc == "relaxation_time" else float(sc)
            tol_phys = S0_REL_TOL * max(abs(tru), scale)
            tol = float(min(tol_phys, tol_eff[q]))
            if bias["n_nonfinite"] > NONFINITE_MAX_FRAC * R:
                status = "fail (non-finite)"
            else:
                status = "pass" if abs(bias["median"]) < tol else "fail"
            res[q] = {"status": status, "median_bias": bias["median"], "tol": tol,
                      "tol_physical": float(tol_phys), "tol_effect": float(tol_eff[q]),
                      "binding": "effect" if tol_eff[q] < tol_phys else "physical",
                      "basis": basis[q], "truth_median": tru,
                      "n_nonfinite": bias["n_nonfinite"], "R": R}
            if q == "rho1" and c["process"] in S0_RHO1_REPORT_ONLY:
                res[q].update(status=S0_RHO1_REPORT_STATUS, status_if_gated=status,
                              rederivation5=s0_rho1_report(pairs, rho1_reference, per))
        res["pass"] = all(res[q]["status"] == "pass" for q in s0_gated_quantities(c["process"]))
        out[per] = res
    out["pass"] = bool(all(out[per]["pass"] for per in PERIODS))
    out["cell"], out["pipeline"], out["R"], out["replicates"] = c, c["pipeline"], R, ids
    out["seeds"] = "s0_seeds(process, pipeline), checked per replicate"
    out["predicted_effect"] = (None if predicted_effect is None else
                               {k: predicted_effect.get(k) for k in
                                ("process", "pipeline", "effect", "status",
                                 "frozen_commit", "per_cell")})
    out["label"] = (f"S0 {c['process']} {c['pipeline']} N={c['N']} R={R}: "
                    f"pre {'pass' if out['pre']['pass'] else 'fail'}, during "
                    f"{'pass' if out['during']['pass'] else 'fail'}: "
                    f"{'pass' if out['pass'] else 'fail'}")
    return out


def predict_bias(composition, true_params, period, standardisation):
    """Predicted V0 bias of each QUANTITIES entry for one replicate and period.

    composition     : DataFrame with COMPOSITION_COLUMNS, rows of that replicate
    true_params     : dict with TRUE_PARAM_KEYS for that period
    standardisation : DataFrame with STANDARDISATION_COLUMNS (same replicate)
    returns         : dict {quantity: predicted estimate − truth(...)}, NaN where
                      the derivation gives no closed form ('simulation only')
    """
    raise NotImplementedError


def truth(true_params, standardisation, period, cell=None):
    """Truth in ξ units: the true A, B mapped by the complete panel's YEARLY
    (mu_x, sigma_x), pooled over the period's transitions by the same
    projection that predict_bias uses (src.k1_theory.truth).  k1_sim calls
    this function; it does not re-implement it.  Pass the cell meta for
    every truth that enters a gate: the table is then checked to be the
    cell's complete panel (contract 9: n constant in each window, the window
    sizes summing to N)."""
    raise NotImplementedError


def estimate(pairs):
    """pairs: DataFrame with CountyCode, Year_t, xi_t, delta_xi (one period).
    Returns {quantity: point estimate}: fit, then ``observables`` (this
    module) on the fitted parameters with xi_data = pairs.xi_t, and
    rho1 = ``rho1_from_pairs``.  NaN results are kept and counted."""
    raise NotImplementedError
