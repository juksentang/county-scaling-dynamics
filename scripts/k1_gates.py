"""A rule the installed src.k1_contract does not provide (k1a_cell_verdict with
a null baseline, s0_check with predicted effect sizes, i.e. contract < 4) is
never substituted: those rows are 'not_evaluable_contract_lacks_rule'.
Cells whose true process is outside the theory's family (theory_compatible
False, src.k1_sim.theory_compat) are 'blocked_theory_incompatible'.

Usage:
    python3 scripts/k1_gates.py [--tables results/tables]
"""
from __future__ import annotations

def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd

from src import k1_sim as S

QUANTITIES = ("xi_trap", "two_delta_V", "tau_mfpt", "mean_diffusion", "B_ratio_w", "rho1")
K1B_COV, K1C_TOL, K1C_COV = 0.90, 0.30, 0.90
K2_POWER, K2_FPR, K2_LO = 0.80, 0.10, 0.5
K3_RATIO = (0.5, 2.0)
N_S0 = 20_000
R_REGISTERED = 500
GATING_PIPELINE = "saai"
GATED_ROW = "projection"
OBS_RULES = ("limit",)
CI_RULES = ("tol0", "tol5", "literal")
PRIMARY_OBS, PRIMARY_CI = "limit", "tol0"
K1A_CELLS = (("i_ou", "d"), ("i_ou", "e"), ("ii_cubic", "d"), ("ii_cubic", "e"))
REPORT_ROWS = ("projection", "fallback", "closed_form", "zeroth_order", "zeroth_order_mbar")


def _contract():
    from src import k1_contract as C
    return C


def _row(gate, r, stat, value, thr, status, gating, rule_set, **kw):
    out = {"gate": gate, "rule_set": rule_set, "cell_index": r.get("cell_index"),
           "process": r.get("process"), "mechanism": r.get("mechanism"),
           "pipeline": r.get("pipeline"), "saai_spec": r.get("saai_spec"),
           "variant": r.get("variant"), "obs_rule": r.get("obs_rule"),
           "ci_nan_rule": r.get("ci_nan_rule"), "period": r.get("period"),
           "quantity": r.get("quantity"), "statistic": stat, "value": value,
           "threshold": thr, "status": status, "gating": bool(gating)}
    out.update(kw)
    R = out.get("R")
    if gating and status not in ("missing",) and not str(status).startswith(("blocked", "not_evaluable")) \
            and (R is None or not np.isfinite(R) or int(R) < R_REGISTERED):
        out["status"] = "fail_R"
        out["status_before_R_check"] = status
    return out


def _rel(num, den):
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
        return np.nan
    return abs(num) / abs(den)


_BASE_COLS = ("cell_index", "process", "mechanism", "n_override", "gate_use", "variant",
              "period", "quantity", "obs_rule", "ci_nan_rule", "pipeline", "saai_spec",
              "theory_compatible", "replicate", "row", "structure")


EXPLORATORY_MARKER = "exploratory"


def refuse_exploratory(obj, what: str = "table") -> None:
    if isinstance(obj, pd.DataFrame):
        marked = EXPLORATORY_MARKER in obj.columns
    elif isinstance(obj, dict):
        marked = bool(obj.get(EXPLORATORY_MARKER))
    elif isinstance(obj, list):
        marked = any(isinstance(r, dict) and r.get(EXPLORATORY_MARKER) for r in obj[:1])
    else:
        marked = False
    if marked:
        raise SystemExit(f"{what} carries the exploratory (amendment 11 (b)) marker: k1_gates never "
                         "reads exploratory outputs")


def _std_cells(df):
    if df is None:
        df = pd.DataFrame()
    refuse_exploratory(df, "a gate input table")
    df = df.copy()
    if df.empty:
        for c in _BASE_COLS:
            if c not in df.columns:
                df[c] = pd.Series(dtype=object)
    if "n_override" in df:
        df["n_override"] = pd.to_numeric(df["n_override"], errors="coerce")
    if "obs_rule" not in df:
        df["obs_rule"] = PRIMARY_OBS
    if "gate_use" not in df:
        df["gate_use"] = np.where(df["n_override"] == N_S0, "S0", "registered")
    if "theory_compatible" not in df:
        df["theory_compatible"] = True
    return df


def _sel(df, process, mechanism):
    m = (df["process"] == process) & (df["mechanism"] == mechanism) & df["n_override"].isna()
    if "gate_use" in df:
        m &= ~df["gate_use"].isin(["K1a_null", "control", "control_null", "S0",
                                   "K1a_shocked_null", S.SENS_NO2006_GATE_USE])
    return df[m]


def _rs(obs, ci=None, extra=None):
    return "/".join(x for x in (obs, ci, extra) if x)


def _missing(gate, process, mech, what, rule_set, gating=True):
    return _row(gate, {"process": process, "mechanism": mech}, what, np.nan, None,
                "missing", gating, rule_set)


def _compatible(df_or_row) -> bool:
    v = df_or_row["theory_compatible"]
    if isinstance(v, pd.Series):
        return bool(v.astype(str).str.lower().isin(["true", "1"]).all())
    return str(v).lower() in ("true", "1")


K1A_PROCESSES = ("i_ou", "ii_cubic")
SHOCK_SHARE_FLAG = "TRUTH_ABSORBS_PROVINCE_SHOCK_SHARE"
BLOCK_SHOCK_SHARE = "blocked_truth_lacks_shock_share_term"
NO_RECORD_PROCESSES = ()


def _refused(exc) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def theory_has_shock_share_term() -> bool:
    try:
        from src import k1_theory as T
    except Exception:
        return False
    C = _contract()
    return bool(getattr(T, SHOCK_SHARE_FLAG, False) or getattr(C, SHOCK_SHARE_FLAG, False))


def index_predict_rows(praw: list | None) -> dict:
    """{(cell_index, pipeline, row): [predict_row outputs]} — the outputs are
    handed to the contract unchanged (it checks their digests)."""
    idx = {}
    for r in praw or []:
        idx.setdefault((int(r["cell_index"]), r["pipeline"], r["diag"].get("row")), []).append(r)
    return idx


def _find_cell(cell_meta: dict, sim_process: str, mech: str, gate_use: str,
               structure=None, pipeline: str = GATING_PIPELINE):
    for ci, d in cell_meta.items():
        m = (d or {}).get(pipeline) or {}
        if (m.get("sim_process") == sim_process and m.get("exit_mechanism") == mech
                and m.get("gate_use") == gate_use
                and (structure is None or m.get("structure") == structure)):
            return int(ci)
    return None


S0_PROCESSES = tuple(dict.fromkeys(S.PROCESS_KEY[p] for p in S.PROCESS_NAMES))
GATE_S0_SCOPE = {"K1a": ("ou", "cubic"), "K1b": ("ou", "cubic", "cubic_mult"),
                 "K1c": ("cubic_shift",),
                 "K2": ("cubic_bistable_0", "cubic_bistable_1", "cubic_bistable_2",
                        "cubic_bistable_4"),
                 "K3": ("ou", "cubic")}
NEURAL_S0_SCOPE = {"K3": ("ou", "cubic")}
NEURAL_S0_PIPELINES = ("saai",)
S0_GATES = ("S0", "S0_neural")


def _s0_record(process: str, pipeline: str) -> tuple[dict | None, str, str]:
    C = _contract()
    try:
        rec = C._require_s0_pass(process, pipeline)
        full = json.loads(C.s0_record_path(process, pipeline).read_text())
        return {**full, **rec}, "pass", f"verified ({rec.get('verified_by')})"
    except Exception as exc:
        why = _refused(exc)
    try:
        path = C.s0_record_path(process, pipeline)
        full = json.loads(path.read_text()) if path.exists() else None
    except Exception:
        full = None
    if full is not None and full.get("pass") is False:
        stale = s0_record_staleness(full, process, pipeline)
        if stale:
            return full, STALE_S0_RECORD, (f"stale failed S0 record (attempt "
                                           f"{full.get('attempt')}): {'; '.join(stale)}")
        return full, "fail", f"failed S0 record (logged attempt {full.get('attempt')}): {why}"
    return full, "not_evaluable_no_s0_record", why


STALE_S0_RECORD = "not_evaluable_stale_s0_record"


def _neural_s0_record(process: str, pipeline: str) -> tuple[dict | None, str, str]:
    try:
        from src import k1_neural_k3 as K3
    except Exception as exc:
        return None, "not_evaluable_no_s0_record", f"neural module: {_refused(exc)}"
    try:
        rec = K3.require_neural_s0_pass(process, pipeline)
        with K3._neural_s0_paths():
            full = json.loads(_contract().s0_record_path(process, pipeline).read_text())
        return {**full, **rec}, "pass", "verified neural S0"
    except Exception as exc:
        why = _refused(exc)
    with K3._neural_s0_paths():
        try:
            path = _contract().s0_record_path(process, pipeline)
            full = json.loads(path.read_text()) if path.exists() else None
        except Exception:
            full = None
        if full is not None and full.get("pass") is False:
            stale = s0_record_staleness(full, process, pipeline)
            if stale:
                return full, STALE_S0_RECORD, f"stale failed neural S0 record: {'; '.join(stale)}"
            return full, "fail", f"failed neural S0 record (attempt {full.get('attempt')}): {why}"
    return full, "not_evaluable_no_s0_record", why


def s0_record_staleness(full: dict, process: str, pipeline: str) -> list[str]:
    C = _contract()
    out = []
    try:
        fr = C._require_frozen().get("frozen_commit")
    except Exception as exc:
        fr = None
        out.append(f"no current theory freeze to compare with ({type(exc).__name__})")
    if fr is not None and full.get("frozen_commit") != fr:
        out.append(f"made under freeze {str(full.get('frozen_commit'))[:10]}, current "
                   f"{str(fr)[:10]}")
    if full.get("contract_version") != C.CONTRACT_VERSION:
        out.append(f"contract {full.get('contract_version')}, current {C.CONTRACT_VERSION}")
    code = full.get("s0_code")
    if not isinstance(code, dict):
        out.append("no s0_code digest (contract 9)")
    else:
        try:
            bad = C.s0_code_mismatch(code)
        except Exception as exc:
            bad = [f"<digest failed: {type(exc).__name__}>"]
        if bad:
            out.append(f"S0 code changed since: {bad[:6]}")
    try:
        started = C._started(C._entries(process, pipeline))
        if full.get("attempt") != len(started):
            out.append(f"attempt {full.get('attempt')} is not the last logged attempt "
                       f"({len(started)} started)")
    except Exception as exc:
        out.append(f"attempt log unreadable ({type(exc).__name__})")
    return out


def gate_s0(shock_share_w: float | None = None, record=None,
            neural_record=None) -> list[dict]:
    rs = _rs(PRIMARY_OBS)
    record = record or _s0_record
    neural_record = neural_record or _neural_s0_record
    w = S._w_shock() if shock_share_w is None else float(shock_share_w)
    share_ok = theory_has_shock_share_term()
    block_share = w > 0 and not share_ok
    rows = []
    neural = sorted({p for v in NEURAL_S0_SCOPE.values() for p in v})
    units = ([("S0", proc, pl, record) for proc in S0_PROCESSES for pl in S.PIPELINES]
             + [("S0_neural", proc, pl, neural_record) for proc in neural
                for pl in NEURAL_S0_PIPELINES])
    for gate, proc, pl, rd in units:
        rec, st0, note = rd(proc, pl)
        notes = [note]
        if block_share:
            notes.append(f"mean_diffusion: truth lacks the province shock-share term at "
                         f"w = {w:.4f} (k1_theory.{SHOCK_SHARE_FLAG} not set)")
        base = {"process": proc, "mechanism": "a", "pipeline": pl,
                "saai_spec": S.SAAI_SPEC_LABEL[pl], "variant": "V0",
                "obs_rule": PRIMARY_OBS}
        res = (rec or {}).get("result") if st0 in ("pass", "fail") else None
        if not res:
            rows.append(_row(gate, {**base, "period": "*", "quantity": "*"},
                             "committed_s0_record", np.nan, None, st0, True, rs,
                             note=" | ".join(notes), R=np.nan))
            continue
        for per in S.PERIODS:
            for q in QUANTITIES:
                d = (res.get(per) or {}).get(q) or {}
                st = d.get("status", "missing")
                if st0 == "pass" and st != "pass":
                    st = "not_evaluable_record_inconsistent"
                if block_share and q == "mean_diffusion":
                    st = BLOCK_SHOCK_SHARE
                rows.append(_row(gate, {**base, "period": per, "quantity": q},
                                 "median_bias_vs_tol", d.get("median_bias", np.nan),
                                 d.get("tol", np.nan), st, True, rs,
                                 contract_status=d.get("status"),
                                 tol_physical=d.get("tol_physical"),
                                 tol_effect=d.get("tol_effect"), binding=d.get("binding"),
                                 basis=d.get("basis"), truth_median=d.get("truth_median"),
                                 n_nonfinite=d.get("n_nonfinite"), R=res.get("R"),
                                 N_meta=(res.get("cell") or {}).get("N"),
                                 s0_attempt=(rec or {}).get("attempt"),
                                 s0_record_commit=(rec or {}).get("code_commit"),
                                 note=" | ".join(notes)))
    return rows


def _contrasts_from_csv(P: pd.DataFrame, ci: int, pl: str) -> list[dict]:
    """The simulated contrast_bias of each replicate (V0 estimate vs the
    shared truth, computed by the aggregator with contract.contrast_bias),
    with its integer replicate id; failed replicates are NaN, kept."""
    g = P[(P["cell_index"] == ci) & (P["pipeline"] == pl) & (P["row"] == GATED_ROW)
          & (P["period"] == "contrast")]
    w = g.pivot_table(index="replicate", columns="quantity", values="sim_bias", dropna=False)
    return [{"replicate": int(r), **{q: float(row.get(q, np.nan)) for q in QUANTITIES}}
            for r, row in w.sort_index().iterrows()]


def k1a_uses_contract_own_offset_rule() -> bool:
    return "zeroth_order" in getattr(_contract(), "OWN_OFFSET_ROWS", ())


def law_claims_gated(verdicts: list) -> dict:
    bad = [v.get("label") for v in verdicts if v.get("report_only")]
    if bad:
        raise ValueError(f"report-only verdicts given to law_claims: {bad}")
    return _contract().law_claims(verdicts)


SHOCKED_NULL_MISSING = "not_evaluable_no_dated_shock_null"


def combine_with_shocked_null(v_reg: dict, v_shk: dict | None) -> dict:
    out = {**v_reg, "checks": dict(v_reg["checks"]),
           "checks_shock_free_null": dict(v_reg["checks"]),
           "checks_dated_shock_null": dict((v_shk or {}).get("checks") or {}),
           "null_rule": 'outside both the shock-free and the dated-shock null bands'}
    for chk, st in v_reg["checks"].items():
        if not chk.endswith("_sign"):
            continue
        st2 = (v_shk or {}).get("checks", {}).get(chk) if v_shk else None
        if st2 is None:
            out["checks"][chk] = SHOCKED_NULL_MISSING if st == "pass" or not \
                str(st).startswith("fail") else st
        elif st == "pass" and st2 == "pass":
            out["checks"][chk] = "pass"
        elif str(st).startswith("fail"):
            out["checks"][chk] = st
        elif str(st2).startswith("fail"):
            out["checks"][chk] = f"{st2} (dated-shock null)"
        elif st != "pass":
            out["checks"][chk] = st
        else:
            out["checks"][chk] = f"{st2} (dated-shock null)"
    if v_shk is None:
        out["law_status"] = (v_reg["law_status"] if v_reg["law_status"] != "pass"
                             else SHOCKED_NULL_MISSING)
    elif v_reg["law_status"] == "pass" and v_shk["law_status"] != "pass":
        out["law_status"] = f"{v_shk['law_status']} (dated-shock null)"
    return out


def gate_k1a(pred: pd.DataFrame | None, cell_meta: dict,
             praw_idx: dict | None = None) -> tuple[list[dict], dict | None]:
    C = _contract()
    rs = _rs(PRIMARY_OBS)
    rows = []
    if pred is None or praw_idx is None:
        return [_missing("K1a", p, m, f"no {'k1_predictions.csv' if pred is None else 'predict rows'}",
                         rs) for p, m in K1A_CELLS], None
    P = _std_cells(pred)
    compat = {int(ci): _compatible(g.iloc[0]) for ci, g in P.groupby("cell_index")}
    verdicts_for_claims = []
    variants = (("K1a/K1b", "K1a_null", "registered"), ("control", "control_null", "registered"),
                ("K1a/K1b", "K1a_shocked_null", "shocked"),
                (S.SENS_NO2006_GATE_USE, "K1a_null", "sens_no2006"))
    blocked_claims = []
    for gate_use, null_use, null_variant in variants:
        for proc, mech in K1A_CELLS:
            if gate_use == S.SENS_NO2006_GATE_USE and mech != "e":
                continue
            ci = _find_cell(cell_meta, proc, mech, gate_use)
            ni = _find_cell(cell_meta, proc, "a", null_use, structure=mech)
            registered = gate_use == "K1a/K1b" and null_variant == "registered"
            if ci is None or ni is None:
                rows.append(_missing("K1a", proc, mech, f"{gate_use} cell or its {null_use} null "
                                     "not in the aggregate", rs, gating=registered))
                continue
            for pl in S.PIPELINES:
                cm, nm = cell_meta[ci].get(pl), cell_meta[ni].get(pl)
                if cm is None or nm is None:
                    if registered and pl == GATING_PIPELINE:
                        rows.append(_missing("K1a", proc, mech, f"no {pl} cell meta", rs))
                    continue
                report_only = (not registered or bool((cm or {}).get("report_only"))
                               or bool((nm or {}).get("report_only")))
                sims = _contrasts_from_csv(P, ci, pl)
                nulls = _contrasts_from_csv(P, ni, pl)
                for row in REPORT_ROWS:
                    preds = praw_idx.get((ci, pl, row), [])
                    if not preds and pl != GATING_PIPELINE and row in ("closed_form",
                                                                      "zeroth_order",
                                                                      "zeroth_order_mbar"):
                        continue
                    gating = (registered and not report_only and pl == GATING_PIPELINE
                              and row == GATED_ROW)
                    base = {"cell_index": ci, "process": proc, "mechanism": mech, "pipeline": pl,
                            "saai_spec": S.SAAI_SPEC_LABEL[pl], "variant": "V0",
                            "obs_rule": PRIMARY_OBS, "period": "contrast"}
                    kw = dict(row=row, null_cell=ni, null_variant=null_variant,
                              report_only=report_only, R=len(sims))
                    claim_unit = registered and not report_only and pl == GATING_PIPELINE
                    ok_compat = compat.get(ci, False) and compat.get(ni, False)
                    try:
                        v = C.k1a_cell_verdict(
                            preds, sims, nulls, cm, nm, row=row,
                            null_predictions=(praw_idx.get((ni, pl, row), [])
                                              if row in C.OWN_OFFSET_ROWS else None))
                    except Exception as exc:
                        if claim_unit:
                            blocked_claims.append(f"cell {ci} {row}: refused")
                        rows.append(_row("K1a", base, f"{row}:k1a_cell_verdict", np.nan, None,
                                         "not_evaluable_refused", gating, rs,
                                         note=_refused(exc), **kw))
                        continue
                    if registered:
                        si = _find_cell(cell_meta, proc, "a", "K1a_shocked_null",
                                        structure=mech)
                        sm = (cell_meta.get(si) or {}).get(pl) if si is not None else None
                        v_shk = None
                        if sm is not None:
                            try:
                                v_shk = C.k1a_cell_verdict(
                                    preds, sims, _contrasts_from_csv(P, si, pl), cm, sm,
                                    row=row,
                                    null_predictions=(praw_idx.get((si, pl, row), [])
                                                      if row in C.OWN_OFFSET_ROWS else None))
                            except Exception:
                                v_shk = None
                        v = combine_with_shocked_null(v, v_shk)
                        kw["shocked_null_cell"] = si
                    v = {**v, "report_only": report_only, "null_variant": null_variant}
                    if claim_unit:
                        if ok_compat:
                            verdicts_for_claims.append(v)
                        else:
                            blocked_claims.append(f"cell {ci} / null {ni} {row}: "
                                                  "blocked_theory_incompatible")
                    for chk, st in v["checks"].items():
                        q = chk.rsplit("_", 1)[0]
                        rep = v["reported"][q]
                        st2 = st if ok_compat else "blocked_theory_incompatible"
                        rows.append(_row(
                            "K1a", {**base, "quantity": q}, f"{row}:{chk}", rep["rel_error"],
                            "≤0.30" if chk.endswith("_rel") else "same sign, beyond null band",
                            st2.replace(" ", "_"), gating, rs, R_null=v["R_null"],
                            median_pred=rep["pred"]["median"],
                            median_sim=rep["sim"]["median"], null_lo=rep["null"]["lo"],
                            null_hi=rep["null"]["hi"], n_pred_nonfinite=rep["pred"]["n_nonfinite"],
                            n_sim_nonfinite=rep["sim"]["n_nonfinite"],
                            own_offset_null_pred_median=(rep.get("own_offset") or {}
                                                         ).get("median"),
                            law_status=v["law_status"], valid_for_law=v["valid_for_law"],
                            **{**kw, "R": v["R"]}))
    claims = None
    if blocked_claims:
        claims = {"not_evaluable": sorted(set(blocked_claims))}
    elif len(verdicts_for_claims) == len(K1A_CELLS) * len(REPORT_ROWS):
        try:
            claims = law_claims_gated(verdicts_for_claims)
        except Exception as exc:
            claims = {"not_evaluable": [_refused(exc)]}
    return rows, claims


def gate_shocked_null(sn: pd.DataFrame | None) -> list[dict]:
    rs = _rs(PRIMARY_OBS)
    if sn is None or sn.empty:
        return [_missing("shocked_null", p, m, "no dated-shock null replicates", rs,
                         gating=False) for p, m in K1A_CELLS]
    rows = []
    for (ci, pl, per, q), g in _std_cells(sn).groupby(["cell_index", "pipeline", "period",
                                                        "quantity"]):
        b = (g["estimate"] - g["truth"]).to_numpy(float)
        ok = np.isfinite(b)
        med = float(np.median(b[ok])) if ok.any() else np.nan
        tm = float(np.nanmedian(g["truth"])) if np.isfinite(g["truth"]).any() else np.nan
        r = g.iloc[0].to_dict()
        rows.append(_row("shocked_null", {**r, "period": per, "quantity": q, "variant": "V0"},
                         "median_bias_vs_5pct_truth", med, 0.05 * abs(tm) if np.isfinite(tm)
                         else np.nan, "report", False, rs, n_nonfinite=int((~ok).sum()),
                         truth_median=tm, R=int(g["replicate"].nunique()),
                         structure=r.get("structure")))
    return rows


def _ci_gates(rule: str) -> bool:
    return rule == PRIMARY_CI


K1B_NOT_EVALUABLE = {("i_ou", "tau_mfpt")}
K1B_NOT_EVALUABLE_STATUS = "report_not_evaluable_amendment_7"


def gate_k1b(cov: pd.DataFrame) -> list[dict]:
    rows = []
    c0 = _std_cells(cov)
    for obs in OBS_RULES:
        for cir in CI_RULES:
            rs = _rs(obs, cir)
            for proc in ("i_ou", "ii_cubic", "iii_mult"):
                for mech in ("b", "c", "d", "e"):
                    c = _sel(c0, proc, mech)
                    c = c[(c["variant"] == "V1") & (c["period"] == "contrast")
                          & c["quantity"].isin(["xi_trap", "tau_mfpt"])
                          & (c["obs_rule"] == obs) & (c["ci_nan_rule"] == cir)]
                    if c.empty:
                        rows.append(_missing("K1b", proc, mech, "cell not in coverage", rs))
                        continue
                    for _, r in c.iterrows():
                        v = r["coverage"]
                        st = ("blocked_theory_incompatible" if not _compatible(r) else
                              "undefined_truth" if r["n_truth_nan"] > 0
                              else "pass" if v >= K1B_COV else "fail")
                        gating = r["pipeline"] == GATING_PIPELINE and _ci_gates(cir)
                        if (proc, r["quantity"]) in K1B_NOT_EVALUABLE:
                            st, gating = K1B_NOT_EVALUABLE_STATUS, False
                        rows.append(_row("K1b", r, "V1_ci95_coverage", v, K1B_COV, st,
                                         gating, rs, R=int(r["R"]),
                                         n_ci_failed=int(r["n_ci_failed"]),
                                         n_truth_nan=int(r["n_truth_nan"])))
    return rows


K1C_DESIGN_OPEN = "not_evaluable_design_open"


def posctrl_design_error(design_constants: dict | None) -> str | None:
    cells = {c["cell_index"] for c in S.registered_cells()
             if c["process"] == "v_posctrl" and c["gate_use"] == "K1c"}
    errs = []
    for ci, d in (design_constants or {}).items():
        if int(ci) in cells:
            if not isinstance(d, dict) or not d or "error" in d:
                errs.append(f"cell {ci}: {(d or {}).get('error', 'no design constants')}")
    return "; ".join(errs) or None


def gate_k1c(bias: pd.DataFrame, cov: pd.DataFrame, power: pd.DataFrame,
             design_error: str | None = None) -> list[dict]:
    """K1c; every gating row is K1C_DESIGN_OPEN while `design_error` is set
    (posctrl_design_error)."""
    rows = []
    b0, c0, p0 = _std_cells(bias), _std_cells(cov), _std_cells(power)
    sel = lambda d: d[(d["variant"] == "V1") & (d["period"] == "contrast")
                      & (d["quantity"] == "xi_trap")]
    for obs in OBS_RULES:
        for cir in CI_RULES:
            rs = _rs(obs, cir)
            for mech in ("a", "d", "e"):
                b, c, p = (sel(_sel(x, "v_posctrl", mech)) for x in (b0, c0, p0))
                b = b[b["obs_rule"] == obs]
                c = c[(c["obs_rule"] == obs) & (c["ci_nan_rule"] == cir)]
                p = p[(p["obs_rule"] == obs) & (p["ci_nan_rule"] == cir)]
                if b.empty or c.empty:
                    rows.append(_missing("K1c", "v_posctrl", mech, "cell not in tables", rs))
                    continue
                for _, r in b.iterrows():
                    v = _rel(r["median_bias"], r["median_truth"])
                    st = ("blocked_theory_incompatible" if not _compatible(r) else
                          "undefined_truth" if r.get("n_truth_nan", 0) > 0 else
                          "fail_nan" if r["n_est_nan"] > 0 else
                          "undefined" if not np.isfinite(v) else
                          "pass" if v <= K1C_TOL else "fail")
                    rows.append(_row("K1c", r, "V1_relative_bias_xi_shift", v, K1C_TOL, st,
                                     r["pipeline"] == GATING_PIPELINE and _ci_gates(cir), rs,
                                     n_est_nan=int(r["n_est_nan"]), R=int(r["R"])))
                for _, r in c.iterrows():
                    v = r["coverage"]
                    st = ("blocked_theory_incompatible" if not _compatible(r) else
                          "undefined_truth" if r["n_truth_nan"] > 0
                          else "pass" if v >= K1C_COV else "fail")
                    rows.append(_row("K1c", r, "V1_ci95_coverage_xi_shift", v, K1C_COV, st,
                                     r["pipeline"] == GATING_PIPELINE and _ci_gates(cir), rs,
                                     R=int(r["R"]),
                                     n_ci_failed=int(r["n_ci_failed"])))
                for _, r in p.iterrows():
                    rows.append(_row("K1c", r, "V1_power_ci_excludes_0",
                                     r["power_excl_null_failed_as_neg"], None, "report", False,
                                     rs, R=int(r["R"]), n_ci_failed=int(r["n_ci_failed"])))
    if design_error:
        for r in rows:
            if r["gating"]:
                r["status_if_design_ok"] = r["status"]
                r["status"] = K1C_DESIGN_OPEN
                r["note"] = f"(v) design search failed: {design_error}"
    return rows


K2_GATED_MECHANISMS = ()
K2_REPORT_MECHANISMS = ("a", "e")
K2_REPORT_STATUS = 'report_only_amendment10'
K2_VERDICT = "report-only (amendment 10)"
K2_TRUTH_FLAG = "truth biased for w > 0 (re-derivation 8)"
K2_REGISTERED_THRESHOLD = {"power_lo_gt_0.5": K2_POWER, "false_positive_rate_lo_gt_0.5": K2_FPR}
K2_REPORT_NOTE = ('amendment 10 (a): K2 is not gated, every (iv) cell is report only; '
                  "'no barrier' is written only as a one-sided bound; detection = CI "
                  'lower bound of 2ΔV > 0.5, reported against the nominal 2ΔV and by '
                  "each replicate's truth 2ΔV bin")


def gate_k2(power: pd.DataFrame, power_vs_truth: pd.DataFrame | None = None) -> list[dict]:
    rows = []
    p0 = _std_cells(power)
    for obs in OBS_RULES:
        for cir in CI_RULES:
            for var in ("V1", "V0"):
                rs = _rs(obs, cir, var)
                for k in (0, 1, 2, 4):
                    for mech in K2_REPORT_MECHANISMS:
                        p = _sel(p0, f"iv_bistable_{k}", mech)
                        p = p[(p["quantity"] == "two_delta_V") & p["period"].isin(["pre", "during"])
                              & (p["obs_rule"] == obs) & (p["ci_nan_rule"] == cir)
                              & (p["variant"] == var)]
                        if p.empty:
                            rows.append(_missing("K2", f"iv_bistable_{k}", mech,
                                                 "cell not in power", rs, gating=False))
                            continue
                        for _, r in p.iterrows():
                            kw = dict(R=int(r["R"]), n_ci_failed=int(r["n_ci_failed"]),
                                      median_truth=r.get("median_truth"),
                                      nominal_two_delta_V=float(k), truth_flag=K2_TRUTH_FLAG,
                                      note=K2_REPORT_NOTE)
                            if k == 0:
                                stat, v = ("false_positive_rate_lo_gt_0.5",
                                           r["detect_lo_gt_0p5_failed_as_pos"])
                                kw["fpr_failed_as_neg"] = r["detect_lo_gt_0p5_failed_as_neg"]
                            elif k == 2:
                                stat, v = "power_lo_gt_0.5", r["detect_lo_gt_0p5_failed_as_neg"]
                                kw["power_failed_as_pos"] = r["detect_lo_gt_0p5_failed_as_pos"]
                            else:
                                stat, v = ("detection_rate_lo_gt_0.5",
                                           r["detect_lo_gt_0p5_failed_as_neg"])
                                kw["detect_failed_as_pos"] = r["detect_lo_gt_0p5_failed_as_pos"]
                            kw["registered_threshold"] = K2_REGISTERED_THRESHOLD.get(stat)
                            rows.append(_row("K2", r, stat, v, None,
                                             "blocked_theory_incompatible" if not _compatible(r)
                                             else K2_REPORT_STATUS, False, rs, **kw))
    rows += k2_report_by_truth_bin(power_vs_truth)
    return rows


def k2_report_by_truth_bin(pvt: pd.DataFrame | None) -> list[dict]:
    rows = []
    p0 = _std_cells(pvt)
    for k in (0, 1, 2, 4):
        for mech in K2_REPORT_MECHANISMS:
            p = _sel(p0, f"iv_bistable_{k}", mech)
            if "quantity" in p:
                p = p[p["quantity"] == "two_delta_V"]
            if p.empty:
                rows.append(_missing("K2", f"iv_bistable_{k}", mech,
                                     "cell not in power_vs_truth", _rs(PRIMARY_OBS, PRIMARY_CI,
                                                                         "V1"), gating=False))
                continue
            for _, r in p.iterrows():
                rs = _rs(r.get("obs_rule") or PRIMARY_OBS, r.get("ci_nan_rule"), r.get("variant"))
                rows.append(_row("K2", r, "detection_rate_lo_gt_0.5_by_truth_bin",
                                 r["detect_failed_as_neg"], None,
                                 "blocked_theory_incompatible" if not _compatible(r)
                                 else K2_REPORT_STATUS, False, rs,
                                 truth_bin=r.get("truth_bin"), n=int(r["n"]),
                                 n_ci_failed=int(r["n_ci_failed"]),
                                 detect_failed_as_pos=r.get("detect_failed_as_pos"),
                                 median_truth=r.get("median_truth"),
                                 nominal_two_delta_V=float(k), truth_flag=K2_TRUTH_FLAG,
                                 note=K2_REPORT_NOTE))
    return rows


K3_DIAG_GATE = "K3_diag"


def gate_k3(k3_est: pd.DataFrame | None, cell_meta: dict,
            praw_idx: dict | None = None) -> list[dict]:
    rs = _rs(PRIMARY_OBS)
    if k3_est is None or not len(k3_est):
        return [_missing("K3", p, m, "no k1_k3_estimates rows", rs) for p, m in K1A_CELLS]
    C = _contract()
    try:
        from src import k1_neural_k3 as K3
        tables = []
        for proc, mech in K1A_CELLS:
            ci = _find_cell(cell_meta, proc, mech, "K1a/K1b")
            ni = _find_cell(cell_meta, proc, "a", "K1a_null", structure=mech)
            if ci is None or ni is None:
                raise ValueError(f"K3 cell {proc} x ({mech}) or its K1a_null not in the aggregate")
            cm, nm = cell_meta[ci]["saai"], cell_meta[ni]["saai"]
            bare = lambda m: {k: v for k, v in m.items() if k != "valid_for_law"}
            nn, ml, tr = K3.from_sim_long(k3_est, cell=bare(cm))
            bn, bm, bt = K3.from_sim_long(k3_est, cell=bare(nm))
            pc = S.PROCESS_KEY[proc]
            entries = []
            for p2, m2 in K1A_CELLS:
                if p2 == proc:
                    c2 = _find_cell(cell_meta, p2, m2, "K1a/K1b")
                    entries.append({"cell": bare(cell_meta[c2]["saai"]),
                                    "predictions": (praw_idx or {}).get((c2, "saai",
                                                                         "projection"), [])})
            pe = C.predicted_effect_sizes(pc, "saai", entries)
            tables.append(K3.k3_cell_table(nn, ml, tr, cell=bare(cm),
                                           baseline={"neural": bn, "mle": bm, "truth": bt,
                                                     "cell": bare(nm)},
                                           predicted_effect=pe))
        table = pd.concat(tables, ignore_index=True)
        v = K3.k3_verdict(table)
    except Exception as exc:
        return [_row("K3", {"process": "*", "mechanism": "*", "pipeline": "saai",
                            "variant": "V0", "obs_rule": PRIMARY_OBS},
                     "k3_verdict", np.nan, None, "not_evaluable_refused", True, rs,
                     note=_refused(exc), R=np.nan)]
    tgt = {"band": "contrast_band", "excess": "contrast_excess"}.get(v["gate_target"])
    g = table[(table["role"] == "gate_candidate") & (table["target"] == tgt)] if tgt else table.iloc[0:0]
    rows = []
    for _, r in g.iterrows():
        key = "/".join((K3.canonical_process(r["process"]), r["exit_cell"], r["quantity"]))
        rows.append(_row("K3", {"process": r["process"], "mechanism": r["exit_cell"],
                                "pipeline": "saai", "variant": "V0", "obs_rule": PRIMARY_OBS,
                                "period": "contrast", "quantity": r["quantity"]},
                         f"k3_{v['gate_target']}", np.nan, "same side / ratio in [0.5, 2]",
                         "pass" if bool(r["pass"]) else "fail", True, rs,
                         R=int(r["R"]), comparison_kind=v["comparison_kinds"].get(key),
                         target=v["gate_target"], floor_rule=v["floor_rule"]))
    rows.append(_row("K3", {"process": "*", "mechanism": "*", "pipeline": "saai",
                            "variant": "V0", "obs_rule": PRIMARY_OBS},
                     "k3_verdict", float(v["n_gate_pass"]), v["n_expected"],
                     "pass" if v["K3_pass"] else "fail", True, rs,
                     R=K3.K3_R if v["K3_pass"] or v["n_gate"] else np.nan,
                     target=v["gate_target"], floor_rule=v["floor_rule"],
                     claim_estimator_independent=v["claim_estimator_independent"],
                     vacuous_passes=";".join(v["vacuous_passes"]),
                     config=v["config"], config_note=v["config_note"]))
    return rows


def gate_k3_diagnostic(bias: pd.DataFrame) -> list[dict]:
    rows = []
    b0 = _std_cells(bias)
    for obs in OBS_RULES:
        rs = _rs(obs)
        for proc in ("i_ou", "ii_cubic"):
            for mech in ("d", "e"):
                c = _sel(b0, proc, mech)
                c = c[(c["pipeline"] == "saai") & (c["obs_rule"] == obs)]
                nn = c[c["variant"] == "V0_neural"]
                if nn.empty:
                    rows.append(_missing(K3_DIAG_GATE, proc, mech, "no neural rows", rs, gating=False))
                    continue
                units = [("xi_trap", "during"), ("xi_trap", "contrast")] + (
                    [("xi_trap", "pre")] if mech == "e" else [])
                for q, period in units:
                    m = c[(c["variant"] == "V0") & (c["period"] == period) & (c["quantity"] == q)]
                    n = nn[(nn["period"] == period) & (nn["quantity"] == q)]
                    r = (n.iloc[0] if len(n) else m.iloc[0]).to_dict() if (len(n) or len(m)) else {
                        "process": proc, "mechanism": mech, "period": period, "quantity": q}
                    bm = float(m["median_bias"].iloc[0]) if len(m) else np.nan
                    bn = float(n["median_bias"].iloc[0]) if len(n) else np.nan
                    nan_m = int(m["n_bias_nan"].iloc[0] if "n_bias_nan" in m else
                                m["n_est_nan"].iloc[0]) if len(m) else 0
                    nan_n = int(n["n_bias_nan"].iloc[0] if "n_bias_nan" in n else
                                n["n_est_nan"].iloc[0]) if len(n) else 0
                    tn = int(m["n_truth_nan"].iloc[0]) if len(m) and "n_truth_nan" in m else 0
                    R = min(int(m["R"].iloc[0]) if len(m) else 0, int(n["R"].iloc[0]) if len(n) else 0)
                    ratio = _rel(bn, bm)
                    ok = (np.isfinite(ratio) and np.sign(bn) == np.sign(bm) != 0
                          and K3_RATIO[0] <= ratio <= K3_RATIO[1])
                    st = ("missing" if not (len(m) and len(n)) else
                          "blocked_theory_incompatible" if not _compatible(r) else
                          "undefined_truth" if tn > 0 else
                          "fail_nan" if nan_m > 0 or nan_n > 0 else
                          "undefined" if not np.isfinite(ratio) else
                          "pass" if ok else "fail")
                    rows.append(_row(K3_DIAG_GATE, r, "neural_over_mle_V0_bias", ratio,
                                     list(K3_RATIO), st, False, rs, median_bias_mle=bm, median_bias_neural=bn,
                                     n_nan_mle=nan_m, n_nan_neural=nan_n, R=R))
    return rows


PRIMARY_RULE_SET = {"S0": _rs(PRIMARY_OBS), "S0_neural": _rs(PRIMARY_OBS),
                    "K1a": _rs(PRIMARY_OBS),
                    "K1b": _rs(PRIMARY_OBS, PRIMARY_CI), "K1c": _rs(PRIMARY_OBS, PRIMARY_CI),
                    "K2": _rs(PRIMARY_OBS, PRIMARY_CI, "V1"), "K3": _rs(PRIMARY_OBS),
                    K3_DIAG_GATE: _rs(PRIMARY_OBS)}


UNSIGNED_RULE_GATES = ("K1b", "K1c", "K2")
REPORT_ONLY_GATES = ("K2",)
SIM_RULES_SIGNED_MARK = "**Simulation rules signed by the author:** signed ("


def sim_rules_signed(plan_path: str | None = None) -> bool:
    import re
    path = plan_path or os.path.join(ROOT, 'results', 'plan.md')
    try:
        txt = open(path, encoding="utf-8").read()
    except OSError:
        return False
    a = txt.find("### Simulation rules draft")
    if a < 0:
        return False
    b = txt.find("\n### ", a + 1)
    sec = txt[a:b if b > 0 else None]
    ok = re.search(r"\*\*Simulation rules signed by the author:\*\* signed \(\d{4}-\d{2}-\d{2}, author\)", sec)
    return bool(ok) and "[author input needed" not in sec


def verdicts(rows: pd.DataFrame, signed: bool | None = None) -> pd.DataFrame:
    out = []
    for (gate, rs), g in rows.groupby(["gate", "rule_set"], sort=False):
        gg = g[g["gating"]]
        st = gg["status"].astype(str)
        if gg.empty or (st == "missing").all():
            v = "not_run"
        elif st.str.startswith("fail").any():
            v = "fail"
        elif (st == "pass").all():
            v = "pass"
        else:
            v = "not_evaluable"
        out.append({"gate": gate, "rule_set": rs,
                    "primary": PRIMARY_RULE_SET.get(gate) == rs, "verdict": v,
                    "n_gating_rows": len(gg), "n_pass": int((st == "pass").sum()),
                    "n_fail": int(st.str.startswith("fail").sum()),
                    "n_fail_R": int((st == "fail_R").sum()),
                    "n_fail_nan": int(st.str.startswith("fail_nan").sum()
                                      + st.str.contains("non-finite").sum()),
                    "n_undefined": int(st.str.startswith("undefined").sum()),
                    "n_not_evaluable": int(st.str.startswith("not_evaluable").sum()),
                    "n_indistinguishable_from_null": int(st.str.contains("indistinguishable").sum()),
                    "n_blocked": int(st.str.startswith("blocked").sum()),
                    "n_missing": int((st == "missing").sum())})
    ver = pd.DataFrame(out)
    ver["verdict_unblocked"] = ver["verdict"]
    signed = public_guard_skipped(True) if signed is None else signed
    if not signed:
        ver.loc[ver["gate"].isin(UNSIGNED_RULE_GATES), "verdict"] = "not_evaluable_unsigned_sim_rules"
    blocked = s0_blocked_gates(rows, ver["gate"].unique())
    ver.loc[ver["gate"].isin(blocked), "verdict"] = "blocked_by_S0"
    k2 = ver["gate"].isin(REPORT_ONLY_GATES)
    ver.loc[k2, ["verdict", "verdict_unblocked"]] = K2_VERDICT
    ver["gated"] = ~k2
    return ver


def s0_pass_by_process(rows: pd.DataFrame, gate: str = "S0") -> dict:
    """{process: True iff every primary gating row of `gate` for that process
    (all its pipelines, periods, quantities) is 'pass'} — from the rows
    gate_s0 wrote; a process without rows is absent (= not passed)."""
    if rows is None or not len(rows) or "process" not in rows.columns:
        return {}
    g = rows[(rows["gate"] == gate) & (rows["rule_set"] == PRIMARY_RULE_SET[gate])
             & rows["gating"].astype(bool)]
    return {p: bool((x["status"].astype(str) == "pass").all() and len(x))
            for p, x in g.groupby("process")}


def s0_blocked_gates(rows: pd.DataFrame, gates) -> set:
    mle, nn = s0_pass_by_process(rows, "S0"), s0_pass_by_process(rows, "S0_neural")
    covered = set()
    if rows is not None and len(rows) and {"process", "pipeline"} <= set(rows.columns):
        g = rows[rows["gate"] == "S0"]
        covered = {p for p, x in g.groupby("process")
                   if set(S.PIPELINES) <= set(x["pipeline"].dropna())}
    out = set()
    for gate in gates:
        if gate in S0_GATES:
            continue
        need = GATE_S0_SCOPE.get(gate, S0_PROCESSES)
        ok = all(mle.get(p, False) and p in covered for p in need)
        ok = ok and all(nn.get(p, False) for p in NEURAL_S0_SCOPE.get(gate, ()))
        if not ok:
            out.add(gate)
    return out


def evaluate(bias, cov, power, pred, cell_meta=None, praw=None, shocked=None,
             shock_share_w=None, signed=None, s0_record=None, k1c_design_error=None,
             k3_est=None, neural_s0_record=None, power_vs_truth=None):
    for name, obj in (("bias", bias), ("coverage", cov), ("power", power),
                      ("predictions", pred), ("shocked null", shocked), ("K3", k3_est),
                      ("power vs truth", power_vs_truth), ("predict rows", praw)):
        refuse_exploratory(obj, f"the {name} input")
    idx = index_predict_rows(praw) if praw is not None else None
    k1a_rows, claims = gate_k1a(pred, cell_meta or {}, idx)
    rows = (gate_s0(shock_share_w, s0_record, neural_s0_record) + k1a_rows
            + gate_k1b(cov) + gate_k1c(bias, cov, power, k1c_design_error)
            + gate_k2(power, power_vs_truth)
            + gate_k3(k3_est, cell_meta or {}, idx) + gate_k3_diagnostic(bias)
            + gate_shocked_null(shocked))
    df = pd.DataFrame(rows)
    ver = verdicts(df, signed)
    blocked = s0_blocked_gates(df, df["gate"].unique())
    df["blocked_by_S0"] = df["gate"].isin(blocked)
    if claims is not None and "K1a" in s0_blocked_gates(df, ["K1a"]):
        claims = {"blocked_by_S0": True, "unblocked": claims}
    return df, ver, claims


def _cell_meta_from_sim(sim_meta: dict) -> dict:
    cm = sim_meta.get("contract_cell_meta") or {}
    return {int(k): v for k, v in cm.items() if v}


def same_code_as_aggregate(agg_commit, git=None) -> None:
    from k1_run_cells import same_code
    if not agg_commit or len(str(agg_commit)) != 40:
        raise SystemExit("the aggregate meta has no aggregator commit")
    why = same_code(str(agg_commit))
    if why:
        raise SystemExit(f"the aggregate was written at another code state: {why}; "
                         "re-run k1_run_cells.py aggregate at this commit first")


def main(argv=None):
    from k1_run_cells import is_under, refuse_exploratory_paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.path.join(ROOT, "results", "tables"))
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args(argv)
    refuse_exploratory_paths([a.tables], "--tables")
    git = lambda *x: subprocess.check_output(["git", "-C", ROOT, *x], text=True).strip()
    if git("status", "--porcelain") and not a.dev:
        raise SystemExit("refusing to evaluate gates on a dirty tree (amendment 1 (C))")
    in_results = is_under(a.tables)
    if a.dev and in_results:
        raise SystemExit("--dev never writes to (or below) results/tables")
    with open(os.path.join(a.tables, "k1_sim_meta.json")) as f:
        sim_meta = json.load(f)
    refuse_exploratory(sim_meta, "k1_sim_meta.json")
    registered = bool(sim_meta.get("registered", False))
    if not registered and in_results:
        raise SystemExit("the tables come from a non-registered aggregate: refusing to write "
                         "gates into (or below) results/tables")
    def t(n):
        try:
            df = pd.read_csv(os.path.join(a.tables, n))
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        refuse_exploratory(df, n)
        return df
    opt = lambda n: t(n) if os.path.exists(os.path.join(a.tables, n)) else None
    if registered and not a.dev:
        same_code_as_aggregate(sim_meta.get("aggregator_git"), git)
    from k1_run_cells import PREDICT_ROWS_FILE, read_predict_rows
    pr_path = os.path.join(a.tables, PREDICT_ROWS_FILE)
    praw = read_predict_rows(pr_path) if os.path.exists(pr_path) else None
    refuse_exploratory(praw, PREDICT_ROWS_FILE)
    rows, ver, claims = evaluate(t("k1_sim_bias.csv"), t("k1_sim_coverage.csv"),
                                 t("k1_power.csv"), opt("k1_predictions.csv"),
                                 _cell_meta_from_sim(sim_meta), praw,
                                 opt("k1_shocked_null_replicates.csv"),
                                 sim_meta.get("province_shock_share_w"),
                                 k1c_design_error=posctrl_design_error(
                                     sim_meta.get("design_constants")),
                                 k3_est=opt("k1_k3_estimates.csv"),
                                 power_vs_truth=opt("k1_power_vs_truth.csv"))
    label = "registered" if registered else "NON-REGISTERED"
    rows.insert(0, "run_label", label)
    ver.insert(0, "run_label", label)
    rows.to_csv(os.path.join(a.tables, "k1_gates.csv"), index=False)
    ver.to_csv(os.path.join(a.tables, "k1_gates_verdict.csv"), index=False)
    C = _contract()
    meta = {"git": git("rev-parse", "HEAD"), "dev": a.dev, "run_label": label,
            "sim_meta_code_commit": sim_meta.get("code_commit"),
            "theory_frozen": sim_meta.get("theory_frozen"),
            "contract_version": getattr(C, "CONTRACT_VERSION", None),
            "sim_rules_signed": public_guard_skipped(True),
            "theory_has_shock_share_term": theory_has_shock_share_term(),
            "own_offset_rule": "src.k1_contract.k1a_cell_verdict (amendment 3 (f) option 2)",
            "s0_source": "committed S0 records (scripts/k1_s0.py, contract 7)",
            "gating_pipeline": GATING_PIPELINE, "gated_row": GATED_ROW,
            "primary_rule_sets": PRIMARY_RULE_SET, "law_claims": claims,
            "rule_sources": "CI NaN rule (tol0 gating), K2 variant (V1, saai), G option "
                            "(3), E sensitivity: simulation rules adopted 2026-10-22; K3: "
                            "k3_verdict ('band', 'effect_scaled'); S0 scope "
                            "per gate: amendment 4 D4; K2 not gated, (iv) × (a), (e) "
                            "report only against the nominal and the truth-binned 2ΔV, "
                            "truth flagged biased for w > 0: amendment 10 (a)",
            "K2": K2_VERDICT, "K2_truth_flag": K2_TRUTH_FLAG,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.tables, "k1_gates_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(ver.to_string(index=False))


if __name__ == "__main__":
    main()
