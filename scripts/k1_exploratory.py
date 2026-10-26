"""cells      list the exploratory cells (registered Step 2 indices)
ntasks     number of array tasks (--reps-per-chunk replicates each)
run        one array task (or --cells/--reps) → chunks in
           results/records/k1_exploratory_chunks/ (git-ignored)
aggregate  chunks → results/tables/k1_exploratory/ (or, --partial,
           results/records/k1_exploratory_partial/)
"""
from __future__ import annotations

def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import argparse
import contextlib
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd

from src import k1_contract as C
from src import k1_sim as S
import k1_gates as G
import k1_run_cells as RC

EXPLORATORY_LABEL = "exploratory = true, not a verdict (amendment 11)"
EXPLORATORY_LABEL_ZH = "exploratory, not a verdict (amendment 11)"
MARKER = RC.EXPLORATORY_MARKER
assert MARKER == G.EXPLORATORY_MARKER
RUN_LABEL = "EXPLORATORY (amendment 11 (b)) — never gated"

PLAN_REL = 'plan.md'
AMENDMENT_SECTION = "### amendment 11"
AMENDMENT_APPROVAL = r"\*\*Approved by the author:\*\* approved \(\d{4}-\d{2}-\d{2}, author\)"
FREEZE_COMMIT = ''
FREEZE_PREFIX = FREEZE_COMMIT[:7]

TABLES_DIR = RC.EXPLORATORY_TABLES
CHUNK_DIR = os.path.join(ROOT, 'results', 'records', "k1_exploratory_chunks")
PARTIAL_DIR = os.path.join(ROOT, 'results', 'records', "k1_exploratory_partial")
CHUNK_PREFIX = "k1exp_"
PREDICT_ROWS_FILE = "k1_exploratory_predict_rows.json.gz"

EXIT_CELLS = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)
NULL_CELLS = (34, 35, 36, 37, 46, 47, 48, 49)
EXPLORATORY_CELLS = EXIT_CELLS + NULL_CELLS
REPS_PER_CHUNK = 5
CONTROL_CELLS = (38, 39, 40, 41)
CONTROL_NULL_CELLS = (42, 43, 44, 45)
SENS_CELLS = (50, 51)
SENS_NULL_CELLS = (35, 37)
CONTROLS_SET_CELLS = tuple(sorted(CONTROL_CELLS + CONTROL_NULL_CELLS + SENS_CELLS
                                  + SENS_NULL_CELLS))
CONTROL_LABEL = "report-only control, exploratory, not a verdict (amendment 11)"
CONTROL_LABEL_ZH = "report-only control, exploratory, not a verdict (amendment 11)"
CONTROL_RUN_LABEL = ('REPORT-ONLY CONTROL, EXPLORATORY (amendment 11 (b); decision of the '
                     'authors, 2026-10-24) — never gated')
CONTROL_VARIANT_GATE_USES = ("control", S.SENS_NO2006_GATE_USE)

SETS = {
    "main": {"cells": EXPLORATORY_CELLS, "label": EXPLORATORY_LABEL,
             "label_zh": EXPLORATORY_LABEL_ZH, "run_label": RUN_LABEL,
             "chunk_dir": CHUNK_DIR, "partial_dir": PARTIAL_DIR,
             "chunk_prefix": CHUNK_PREFIX, "out_prefix": "k1_exploratory_",
             "predict_rows": PREDICT_ROWS_FILE},
    "controls": {"cells": CONTROLS_SET_CELLS, "label": CONTROL_LABEL,
                 "label_zh": CONTROL_LABEL_ZH, "run_label": CONTROL_RUN_LABEL,
                 "chunk_dir": os.path.join(ROOT, 'results', 'records',
                                           "k1_exploratory_controls_chunks"),
                 "partial_dir": os.path.join(ROOT, 'results', 'records',
                                             "k1_exploratory_controls_partial"),
                 "chunk_prefix": "k1expc_", "out_prefix": "k1_exploratory_controls_",
                 "predict_rows": "k1_exploratory_controls_predict_rows.json.gz"},
}


def cell_set(name: str) -> dict:
    if name not in SETS:
        raise SystemExit(f"unknown cell set {name!r} (main, controls)")
    return SETS[name]


NEURAL_NOTE = ("not run (amendment 11 (b): exploratory MLE only; neural S0 deferred, amendment 7 (S3); "
               "K3 blocked)")

S0_CAVEAT_PROCESS, S0_CAVEAT_PIPELINE, S0_CAVEAT_PERIOD, S0_CAVEAT_Q = (
    "cubic", "saai", "during", "mean_diffusion")
PRECHECK_UNITS = os.path.join(ROOT, "results", "tables", "k1_s0_scope_f2",
                              "k1_s0_scope_precheck_units.csv")
PRECHECK_CAVEAT_CELL, PRECHECK_CAVEAT_Q = 10, "tau_mfpt"


def control_cells() -> list[dict]:
    allc = S.registered_cells()
    want = {**{i: "control" for i in CONTROL_CELLS},
            **{i: "control_null" for i in CONTROL_NULL_CELLS},
            **{i: S.SENS_NO2006_GATE_USE for i in SENS_CELLS},
            **{i: "K1a_null" for i in SENS_NULL_CELLS}}
    got = [c for c in allc if c["cell_index"] in want]
    for c in got:
        ci = c["cell_index"]
        if c["gate_use"] != want[ci] or c["B"] != 0:
            raise SystemExit(f"registered cell {ci} moved: gate_use {c['gate_use']!r}, B {c['B']}")
        key = (c["process"], c.get("structure") or c["mechanism"])
        if key not in S.K1A_CELLS:
            raise SystemExit(f"cell {ci} is not a variant of a K1a cell")
    for ci in SENS_CELLS:
        c = allc[ci]
        if c["mechanism"] != "e" or tuple(c.get("drop_breaks") or ()) != S.SENS_DROP_BREAKS:
            raise SystemExit(f"cell {ci} is not the registered no-2006 (e) cell")
        n = allc[SENS_NULL_CELLS[SENS_CELLS.index(ci)]]
        if (n["process"], n["structure"], bool(n.get("dated_shocks"))) != (c["process"], "e", False):
            raise SystemExit(f"cell {n['cell_index']} is not the K1a_null band of cell {ci}")
    for ci in CONTROL_CELLS:
        c = allc[ci]
        if c["re_placement"] != S.CONTROL_RE_PLACEMENT or c["gated"]:
            raise SystemExit(f"cell {ci} is not a persistent report-only control")
        if S.null_cell_for(c)["cell_index"] not in CONTROL_NULL_CELLS:
            raise SystemExit(f"the null of control cell {ci} is not in the control set")
    if tuple(c["cell_index"] for c in got) != CONTROLS_SET_CELLS:
        raise SystemExit("control set cells missing from the registered grid")
    if any(S.is_precheck_a_cell(c) for c in got):
        raise SystemExit("a control-set cell is a precheck cell (other seed stream)")
    return got


def set_cells(name: str = "main") -> list[dict]:
    return control_cells() if cell_set(name) is SETS["controls"] else exploratory_cells()


def exploratory_cells() -> list[dict]:
    """The registered K1a/K1b exit cells and the matched K1a null cells, in
    registered order; checked against the registered gate uses."""
    allc = S.registered_cells()
    exits = [c for c in allc if c["gate_use"] == "K1a/K1b" and c["mechanism"] != "a"]
    nulls = [c for c in allc if c["gate_use"] in ("K1a_null", "K1a_shocked_null")]
    got = tuple(c["cell_index"] for c in exits + nulls)
    if got != EXPLORATORY_CELLS:
        raise SystemExit(f"registered cells moved: K1a/K1b exits + K1a nulls are {got}, "
                         f"expected {EXPLORATORY_CELLS}")
    for c in nulls:
        if (c["process"], c["structure"]) not in S.K1A_CELLS or c["B"] != 0:
            raise SystemExit(f"cell {c['cell_index']} is not a matched K1a null")
    if any(S.is_precheck_a_cell(c) for c in exits + nulls):
        raise SystemExit("an exploratory cell is a precheck cell (other seed stream)")
    return exits + nulls


def is_exploratory_cell(cell: dict, set_name: str = "main") -> bool:
    ci = cell.get("cell_index")
    return (ci in cell_set(set_name)["cells"] and ci < len(S.registered_cells())
            and cell == S.registered_cells()[ci])


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", ROOT, *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


def plan_text_at_head() -> str:
    try:
        return _git("show", f"HEAD:{PLAN_REL}")
    except Exception as exc:
        raise SystemExit(f"cannot read {PLAN_REL} at HEAD ({type(exc).__name__}): the "
                         "exploratory driver runs only in a git checkout") from exc


def amendment_approved(plan_text: str) -> bool:
    i = plan_text.find(AMENDMENT_SECTION)
    while i >= 0:
        line_end = plan_text.find("\n", i)
        head = plan_text[i:line_end if line_end >= 0 else None]
        if re.match(r"### amendment 11(?!\d)", head):
            nxt = plan_text.find("\n### ", i + 1)
            body = plan_text[i:nxt if nxt >= 0 else None]
            return re.search(AMENDMENT_APPROVAL, body) is not None
        i = plan_text.find(AMENDMENT_SECTION, i + 1)
    return False


def require_preconditions() -> str:
    if not public_guard_skipped(True):
        raise SystemExit("the amendment 11 approval line is not in the registration record at "
                         "HEAD: the exploratory K1a/K1b run is refused")
    env = os.environ.get("K1_THEORY_FROZEN", "").strip()
    if len(env) < 7 or not FREEZE_COMMIT.startswith(env):
        raise SystemExit(f"K1_THEORY_FROZEN={env!r}: the exploratory run exists only under the "
                         f"second freeze {FREEZE_PREFIX}… (amendment 11)")
    assert_theory_frozen = public_guard_skipped
    got = assert_theory_frozen()
    if got != FREEZE_COMMIT:
        raise SystemExit(f"the theory-freeze guard returned {got}, not {FREEZE_COMMIT}")
    return got


def allowed_output(path: str) -> bool:
    """Only results/tables/k1_exploratory/ and results/records/k1_exploratory_*."""
    return RC.is_exploratory_path(path)


def check_output(path: str) -> str:
    if not allowed_output(path):
        raise SystemExit(f"{path}: the exploratory driver writes only to "
                         "results/tables/k1_exploratory/ and results/records/k1_exploratory_*")
    return path


def caveats() -> dict:
    rec_path = os.path.join(ROOT, C.S0_RECORD.format(process=S0_CAVEAT_PROCESS,
                                                     pipeline=S0_CAVEAT_PIPELINE))
    with open(rec_path) as f:
        full = json.load(f)
    res = full["result"]
    md = res[S0_CAVEAT_PERIOD][S0_CAVEAT_Q]
    eff = float(res["predicted_effect"]["effect"][S0_CAVEAT_Q])
    bias, tol = float(md["median_bias"]), float(md["tol"])
    share = abs(bias) / eff
    u = pd.read_csv(PRECHECK_UNITS)
    r = u[(u["cell_index"] == PRECHECK_CAVEAT_CELL) & (u["quantity"] == PRECHECK_CAVEAT_Q)
          & (u["period"] == "contrast")]
    if len(r) != 1:
        raise SystemExit(f"precheck unit cell {PRECHECK_CAVEAT_CELL} {PRECHECK_CAVEAT_Q} not "
                         f"found in {os.path.relpath(PRECHECK_UNITS, ROOT)}")
    cov = float(r["coverage"].iloc[0])
    band = list(S.COVERAGE_BAND)
    s0_status = "pass" if full.get("pass") else "FAIL"
    text = (f"{EXPLORATORY_LABEL_ZH}. S0 {S0_CAVEAT_PROCESS}/{S0_CAVEAT_PIPELINE} under freeze "
            f"{FREEZE_PREFIX} (attempt {full.get('attempt')}): {s0_status} — "
            f"{S0_CAVEAT_PERIOD} {S0_CAVEAT_Q} median bias {bias:.6f} = {100 * share:.1f}% of "
            f"the predicted contrast |{eff:.6f}| (tolerance {100 * C.S0_REL_TOL:.0f}% = "
            f"{tol:.6f}); (a) precheck scope K1b: cell {PRECHECK_CAVEAT_CELL} τ-ratio contrast "
            f"coverage {cov:.3f} outside {band}; neural S0 deferred, no neural estimator run. "
            "Descriptive evidence only (amendment 11 (b)): never 'passed a pre-registered "
            "criterion', never a 'law'.")
    return {"text": text, "s0_record": os.path.relpath(rec_path, ROOT),
            "s0_pass": bool(full.get("pass")), "s0_median_bias": bias, "s0_tol": tol,
            "s0_predicted_effect": eff, "s0_bias_share_of_effect": share,
            "s0_tol_share": C.S0_REL_TOL,
            "precheck_units": os.path.relpath(PRECHECK_UNITS, ROOT),
            "precheck_cell": PRECHECK_CAVEAT_CELL, "precheck_quantity": PRECHECK_CAVEAT_Q,
            "precheck_coverage": cov, "precheck_band": band,
            "neural": NEURAL_NOTE}


def label_frame(df: pd.DataFrame, caveat_text: str, set_name: str = "main") -> pd.DataFrame:
    cs = cell_set(set_name)
    df = df.copy()
    for col in (MARKER, "caveat", "run_label"):
        if col in df.columns:
            df = df.drop(columns=col)
    df.insert(0, "caveat", caveat_text)
    df.insert(0, MARKER, cs["label"])
    df.insert(0, "run_label", cs["run_label"])
    return df


def label_gate_rows(rows: list[dict], caveat_text: str, set_name: str = "main") -> pd.DataFrame:
    """k1_gates rows → exploratory rows: never gating, outcome prefixed."""
    df = pd.DataFrame(rows)
    if df.empty:
        return label_frame(df, caveat_text, set_name)
    df = df.rename(columns={"gate": "registered_rule", "status": "exploratory_outcome"})
    df["exploratory_outcome"] = "exploratory:" + df["exploratory_outcome"].astype(str)
    if "status_before_R_check" in df:
        df["status_before_R_check"] = df["status_before_R_check"].map(
            lambda s: s if pd.isna(s) else f"exploratory:{s}")
    if "law_status" in df:
        df["law_status"] = df["law_status"].map(
            lambda s: s if pd.isna(s) else f"exploratory:{s}")
    df["would_gate_if_registered"] = df["gating"].astype(bool)
    df["gating"] = False
    return label_frame(df, caveat_text, set_name)


ALLOWED_S0_FAILURE = (S0_CAVEAT_PROCESS, S0_CAVEAT_PIPELINE)
ALLOWED_S0_FAILED_UNITS = frozenset({(S0_CAVEAT_PERIOD, S0_CAVEAT_Q)})
ALLOWED_PRECHECK_SCOPE = "K1b"


class ExploratoryRefused(S.GuardRefused):
    """A refusal the exploratory driver does NOT tolerate (fail closed)."""


def _s0_failed_units(result: dict, process: str) -> set:
    gated = C.s0_gated_quantities(process)
    bad = set()
    for per in C.PERIODS:
        body = result.get(per)
        if not isinstance(body, dict):
            raise ExploratoryRefused(f"S0 record body of {process} lacks the {per!r} period")
        for q in C.QUANTITIES:
            want = "pass" if q in gated else C.S0_RHO1_REPORT_STATUS
            if (body.get(q) or {}).get("status") != want:
                bad.add((per, q))
    return bad


def s0_checks_except_pass(process: str, pipeline: str) -> dict:
    if (process, pipeline) != ALLOWED_S0_FAILURE:
        raise ExploratoryRefused(f"S0 {process}/{pipeline}: only the S0 failure of "
                                 f"{'/'.join(ALLOWED_S0_FAILURE)} is tolerated (amendment 11)")
    rec = C._require_frozen()
    path = C.s0_record_path(process, pipeline)
    s0 = json.loads(path.read_text())
    if s0.get("pass") is not False:
        raise ExploratoryRefused(f"S0 {process}/{pipeline}: record pass is "
                                 f"{s0.get('pass')!r}, not the logged failure")
    if s0.get("frozen_commit") != rec.get("frozen_commit"):
        raise ExploratoryRefused("S0 record was made under another theory freeze")
    if s0.get("contract_version") != C.CONTRACT_VERSION:
        raise ExploratoryRefused("S0 record was made under another CONTRACT_VERSION")
    if s0.get("clean_tree") is not True or not s0.get("code_commit"):
        raise ExploratoryRefused("S0 record lacks a clean producing tree / code commit")
    if s0.get("seeds") != C.s0_seeds(process, pipeline):
        raise ExploratoryRefused("S0 record seeds are not the registered s0_seeds")
    result = s0.get("result") or {}
    if s0.get("result_sha256") != C._sha(result):
        raise ExploratoryRefused("S0 record result does not match its hash")
    if result.get("pass") is not False:
        raise ExploratoryRefused("S0 record body pass is not the logged failure")
    failed = _s0_failed_units(result, process)
    if failed != set(ALLOWED_S0_FAILED_UNITS):
        raise ExploratoryRefused(f"S0 {process}/{pipeline} failed on {sorted(failed)}, not "
                                 f"only {sorted(ALLOWED_S0_FAILED_UNITS)}: the caveat would "
                                 "mislabel it")
    as_if = json.loads(json.dumps(result))
    for per, q in failed:
        as_if[per][q] = {**(as_if[per].get(q) or {}), "status": "pass"}
    as_if["pass"] = True
    try:
        C.validate_s0_result(as_if, process, pipeline)
    except Exception as exc:
        raise ExploratoryRefused(f"S0 record body: {exc}") from exc
    entries = C._entries(process, pipeline)
    started = C._started(entries)
    same = [e for e in entries if e.get("attempt") == s0.get("attempt")
            and e.get("frozen_commit") == s0.get("frozen_commit")]
    st = [e for e in same if e.get("stage") == "started"]
    res = [e for e in same if e.get("stage") == "result"
           and e.get("result_sha256") == s0.get("result_sha256")]
    if len(st) != 1 or len(res) != 1 or len(same) != 2 \
            or s0.get("attempt") != len(started) or res[0].get("pass") is not False:
        raise ExploratoryRefused(f"S0 record attempt {s0.get('attempt')} is not the last logged "
                                 f"attempt of {process}/{pipeline} or lacks its started / "
                                 "failed-result log entries")
    try:
        C._check_s0_code(s0)
    except Exception as exc:
        raise ExploratoryRefused(f"S0 {process}/{pipeline}: {exc}") from exc
    return s0


def checked_s0(process: str, pipeline: str, require=None) -> str:
    require = require or C._require_s0_pass
    try:
        require(process, pipeline)
        return f"S0 {process}/{pipeline}: pass verified (code closure checked)"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if not (isinstance(exc, RuntimeError) and (process, pipeline) == ALLOWED_S0_FAILURE
                and str(exc).startswith(f"S0 did not pass for {process}/{pipeline}:")):
            raise ExploratoryRefused(f"S0 {process}/{pipeline} refused: {msg}") from exc
    s0_checks_except_pass(process, pipeline)
    return (f"S0 {process}/{pipeline}: registered refusal not enforced (amendment 11 (b)): "
            f"{msg[:200]}; every other S0 check incl. the code closure passed")


def checked_precheck(src: dict) -> str:
    """The registered coverage precheck on an exit cell, with the one
    tolerated refusal (2).  Any other refusal raises ExploratoryRefused."""
    try:
        S._check_coverage_precheck(src)
        return "coverage precheck: pass"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        cause = exc
    try:
        g = S.coverage_precheck_scope(src)
        assert_theory_frozen = public_guard_skipped
        res = S.coverage_precheck(frozen=assert_theory_frozen())
    except Exception as exc:
        raise ExploratoryRefused(f"coverage precheck refused: {msg}; {exc}") from exc
    sc = (res.get("scopes") or {}).get(g)
    oob = list((sc or {}).get("out_of_band") or [])
    want = f"cell {PRECHECK_CAVEAT_CELL} "
    if not (isinstance(cause, RuntimeError) and g == ALLOWED_PRECHECK_SCOPE and sc
            and sc.get("pass") is False and len(oob) == 1 and oob[0].startswith(want)
            and f" {PRECHECK_CAVEAT_Q} contrast:" in oob[0]
            and str(cause).startswith(f"coverage precheck: scope {g}:")):
        raise ExploratoryRefused(f"coverage precheck refused outside the tolerated scope "
                                 f"{ALLOWED_PRECHECK_SCOPE} / cell {PRECHECK_CAVEAT_CELL} "
                                 f"{PRECHECK_CAVEAT_Q}: {msg}") from cause
    return f"coverage precheck: registered refusal not enforced (amendment 11 (b)): {msg[:200]}"


def exploratory_estimation_checks(cell: dict, pipelines, require=None) -> str:
    src = S.exit_source_cell(cell)
    notes = []
    for pl in pipelines:
        meta = S.contract_cell_meta(src, pl)
        c = C.check_cell_config(meta)
        if c["exit_mechanism"] != "a":
            notes.append(checked_s0(c["process"], pl, require))
    notes.append(checked_precheck(src))
    return "; ".join(notes)


def exploratory_exit_guards(orig, record: dict, set_name: str = "main"):
    def guard(cell, estimate, pipelines=S.PIPELINES, neural=None):
        if not is_exploratory_cell(cell, set_name):
            raise S.GuardRefused(f"cell {cell.get('cell_index')}: not an exploratory cell of "
                                 f"amendment 11 (b) (set {set_name}); only the registered guards "
                                 "apply to it")
        if neural is not None:
            raise S.GuardRefused("the exploratory run is MLE only (neural S0 deferred)")
        sim = orig(cell, False, pipelines, None)
        if not estimate:
            return sim
        ci = cell["cell_index"]
        if ci not in record:
            if S.exit_source_cell(cell) is None:
                orig(cell, True, pipelines, None)
                record[ci] = "registered estimation guards pass"
            else:
                try:
                    record[ci] = exploratory_estimation_checks(cell, pipelines)
                except S.GuardRefused:
                    raise
                except Exception as exc:
                    raise ExploratoryRefused(f"cell {ci}: {type(exc).__name__}: {exc}") from exc
                S.neural_preflight(cell, None, True)
        return f"exploratory (amendment 11 (b)); {sim}; {record[ci]}"
    guard.exploratory = True
    return guard


@contextlib.contextmanager
def relaxed_estimation_guards(record: dict, set_name: str = "main"):
    orig = S.exit_guards
    if getattr(orig, "exploratory", False):
        raise RuntimeError("exploratory guards are already installed")
    S.exit_guards = exploratory_exit_guards(orig, record, set_name)
    try:
        yield
    finally:
        S.exit_guards = orig


@contextlib.contextmanager
def s0_not_enforced(record: dict):
    orig = C._require_s0_pass
    if getattr(orig, "exploratory", False):
        raise RuntimeError("the S0 relaxation is already installed")

    def relaxed(process, pipeline):
        C._require_frozen()
        got = {}

        def require(p, q):
            got["rec"] = orig(p, q)
        note = checked_s0(process, pipeline, require)
        if "rec" in got:
            return got["rec"]
        record[f"{process}/{pipeline}"] = note
        return None
    relaxed.exploratory = True
    C._require_s0_pass = relaxed
    try:
        yield
    finally:
        C._require_s0_pass = orig


def task_plan(per_chunk: int = REPS_PER_CHUNK, set_name: str = "main"):
    return RC.task_plan(set_cells(set_name), S.R_REGISTERED, per_chunk)


def _retag_chunk(stage: str, out: str, caveat_text: str, extra: dict,
                 set_name: str = "main") -> list[str]:
    """Move run_chunk's files from the staging dir into `out` with the
    exploratory prefix, add the label columns to every table and the
    exploratory fields to the meta."""
    written = []
    for meta_path in glob.glob(os.path.join(stage, "k1chunk_*.meta.json")):
        stem = meta_path[:-len(".meta.json")]
        with open(meta_path) as f:
            meta = json.load(f)
        new_stem = os.path.join(out, cell_set(set_name)["chunk_prefix"]
                                + os.path.basename(stem)[len("k1chunk_"):])
        files = {}
        for k, fn in meta["files"].items():
            src = os.path.join(stage, fn)
            df = RC.read_table(src[: -len(".parquet")] if fn.endswith(".parquet")
                               else src[: -len(".csv.gz")])
            fmt = "parquet" if fn.endswith(".parquet") else "csv.gz"
            files[k] = os.path.basename(RC.write_table(label_frame(df, caveat_text, set_name),
                                                       f"{new_stem}.{k}", fmt))
        meta.update(files=files, **extra)
        with open(new_stem + ".meta.json", "w") as f:
            json.dump(meta, f, indent=1, default=float)
        written.append(new_stem)
    return written


def run_task(cell, r0, r1, out=None, B=None, frozen=None, set_name: str = "main"):
    """One chunk of an exploratory cell (B = the registered B unless a test
    passes another; the meta records it and the aggregator refuses it)."""
    cs = cell_set(set_name)
    out = out or cs["chunk_dir"]
    check_output(out)
    if not is_exploratory_cell(cell, set_name):
        raise SystemExit(f"cell {cell.get('cell_index')} is not an exploratory cell "
                         f"(set {set_name})")
    frozen = frozen or public_guard_skipped(FREEZE_COMMIT)
    commit, dev = RC.code_commit(False)
    cav = caveats()
    os.makedirs(out, exist_ok=True)
    stage = os.path.join(out, f".staging_{os.getpid()}_{cell['cell_index']}_{r0}")
    os.makedirs(stage, exist_ok=False)
    record: dict = {}
    fmt = "parquet" if RC._has_pyarrow() else "csv.gz"
    try:
        with relaxed_estimation_guards(record, set_name):
            RC.run_chunk(cell, r0, r1, cell["B"] if B is None else B, stage, commit, dev,
                         frozen, fmt, None, NEURAL_NOTE)
        extra = {MARKER: cs["label"], "exploratory_zh": cs["label_zh"],
                 "caveat": cav["text"], "caveats": cav, "amendment": "amendment 11 (b)",
                 "cell_set": set_name,
                 "relaxed_estimation_guards": {str(k): v for k, v in record.items()},
                 "driver": "scripts/k1_exploratory.py"}
        if set_name != "main":
            extra["decision"] = "decision of the authors, 2026-10-24"
        return _retag_chunk(stage, out, cav["text"], extra, set_name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cmd_run(a):
    set_name = getattr(a, "set", "main") or "main"
    cs = cell_set(set_name)
    out = check_output(a.out or cs["chunk_dir"])
    cells = set_cells(set_name)
    frozen = public_guard_skipped(FREEZE_COMMIT)
    if a.task is not None:
        plan = task_plan(a.reps_per_chunk, set_name)
        jobs = [plan[a.task]]
    else:
        want = [int(x) for x in a.cells.split(",")] if a.cells else None
        if want is None:
            raise SystemExit("give --task or --cells")
        bad = [c for c in want if c not in cs["cells"]]
        if bad:
            raise SystemExit(f"cells {bad} are not exploratory cells {cs['cells']} "
                             f"(set {set_name})")
        r0, r1 = map(int, a.reps.split(":"))
        jobs = [(c, r0, r1) for c in cells if c["cell_index"] in want]
    for c, r0, r1 in jobs:
        for stem in run_task(c, r0, r1, out, frozen=frozen, set_name=set_name):
            print(f"wrote {os.path.relpath(stem, ROOT)}.*", flush=True)


def _registered_cell_from_json(cell):
    """A chunk meta's cell after its JSON round trip: tuples (the no-2006
    cells' drop_breaks) come back as lists.  If the cell equals its
    registered cell up to that round trip, the registered dict itself is
    returned (so the exact-equality checks downstream apply); otherwise the
    cell is returned unchanged and those checks refuse it."""
    if not isinstance(cell, dict):
        return cell
    ci = cell.get("cell_index")
    allc = S.registered_cells()
    if isinstance(ci, int) and 0 <= ci < len(allc):
        reg = allc[ci]
        if json.loads(json.dumps(reg, default=list)) == cell:
            return reg
    return cell


def load_chunks(dirs, set_name: str = "main") -> tuple[dict, list[dict]]:
    """Exploratory chunks only (prefix k1exp_ / k1expc_, the set's marker in
    the meta and in every table); the label columns are dropped before the
    tables are used."""
    cs = cell_set(set_name)
    label = cs["label"]
    metas = []
    for d in dirs:
        for m in sorted(glob.glob(os.path.join(d, cs["chunk_prefix"] + "*.meta.json"))):
            with open(m) as f:
                meta = json.load(f)
            if meta.get(MARKER) != label:
                raise SystemExit(f"{m}: chunk meta without the exploratory label: refusing")
            meta["_stem"] = m[:-len(".meta.json")]
            meta["cell"] = _registered_cell_from_json(meta.get("cell"))
            metas.append(meta)
    if not metas:
        raise SystemExit("no exploratory chunks found")
    tabs = {}
    for k in RC.TABLES:
        parts = []
        for m in metas:
            df = RC.read_table(f"{m['_stem']}.{k}")
            if len(df) and (MARKER not in df.columns
                            or not (df[MARKER] == label).all()):
                raise SystemExit(f"{m['_stem']}.{k}: table without the exploratory label")
            parts.append(df.drop(columns=[c for c in (MARKER, "caveat", "run_label")
                                          if c in df.columns]))
        tabs[k] = pd.concat(parts, ignore_index=True)
    return tabs, metas


def check_exploratory_chunks(metas, partial: bool, set_name: str = "main") -> list[str]:
    problems = []
    for m in metas:
        c = m["cell"]
        if not is_exploratory_cell(c, set_name):
            raise SystemExit(f"chunk of cell {c.get('cell_index')}: not an exploratory cell")
        if m.get("seed_stream") != "registered":
            raise SystemExit(f"{m['_stem']}: seed stream {m.get('seed_stream')!r}")
        want = [S.replicate_seed(c, r) for r in range(*m["replicates"])]
        if m.get("seeds") != want:
            raise SystemExit(f"{m['_stem']}: seeds differ from the registered stream")
        if m.get("theory_frozen") != FREEZE_COMMIT:
            raise SystemExit(f"{m['_stem']}: theory freeze {m.get('theory_frozen')}")
    present = {m["cell"]["cell_index"] for m in metas}
    missing = [c for c in cell_set(set_name)["cells"] if c not in present]
    if missing:
        problems.append(f"cells {missing} have no chunk (amendment 11 (b): every cell is reported)")
    if problems and not partial:
        raise SystemExit("incomplete exploratory grid (pass --partial for a labelled partial "
                         "aggregate in results/records/k1_exploratory_partial):\n  "
                         + "\n  ".join(problems))
    return problems


def _round_trip(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly what k1_gates would read from the CSV the registered aggregator
    writes."""
    if df is None or not len(df):
        return pd.DataFrame()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def control_rows(k1a: list[dict]) -> tuple[list[dict], int]:
    """The 'controls' set: keep the k1_gates K1a rows of the control and
    no-2006 variants (cells 38–41, 50, 51, and any 'missing' row of those
    variants); drop the rows of the variants whose cells are not in this set
    (registered K1a, dated-shock band: reported by the 'main' set).  A kept
    gating row is refused (never possible for these variants)."""
    keep_cells = set(CONTROL_CELLS + SENS_CELLS)
    keep, n_drop = [], 0
    for r in k1a:
        ci = r.get("cell_index")
        own = (ci is not None and not pd.isna(ci) and int(ci) in keep_cells) or (
            r.get("status") == "missing"
            and str(r.get("statistic", "")).startswith(tuple(f"{g} cell"
                                                             for g in CONTROL_VARIANT_GATE_USES)))
        if own:
            if r.get("gating"):
                raise SystemExit(f"a control-set K1a row is gating: {r}")
            keep.append(r)
        else:
            n_drop += 1
    return keep, n_drop


def exploratory_statistics(tabs: dict, cov: pd.DataFrame, pred, praw, sn, metas,
                           set_name: str = "main"):
    cell_meta = {int(m["cell"]["cell_index"]): m.get("contract_cell_meta") or {} for m in metas}
    s0_relaxed: dict = {}
    with s0_not_enforced(s0_relaxed):
        k1a, _claims = G.gate_k1a(pred, cell_meta, G.index_predict_rows(praw)
                                  if praw is not None else None)
    if set_name != "main":
        k1a, n_drop = control_rows(k1a)
        return k1a, [], [], s0_relaxed, n_drop
    k1b = G.gate_k1b(cov)
    shk = G.gate_shocked_null(sn)
    dropped = [r for r in k1a if r["status"] == "missing" and not r["gating"]]
    k1a = [r for r in k1a if not (r["status"] == "missing" and not r["gating"])]
    return k1a, k1b, shk, s0_relaxed, len(dropped)


RERUN_REL_TOL = 1e-12


def rerun_null_check(merged: dict) -> dict:
    """Set 'controls': cells 35, 37 were re-run at this commit with the
    registered seeds; compare them with the committed 'main' tables
    (results/tables/k1_exploratory/k1_exploratory_{sim_bias,predictions}).
    Reported, never a refusal."""
    out = {}
    for k in ("sim_bias", "predictions"):
        fn = os.path.join(TABLES_DIR, f"k1_exploratory_{k}.csv" + (".gz" if k == "predictions"
                                                                  else ""))
        new = merged.get(k)
        if new is None or not os.path.exists(fn):
            out[k] = {"status": "not compared (table missing)"}
            continue
        old = pd.read_csv(fn, low_memory=False)
        old = old.drop(columns=[c for c in (MARKER, "caveat", "run_label") if c in old.columns])
        buf = io.StringIO()
        new.to_csv(buf, index=False)
        buf.seek(0)
        new = pd.read_csv(buf, low_memory=False)
        old = old[old["cell_index"].isin(SENS_NULL_CELLS)]
        new = new[new["cell_index"].isin(SENS_NULL_CELLS)]
        cols = [c for c in new.columns if c in old.columns]
        num = [c for c in cols if pd.api.types.is_numeric_dtype(new[c])
               and pd.api.types.is_numeric_dtype(old[c]) and c != "cell_index"]
        keys = [c for c in cols if c not in num]
        o = old[cols].sort_values(keys, kind="mergesort").reset_index(drop=True)
        n = new[cols].sort_values(keys, kind="mergesort").reset_index(drop=True)
        rec = {"rows_main": int(len(o)), "rows_rerun": int(len(n)),
               "columns_compared": len(cols)}
        if len(o) != len(n) or not o[keys].astype(str).equals(n[keys].astype(str)):
            rec["status"] = "differ (rows / keys)"
        else:
            a_, b_ = o[num].to_numpy(float), n[num].to_numpy(float)
            both_nan = np.isnan(a_) & np.isnan(b_)
            d = np.where(both_nan, 0.0, np.abs(a_ - b_))
            mx = float(np.nanmax(d)) if d.size else 0.0
            scale = np.maximum(np.abs(a_), np.abs(b_))
            rel = np.where(both_nan | (d == 0), 0.0, d / np.where(scale > 0, scale, 1.0))
            mr = float(np.nanmax(rel)) if rel.size else 0.0
            rec["max_abs_diff"] = mx if np.isfinite(mx) else "nan mismatch"
            rec["max_rel_diff"] = mr if np.isfinite(mr) else "nan mismatch"
            rec["nan_pattern_equal"] = bool((np.isnan(a_) == np.isnan(b_)).all())
            if not rec["nan_pattern_equal"] or not (np.isfinite(mr) and mr <= RERUN_REL_TOL):
                rec["status"] = "differ (values)"
            else:
                rec["status"] = ("identical" if mx == 0.0 else
                                 f"identical to CSV round-off (max relative diff ≤ {RERUN_REL_TOL:g})")
        out[k] = rec
    return out


def cmd_aggregate(a):
    set_name = getattr(a, "set", "main") or "main"
    cs = cell_set(set_name)
    partial = bool(a.partial)
    out = check_output(a.out or (cs["partial_dir"] if partial else TABLES_DIR))
    if partial and RC.is_under(out, TABLES_DIR):
        raise SystemExit("a --partial aggregate never goes to results/tables/k1_exploratory")
    inputs = a.inputs or [cs["chunk_dir"]]
    for d in inputs:
        check_output(d)
    frozen = public_guard_skipped(FREEZE_COMMIT)
    if RC.tree_dirty():
        raise SystemExit("refusing to aggregate on a dirty tree (amendment 1 (C))")
    tabs, metas = load_chunks(inputs, set_name)
    problems = check_exploratory_chunks(metas, partial, set_name)
    commit, report, reg_problems, mask_sha = RC.check_chunks(
        metas, registered=not partial, frozen=frozen, exploratory=True)
    why = RC.same_code(commit)
    if why:
        raise SystemExit(f"chunks were run at code commit {str(commit)[:10]}, not at the "
                         f"aggregator's code state: {why}")
    est, comp, std = tabs["estimates"], tabs["composition"], tabs["standardisation"]
    resid = comp["identity_residual"].abs().max() if len(comp) else 0.0
    if not resid < 1e-10:
        raise SystemExit(f"Lemma 1 identity violated in the chunks: {resid:.2e}")
    cav = caveats()
    cf = RC._cells_frame(metas)
    raw = {"sim_bias": RC.bias_table(est), "sim_coverage": RC.coverage_table(est),
           "power": RC.power_table(est), "refit_contribution": RC.refit_table(est),
           "shocked_null_replicates": RC.shocked_null_table(est, metas)}
    pred, pred_note, praw = RC.predictions_table(comp, std, est, metas)
    if pred is not None:
        raw["predictions"] = pred
    merged = {k: (cf.merge(df, on="cell_index", how="right") if len(df) else df)
              for k, df in raw.items()}
    rt = {k: _round_trip(df) for k, df in merged.items()}
    k1a, k1b, shk, s0_relaxed, n_dropped = exploratory_statistics(
        tabs, rt["sim_coverage"], rt.get("predictions"), praw, rt["shocked_null_replicates"],
        metas, set_name)
    main = set_name == "main"
    pre = cs["out_prefix"]
    os.makedirs(out, exist_ok=True)
    written = {}
    for k, df in merged.items():
        name = f"{pre}{k}.csv" + (".gz" if k == "predictions" else "")
        label_frame(df, cav["text"], set_name).to_csv(check_output(os.path.join(out, name)),
                                                      index=False)
        written[name] = len(df)
    gate_sets = (("k1a", k1a), ("k1b", k1b), ("shocked_null", shk)) if main else (("k1a", k1a),)
    for k, rows in gate_sets:
        name = f"{pre}{k}.csv"
        df = label_gate_rows(rows, cav["text"], set_name)
        df.to_csv(check_output(os.path.join(out, name)), index=False)
        written[name] = len(df)
    if main:
        allr = label_gate_rows(k1a + k1b, cav["text"])
        summ = allr[allr["would_gate_if_registered"]] if len(allr) else allr
        summ.to_csv(check_output(os.path.join(out, "k1_exploratory_summary.csv")), index=False)
        written["k1_exploratory_summary.csv"] = len(summ)
    if praw is not None:
        import gzip
        with gzip.open(check_output(os.path.join(out, cs["predict_rows"])), "wt",
                       encoding="utf-8") as f:
            json.dump({MARKER: cs["label"], "caveat": cav["text"], "rows": praw}, f,
                      allow_nan=True)
        written[cs["predict_rows"]] = len(praw)
    set_cells_ = list(cs["cells"])
    meta = {MARKER: cs["label"], "exploratory_zh": cs["label_zh"], "cell_set": set_name,
            "run_label": cs["run_label"], "partial": partial, "never_gated": True,
            "caveat": cav["text"], "caveats": cav, "amendment": "amendment 11 (b)",
            "theory_frozen": frozen, "code_commit": commit,
            "aggregator_git": _git("rev-parse", "HEAD"),
            "cells": set_cells_,
            **({"exit_cells": list(EXIT_CELLS), "null_cells": list(NULL_CELLS)} if main else
               {"control_cells": list(CONTROL_CELLS), "control_null_cells": list(CONTROL_NULL_CELLS),
                "sens_no2006_cells": list(SENS_CELLS),
                "sens_null_cells_rerun": list(SENS_NULL_CELLS),
                "decision": "decision of the authors, 2026-10-24",
                "rerun_null_check_vs_main": rerun_null_check(merged)}),
            "per_cell": {int(k): v for k, v in report.items()},
            "problems": problems + reg_problems, "mask_content_sha256": mask_sha,
            "seed_rule": RC.SEED_RULE + " (registered stream of the second freeze)",
            "R_registered": S.R_REGISTERED, "B_registered": S.B_REGISTERED,
            "bootstrap_note": RC.BOOTSTRAP_DISCLOSURE,
            "relaxed_estimation_guards": {os.path.basename(m["_stem"]):
                                          m.get("relaxed_estimation_guards") for m in metas},
            "s0_pass_not_enforced_in_k1a_statistics": s0_relaxed,
            "statistics": ("k1_gates.gate_k1a / gate_k1b / gate_shocked_null on these tables; "
                           "rows relabelled (registered_rule, exploratory_outcome, gating False)"
                           if main else
                           "k1_gates.gate_k1a 'control' and 'sens_no2006' variant rows on these "
                           "tables (rows of variants without cells in this set dropped, counted); "
                           "relabelled (registered_rule, exploratory_outcome, gating False)"),
            "law_claims": "not computed (amendment 11 (b): no law claim from exploratory cells)",
            "dropped_missing_report_only_rows": n_dropped,
            "not_run": ("persistent controls 38–45, no-2006 sensitivity cells 50–51, neural "
                        "estimator (outside the amendment 11 (b) driver's cell set)" if main else
                        "neural estimator; K1b and dated-shock statistics (no such cells in "
                        "this set; reported by the 'main' set)"),
            "predictions": pred_note, "neural": NEURAL_NOTE,
            "failed_replicates": {os.path.basename(m["_stem"]): m.get("failed_replicates", [])
                                  for m in metas if m.get("failed_replicates")},
            "slurm_jobs": sorted({str(m.get("slurm_job")) for m in metas}),
            "versions": sorted({json.dumps(m.get("versions"), sort_keys=True) for m in metas}),
            "n_chunks": len(metas),
            "core_seconds": float(sum(sum(m["seconds_per_replicate"]) for m in metas)),
            "files": written, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(check_output(os.path.join(out, f"{pre}meta.json")), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    for n, k in written.items():
        print(f"wrote {n} ({k} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ce = sub.add_parser("cells")
    n = sub.add_parser("ntasks")
    n.add_argument("--reps-per-chunk", type=int, default=REPS_PER_CHUNK)
    r = sub.add_parser("run")
    r.add_argument("--task", type=int, default=None)
    r.add_argument("--reps-per-chunk", type=int, default=REPS_PER_CHUNK)
    r.add_argument("--cells", default=None, help="exploratory cell indices (with --reps)")
    r.add_argument("--reps", default="0:1")
    r.add_argument("--out", default=None, help="default: the set's chunk dir")
    g = sub.add_parser("aggregate")
    g.add_argument("--in", dest="inputs", nargs="+", default=None,
                   help="default: the set's chunk dir")
    g.add_argument("--out", default=None)
    g.add_argument("--partial", action="store_true")
    for sp in (ce, n, r, g):
        sp.add_argument("--set", choices=sorted(SETS), default="main",
                        help='cell set: main (amendment 11 (b)) or controls (decision of the '
                             'authors, 2026-10-24)')
    a = ap.parse_args(argv)
    if a.cmd == "cells":
        for c in set_cells(a.set):
            print(c["cell_index"], c["process"], c["mechanism"], c["gate_use"],
                  c.get("structure"), f"B={c['B']}")
        return
    if a.cmd == "ntasks":
        if not public_guard_skipped(True):
            raise SystemExit("the amendment 11 approval line is not in the registration record at HEAD")
        print(len(task_plan(a.reps_per_chunk, a.set)))
        return
    if a.cmd == "run":
        cmd_run(a)
    else:
        cmd_aggregate(a)


if __name__ == "__main__":
    main()
