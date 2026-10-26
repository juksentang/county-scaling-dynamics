"""run        cell × replicate chunks → one file set per chunk
    aggregate  chunks → results/tables/k1_sim_bias.csv, k1_sim_coverage.csv,
               k1_power.csv, k1_refit_contribution.csv, k1_predictions.csv
               and k1_sim_meta.json

Mask provenance: every chunk records the presence mask's file sha256 and its
format-independent content sha256 (src/k1_sim.mask_content_sha256); the aggregator refuses chunks whose
content hashes differ.  Non-registered (--non-registered) and --dev outputs
can never be written inside results/tables (path-prefix check).
"""
from __future__ import annotations

def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src import k1_sim as S
from src import k1_contract as C
from src.k1_contract import PERIOD_CONTRASTS, QUANTITIES

TABLES = ("estimates", "composition", "truth", "standardisation")
SEED_RULE = ("SEED_BASE 20261021 + cell_index*1e4 + replicate; the (a) precheck cells "
             "(amendment 7): PRECHECK_SEED_BASE + cell_index*1e4 + replicate (implcheck stream: "
             "IMPLCHECK_SEED_BASE)")
SHIP_MANIFEST = os.path.join(ROOT, "K1_SHIP.json")


RESULTS_TABLES = os.path.join(ROOT, "results", "tables")


def is_under(path: str, base: str = RESULTS_TABLES) -> bool:
    """True if path is base or anything inside it (path-prefix check)."""
    p, b = os.path.realpath(os.path.abspath(path)), os.path.realpath(os.path.abspath(base))
    try:
        return os.path.commonpath([p, b]) == b
    except ValueError:
        return False


EXPLORATORY_TABLES = os.path.join(ROOT, "results", "tables", "k1_exploratory")
EXPLORATORY_AUDIT_DIR = os.path.join(ROOT, 'results', 'records')
EXPLORATORY_AUDIT_PREFIX = "k1_exploratory"
EXPLORATORY_MARKER = "exploratory"


def is_exploratory_path(path: str) -> bool:
    """True for results/tables/k1_exploratory (and below) and for every
    results/records/k1_exploratory* entry (and below)."""
    if is_under(path, EXPLORATORY_TABLES):
        return True
    p = os.path.realpath(os.path.abspath(path))
    base = os.path.realpath(EXPLORATORY_AUDIT_DIR)
    try:
        if os.path.commonpath([p, base]) != base or p == base:
            return False
    except ValueError:
        return False
    return os.path.relpath(p, base).split(os.sep)[0].startswith(EXPLORATORY_AUDIT_PREFIX)


def refuse_exploratory_paths(paths, what: str) -> None:
    bad = [p for p in paths if is_exploratory_path(p)]
    if bad:
        raise SystemExit(f"{what} {bad[:3]} is an exploratory (amendment 11 (b)) path: the registered "
                         "aggregator / gates never read or write exploratory outputs")


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()


def in_git() -> bool:
    try:
        _git("rev-parse", "--git-dir")
        return True
    except Exception:
        return False


def tree_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--untracked-files=normal"))


NON_CODE_PREFIXES = ("results/tables/", "results/records/")


def same_code(commit: str, head: str | None = None) -> str | None:
    if not commit or len(str(commit)) != 40:
        return f"not a full commit hash: {commit!r}"
    head = head or _git("rev-parse", "HEAD")
    if commit == head:
        return None
    if subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor", commit, head],
                      capture_output=True).returncode:
        return f"{commit[:10]} is not an ancestor of HEAD {head[:10]}"
    changed = [f for f in _git("diff", "--name-only", commit, head).splitlines() if f]
    code = [f for f in changed if not f.startswith(NON_CODE_PREFIXES)]
    if code:
        return f"code changed between {commit[:10]} and HEAD {head[:10]}: {code[:5]}"
    return None


SHIP_EXEMPT = ("K1_SHIP.json",)


def verify_ship(root=ROOT, manifest=None) -> str:
    manifest = manifest or os.path.join(root, "K1_SHIP.json")
    with open(manifest) as f:
        man = json.load(f)
    files = man.get("files")
    if not files:
        raise SystemExit("K1_SHIP.json has no per-file sha256 list: cannot verify the shipped tree")
    bad = [p for p, h in files.items()
           if not os.path.exists(os.path.join(root, p)) or _sha(os.path.join(root, p)) != h]
    if bad:
        raise SystemExit(f"shipped files modified or missing (first: {bad[:5]}): refusing")
    extra = []
    for d, _, fs in os.walk(root):
        rel_d = os.path.relpath(d, root)
        if rel_d.split(os.sep)[0] in ("chunks", "logs", "data_clean") or "__pycache__" in d:
            continue
        for fn in fs:
            rel = os.path.normpath(os.path.join(rel_d, fn))
            if fn.endswith(".py") and rel not in files and rel not in SHIP_EXEMPT:
                extra.append(rel)
    if extra:
        raise SystemExit(f"unlisted .py files in the shipped tree: {extra[:5]}: refusing")
    return man["commit"]


def code_commit(dev: bool) -> tuple[str, bool]:
    if in_git():
        head = _git("rev-parse", "HEAD")
        if tree_dirty():
            if not dev:
                raise SystemExit("refusing to run on a dirty tree (amendment 1 (C)); "
                                 "commit first, or pass --dev for throw-away output")
            return head + "-dirty", True
        return head, bool(dev)
    if os.path.exists(SHIP_MANIFEST):
        try:
            return verify_ship(), bool(dev)
        except SystemExit:
            if not dev:
                raise
            return "shipped-unverified", True
    if not dev:
        raise SystemExit("no git and no K1_SHIP.json: cannot identify the code commit")
    return "unknown", True


S0_REFUSAL = ("S0 cells run only through scripts/k1_s0.py with src.k1_s0_backend (contract 7, "
              "amendment 3 (j)): every S0 attempt is started and logged before it runs, so no S0 "
              "number is produced outside the attempt log")


def refuse_s0_cells(cells: list[dict]) -> None:
    bad = [c["cell_index"] for c in cells if c.get("gate_use") == "S0"]
    if bad:
        raise SystemExit(f"cells {bad}: {S0_REFUSAL}")


def precheck_freeze(cells: list[dict], registered: bool, frozen: str | None) -> str | None:
    if frozen is not None or not registered:
        return frozen
    if any(S.exit_source_cell(c) is not None for c in cells):
        return frozen
    assert_theory_frozen = public_guard_skipped
    return assert_theory_frozen()


def guard_cells(cells: list[dict]) -> str | None:
    """Refuse exit mechanisms (b)-(e) and the dated-shock nulls before the
    theory freeze."""
    if all(S.exit_source_cell(c) is None for c in cells):
        return None
    assert_theory_frozen = public_guard_skipped
    return assert_theory_frozen()


def _has_pyarrow() -> bool:
    try:
        import pyarrow
        return True
    except Exception:
        return False


def write_table(df: pd.DataFrame, stem: str, fmt: str) -> str:
    if fmt == "parquet":
        path = stem + ".parquet"
        df.to_parquet(path, index=False)
    else:
        path = stem + ".csv.gz"
        df.to_csv(path, index=False, float_format="%.17g")
    return path


def read_table(stem: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv.gz"):
        if os.path.exists(stem + ext):
            return pd.read_parquet(stem + ext) if ext == ".parquet" else pd.read_csv(stem + ext)
    raise FileNotFoundError(stem)


def parse_cells(spec: str) -> list[dict]:
    allc = S.registered_cells()
    if spec in ("all", "registered"):
        return [c for c in allc if c["gate_use"] != "S0"]
    if spec == "a":
        return [c for c in allc if c["mechanism"] == "a" and c["gate_use"] != "S0"]
    out = []
    for tok in spec.split(","):
        if "-" in tok:
            a, b = map(int, tok.split("-"))
            out += allc[a:b + 1]
        else:
            out.append(allc[int(tok)])
    return out


def task_plan(cells, R, per_chunk):
    plan = []
    for c in cells:
        for a in range(0, R, per_chunk):
            plan.append((c, a, min(a + per_chunk, R)))
    return plan


def load_neural():
    try:
        from src.k1_neural_k3 import sim_estimate as neural
        return neural, "src.k1_neural_k3.sim_estimate"
    except Exception as exc:
        return None, f"not importable ({type(exc).__name__})"


def _bare_cell(cell: dict) -> dict:
    return {k: v for k, v in cell.items() if k != "valid_for_law"}


def _versions() -> dict:
    import scipy
    import statsmodels
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__}


def run_chunk(cell, r0, r1, B, out, commit, dev, frozen, fmt, neural, neural_note,
              composition_only=False, seed_stream="registered", pipelines=S.PIPELINES):
    import traceback
    TheoryNotFrozen = public_guard_skipped(S.GuardRefused)
    mask = S.load_mask()
    mprov = S.mask_provenance(mask)
    ship_mask = os.path.join(ROOT, "K1_MASK.json")
    if not in_git() and os.path.exists(ship_mask):
        with open(ship_mask) as f:
            want = json.load(f)["content_sha256"]
        if want != mprov["content_sha256"] and not dev:
            raise SystemExit(f"mask content {mprov['content_sha256'][:12]} != shipped "
                             f"K1_MASK.json {want[:12]}: refusing")
    S.check_cell(cell)
    refuse_s0_cells([cell])
    estimate = not composition_only
    guard = S.exit_guards(cell, estimate, neural=neural if estimate else None)
    B_used = 0 if (cell["B"] == 0 or composition_only) else B
    parts = {k: [] for k in TABLES}
    timing, deltas, offdiag, tp, proc_desc = [], None, None, None, None
    failures = []
    for rep in range(r0, r1):
        t0 = time.time()
        try:
            res = S.run_replicate(cell, rep, mask, B=B_used, neural=neural if estimate else None,
                                  estimate=estimate, seed_stream=seed_stream,
                                  pipelines=pipelines)
        except (TheoryNotFrozen, S.GuardRefused):
            raise
        except Exception as exc:
            if S.is_guard_refusal(exc):
                raise
            res = S.failed_replicate(cell, rep, B_used, exc, neural=neural if estimate else None,
                                     pipelines=pipelines, seed_stream=seed_stream)
            if composition_only:
                res["estimates"] = res["estimates"].iloc[0:0]
            failures.append({"replicate": rep, "status": res["status"],
                             "traceback": traceback.format_exc()[-2000:]})
        timing.append(time.time() - t0)
        for k in TABLES:
            parts[k].append(res[k])
        if res["deltas"] is not None:
            deltas = {p: {str(y): v for y, v in d[0].items()} for p, d in res["deltas"].items()}
            offdiag = {p: {str(y): v for y, v in d[1].items()} for p, d in res["deltas"].items()}
        if res["true_params"] is not None:
            tp, proc_desc = res["true_params"], res["true_process"]
    try:
        design = S.design_constants(cell["process"])
    except Exception as exc:
        design = {"error": f"{type(exc).__name__}: {exc}"}
    stem = os.path.join(out, f"k1chunk_c{cell['cell_index']:02d}_r{r0:04d}-{r1:04d}")
    files = {k: os.path.basename(write_table(pd.concat(parts[k], ignore_index=True),
                                             f"{stem}.{k}", fmt)) for k in TABLES}
    meta = {"cell": cell, "replicates": [r0, r1], "B": B_used, "B_requested": B,
            "mode": "composition_only" if composition_only else "full", "guard": guard,
            "seeds": ({pl: [S.replicate_seed(cell, r, pl) for r in range(r0, r1)]
                       for pl in S.PIPELINES} if cell["gate_use"] == "S0"
                      else [S.replicate_seed(cell, r, stream=seed_stream) for r in range(r0, r1)]),
            "seed_rule": ("S0: src.k1_contract.s0_seeds(process, pipeline)[replicate] "
                          "(amendment 3 (j)), one panel per pipeline" if cell["gate_use"] == "S0"
                          else SEED_RULE) + "; streams [seed, 0..2] + bootstrap [seed, 4]",
            "seed_stream": seed_stream, "pipelines": list(pipelines),
            "bootstrap_scheme": S.BOOTSTRAP_SCHEME,
            "code_commit": commit, "dev": dev, "theory_frozen": frozen,
            "true_params": tp, "true_process": proc_desc,
            "design_constants": design,
            "offsets_delta": deltas, "offsets_diagnostics": offdiag,
            "failed_replicates": failures, "obs_rules": list(S.OBS_RULES),
            "mask": mprov,
            "contract_cell_meta": {pl: S.contract_cell_meta(cell, pl) for pl in S.PIPELINES},
            "theory_compat": {p: _compat(cell, p) for p in S.PERIODS},
            "ci_nan_rules": S.CI_NAN_RULES, "neural": neural_note if S.is_neural_cell(cell) else "n/a",
            "files": files, "seconds_per_replicate": timing,
            "versions": _versions(), "host": platform.node(), "cluster": os.environ.get("CC_CLUSTER", platform.node()),
            "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID")),
            "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(stem + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print(f"cell {cell['cell_index']} reps {r0}-{r1}: {np.mean(timing):.1f} s/replicate "
          f"(B={B_used})", flush=True)


def _compat(cell, period):
    try:
        return S.theory_compat(cell["process"], period)
    except Exception as exc:
        return {"compatible": False, "reason": f"{type(exc).__name__}: {exc}"}


def cmd_run(a):
    refuse_exploratory_paths([a.out], "run output")
    cells = parse_cells(a.cells)
    refuse_s0_cells(cells)
    frozen = guard_cells(cells)
    neural, note = load_neural()
    if a.composition_only:
        neural, note = None, "composition-only mode: no estimator"
    elif getattr(a, "no_neural", False):
        neural, note = None, "not run (--no-neural): no V0_neural rows; K3 reads not_run"
    elif neural is None and any(S.is_neural_cell(c) for c in cells):
        raise SystemExit(f"neural estimator {note}: refusing to run neural cells without it "
                         "(pass --no-neural to run the MLE only, recorded in the chunk meta)")
    for c in cells:
        S.exit_guards(c, not a.composition_only, neural=neural)
    commit, dev = code_commit(a.dev)
    os.makedirs(a.out, exist_ok=True)
    fmt = a.format or ("parquet" if _has_pyarrow() else "csv.gz")
    if a.task is not None:
        plan = task_plan(cells, a.R, a.reps_per_chunk)
        jobs = [plan[a.task]]
    else:
        r0, r1 = map(int, a.reps.split(":"))
        jobs = [(c, r0, r1) for c in cells]
    pipelines = tuple(a.pipelines.split(",")) if a.pipelines else S.PIPELINES
    if any(p not in S.PIPELINES for p in pipelines):
        raise SystemExit(f"--pipelines must be a subset of {S.PIPELINES}")
    if a.seed_stream != "registered":
        bad = [c["cell_index"] for c in cells if not S.is_precheck_a_cell(c)]
        if bad:
            raise SystemExit(f"cells {bad}: the {a.seed_stream!r} seed stream exists only for "
                             "the (a) precheck cells")
    for c, r0, r1 in jobs:
        run_chunk(c, r0, r1, a.B, a.out, commit, dev, frozen, fmt, neural, note,
                  composition_only=a.composition_only, seed_stream=a.seed_stream,
                  pipelines=pipelines)


def load_chunks(dirs) -> tuple[dict, list[dict]]:
    metas = []
    for d in dirs:
        for m in sorted(glob.glob(os.path.join(d, "k1chunk_*.meta.json"))):
            with open(m) as f:
                meta = json.load(f)
            meta["_stem"] = m[:-len(".meta.json")]
            metas.append(meta)
    if not metas:
        raise SystemExit("no chunks found")
    tabs = {k: pd.concat([read_table(f"{m['_stem']}.{k}") for m in metas], ignore_index=True)
            for k in TABLES}
    return tabs, metas


def check_chunks(metas, allow_dev=False, registered=True, frozen=None, exploratory=False):
    marked = [bool(m.get(EXPLORATORY_MARKER)) for m in metas]
    if not exploratory and any(marked):
        raise SystemExit("exploratory (amendment 11 (b)) chunks are never aggregated by the registered "
                         "aggregator: refusing")
    if exploratory and not all(marked):
        raise SystemExit("a chunk without the exploratory marker in an exploratory aggregate: "
                         "refusing")
    problems = []
    commits = {m["code_commit"] for m in metas}
    if len(commits) != 1:
        raise SystemExit(f"chunks come from more than one code commit: {sorted(commits)}")
    if any(m["dev"] for m in metas) and not allow_dev:
        raise SystemExit("dev chunks (dirty tree) do not count: refusing to aggregate")
    seen, by_cell = {}, {}
    for m in metas:
        ci = m["cell"]["cell_index"]
        for r in range(*m["replicates"]):
            if (ci, r) in seen:
                raise SystemExit(f"replicate {r} of cell {ci} appears in two chunks")
            seen[(ci, r)] = m["_stem"]
        if m["cell"] != S.registered_cells()[ci]:
            raise SystemExit(f"cell {ci} of a chunk differs from registered_cells()")
        S.check_cell(m["cell"])
        by_cell.setdefault(ci, []).append(m)
    masks = {(m.get("mask") or {}).get("content_sha256") for m in metas}
    if None in masks:
        raise SystemExit("a chunk does not record the mask content hash: refusing")
    if len(masks) != 1:
        raise SystemExit(f"chunks were run on {len(masks)} different presence masks: {sorted(masks)}")
    for m in metas:
        if m["cell"]["mechanism"] != "a" and m.get("theory_frozen") != frozen:
            raise SystemExit(f"chunk {os.path.basename(m['_stem'])} ran under theory freeze "
                             f"{m.get('theory_frozen')} but the aggregator verifies {frozen}")
    for ci, ms in by_cell.items():
        dc = {json.dumps(m.get("design_constants"), sort_keys=True, default=str) for m in ms}
        if len(dc) != 1:
            raise SystemExit(f"cell {ci}: design constants differ between chunks")
        nf = {str(m.get("neural")) for m in ms}
        if len(nf) != 1:
            raise SystemExit(f"cell {ci}: chunks differ in the neural field {sorted(nf)}")
        od = {json.dumps(m.get("offsets_delta"), sort_keys=True, default=str) for m in ms
              if m.get("offsets_delta") is not None}
        if len(od) > 1:
            raise SystemExit(f"cell {ci}: offsets δ differ between chunks")
        for m in ms:
            for p, d in (m.get("offsets_diagnostics") or {}).items():
                for y, v in d.items():
                    if not v.get("converged"):
                        raise SystemExit(f"cell {ci} {p} break {y}: offset calibration not "
                                         f"converged (target {v.get('target')}, achieved "
                                         f"{v.get('achieved')}, bracketed {v.get('bracketed')})")
    report = {}
    for ci, ms in by_cell.items():
        modes = {m.get("mode", "full") for m in ms}
        if len(modes) != 1:
            raise SystemExit(f"cell {ci}: chunks mix modes {sorted(modes)} (composition-only "
                             "and full chunks of one cell are aggregated separately)")
        mode = modes.pop()
        reps = sorted(r for (c, r) in seen if c == ci)
        Bs = sorted({m["B"] for m in ms})
        B_exp = 0 if mode == "composition_only" else S.registered_cells()[ci]["B"]
        complete = reps == list(range(S.R_REGISTERED))
        report[ci] = {"R": len(reps), "complete_0_to_R-1": complete, "B": Bs,
                      "B_expected": B_exp, "mode": mode,
                      "n_failed_replicates": sum(len(m.get("failed_replicates", [])) for m in ms)}
        if not complete:
            problems.append(f"cell {ci}: replicates {reps[:3]}..{reps[-3:]} (n={len(reps)}) "
                            f"are not 0..{S.R_REGISTERED - 1}")
        if Bs != [B_exp]:
            problems.append(f"cell {ci}: B = {Bs}, registered {B_exp}")
    vers = {json.dumps(m.get("versions"), sort_keys=True) for m in metas}
    if len(vers) != 1:
        problems.append(f"chunks come from {len(vers)} package-version sets")
    if problems and registered:
        raise SystemExit("not a registered run (pass --non-registered to aggregate anyway, "
                         "labelled):\n  " + "\n  ".join(problems))
    return commits.pop(), report, problems, masks.pop()


def _cells_frame(metas) -> pd.DataFrame:
    rows = {}
    for m in metas:
        c = m["cell"]
        tc = m.get("theory_compat") or {}
        rows[c["cell_index"]] = {"cell_index": c["cell_index"], "process": c["process"],
                                 "mechanism": c["mechanism"], "n_override": c["n_override"],
                                 "structure": c.get("structure"),
                                 "re_placement": c["re_placement"], "gated": c["gated"],
                                 "gate_use": c["gate_use"],
                                 "theory_compatible": bool(tc) and all(
                                     bool(v.get("compatible")) for v in tc.values())}
    return pd.DataFrame(rows.values())


KEYS = ["cell_index", "pipeline", "saai_spec", "variant", "obs_rule", "period", "quantity"]


def _excludes_null(lo, hi, q, period):
    null = 1.0 if (period == "contrast" and PERIOD_CONTRASTS[q] == "ratio") else 0.0
    return (lo > null) | (hi < null)


def bias_table(est: pd.DataFrame) -> pd.DataFrame:
    if est.empty:
        return pd.DataFrame()
    e = est.copy()
    e["bias"] = e["estimate"] - e["truth"]
    e["rep_failed"] = e["replicate_status"].astype(str) != "ok"
    g = e.groupby(KEYS)
    out = g.agg(R=("replicate", "nunique"),
                n_replicate_failed=("rep_failed", "sum"),
                n_est_nan=("estimate", lambda v: int(v.isna().sum())),
                n_truth_nan=("truth", lambda v: int(v.isna().sum())),
                n_bias_nan=("bias", lambda v: int(v.isna().sum())),
                median_estimate=("estimate", "median"), median_truth=("truth", "median"),
                median_bias=("bias", "median"), mean_bias=("bias", "mean"),
                sd_bias=("bias", "std")).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_median_bias"] = out["median_bias"] / out["median_truth"].abs()
    out["note"] = "medians over finite values; n_*_nan count the rest (gates fail on them)"
    return out


def ci_status(e: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Per-replicate CI usability under one CI_NAN_RULES rule: tol0 / tol5
    'failed' when not finite or the NaN share of its B draws exceeds the
    tolerance; 'literal' uses the bounds with NaN draws at the widening
    extreme (may be ±inf, never failed unless B = 0).  Failed CIs stay in
    every denominator.  Cells with B = 0 have no CI (failed)."""
    e = e.copy()
    tol = S.CI_NAN_RULES[rule]
    if tol is None:
        e["ci_lo"], e["ci_hi"] = e["ci_lo_lit"], e["ci_hi_lit"]
        e["ci_failed"] = e["ci_lo"].isna() | e["ci_hi"].isna() | (e["B"] <= 0)
    else:
        finite = np.isfinite(e["ci_lo"]) & np.isfinite(e["ci_hi"])
        e["ci_failed"] = ~finite | (e["n_nan_boot"] > tol * e["B"]) | (e["B"] <= 0)
    e["ci_nan_rule"] = rule
    return e


CI_RULE_STATUS = {
    "tol0": "primary (gating): CI usable only with no NaN draw",
    "tol5": "reported only: drops up to 5 % of the draws; not the registered rule",
    "literal": "reported only, ANTI-CONSERVATIVE: NaN draws at ±inf make the interval cover "
               "any truth once n_nan >= k = ceil(0.025 B); never in a gating path",
}


BOOTSTRAP_DISCLOSURE = S.BOOTSTRAP_SCHEME + ("; percentile CI (symmetric order statistics); "
                                             "the 'during' window has 8–9 provinces (few "
                                             "clusters: known limitation)")


def coverage_table(est: pd.DataFrame) -> pd.DataFrame:
    if est.empty:
        return pd.DataFrame()
    outs = []
    for rule in S.CI_NAN_RULES:
        e = ci_status(est[est["variant"].isin(["V0", "V1"]) & (est["B"] > 0)], rule)
        e["truth_nan"] = ~np.isfinite(e["truth"])
        e["covered"] = (~e["ci_failed"] & ~e["truth_nan"]
                        & (e["ci_lo"] <= e["truth"]) & (e["truth"] <= e["ci_hi"]))
        g = e.groupby(KEYS + ["ci_nan_rule"])
        out = g.agg(R=("replicate", "nunique"), n_covered=("covered", "sum"),
                    n_ci_failed=("ci_failed", "sum"), n_truth_nan=("truth_nan", "sum"),
                    n_nan_boot_total=("n_nan_boot", "sum"), B=("B", "max")).reset_index()
        out["coverage"] = out["n_covered"] / out["R"]
        den = out["R"] - out["n_ci_failed"]
        out["coverage_excl_failed"] = np.where(den > 0, out["n_covered"] / den.clip(lower=1),
                                               np.nan)
        out["ci_rule_status"] = CI_RULE_STATUS[rule]
        out["ci_method"] = S.QUANTILE_METHOD
        out["bootstrap_scheme"] = BOOTSTRAP_DISCLOSURE
        outs.append(out)
    return pd.concat(outs, ignore_index=True)


def power_table(est: pd.DataFrame) -> pd.DataFrame:
    if est.empty:
        return pd.DataFrame()
    rows = []
    for rule in S.CI_NAN_RULES:
        e = ci_status(est[est["variant"].isin(["V0", "V1"]) & (est["B"] > 0)], rule)
        for key, g in e.groupby(KEYS):
            q, period = key[6], key[5]
            ok = ~g["ci_failed"]
            excl = ok & _excludes_null(g["ci_lo"], g["ci_hi"], q, period)
            row = dict(zip(KEYS, key))
            row.update(ci_nan_rule=rule, ci_rule_status=CI_RULE_STATUS[rule],
                       R=g["replicate"].nunique(),
                       n_ci_failed=int((~ok).sum()),
                       median_truth=float(g["truth"].median()),
                       n_truth_nan=int(g["truth"].isna().sum()),
                       power_excl_null_failed_as_neg=float(excl.mean()),
                       power_excl_null_failed_as_pos=float((excl | ~ok).mean()))
            if q == "two_delta_V" and period != "contrast":
                det = ok & (g["ci_lo"] > 0.5)
                row.update(detect_lo_gt_0p5_failed_as_neg=float(det.mean()),
                           detect_lo_gt_0p5_failed_as_pos=float((det | ~ok).mean()),
                           truth_q10=float(g["truth"].quantile(0.1)),
                           truth_q90=float(g["truth"].quantile(0.9)))
            rows.append(row)
    return pd.DataFrame(rows)


TRUTH_BINS = (-np.inf, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, np.inf)


def power_vs_truth_table(est: pd.DataFrame) -> pd.DataFrame:
    """K2 detection (CI lower bound of 2ΔV > 0.5) against each replicate's own
    truth, binned; NaN truths are their own bin, nothing is dropped."""
    if est.empty:
        return pd.DataFrame()
    rows = []
    e0 = est[est["variant"].isin(["V0", "V1"]) & (est["quantity"] == "two_delta_V")
             & est["period"].isin(["pre", "during"]) & (est["B"] > 0)]
    for rule in S.CI_NAN_RULES:
        e = ci_status(e0, rule)
        e["det"] = ~e["ci_failed"] & (e["ci_lo"] > 0.5)
        e["truth_bin"] = pd.cut(e["truth"], TRUTH_BINS).astype(str)
        e.loc[e["truth"].isna(), "truth_bin"] = "truth_nan"
        for key, g in e.groupby(KEYS + ["truth_bin"]):
            row = dict(zip(KEYS + ["truth_bin"], key))
            row.update(ci_nan_rule=rule, n=len(g), n_ci_failed=int(g["ci_failed"].sum()),
                       detect_failed_as_neg=float(g["det"].mean()),
                       detect_failed_as_pos=float((g["det"] | g["ci_failed"]).mean()),
                       median_truth=float(g["truth"].median()))
            rows.append(row)
    return pd.DataFrame(rows)


def refit_table(est: pd.DataFrame) -> pd.DataFrame:
    """Contribution of the yearly SAAI refit: bias(saai) − bias(direct), paired
    by replicate (same latent panel).  It measures the refit term under the
    simulated covariates (iid per county, independent of x and province);
    the (a) precheck showed it is NOT ≈ 0 for 2ΔV in the bistable cells
    (src/k1_sim.py docstring, REFIT_INFORMATIVE)."""
    if est.empty:
        return pd.DataFrame()
    e = est[est["variant"].isin(["V0", "V1"])].copy()
    e["bias"] = e["estimate"] - e["truth"]
    k = ["cell_index", "replicate", "variant", "obs_rule", "period", "quantity"]
    w = e.pivot_table(index=k, columns="pipeline", values="bias", dropna=False).reset_index()
    if not {"saai", "direct"} <= set(w.columns):
        return pd.DataFrame()
    w["refit"] = w["saai"] - w["direct"]
    g = w.groupby(["cell_index", "variant", "obs_rule", "period", "quantity"])
    out = g.agg(R=("replicate", "nunique"), median_bias_saai=("saai", "median"),
                median_bias_direct=("direct", "median"),
                median_refit_contribution=("refit", "median"),
                mean_refit_contribution=("refit", "mean"),
                n_nan=("refit", lambda v: int(v.isna().sum()))).reset_index()
    out.insert(1, "saai_spec", "multi minus direct_x")
    out["informative"] = bool(S.REFIT_INFORMATIVE)
    out["note"] = ("refit contribution under covariates iid per county, independent of x and "
                   "province; whether this covariate design is the right one for the "
                   "registered limitation / K2 is left to the authors")
    return out


K3_ESTIMATES_FILE = "k1_k3_estimates.csv"
K3_EST_COLS = ("cell_index", "replicate", "pipeline", "saai_spec", "variant", "obs_rule",
               "period", "quantity", "estimate", "truth")


def k3_estimates_table(est: pd.DataFrame, metas) -> pd.DataFrame:
    cells = {m["cell"]["cell_index"] for m in metas if S.is_neural_cell(m["cell"])}
    if not len(est) or not cells:
        return pd.DataFrame(columns=list(K3_EST_COLS))
    e = est[est["cell_index"].isin(cells) & (est["pipeline"] == "saai")
            & est["variant"].isin(["V0", "V0_neural"]) & est["period"].isin(list(S.PERIODS))]
    return e[[c for c in K3_EST_COLS if c in e.columns]].reset_index(drop=True)


PRED_GATE_USES = ("K1a/K1b", "K1a_null", "control", "control_null", "K1a_shocked_null",
                  S.SENS_NO2006_GATE_USE)
PRED_ROWS_ALL_PIPELINES = ("projection", "fallback")
PRED_ROWS_SAAI_ONLY = ("closed_form", "zeroth_order", "zeroth_order_mbar")
DIAG_KEEP = ("cell_checked", "cell_key", "row", "design", "replicate", "period",
             "theory_version", "contract_version", "true_params", "true_params_check",
             "digest", "valid_for_law_cell",
             "n_transitions", "invalid", "prl_condition_1_inputs")
PREDICT_ROWS_FILE = "k1_predict_rows.json.gz"


def _keep_diag(d: dict) -> dict:
    out = {k: d.get(k) for k in DIAG_KEEP}
    if out["cell_key"] is not None:
        out["cell_key"] = [list(x) for x in out["cell_key"]]
    return out


def write_predict_rows(path: str, rows: list[dict]) -> None:
    """The k1_theory.predict_row outputs, exactly (JSON keeps float repr, so
    prediction_digest still matches after the round trip)."""
    import gzip
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f, allow_nan=True)


def read_predict_rows(path: str) -> list[dict]:
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def predictions_table(comp, std, est, metas, rows=None):
    """Returns (table, note, raw): table = the flat per-replicate CSV (report;
    contrast rows by src.k1_contract.contrast_bias for prediction and
    simulation); raw = the predict_row outputs themselves, one per
    (cell, replicate, pipeline, row, period), each with cell_index and
    pipeline added — the ONLY prediction input k1_gates hands to the
    contract gate functions (they validate the digest and compute every
    contrast themselves).  A failure is a NaN row with pred_error in the
    table and no raw output (the contract then refuses the cell: R ≠ 500),
    never dropped silently.
    """
    try:
        from src import k1_theory as T
        from src import k1_contract as C
    except Exception as exc:
        return None, f"src.k1_theory not importable ({type(exc).__name__}: {exc})", None
    tp = {m["cell"]["cell_index"]: m["true_params"] for m in metas
          if m.get("true_params") is not None}
    cells = {m["cell"]["cell_index"]: m["cell"] for m in metas}
    cmeta = {m["cell"]["cell_index"]: m.get("contract_cell_meta") or {} for m in metas}
    want = {ci for ci, c in cells.items() if c["gate_use"] in PRED_GATE_USES}
    e = est[(est["variant"] == "V0") & est["cell_index"].isin(want)] if len(est) else est
    ei = (e.set_index(["cell_index", "replicate", "pipeline", "period", "quantity"])["estimate"]
          if len(e) else pd.Series(dtype=float))
    cg = {k: g for k, g in comp.groupby(["cell_index", "replicate", "pipeline"])} if len(comp) else {}
    sg = {k: g for k, g in std.groupby(["cell_index", "replicate"])} if len(std) else {}
    out, raw = [], []
    reps = sorted({(m["cell"]["cell_index"], r, pl) for m in metas
                   if m["cell"]["cell_index"] in want
                   for r in range(*m["replicates"]) for pl in S.PIPELINES})
    for ci, rep, pl in reps:
        c, st = cg.get((ci, rep, pl)), sg.get((ci, rep))
        sim = {p: {q: float(ei.get((ci, rep, pl, p, q), np.nan)) for q in QUANTITIES}
               for p in S.PERIODS}
        cell = cmeta.get(ci, {}).get(pl)
        try:
            if c is None or st is None or ci not in tp:
                raise RuntimeError("replicate failed: no composition / standardisation")
            if not cell:
                raise RuntimeError("chunk meta lacks the contract cell meta")
            st_all = st.drop(columns=[c for c in ("cell_index", "replicate", "period")
                                      if c in st.columns]).reset_index(drop=True)
            tru = {p: T.truth(tp[ci][p], st_all, p, cell=_bare_cell(cell))
                   for p in S.PERIODS}
            terr = ""
        except Exception as exc:
            tru = {p: {q: np.nan for q in QUANTITIES} for p in S.PERIODS}
            terr = f"truth:{type(exc).__name__}"
        row_list = PRED_ROWS_ALL_PIPELINES + (PRED_ROWS_SAAI_ONLY if pl == "saai" else ())
        for row in (rows or row_list):
            pe, err = {}, terr
            for p in S.PERIODS:
                try:
                    if terr:
                        raise RuntimeError(terr)
                    if not cell:
                        raise RuntimeError("chunk meta lacks the contract cell meta")
                    cp = c[c["period"] == p].reset_index(drop=True)
                    res = T.predict_row(row, cp, tp[ci][p], p, st_all, cell)
                    pe[p] = {q: float(res["estimate"][q]) for q in QUANTITIES}
                    raw.append({"cell_index": int(ci), "pipeline": pl,
                                "estimate": {q: res["estimate"][q] for q in QUANTITIES}
                                | {"id_status": res["estimate"].get("id_status")},
                                "truth": {q: res["truth"][q] for q in QUANTITIES}
                                | {"id_status": res["truth"].get("id_status")},
                                "diag": _keep_diag(res["diag"])})
                except Exception as exc:
                    pe[p] = {q: np.nan for q in QUANTITIES}
                    err = err or f"{type(exc).__name__}: {str(exc)[:120]}"
            pc = C.contrast_bias(pe["pre"], pe["during"], tru["pre"], tru["during"])
            sc = C.contrast_bias(sim["pre"], sim["during"], tru["pre"], tru["during"])
            for q in QUANTITIES:
                for p in S.PERIODS:
                    out.append({"cell_index": ci, "replicate": rep, "pipeline": pl,
                                "saai_spec": S.SAAI_SPEC_LABEL[pl], "obs_rule": S.OBS_RULES[0],
                                "row": row, "period": p, "quantity": q,
                                "pred_estimate": pe[p][q], "sim_estimate": sim[p][q],
                                "truth": tru[p][q], "pred_bias": pe[p][q] - tru[p][q],
                                "sim_bias": sim[p][q] - tru[p][q], "pred_error": err})
                out.append({"cell_index": ci, "replicate": rep, "pipeline": pl,
                            "saai_spec": S.SAAI_SPEC_LABEL[pl], "obs_rule": S.OBS_RULES[0],
                            "row": row, "period": "contrast", "quantity": q,
                            "pred_estimate": np.nan, "sim_estimate": np.nan, "truth": np.nan,
                            "pred_bias": pc[q], "sim_bias": sc[q], "pred_error": err})
    return (pd.DataFrame(out), "src.k1_theory.predict_row(row, ..., cell=...)", raw)


def shocked_null_table(est: pd.DataFrame, metas) -> pd.DataFrame:
    ids = {m["cell"]["cell_index"] for m in metas if m["cell"]["gate_use"] == "K1a_shocked_null"}
    if not len(est):
        return pd.DataFrame()
    e = est[est["cell_index"].isin(ids) & (est["variant"] == "V0")
            & est["period"].isin(list(S.PERIODS))]
    return e[["cell_index", "replicate", "pipeline", "saai_spec", "obs_rule", "period",
              "quantity", "estimate", "truth", "replicate_status"]].reset_index(drop=True)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cmd_aggregate(a):
    if in_git() and tree_dirty() and not a.dev:
        raise SystemExit("refusing to aggregate on a dirty tree (amendment 1 (C))")
    if a.dev and is_under(a.out):
        raise SystemExit("--dev output never goes to (or below) results/tables")
    registered = not a.non_registered
    if not registered and is_under(a.out):
        raise SystemExit("--non-registered output never goes to (or below) results/tables")
    refuse_exploratory_paths(list(a.inputs) + [a.out], "aggregate path")
    tabs, metas = load_chunks(a.inputs)
    cells = [m["cell"] for m in metas]
    refuse_s0_cells(cells)
    frozen = precheck_freeze(cells, registered, guard_cells(cells))
    commit, report, problems, mask_sha = check_chunks(metas, allow_dev=a.dev,
                                                      registered=registered, frozen=frozen)
    why = same_code(commit) if in_git() else "the aggregator is not in a git checkout"
    if why and not a.dev:
        raise SystemExit(f"chunks were run at code commit {str(commit)[:10]}, not at the "
                         f"aggregator's code state: {why}; re-run the chunks at this commit")
    est, comp, std = tabs["estimates"], tabs["composition"], tabs["standardisation"]
    cf = _cells_frame(metas)
    streams = sorted({m.get("seed_stream", "registered") for m in metas})
    schemes = sorted({str(m.get("bootstrap_scheme")) for m in metas})
    pipes = sorted({",".join(m.get("pipelines") or S.PIPELINES) for m in metas})
    if len(streams) != 1 or len(schemes) != 1 or len(pipes) != 1:
        raise SystemExit(f"chunks mix seed streams {streams}, bootstrap schemes {schemes} or "
                         f"pipeline sets {pipes}")
    seed_stream, boot_scheme = streams[0], schemes[0]
    if registered and (seed_stream != "registered" or boot_scheme != S.BOOTSTRAP_SCHEME
                       or pipes[0] != ",".join(S.PIPELINES)):
        raise SystemExit(f"a registered aggregate needs the registered seed stream, the "
                         f"amendment 7 bootstrap and both pipelines (got {seed_stream!r}, "
                         f"{boot_scheme!r}, {pipes[0]!r}): pass --non-registered")

    resid = comp["identity_residual"].abs().max() if len(comp) else 0.0
    if not resid < 1e-10:
        raise SystemExit(f"Lemma 1 identity violated in the chunks: {resid:.2e}")

    os.makedirs(a.out, exist_ok=True)
    out = {"k1_sim_bias.csv": bias_table(est), "k1_sim_coverage.csv": coverage_table(est),
           "k1_power.csv": power_table(est), "k1_power_vs_truth.csv": power_vs_truth_table(est),
           "k1_refit_contribution.csv": refit_table(est)}
    out["k1_shocked_null_replicates.csv"] = shocked_null_table(est, metas)
    out[K3_ESTIMATES_FILE] = k3_estimates_table(est, metas)
    pred, pred_note, praw = predictions_table(comp, std, est, metas)
    if pred is not None:
        out["k1_predictions.csv"] = pred
    run_label = "registered" if registered else "NON-REGISTERED"
    for name, df in out.items():
        if len(df):
            df = cf.merge(df, on="cell_index", how="right")
            df.insert(0, "run_label", run_label)
        df.to_csv(os.path.join(a.out, name), index=False)
        print(f"wrote {name} ({len(df)} rows)")

    if pred is not None:
        write_predict_rows(os.path.join(a.out, PREDICT_ROWS_FILE), praw)
        print(f"wrote {PREDICT_ROWS_FILE} ({len(praw)} predict_row outputs)")

    def _first(key):
        return {m["cell"]["cell_index"]: m.get(key) for m in metas}
    meta = {"aggregator_git": _git("rev-parse", "HEAD") if in_git() else "unknown",
            "run_label": run_label, "registered": registered,
            "registration_problems": problems,
            "code_commit": commit, "theory_frozen": frozen,
            "k1_theory_sha256": _sha(os.path.join(ROOT, "src", "k1_theory.py"))
            if pred is not None else None,
            "predictions": pred_note, "dev": bool(a.dev),
            "mask_content_sha256": mask_sha,
            "mask_files": sorted({json.dumps(m.get("mask"), sort_keys=True) for m in metas}),
            "seed_rule": SEED_RULE + " (S0: scripts/k1_s0.py, src.k1_contract.s0_seeds)",
            "seed_stream": seed_stream, "precheck_seed_base": C.PRECHECK_SEED_BASE,
            "implcheck_seed_base": C.IMPLCHECK_SEED_BASE,
            "bootstrap_scheme": boot_scheme, "pipelines": pipes,
            "chunks_same_code_as_aggregator": why is None,
            "per_cell": {int(k): v for k, v in report.items()},
            "R_registered": S.R_REGISTERED, "B_registered": S.B_REGISTERED,
            "saai_spec": S.SAAI_SPEC_LABEL, "obs_rules": list(S.OBS_RULES),
            "degenerate_convention": "limit (src.k1_contract.observables, amendment 2 (h))",
            "ci_nan_rules": S.CI_NAN_RULES, "primary_ci_rule": S.PRIMARY_CI_RULE,
            "quantile_method": S.QUANTILE_METHOD,
            "refit_informative": bool(S.REFIT_INFORMATIVE),
            "province_shock_share_w": S._w_shock(),
            "identity_max_residual": float(resid),
            "contract_cell_meta": _first("contract_cell_meta"),
            "theory_compat": _first("theory_compat"),
            "failed_replicates": {os.path.basename(m["_stem"]): m.get("failed_replicates", [])
                                  for m in metas if m.get("failed_replicates")},
            "offsets_diagnostics": {m["cell"]["cell_index"]: m.get("offsets_diagnostics")
                                    for m in metas if m.get("offsets_diagnostics")},
            "design_constants": _first("design_constants"),
            "clusters": sorted({str(m["cluster"]) for m in metas}),
            "slurm_jobs": sorted({str(m["slurm_job"]) for m in metas}),
            "neural": sorted({m["neural"] for m in metas}),
            "versions": sorted({json.dumps(m.get("versions"), sort_keys=True) for m in metas}),
            "n_chunks": len(metas),
            "core_seconds": float(sum(sum(m["seconds_per_replicate"]) for m in metas)),
            "bootstrap_note": BOOTSTRAP_DISCLOSURE,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "k1_sim_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("wrote k1_sim_meta.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cells", required=True, help="'a', 'all', or indices like 0,5,26-33")
    r.add_argument("--reps", default="0:1", help="r0:r1 (ignored with --task)")
    r.add_argument("--task", type=int, default=None, help="array task index into the task list")
    r.add_argument("--R", type=int, default=S.R_REGISTERED)
    r.add_argument("--reps-per-chunk", type=int, default=10)
    r.add_argument("--B", type=int, default=S.B_REGISTERED)
    r.add_argument("--out", required=True)
    r.add_argument("--format", choices=["parquet", "csv.gz"], default=None)
    r.add_argument("--dev", action="store_true", help="allow a dirty tree; output never counts")
    r.add_argument("--composition-only", action="store_true",
                   help="amendment 3 (d) step 2: composition / standardisation / truth only, no "
                        "estimator (the only way exit cells may run before S0 has passed)")
    r.add_argument("--seed-stream", choices=list(S.SEED_STREAMS), default="registered",
                   help="'implcheck': the amendment 7 implementation check's third seed stream on "
                        "the (a) precheck cells (diagnostic; aggregate --non-registered only)")
    r.add_argument("--pipelines", default=None, help="subset of saai,direct (default both)")
    r.add_argument("--no-neural", action="store_true",
                   help="run the MLE only on neural cells (no V0_neural rows; recorded in "
                        "the chunk meta; K3 then reads not_run, never fail)")
    n = sub.add_parser("ntasks")
    n.add_argument("--cells", required=True)
    n.add_argument("--R", type=int, default=S.R_REGISTERED)
    n.add_argument("--reps-per-chunk", type=int, default=10)
    g = sub.add_parser("aggregate")
    g.add_argument("--in", dest="inputs", nargs="+", required=True)
    g.add_argument("--out", default=os.path.join(ROOT, "results", "tables"))
    g.add_argument("--dev", action="store_true")
    g.add_argument("--non-registered", action="store_true",
                   help="aggregate an incomplete / non-registered grid; every output labelled")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "ntasks":
        cells = parse_cells(a.cells)
        print(len(task_plan(cells, a.R, a.reps_per_chunk)))
    else:
        cmd_aggregate(a)


if __name__ == "__main__":
    main()
