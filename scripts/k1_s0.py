"""Truth check (S0) of the simulation study, end to end, for one process and pipeline
(run --replicates r0:r1: half-open range of replicate indices, default 0:500).

    python3 scripts/k1_s0.py start  --process cubic --pipeline saai
        clean git checkout: checks the backend's precheck and the cell metas
        (simulating nothing), appends a 'started' entry to
        results/records/k1_s0_attempts.jsonl and commits it
    python3 scripts/k1_s0.py run    --process cubic --pipeline saai --replicates 0:50 --out RAW_DIR [--workers 8]
        from the start commit (HEAD must be the start commit), in chunks if needed
    python3 scripts/k1_s0.py record --process cubic --pipeline saai --raw RAW_DIR
        clean checkout, every chunk in RAW_DIR: recomputes truths, predictions and
        the verdict from the raw simulation outputs, writes
        results/records/k1_s0_<process>_<pipeline>.json and the result log entry,
        and commits both, pass or fail
    python3 scripts/k1_s0.py record ... --dry-run
        recomputes the result of a committed record and compares its sha256

An exception inside the estimator or the truth of one replicate is recorded
as NaN estimates of that replicate with its error text; the replicate stays
in the denominator. A refusal (a seed or cell check) stops the run.
"""
from __future__ import annotations

def public_guard_skipped(value=None, *args, **kwargs):
    """Integrity guard that checks the authors' private version-controlled record:
    disabled in the public copy. Returns `value` unchanged."""
    return value


import argparse
import gzip
import importlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import k1_contract as C

S0_BACKEND = C.S0_BACKEND
ESTIMATORS = ("mle", "neural")
NEURAL_PROCESSES = ("ou", "cubic")
NEURAL_PIPELINES = ("saai",)


def _neural():
    from src import k1_neural_k3 as K3
    return K3


def _paths(estimator: str):
    import contextlib
    if estimator == "mle":
        return contextlib.nullcontext()
    return _neural()._neural_s0_paths()


def _check_estimator(estimator: str, process: str, pipeline: str) -> None:
    if estimator not in ESTIMATORS:
        raise RuntimeError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    if estimator == "neural":
        if process not in NEURAL_PROCESSES or pipeline not in NEURAL_PIPELINES:
            raise RuntimeError(f"the neural S0 exists for {NEURAL_PROCESSES} × "
                               f"{NEURAL_PIPELINES} only (draft N2 (I)), got "
                               f"{process}/{pipeline}")
        from src import k1_neural_sde as nsde
        if not nsde.TUNING_FROZEN:
            raise RuntimeError("neural S0 refused: the neural configuration is not frozen "
                               "(amendment N1 / N2 (A))")


def _is_refusal(exc: BaseException) -> bool:
    """A guard refusal (never a replicate failure)."""
    return bool(getattr(exc, "guard_refusal", False)) or type(exc).__name__ in (
        "S0Blocked", "TheoryNotFrozen")
BACKEND_API = {
    "cell_meta": "cell_meta(process, exit_mechanism, pipeline, s0: bool) -> the cell "
                 "meta (CELL_META_KEYS) of the gated 'shock' cell; s0=True: the registered S0 "
                 "cell (exit 'a', S0_N counties per window: C.check_s0_N)",
    "simulate_no_exit": "simulate_no_exit(cell, replicate, seed) -> {'pairs': {'pre': "
                        "DataFrame, 'during': DataFrame} (contract estimate() input), "
                        "'standardisation': DataFrame (STANDARDISATION_COLUMNS of the "
                        "replicate's COMPLETE panel: every simulated county of each "
                        "window in every year, n constant within a window, window "
                        "sizes summing to the cell N), 'true_params': {'pre': dict, 'during': "
                        "dict} (TrueProcess.as_dict, TRUE_PROCESS_KEYS)}",
    "simulate_composition": "simulate_composition(cell, replicate) -> {'composition': "
                            "DataFrame (COMPOSITION_COLUMNS, both periods), "
                            "'standardisation': DataFrame (STANDARDISATION_COLUMNS of "
                            "the replicate's COMPLETE panel: every simulated county of "
                            "each window in every year, window sizes summing to the "
                            "cell N; never the observed panel), "
                            "'true_params': {...}, 'seed': int (the Step 2 seed "
                            "used)} — exit cells, compositions only, no estimator",
    "estimate": "estimate(pairs) -> {quantity: value} (contract estimate(): fit, "
                "observables, rho1_from_pairs)",
}


def load_backend(name: str = S0_BACKEND):
    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        raise RuntimeError(f"S0 backend {name!r} is not importable ({e}): the simulation "
                           f"line provides it with {sorted(BACKEND_API)}") from e
    miss = [f for f in BACKEND_API if not callable(getattr(mod, f, None))]
    if miss:
        raise RuntimeError(f"S0 backend {name!r} lacks {miss}")
    return mod


def _git(*a) -> str:
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True,
                          check=True).stdout.strip()


def code_commit() -> str:
    if C._in_git_checkout():
        if _git("status", "--porcelain"):
            raise RuntimeError("refusing: working tree is not clean (rule 1)")
        return _git("rev-parse", "HEAD")
    arc = C.archive_commit()
    if arc is None:
        raise RuntimeError("not a checkout and not a git-archive export of a committed hash")
    return arc["commit"]


def _df(d) -> dict:
    return pd.DataFrame(d).to_dict(orient="list")


def _rows_to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d)


def _check_raw_code(raw_code) -> None:
    bad = C.s0_code_mismatch(raw_code or {}) if isinstance(raw_code, dict) else ["<missing>"]
    if bad:
        raise RuntimeError(f"the raw S0 chunks were made by code that differs from the start "
                           f"commit's in {bad} (contract 9)")


def start(process: str, pipeline: str, backend_name: str = S0_BACKEND,
          estimator: str = "mle") -> dict:
    pass
    public_guard_skipped()
    C._require_frozen()
    _check_estimator(estimator, process, pipeline)
    backend = load_backend(backend_name)
    pre = getattr(backend, "s0_precheck", None)
    if callable(pre):
        pre(process, pipeline)
    cell = _check_s0_cell(backend.cell_meta(process, "a", pipeline, s0=True), process,
                          pipeline)
    if estimator == "neural":
        _neural().assert_neural_allowed({k: v for k, v in cell.items()
                                         if k != "valid_for_law"})
    if process in {p for p, _ in C.K1A_CELLS}:
        _k1a_cells(backend, process, pipeline)
    with _paths(estimator):
        return C.start_s0_attempt(process, pipeline)


def _check_s0_cell(cell: dict, process: str, pipeline: str) -> dict:
    c = C.check_cell_config(cell)
    want = {"process": process, "pipeline": pipeline, "exit_mechanism": "a",
            "exit_dating": "none", "gated": True, "re_placement": C.GATED_RE_PLACEMENT}
    bad = {k: c[k] for k, v in want.items() if c[k] != v}
    if not bad:
        try:
            C.check_s0_N(c)
        except ValueError as exc:
            bad = {"N": c.get("N"), C.S0_N_WINDOWS_KEY: c.get(C.S0_N_WINDOWS_KEY),
                   "why": str(exc)}
    if bad:
        raise RuntimeError(f"backend S0 cell is not the registered S0 cell: {bad}")
    return c


def _k1a_cells(backend, process: str, pipeline: str) -> list:
    out = []
    for p, mech in C.K1A_CELLS:
        if p != process:
            continue
        c = C.check_cell_config(backend.cell_meta(process, mech, pipeline, s0=False))
        if (c["process"], c["exit_mechanism"], c["pipeline"], c["gated"],
                c["re_placement"]) != (process, mech, pipeline, True, C.GATED_RE_PLACEMENT):
            raise RuntimeError(f"backend K1a cell {process}-{mech} is not the gated "
                               f"'shock' {pipeline} cell: {c}")
        out.append(c)
    return out


def _check_tp(process: str, tp: dict) -> None:
    for per in C.PERIODS:
        C.check_registered_true_params(process, per, tp[per])


NEURAL_ROW_KEYS = ("estimator", "degenerate", "config", "pipeline", "saai_spec", "variant",
                   "period", "exit_mechanism", "cell_process", "N", "cell_index",
                   "rep_seed", "fit_seed", "id_status", "n_stable_roots")


def _estimate_one(backend, estimator: str, cell: dict, r: int, per: str, pairs) -> dict:
    if estimator == "mle":
        e = backend.estimate(pairs)
        return {q: float(e[q]) for q in C.QUANTITIES}
    row = _neural().neural_k3_row(pairs, cell=cell, replicate=r, period=per)
    out = {q: float(row[q]) for q in C.QUANTITIES}
    out.update({k: (row.get(k) if not isinstance(row.get(k), np.generic)
                    else row.get(k).item()) for k in NEURAL_ROW_KEYS})
    return out


def _failed_estimate(estimator: str, cell: dict, r: int, per: str, exc) -> dict:
    out = {q: float("nan") for q in C.QUANTITIES}
    out["replicate_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    if estimator == "neural":
        K3 = _neural()
        from src import k1_neural_sde as nsde
        out.update({**K3.NEURAL_S0_LABELS, **K3.K3_LABELS, "config": nsde.FROZEN_CONFIG,
                    "period": per, "exit_mechanism": cell["exit_mechanism"],
                    "cell_process": cell["process"], "N": int(cell["N"]),
                    "cell_index": cell.get("cell_index"), "id_status": "fit_error"})
    return out


def _s0_replicate(args) -> dict:
    backend_name, cell, r, seed = args[:4]
    estimator = args[4] if len(args) > 4 else "mle"
    backend = load_backend(backend_name)
    sim = backend.simulate_no_exit(cell, r, seed)
    _check_tp(cell["process"], sim["true_params"])
    est, errors = {}, {}
    for per in C.PERIODS:
        try:
            est[per] = _estimate_one(backend, estimator, cell, r, per, sim["pairs"][per])
        except Exception as exc:
            if _is_refusal(exc):
                raise
            est[per] = _failed_estimate(estimator, cell, r, per, exc)
            errors[per] = est[per]["replicate_error"]
    return {"replicate": r, "seed": seed, "estimate": est, "replicate_errors": errors,
            "standardisation": _df(sim["standardisation"])}


def _comp_replicate(args) -> dict:
    backend_name, cell, r = args
    backend = load_backend(backend_name)
    sim = backend.simulate_composition(cell, r)
    _check_tp(cell["process"], sim["true_params"])
    return {"replicate": r, "seed": int(sim["seed"]),
            "composition": _df(sim["composition"]),
            "standardisation": _df(sim["standardisation"])}


def run(process: str, pipeline: str, replicates: range, out_dir: Path,
        workers: int = 1, backend_name: str = S0_BACKEND, estimator: str = "mle") -> Path:
    freeze = C._require_frozen()
    _check_estimator(estimator, process, pipeline)
    with _paths(estimator):
        attempt = C.open_s0_attempt(process, pipeline)
    commit = code_commit()
    if C._in_git_checkout():
        with _paths(estimator):
            C.s0_code_binding(attempt, commit)
    backend = load_backend(backend_name)
    s0cell = _check_s0_cell(backend.cell_meta(process, "a", pipeline, s0=True),
                            process, pipeline)
    C.assert_exit_estimation_allowed(s0cell)
    if estimator == "neural":
        _neural().assert_neural_allowed({k: v for k, v in s0cell.items()
                                         if k != "valid_for_law"})
    k1a = _k1a_cells(backend, process, pipeline) if process in {
        p for p, _ in C.K1A_CELLS} else []
    for c in k1a:
        C.assert_exit_simulation_allowed(c)
    if replicates.start < 0 or replicates.stop > C.R_PER_CELL:
        raise RuntimeError(f"replicates must lie in 0 … {C.R_PER_CELL - 1}")
    seeds = C.s0_seeds(process, pipeline)
    clean = {k: v for k, v in s0cell.items() if k != "valid_for_law"}
    jobs = [(backend_name, clean, r, seeds[r], estimator) for r in replicates]
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    mapper = pool.map if pool else map
    try:
        s0 = list(mapper(_s0_replicate, jobs))
        comps = {}
        for c in k1a:
            cc = {k: v for k, v in c.items() if k != "valid_for_law"}
            comps[c["exit_mechanism"]] = {
                "cell": cc,
                "replicates": list(mapper(_comp_replicate,
                                          [(backend_name, cc, r) for r in replicates]))}
    finally:
        if pool:
            pool.shutdown()
    raw = {"meta": {"script": "scripts/k1_s0.py", "code_commit": commit,
                    "frozen_commit": freeze.get("frozen_commit"),
                    "contract_version": C.CONTRACT_VERSION, "attempt": attempt["attempt"],
                    "process": process, "pipeline": pipeline, "estimator": estimator,
                    "replicates": [replicates.start, replicates.stop],
                    "backend": backend_name, "s0_code": C.s0_code_digest(),
                    "environment": {m.__name__: m.__version__ for m in (np, pd, scipy)}},
           "s0_cell": clean, "s0": s0, "k1a": comps}
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if estimator == "mle" else f"_{estimator}"
    out = out_dir / (f"k1_s0_raw_{process}_{pipeline}{tag}_{replicates.start:03d}_"
                     f"{replicates.stop:03d}.json.gz")
    with gzip.open(out, "wt") as f:
        json.dump(raw, f)
    return out


def load_raw(raw_dir: Path, process: str, pipeline: str, estimator: str = "mle") -> dict:
    """Merge the raw chunks; refuse inconsistent metadata (commit, freeze,
    attempt, cells, estimator) and anything but exactly the replicates 0 … R−1."""
    import re
    pat = re.compile(rf"k1_s0_raw_{re.escape(process)}_{re.escape(pipeline)}"
                     rf"{'' if estimator == 'mle' else '_' + estimator}_\d{{3}}_\d{{3}}\.json\.gz$")
    files = sorted(f for f in raw_dir.glob(f"k1_s0_raw_{process}_{pipeline}_*.json.gz")
                   if pat.fullmatch(f.name))
    if not files:
        raise RuntimeError(f"no raw chunks for {process}/{pipeline} in {raw_dir}")
    merged, meta = None, None
    for fp in files:
        with gzip.open(fp, "rt") as f:
            raw = json.load(f)
        m = {k: v for k, v in raw["meta"].items() if k != "replicates"}
        if m.get("estimator", "mle") != estimator:
            raise RuntimeError(f"{fp.name}: raw chunk of estimator {m.get('estimator')!r}, "
                               f"not {estimator!r}")
        if meta is None:
            meta = m
            merged = {"meta": m, "s0_cell": raw["s0_cell"], "s0": [],
                      "k1a": {k: {"cell": v["cell"], "replicates": []}
                              for k, v in raw["k1a"].items()}}
        elif m != meta or raw["s0_cell"] != merged["s0_cell"] or \
                {k: v["cell"] for k, v in raw["k1a"].items()} != \
                {k: v["cell"] for k, v in merged["k1a"].items()}:
            raise RuntimeError(f"{fp.name}: metadata or cells differ from the other chunks")
        merged["s0"] += raw["s0"]
        for k, v in raw["k1a"].items():
            merged["k1a"][k]["replicates"] += v["replicates"]
    want = list(range(C.R_PER_CELL))
    if sorted(d["replicate"] for d in merged["s0"]) != want:
        raise RuntimeError(f"S0 raw chunks do not hold exactly the replicates 0 … "
                           f"{C.R_PER_CELL - 1}")
    for k, v in merged["k1a"].items():
        if sorted(d["replicate"] for d in v["replicates"]) != want:
            raise RuntimeError(f"K1a cell {k} raw chunks do not hold exactly 0 … "
                               f"{C.R_PER_CELL - 1}")
    return merged


def compute_s0(raw: dict, process: str, pipeline: str) -> dict:
    from src import k1_theory as T
    if raw["meta"]["process"] != process or raw["meta"]["pipeline"] != pipeline:
        raise RuntimeError("raw output is for another process / pipeline")
    estimator = raw["meta"].get("estimator", "mle")
    tp = {per: C.registered_true_params(process, per) for per in C.PERIODS}
    est, tru = {p: [] for p in C.PERIODS}, {p: [] for p in C.PERIODS}
    for d in sorted(raw["s0"], key=lambda d: d["replicate"]):
        std = _rows_to_df(d["standardisation"])
        for per in C.PERIODS:
            est[per].append({**d["estimate"][per], "replicate": d["replicate"],
                             "seed": d["seed"]})
            try:
                t = T.truth(tp[per], std, per, cell=raw["s0_cell"])
            except ValueError as exc:
                if "complete panel" in str(exc):
                    raise
                t = {q: float("nan") for q in C.QUANTITIES}
            tru[per].append({**{q: t[q] for q in C.QUANTITIES},
                             "replicate": d["replicate"], "true_params": tp[per]})
    entries = None
    if raw["k1a"]:
        entries = []
        for mech in sorted(raw["k1a"]):
            v = raw["k1a"][mech]
            preds = []
            for d in sorted(v["replicates"], key=lambda d: d["replicate"]):
                comp, std = _rows_to_df(d["composition"]), _rows_to_df(d["standardisation"])
                for per in C.PERIODS:
                    preds.append(T.predict_row("projection", comp, tp[per], per, std,
                                               v["cell"]))
            entries.append({"cell": v["cell"], "predictions": preds})
    if estimator == "neural":
        res = _neural().neural_s0_check(raw["s0_cell"], est, tru, entries)
    else:
        fn = getattr(load_backend(), "rho1_reference", None)
        ref = fn(process) if (fn is not None and process in C.S0_RHO1_REPORT_ONLY) else None
        res = C.s0_check(raw["s0_cell"], est, tru, entries, rho1_reference=ref)
    res["raw_meta"] = raw["meta"]
    res["replicate_errors"] = {str(d["replicate"]): d.get("replicate_errors")
                               for d in raw["s0"] if d.get("replicate_errors")}
    res["k1a_composition_seeds"] = {
        mech: [d["seed"] for d in sorted(v["replicates"], key=lambda d: d["replicate"])]
        for mech, v in raw["k1a"].items()}
    return res


def _check_raw_attempt(raw: dict, entry: dict) -> None:
    m = raw["meta"]
    if m.get("attempt") != entry.get("attempt") or \
            m.get("frozen_commit") != entry.get("frozen_commit"):
        raise RuntimeError("raw output belongs to another S0 attempt or freeze")


def _committed_record(process: str, pipeline: str, estimator: str = "mle") -> tuple[dict, dict]:
    with _paths(estimator):
        return _committed_record_here(process, pipeline)


def _committed_record_here(process: str, pipeline: str) -> tuple[dict, dict]:
    freeze = C._require_frozen()
    path = C.s0_record_path(process, pipeline)
    if not path.exists():
        raise RuntimeError(f"record --dry-run is the verifier's rerun of a COMMITTED S0 "
                           f"record and {path.relative_to(C.REPO)} does not exist: S0 runs "
                           "once per freeze, and its verdict is seen only through the "
                           "logged record (amendment 3 (j))")
    C._git_committed_unchanged(path)
    C._git_committed_unchanged(C.s0_attempt_log_path())
    rec = json.loads(path.read_text())
    if rec.get("frozen_commit") != freeze.get("frozen_commit"):
        raise RuntimeError("the committed S0 record was made under another freeze")
    st = [e for e in C._entries(process, pipeline) if e.get("stage") == "started"
          and e.get("attempt") == rec.get("attempt")
          and e.get("frozen_commit") == rec.get("frozen_commit")]
    if len(st) != 1:
        raise RuntimeError(f"the committed S0 record's attempt {rec.get('attempt')} has "
                           f"{len(st)} started log entries (need 1)")
    return rec, st[0]


def record(process: str, pipeline: str, raw_dir: Path, dry_run: bool = False,
           estimator: str = "mle") -> dict:
    _check_estimator(estimator, process, pipeline)
    raw = load_raw(raw_dir, process, pipeline, estimator)
    if dry_run:
        rec, entry = _committed_record(process, pipeline, estimator)
        _check_raw_attempt(raw, entry)
        with _paths(estimator):
            start = C.s0_code_binding(entry, raw["meta"]["code_commit"])
        _check_raw_code(raw["meta"].get("s0_code"))
        if rec.get("s0_code") != raw["meta"].get("s0_code"):
            raise RuntimeError("the committed record's s0_code is not the rerun's code")
        if rec.get("start_commit") != start or rec.get("estimates_commit") != start:
            raise RuntimeError("the committed record's start / estimates commit is not the "
                               "attempt's start commit")
        res = compute_s0(raw, process, pipeline)
        sha = C._sha(C._canon_json(res))
        return {"result": res, "result_sha256": sha, "path": None,
                "record_sha256": rec.get("result_sha256"),
                "matches": sha == rec.get("result_sha256")}
    with _paths(estimator):
        attempt = C.open_s0_attempt(process, pipeline)
        _check_raw_attempt(raw, attempt)
        C.s0_code_binding(attempt, raw["meta"]["code_commit"])
    _check_raw_code(raw["meta"].get("s0_code"))
    res = compute_s0(raw, process, pipeline)
    sha = C._sha(C._canon_json(res))
    if estimator == "neural":
        path = _neural().write_neural_s0_record(res, C.s0_seeds(process, pipeline),
                                                raw["meta"]["code_commit"])
    else:
        path = C.write_s0_record(res, C.s0_seeds(process, pipeline),
                                 raw["meta"]["code_commit"])
    return {"result": res, "result_sha256": sha, "path": path}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "run", "record"):
        s = sub.add_parser(name)
        s.add_argument("--process", required=True, choices=list(C.PROCESSES))
        s.add_argument("--pipeline", required=True, choices=list(C.S0_PIPELINES))
        s.add_argument("--estimator", default="mle", choices=list(ESTIMATORS))
        if name == "run":
            s.add_argument("--replicates", default=f"0:{C.R_PER_CELL}")
            s.add_argument("--out", type=Path, default=REPO / "data_clean/k1_s0_raw")
            s.add_argument("--workers", type=int, default=1)
        if name == "record":
            s.add_argument("--raw", type=Path, default=REPO / "data_clean/k1_s0_raw")
            s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "start":
        e = start(a.process, a.pipeline, estimator=a.estimator)
        print(f"S0 attempt {e['attempt']} of {a.process}/{a.pipeline} started at "
              f"{e['start_commit'][:10]}: check out this commit and run")
    elif a.cmd == "run":
        lo, hi = (int(v) for v in a.replicates.split(":"))
        out = run(a.process, a.pipeline, range(lo, hi), a.out, a.workers,
                  estimator=a.estimator)
        print(f"wrote {out}")
    else:
        r = record(a.process, a.pipeline, a.raw, a.dry_run, estimator=a.estimator)
        print(r["result"]["label"])
        if r["path"]:
            print(f"result_sha256 {r['result_sha256']}; committed "
                  f"{r['path'].relative_to(C.REPO)}")
        else:
            print(f"result_sha256 {r['result_sha256']} (verifier rerun, nothing written); "
                  f"committed record {r['record_sha256']}: "
                  + ("MATCH" if r["matches"] else "MISMATCH"))
            if not r["matches"]:
                sys.exit(1)


if __name__ == "__main__":
    main()
