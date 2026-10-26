"""python3 scripts/k1_predicted_effects.py --in results/tables/k1_composition_exit         --out results/tables

Writes k1_predicted_effect_sizes.csv (one row per process × pipeline ×
quantity, with the per-cell medians and non-finite counts) and
k1_predicted_effect_sizes_meta.json.  Refuses a dirty tree, a non-registered
or dev aggregate, an aggregate made at another code state, and any cell that
is not composition-only.
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

import pandas as pd

from src import k1_contract as C

PROCESSES = tuple(dict.fromkeys(p for p, _ in C.K1A_CELLS))


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def main(argv=None) -> None:
    from k1_run_cells import PREDICT_ROWS_FILE, read_predict_rows, same_code
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if _git("status", "--porcelain"):
        raise SystemExit("refusing on a dirty tree (rule 1)")
    with open(os.path.join(a.inp, "k1_sim_meta.json")) as f:
        meta = json.load(f)
    if meta.get("registered") is not True or meta.get("dev") is not False:
        raise SystemExit("the composition aggregate is not a registered run")
    why = same_code(str(meta.get("aggregator_git")))
    if why:
        raise SystemExit(f"the composition aggregate was written at another code state: {why}")
    modes = {int(k): v.get("mode") for k, v in (meta.get("per_cell") or {}).items()}
    if not modes or set(modes.values()) != {"composition_only"}:
        raise SystemExit(f"not a composition-only aggregate: {modes}")
    assert_theory_frozen = public_guard_skipped
    tag = assert_theory_frozen()
    rec = C._require_frozen()
    if meta.get("theory_frozen") != tag:
        raise SystemExit(f"aggregate freeze {meta.get('theory_frozen')} != {tag}")
    praw = read_predict_rows(os.path.join(a.inp, PREDICT_ROWS_FILE))
    idx = {}
    for r in praw:
        idx.setdefault((int(r["cell_index"]), r["pipeline"], r["diag"].get("row")), []).append(r)
    cmeta = {int(k): v for k, v in (meta.get("contract_cell_meta") or {}).items() if v}
    rows, effects = [], {}
    for proc in PROCESSES:
        for pl in C.S0_PIPELINES:
            entries = []
            for p2, mech in C.K1A_CELLS:
                if p2 != proc:
                    continue
                hits = [ci for ci, d in cmeta.items()
                        if (d.get(pl) or {}).get("process") == proc
                        and (d.get(pl) or {}).get("exit_mechanism") == mech
                        and (d.get(pl) or {}).get("gate_use") == "K1a/K1b"]
                if len(hits) != 1:
                    raise SystemExit(f"{proc} × ({mech}) {pl}: {len(hits)} cells in the aggregate")
                ci = hits[0]
                cell = {k: v for k, v in cmeta[ci][pl].items() if k != "valid_for_law"}
                entries.append({"cell": cell, "predictions": idx.get((ci, pl, "projection"), [])})
            pe = C.predicted_effect_sizes(proc, pl, entries)
            effects[f"{proc}/{pl}"] = C._canon_json(pe)
            for q in C.QUANTITIES:
                pc = {k: v.get(q, {}) for k, v in (pe.get("per_cell") or {}).items()}
                rows.append({"process": proc, "pipeline": pl, "quantity": q,
                             "effect": pe["effect"][q], "status": pe["status"][q],
                             **{f"median_{k}": v.get("median") for k, v in pc.items()},
                             **{f"n_nonfinite_{k}": v.get("n_nonfinite") for k, v in pc.items()},
                             **{f"n_total_{k}": v.get("n_total") for k, v in pc.items()},
                             "frozen_commit": pe.get("frozen_commit"),
                             "row": "projection", "design": "observed"})
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "k1_predicted_effect_sizes.csv"), index=False)
    out_meta = {"script": "scripts/k1_predicted_effects.py", "git": _git("rev-parse", "HEAD"),
                "clean_tree": True, "input_dir": os.path.relpath(a.inp, ROOT),
                "input_aggregator_git": meta.get("aggregator_git"),
                "input_code_commit": meta.get("code_commit"),
                "theory_frozen": tag, "freeze_record_frozen_commit": rec.get("frozen_commit"),
                "contract_version": C.CONTRACT_VERSION,
                "slurm_jobs": meta.get("slurm_jobs"), "clusters": meta.get("clusters"),
                "predicted_effect_sizes": effects,
                "note": "S0 tolerance caps (amendment 2 (d), amendment 3 (c)); no estimator output read",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "k1_predicted_effect_sizes_meta.json"), "w") as f:
        json.dump(out_meta, f, indent=1, default=str)
    print(pd.DataFrame(rows)[["process", "pipeline", "quantity", "effect", "status"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
