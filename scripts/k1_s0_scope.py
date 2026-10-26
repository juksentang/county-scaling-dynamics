"""python3 scripts/k1_s0_scope.py --out results/tables"""
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

GATES = ("K1a", "K1b", "K1c", "K2", "K3")
PRECHECK_SCOPE_OF_GATE = {"K1a": "K1b", "K1b": "K1b", "K1c": "K1c", "K2": "K2", "K3": "K1b"}


def _git(*a) -> str:
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def main(argv=None) -> None:
    import k1_gates as G
    from src import k1_sim as S
    assert_theory_frozen = public_guard_skipped
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if _git("status", "--porcelain"):
        raise SystemExit("refusing on a dirty tree (rule 1)")
    frozen = assert_theory_frozen()
    rows = pd.DataFrame(G.gate_s0())
    mle = G.s0_pass_by_process(rows, "S0")
    nn = G.s0_pass_by_process(rows, "S0_neural")
    blocked = G.s0_blocked_gates(rows, GATES)
    pre = S.coverage_precheck(frozen=frozen)
    out = []
    for g in GATES:
        need = G.GATE_S0_SCOPE.get(g, G.S0_PROCESSES)
        need_nn = G.NEURAL_S0_SCOPE.get(g, ())
        miss = [p for p in need if not mle.get(p, False)]
        miss_nn = [p for p in need_nn if not nn.get(p, False)]
        s0_ok = g not in blocked
        sc = (pre.get("scopes") or {}).get(PRECHECK_SCOPE_OF_GATE[g]) or {}
        pc_ok = bool(sc.get("pass"))
        pc_why = (pre.get("why") if not sc else
                  f"scope {PRECHECK_SCOPE_OF_GATE[g]}: {'; '.join(sc.get('out_of_band') or [])}")
        out.append({"gate": g,
                    "verdict_role": (G.K2_VERDICT if g in G.REPORT_ONLY_GATES else "gated"),
                    "s0_scope_mle": ";".join(need),
                    "s0_scope_neural": ";".join(need_nn),
                    "s0_not_passed_mle": ";".join(miss), "s0_not_passed_neural": ";".join(miss_nn),
                    "s0_scope_ok": s0_ok, "precheck_scope": PRECHECK_SCOPE_OF_GATE[g],
                    "precheck_pass": pc_ok, "precheck_all_scopes_pass": bool(pre.get("pass")),
                    "unlocked": bool(s0_ok and pc_ok),
                    "why": "" if (s0_ok and pc_ok) else "; ".join(
                        x for x in ((f"S0 not passed: {';'.join(miss + [p + ' (neural)' for p in miss_nn])}"
                                     if not s0_ok else ""),
                                    ("" if pc_ok else f"precheck: {pc_why}"))
                        if x)})
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(a.out, "k1_s0_scope.csv"), index=False)
    s0_cols = [c for c in ("gate", "process", "pipeline", "period", "quantity", "status",
                           "value", "threshold", "rule_set", "gating", "note") if c in rows.columns]
    rows[s0_cols].to_csv(os.path.join(a.out, "k1_s0_scope_s0_rows.csv"), index=False)
    units = []
    try:
        cov = pd.read_csv(os.path.join(ROOT, S.COVERAGE_PRECHECK_DIR, "k1_sim_coverage.csv"))
        gated = S.coverage_precheck_units()
        for c, q, per in S.coverage_precheck_units(include_not_evaluable=True):
            r = cov[(cov["cell_index"] == c["cell_index"]) & (cov["pipeline"] == "saai")
                    & (cov["variant"] == "V1") & (cov["ci_nan_rule"] == "tol0")
                    & (cov["obs_rule"] == S.OBS_RULES[0]) & (cov["quantity"] == q)
                    & (cov["period"] == per)]
            v = float(r["coverage"].iloc[0]) if len(r) == 1 else float("nan")
            units.append({"cell_index": c["cell_index"], "process": c["process"],
                          "quantity": q, "period": per, "coverage": v,
                          "R": int(r["R"].iloc[0]) if len(r) == 1 else None,
                          "in_band": S.COVERAGE_BAND[0] <= v <= S.COVERAGE_BAND[1],
                          "band": f"{list(S.COVERAGE_BAND)}",
                          "scope": S.coverage_precheck_scope(c),
                          "status": ("gated" if (c, q, per) in gated
                                     else "not evaluable (amendment 7 (S2) item 4)")})
    except (OSError, ValueError, KeyError) as exc:
        units.append({"error": f"{type(exc).__name__}: {exc}"})
    pd.DataFrame(units).to_csv(os.path.join(a.out, "k1_s0_scope_precheck_units.csv"), index=False)
    meta = {"script": "scripts/k1_s0_scope.py", "git": _git("rev-parse", "HEAD"),
            "theory_frozen": frozen, "s0_pass_mle": mle, "s0_pass_neural": nn,
            "precheck": pre, "gate_s0_scope": G.GATE_S0_SCOPE,
            "neural_s0_scope": G.NEURAL_S0_SCOPE,
            "rule": "amendment 4 D4 (S0 per gate) + coverage precheck / amendment 5 (a), amendment 7 (S2) "
                    "(precheck per gate scope)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "k1_s0_scope_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
