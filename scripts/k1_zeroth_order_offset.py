"""python3 scripts/k1_zeroth_order_offset.py

with the level offset (row − truth) per period and the contrast offset
contrast_bias(row_pre, row_during, truth_pre, truth_during) — the amount by
which the row's predicted K1a contrast is shifted when there is no exit at
all.  Province shocks and finite N move the replicate-level values around
these population values; Step 2 reports the offset of every replicate.

Refuses on a dirty tree (rule 1).  Output:
results/tables/k1_zeroth_order_noexit_offset.csv
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import k1_theory as T
from src.k1_contract import (CONTRACT_VERSION, PERIOD_CONTRASTS, QUANTITIES,
                             X_UNITS, contrast_bias, registered_true_params)

CAL = REPO / "src/k1_sim_calibration.json"
OUT = REPO / "results/tables/k1_zeroth_order_noexit_offset.csv"
PERIODS = ("pre", "during")


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                          check=True).stdout.strip()


def registered(cal: dict, process: str, period: str) -> dict:
    """The registered chain (contract registered_true_params, which reads
    the same committed calibration), or for 'cubic_unguarded' the (ii)
    polynomial without its guard (comparison only, not the simulated chain)."""
    if process == "cubic_unguarded":
        return dict(registered_true_params("cubic", period), **T.NO_GUARD)
    return registered_true_params(process, period)


def no_exit_composition(tp: dict, period: str, n: int = 1000) -> tuple:
    std = T.stationary_standardisation(tp, n=n)
    m, s = float(std["mu_x"].iloc[0]), float(std["sigma_x"].iloc[0])
    rows = [dict(replicate=0, period=period, pipeline="direct", year_t=t, n_stay=n,
                 n_leave=0, n_enter=0, mu_x_t=m, sigma_x_t=s, mu_x_t1=m, sigma_x_t1=s,
                 x_units=X_UNITS, mech_term=0.0, xi_bar_stay=0.0, var_stay=1.0,
                 re_placement="shock", exit_dating="none")
            for t in T.period_transitions(period)]
    return pd.DataFrame(rows), std


def rows_for(tp: dict, period: str, linear: bool) -> dict:
    comp, std = no_exit_composition(tp, period)
    out = {"truth": T.truth(tp, std, period),
           "projection": T.predict(comp, tp, period, std)["estimate"],
           "zeroth_order": T.zeroth_order_observables(comp, tp, period, std),
           "zeroth_order_mbar": T.zeroth_order_observables(comp, tp, period, std,
                                                           form="mbar")}
    if linear:
        out["closed_form"] = T.closed_form_observables(comp, tp, period, std)
    return out


def main() -> None:
    clean = not git("status", "--porcelain")
    if not clean:
        sys.exit("refusing: working tree is not clean (rule 1)")
    cal_bytes = CAL.read_bytes()
    cal = json.loads(cal_bytes)
    meta = {"git_commit": git("rev-parse", "HEAD"), "git_clean": clean,
            "theory_version": T.THEORY_VERSION, "contract_version": CONTRACT_VERSION,
            "calibration_sha256": hashlib.sha256(cal_bytes).hexdigest(),
            "calibration_saai_spec": cal.get("saai_spec"),
            "pipeline": "population (cell (a), no exit; pipeline-free)",
            "cell": "a (no exit), stationary complete panel per period"}
    recs = []
    plan = [("ou", "(i) OU, unguarded", PERIODS, True),
            ("cubic", "(ii) cubic, guarded as simulated", PERIODS, False),
            ("cubic_unguarded", "(ii) cubic, UNGUARDED (not the simulated chain)",
             ("pre",), False)]
    for proc, label, periods, linear in plan:
        res = {}
        for per in periods:
            tp = registered(cal, proc, per)
            res[per] = rows_for(tp, per, linear)
            tr = res[per]["truth"]
            for row, vals in res[per].items():
                for q in QUANTITIES:
                    v, t = float(vals[q]), float(tr[q])
                    recs.append({"process": proc, "process_label": label,
                                 "guarded": tp["guard_lo"] is not None, "kind": "level",
                                 "period": per, "row": row, "quantity": q, "value": v,
                                 "truth": t, "offset": v - t,
                                 "rel_offset": (v - t) / t if t != 0 else np.nan,
                                 "contrast_type": ""})
        if len(periods) == 2:
            for row in res["pre"]:
                if row == "truth":
                    continue
                cb = contrast_bias(res["pre"][row], res["during"][row],
                                   res["pre"]["truth"], res["during"]["truth"])
                for q in QUANTITIES:
                    recs.append({"process": proc, "process_label": label,
                                 "guarded": proc == "cubic", "kind": "contrast",
                                 "period": "during-pre", "row": row, "quantity": q,
                                 "value": np.nan, "truth": np.nan, "offset": cb[q],
                                 "rel_offset": np.nan,
                                 "contrast_type": PERIOD_CONTRASTS[q]})
    df = pd.DataFrame(recs).assign(**meta)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.10g")
    show = df[(df["row"] == "zeroth_order")
              & df["quantity"].isin(["xi_trap", "two_delta_V", "tau_mfpt", "rho1"])]
    print(show[["process", "kind", "period", "quantity", "value", "truth", "offset",
                "rel_offset"]].to_string(index=False))
    print(f"wrote {OUT} at {meta['git_commit'][:10]}")


if __name__ == "__main__":
    main()
