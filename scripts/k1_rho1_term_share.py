"""python3 -P scripts/k1_rho1_term_share.py"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TERM = ROOT / "results" / "tables" / "k1_rho1_term_expected.csv"
EFFECT = ROOT / "results" / "tables" / "k1_predicted_effect_sizes.csv"
OUT = ROOT / "results" / "tables" / "k1_rho1_term_share.csv"
PROCESSES = ("ou", "cubic")
LABEL = ("report only: complete-panel finite-province rho1 term vs predicted K1a rho1 "
         "contrast; NOT a bound for the observed panel with exits (not modelled)")


def table(term: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    t = term[(term["variant"] == "registered") & term["process"].isin(PROCESSES)]
    e = eff[(eff["quantity"] == "rho1") & eff["process"].isin(PROCESSES)]
    rows = []
    for _, r in t.iterrows():
        for _, c in e[e["process"] == r["process"]].iterrows():
            ae, eff_ = abs(float(r["term_expected"])), abs(float(c["effect"]))
            rows.append({"process": r["process"], "period": r["period"], "pipeline": c["pipeline"],
                         "n": int(r["n"]), "provinces": int(r["P"]),
                         "term_expected_complete_panel": float(r["term_expected"]),
                         "abs_term": ae, "abs_term_estimator_half": ae / 2,
                         "predicted_rho1_contrast": float(c["effect"]),
                         "share_abs_term": ae / eff_, "share_estimator_half": ae / 2 / eff_,
                         "label": LABEL})
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no registered ou/cubic rows found")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = table(pd.read_csv(TERM), pd.read_csv(EFFECT))
    out.to_csv(a.out, index=False)
    print(out.drop(columns="label").to_string(index=False))


if __name__ == "__main__":
    main()
