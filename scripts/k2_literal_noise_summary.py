"""Output results/tables/k2_literal_noise_summary.csv, one row per
(level, scope, scheme, conv, quantity):
  level 'cell'    : each scope x scheme x conv x quantity;
  level 'scope'   : each scope x quantity, pooled over schemes and conv;
  level 'overall' : each quantity, pooled over everything (the "<= 1.4e-3"
                    reading for xi_trap is this row's max_abs_diff);
  level 'primary' : the registered primary cell (metro, lineage, limit).
Columns: n_rows, n_draws, max_abs_diff, median_abs_diff, n_over_tol,
argmax_row (the row holding the maximum).

    python3 -P scripts/k2_literal_noise_summary.py [--src CSV] [--out CSV]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "tables" / "k2" / "k2_literal_draw_check.csv"
OUT = ROOT / "results" / "tables" / "k2_literal_noise_summary.csv"
PRIMARY = {"scope": "metro", "scheme": "lineage", "conv": "limit"}
LABEL = "report only: summary of the literal-draw check"


def _row(g: pd.DataFrame, level: str, **key) -> dict:
    i = g["abs_diff"].idxmax()
    w = g.loc[i]
    return {"level": level, **key, "n_rows": len(g), "n_draws": g["draw"].nunique(),
            "max_abs_diff": float(g["abs_diff"].max()),
            "median_abs_diff": float(g["abs_diff"].median()),
            "n_over_tol": int(g["over_tol"].astype(str).str.lower().eq("true").sum()),
            "argmax_row": "|".join(str(w[c]) for c in ("scope", "scheme", "draw", "outcome", "period",
                                                    "panel", "link", "variant", "conv")),
            "label": LABEL}


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["quantity"] != "pairs"].copy()
    d["abs_diff"] = pd.to_numeric(d["abs_diff"], errors="coerce")
    if d["abs_diff"].isna().any():
        raise SystemExit("non-numeric abs_diff in the literal-draw check")
    rows = []
    for (s, sc, c, q), g in d.groupby(["scope", "scheme", "conv", "quantity"]):
        rows.append(_row(g, "cell", scope=s, scheme=sc, conv=c, quantity=q))
    for (s, q), g in d.groupby(["scope", "quantity"]):
        rows.append(_row(g, "scope", scope=s, scheme="all", conv="all", quantity=q))
    for q, g in d.groupby("quantity"):
        rows.append(_row(g, "overall", scope="all", scheme="all", conv="all", quantity=q))
    m = np.ones(len(d), bool)
    for k, v in PRIMARY.items():
        m &= (d[k] == v).to_numpy()
    for q, g in d[m].groupby("quantity"):
        rows.append(_row(g, "primary", **PRIMARY, quantity=q))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=LABEL)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = summarise(pd.read_csv(a.src))
    out.to_csv(a.out, index=False)
    print(out[out.quantity == "xi_trap"].to_string(index=False))


if __name__ == "__main__":
    main()
