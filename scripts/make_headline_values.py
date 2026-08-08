"""
Emit the headline scaling values that both the manuscript text and the
Results figures print, so the two cannot drift apart.

Two of these had been computed independently in the two places and disagreed:

  - the sectoral exponents: the text quoted means of the 24 yearly
    cross-sectional estimates for primary/tertiary/secondary but the pooled
    estimate for non-agricultural, mixing estimators inside one sentence,
    while Figure 1(a) printed a third pair (the plotted cross-section and a
    year-fixed-effects pooled fit);
  - the grain interaction slope: the text reported -0.70 from panel_full,
    Figure 3(b) -1.26 from panel_scaling. panel_scaling is the pinned input
    (CLAUDE.md), so the figure was on the right panel and the text was not.

Writes results/tables/headline_scaling.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_merger import load_panel
from src.scaling import (run_scaling_regression, run_interaction_regression,
                         run_scaling_timeseries)
from src.config import SCALING_EQUATIONS

OUT = "results/tables/headline_scaling.csv"

# Baseline sectoral regressions, in the order the text reports them.
SECTORS = [("Primary", "14j"), ("Tertiary", "14i"),
           ("Secondary", "14h"), ("Non-agricultural", "14a")]


def main():
    panel = load_panel("panel_scaling")
    rows = []

    for label, eq in SECTORS:
        cfg = SCALING_EQUATIONS[eq]
        r = run_scaling_regression(panel, y_col=cfg["y"], x_cols=cfg["x"],
                                   year="pooled")
        rows.append({
            "quantity": f"beta_{label.lower()}", "estimator": "pooled OLS",
            "panel": "panel_scaling", "equation": eq,
            "value": r.betas["PopNum"], "se": r.se["PopNum"],
            "ci_lower": r.ci_lower["PopNum"], "ci_upper": r.ci_upper["PopNum"],
            "n": int(r.n_obs),
        })

    # The yearly means the text used to quote, kept so the two estimators can
    # be compared rather than silently swapped.
    ts = run_scaling_timeseries(panel)
    ts = ts[ts["Variable"] == "PopNum"]
    for label, eq in SECTORS:
        s = ts[ts["Equation"] == eq]
        if not len(s):
            continue
        rows.append({
            "quantity": f"beta_{label.lower()}", "estimator": "mean of yearly",
            "panel": "panel_scaling", "equation": eq,
            "value": s["Beta"].mean(), "se": s["Beta"].std() / len(s) ** 0.5,
            "ci_lower": None, "ci_upper": None, "n": len(s),
        })

    gr = run_interaction_regression(panel, y_col="GrainOutput")
    rows.append({
        "quantity": "beta1_grain_interaction", "estimator": "interaction OLS",
        "panel": "panel_scaling", "equation": "eq:interaction",
        "value": gr["beta1"], "se": gr["se_beta1"],
        "ci_lower": None, "ci_upper": None, "n": int(gr["N"]),
        "p_value": gr["p_beta1"],
    })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
