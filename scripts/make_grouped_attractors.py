"""
Emit the urbanisation-group Fokker-Planck attractor positions.

The manuscript reports a Medium-group attractor shift (xi* pre -> post) in the
Results, in SI Text S11 and in a figure caption, but the numbers had no
generating record: the audit could only mark them "no truth record", which is
not a state a correction table can leave a live claim in. This writes them,
with the reliability counts that decide whether a cell may be quoted at all.

Writes results/tables/grouped_fp_attractors.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_merger import load_panel
from src.saai import compute_saai
from src.fokker_planck import estimate_km_by_group, find_drift_zeros

OUT = "results/tables/grouped_fp_attractors.csv"
MIN_RELIABLE = 3  # bins needed before a drift zero is worth reporting


def main():
    panel = load_panel("panel_scaling")
    if "NonAgriGDP" not in panel.columns:
        panel["NonAgriGDP"] = (panel["RegGDP_Secondary"].fillna(0)
                               + panel["RegGDP_Tertiary"].fillna(0))
    saai = compute_saai(panel, y_col="GrossOutputValue",
                        x_cols=["PopNum", "CultivateArea"])
    # compute_saai returns the residual frame; the grouping needs the
    # urbanisation inputs alongside it.
    for col in ("NonAgriGDP", "RegGDP"):
        if col not in saai.columns and col in panel.columns:
            saai = saai.merge(panel[["CountyCode", "Year", col]],
                              on=["CountyCode", "Year"], how="left")
    groups = estimate_km_by_group(saai, group_col="auto", saai_col="SAAI")

    rows = []
    for name, data in groups.items():
        for period, km in data["km"].items():
            zeros = find_drift_zeros(km)
            stable = [z for z in zeros if z["type"] == "stable"]
            n_reliable = int(km.reliable.sum())
            rows.append({
                "group": name,
                "period": period,
                "n_counties": data["n_counties"],
                "n_pairs": int(km.n_pairs_total),
                "n_reliable_bins": n_reliable,
                "n_stable_zeros": len(stable),
                "xi_star": stable[0]["xi"] if stable else None,
                "xi_star_median_of_stable": (
                    float(pd.Series([z["xi"] for z in stable]).median())
                    if stable else None),
                "identified": bool(stable) and n_reliable >= MIN_RELIABLE,
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
