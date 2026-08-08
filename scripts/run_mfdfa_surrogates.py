"""
Yearly MFDFA surrogate test on the SAAI cross-sections: computes the observed
multifractal width delta_h per year and specification (multi/single) against
phase-randomised surrogates, and writes
results/tables/mfdfa_surrogate_yearly.csv.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.data_merger import load_panel
from src.saai import compute_saai, compute_saai_single_factor
from src.dfa import compute_mfdfa_surrogate_test

panel = load_panel("panel_scaling")
specs = {"multi": compute_saai(panel, y_col="GrossOutputValue", x_cols=["PopNum","CultivateArea"]),
         "single": compute_saai_single_factor(panel)}
rows=[]
for spec, sa in specs.items():
    for y in sorted(sa["Year"].unique()):
        sub = sa[sa["Year"]==y].dropna(subset=["SAAI"]).sort_values("CountyCode")
        if len(sub) < 100: continue
        t0=time.time()
        st = compute_mfdfa_surrogate_test(sub["SAAI"].values)
        if "status" in st: continue
        rows.append({"spec":spec,"year":int(y),"n":len(sub),
                     "delta_h":st["delta_h_obs"],"z":st["z_score"],"p":st["p_value"],
                     "surr_mean":st["mean_surr"],"surr_sd":st["std_surr"]})
        print(f"{spec} {y}: dh={st['delta_h_obs']:.3f} z={st['z_score']:.2f} p={st['p_value']:.3f} ({time.time()-t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/tables/mfdfa_surrogate_yearly.csv", index=False)
for spec in ("multi","single"):
    d = df[df["spec"]==spec]
    print(f"SUMMARY {spec}: {len(d)} years | z median={d['z'].median():.2f} | p<0.05: {(d['p']<0.05).sum()}/{len(d)} | p<0.01: {(d['p']<0.01).sum()}/{len(d)}")
