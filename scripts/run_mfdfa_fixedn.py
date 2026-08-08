"""
Fixed-sample-size resampling test for the multifractal width: every annual
SAAI cross-section is downsampled to a common size n* and delta_h is
re-estimated, testing whether the temporal decline of the width survives
at fixed sample size (rather than reflecting changing n across years).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, chi2

from src.data_merger import load_panel
from src.saai import compute_saai, compute_saai_single_factor
from src.dfa import compute_mfdfa, compute_mfdfa_surrogate_test

R = 200
SEED = 20260807
REF_YEARS = [2002, 2007, 2012, 2017]
EXCLUDE = {"multi": {2020, 2021}, "single": {2023}}
YEAR_MAX = {"multi": 2019, "single": 2022}

panel = load_panel("panel_scaling")
specs = {"multi": compute_saai(panel, y_col="GrossOutputValue",
                               x_cols=["PopNum", "CultivateArea"]),
         "single": compute_saai_single_factor(panel)}

rng = np.random.default_rng(SEED)
rows, null_rows = [], []
for spec, sa in specs.items():
    series = {}
    for y in sorted(sa["Year"].unique()):
        if y in EXCLUDE[spec] or y > YEAR_MAX[spec]:
            continue
        sub = sa[sa["Year"] == y].dropna(subset=["SAAI"]).sort_values("CountyCode")
        if len(sub) >= 100:
            series[int(y)] = sub["SAAI"].values
    n_star = min(len(v) for v in series.values())
    print(f"[{spec}] {len(series)} years, n* = {n_star}", flush=True)

    for y, x in series.items():
        t0 = time.time()
        dhs = []
        for r in range(R):
            idx = np.sort(rng.choice(len(x), size=n_star, replace=False))
            mf = compute_mfdfa(x[idx])
            if "h_q" in mf and not np.any(np.isnan(mf["h_q"])):
                dhs.append(mf["h_q"][0] - mf["h_q"][-1])
        dhs = np.array(dhs)
        rows.append({"spec": spec, "year": y, "n_full": len(x), "n_star": n_star,
                     "R_ok": len(dhs), "dh_median": np.median(dhs),
                     "dh_q25": np.quantile(dhs, .25), "dh_q75": np.quantile(dhs, .75)})
        print(f"  {spec} {y}: fixed-n dh median={np.median(dhs):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # surrogate null width at n* for reference years (is the null stable?)
    for y in REF_YEARS:
        if y not in series:
            continue
        x = series[y]
        idx = np.sort(rng.choice(len(x), size=n_star, replace=False))
        st = compute_mfdfa_surrogate_test(x[idx], n_surrogates=199, seed=SEED + y)
        if "status" not in st:
            null_rows.append({"spec": spec, "year": y, "n_star": n_star,
                              "null_mean": st["mean_surr"], "null_sd": st["std_surr"],
                              "dh_this_draw": st["delta_h_obs"], "z_this_draw": st["z_score"]})
            print(f"  {spec} null@{y}: mean={st['mean_surr']:.3f} sd={st['std_surr']:.3f}",
                  flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/tables/mfdfa_fixedn_test.csv", index=False)
pd.DataFrame(null_rows).to_csv("results/tables/mfdfa_fixedn_nulls.csv", index=False)

print("\n=== VERDICT INPUTS ===", flush=True)
for spec in specs:
    d = df[df["spec"] == spec]
    early = d[(d["year"] >= 2000) & (d["year"] <= 2006)]["dh_median"]
    late = d[(d["year"] >= 2013) & (d["year"] <= 2019)]["dh_median"]
    u, p = mannwhitneyu(early, late, alternative="greater")
    rho, rp = spearmanr(d["year"], d["dh_median"])
    print(f"{spec}: early(00-06) median-of-medians={early.median():.3f} vs "
          f"late(13-19)={late.median():.3f} | MW one-sided p={p:.4f} | "
          f"Spearman(dh, year) rho={rho:.3f} p={rp:.4f}")

# Fisher combined p across years (from the full-n per-year surrogate table)
sur = pd.read_csv("results/tables/mfdfa_surrogate_yearly.csv")
for spec in ("multi", "single"):
    pv = sur[sur["spec"] == spec]["p"].clip(1e-4, 1).values
    stat = -2 * np.sum(np.log(pv))
    gp = chi2.sf(stat, 2 * len(pv))
    print(f"Fisher combined [{spec}]: chi2={stat:.1f} df={2*len(pv)} p={gp:.2e} "
          f"(cross-year dependence caveat applies)")
