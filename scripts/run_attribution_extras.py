"""
Attribution extras: E2-P pre-period placebo (with crisis diagnostics and
main-result crisis robustness), B1 endogenous break search, and the B2
event-time figure.

Regenerates: results/tables/placebo_and_crisis_robustness.csv, break_search.csv,
event_time_coefficients.csv, results/figures/event_time_coefficients.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from run_attribution_designs import build_increments, within_ols

CRISIS = {2008, 2009}


def _prep(df, y0, y1, fake_start=None, excl_crisis=False):
    sub = df[(df["Year_t"] >= y0) & (df["Year_t"] <= y1)].copy()
    if excl_crisis:
        sub = sub[~sub["Year_t"].isin(CRISIS) & ~(sub["Year_t"] + 1).isin(CRISIS)]
    during = (sub["Year_t"] >= (fake_start or 2013)).astype(float)
    sub["pov_x"] = (sub["IsPoverty"] == True).astype(float) * during
    sub["neg_x"] = (sub["xi_t"] < 0).astype(float) * during
    return sub


def _fit(sub, terms):
    X = pd.DataFrame({"xi": sub["xi_t"], "xi2": sub["xi_t"] ** 2,
                      "xi3": sub["xi_t"] ** 3,
                      **{t: sub[t] for t in terms}}, index=sub.index)
    return within_ols(sub["log_dxi2"], X, sub["CountyCode"], sub["Year_t"])


def run_placebo(df):
    rows = []
    grid = [("P-S1", "2000-2012", dict(y0=2000, y1=2012, fake_start=2007), ["pov_x"]),
            ("P-S2", "2000-2012", dict(y0=2000, y1=2012, fake_start=2007), ["pov_x", "neg_x"]),
            ("P-S2", "2000-2012 excl 2008-09",
             dict(y0=2000, y1=2012, fake_start=2007, excl_crisis=True), ["pov_x", "neg_x"]),
            ("S1", "2000-2019 excl 2008-09",
             dict(y0=2000, y1=2019, excl_crisis=True), ["pov_x"]),
            ("S2", "2000-2019 excl 2008-09",
             dict(y0=2000, y1=2019, excl_crisis=True), ["pov_x", "neg_x"])]
    for spec, sample, kw, terms in grid:
        sub = _prep(df, **kw)
        res = _fit(sub, terms)
        for t in terms:
            label = t.replace("pov_x", "pov_x_" + ("fake" if "P-" in spec else "during"))\
                     .replace("neg_x", "neg_x_" + ("fake" if "P-" in spec else "during"))
            rows.append({"spec": spec, "sample": sample, "term": label,
                         "coef": res.params[t], "se": res.bse[t],
                         "p": res.pvalues[t], "n": int(res.nobs)})
            print(f"{spec:5s} [{sample}] {label}: {res.params[t]:+.4f} "
                  f"(p={res.pvalues[t]:.4f})")
    pd.DataFrame(rows).to_csv("results/tables/placebo_and_crisis_robustness.csv", index=False)


def run_break_search(df):
    sub = df[(df["Year_t"] >= 2000) & (df["Year_t"] <= 2019)].copy()
    neg = (sub["xi_t"] < 0).astype(float)
    pov = (sub["IsPoverty"] == True).astype(float)
    rows = []
    for margin, dvar in (("state(neg)", neg), ("designation(pov)", pov)):
        for tau in range(2005, 2018):
            sub["m_x_post"] = dvar * (sub["Year_t"] >= tau).astype(float)
            res = _fit(sub, ["m_x_post"])
            b, se = res.params["m_x_post"], res.bse["m_x_post"]
            rows.append({"margin": margin, "tau": tau, "coef": b, "se": se,
                         "wald": (b / se) ** 2, "p": res.pvalues["m_x_post"]})
    b1 = pd.DataFrame(rows)
    b1.to_csv("results/tables/break_search.csv", index=False)
    for margin in b1["margin"].unique():
        m = b1[b1["margin"] == margin]
        star = m.loc[m["wald"].idxmax()]
        print(f"B1 {margin}: tau*={int(star['tau'])} sup-Wald={star['wald']:.1f}")


def run_eventtime(df):
    sub = df[(df["Year_t"] >= 2000) & (df["Year_t"] <= 2019)].copy()
    neg = (sub["xi_t"] < 0).astype(float)
    years = [y for y in range(2000, 2020) if y != 2012]
    for y in years:
        sub[f"negx{y}"] = neg * (sub["Year_t"] == y).astype(float)
    res = _fit(sub, [f"negx{y}" for y in years])
    ev = pd.DataFrame({"year": years,
                       "coef": [res.params[f"negx{y}"] for y in years],
                       "se": [res.bse[f"negx{y}"] for y in years]})
    ev = pd.concat([ev, pd.DataFrame([{"year": 2012, "coef": 0.0, "se": 0.0}])],
                   ignore_index=True).sort_values("year")
    ev.to_csv("results/tables/event_time_coefficients.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.errorbar(ev["year"], ev["coef"], yerr=1.96 * ev["se"], fmt="o-", ms=4,
                lw=1.2, capsize=2.5, color="#2166ac")
    ax.axvline(2012.5, color="#d73027", ls="--", lw=1.2)
    ax.text(2012.7, 1.75, "campaign starts", color="#d73027", fontsize=9)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\log(\Delta\xi)^2$: $\xi<0$ differential (ref. 2012)")
    ax.set_title("Poverty-basin diffusion differential, yearly coefficients")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/figures/event_time_coefficients.png")
    print("B2 event-time figure saved")


if __name__ == "__main__":
    df = build_increments()
    run_placebo(df)
    run_break_search(df)
    run_eventtime(df)
