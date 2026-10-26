"""Usage:
python3 scripts/k1_calibrate.py
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import optimize

from run_attrition_diagnostic import decompose_year, pairs_v1_restandardised, SAAI_SPECS
from src.config import FP_PERIODS
from src.data_merger import load_panel
from src.km_mle import fit_mle, negloglik
from src.saai import compute_saai

OUT = "src/k1_sim_calibration.json"


def _fit_restricted(xi, dxi, free):
    """MLE with the parameters outside `free` fixed at 0 (same likelihood)."""
    full = fit_mle(xi, dxi)["params"]
    idx = np.array(free)

    def nll(theta):
        p = np.zeros(7)
        p[idx] = theta
        return negloglik(p, xi, dxi)

    x0 = full[idx].copy()
    res = optimize.minimize(nll, x0, method="L-BFGS-B")
    p = np.zeros(7)
    p[idx] = res.x
    return p, bool(res.success)


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def province_icc(g: pd.DataFrame, col: str) -> float:
    """One-way ANOVA intraclass correlation of g[col] by ProvinceCode
    (between-province variance share net of the sampling noise of the
    province means); clipped at 0."""
    v = g[col].to_numpy(float)
    grp = g.groupby("ProvinceCode")[col]
    n_p, m_p = grp.size().to_numpy(float), grp.mean().to_numpy(float)
    N, P = float(len(v)), float(len(n_p))
    if P < 2 or N <= P:
        return float("nan")
    ssb = float(np.sum(n_p * (m_p - v.mean()) ** 2))
    ssw = float(np.sum((v - grp.transform("mean").to_numpy(float)) ** 2))
    msb, msw = ssb / (P - 1), ssw / (N - P)
    n0 = (N - np.sum(n_p ** 2) / N) / (P - 1)
    sb2 = max(0.0, (msb - msw) / n0)
    return float(sb2 / (sb2 + msw))


def main():
    if _git("status", "--porcelain"):
        sys.exit("refusing: the working tree is not clean (amendment 1 (C)); "
                 "commit first so that the JSON records the code that produced it")
    panel_path = "data_clean/panel_scaling.parquet"
    with open(panel_path, "rb") as f:
        panel_sha = hashlib.sha256(f.read()).hexdigest()
    panel = load_panel("panel_scaling")
    spec = SAAI_SPECS["multi"]
    sa = compute_saai(panel, **spec)

    fits = {}
    for period in ("pre", "during"):
        pr = pairs_v1_restandardised(panel, sa, spec, period)
        xi, dxi = pr["xi_t"].to_numpy(), pr["delta_xi"].to_numpy()
        p_full = fit_mle(xi, dxi)
        p_cubic, ok_c = _fit_restricted(xi, dxi, [0, 1, 2, 3, 4])
        p_ou, ok_o = _fit_restricted(xi, dxi, [0, 1, 4])
        fits[period] = {
            "n_pairs": int(len(pr)),
            "support_p01_p99": [float(np.percentile(xi, 1)), float(np.percentile(xi, 99))],
            "full": [float(v) for v in p_full["params"]],
            "cubic_constB": [float(v) for v in p_cubic],
            "ou": [float(v) for v in p_ou],
            "converged": {"full": p_full["converged"], "cubic": ok_c, "ou": ok_o},
        }

    m = panel[(panel["GrossOutputValue"] > 0) & (panel["PopNum"] > 0)
              & (panel["CultivateArea"] > 0)].copy()
    m = m.dropna(subset=["GrossOutputValue", "PopNum", "CultivateArea"])
    lx = np.column_stack([np.log(m["PopNum"]), np.log(m["CultivateArea"])])
    ly = np.log(m["GrossOutputValue"].to_numpy())
    X = np.column_stack([np.ones(len(m)), lx])
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    m["lp"], m["ll"] = lx[:, 0], lx[:, 1]
    cmeans = m.groupby("CountyCode")[["lp", "ll"]].mean()
    m = m.sort_values(["CountyCode", "Year"])
    d = m.groupby("CountyCode")[["lp", "ll", "Year"]].diff()
    d = d[d["Year"] == 1]
    resid_sd = []
    for y, g in m.groupby("Year"):
        if len(g) >= 10:
            Xy = np.column_stack([np.ones(len(g)), g["lp"], g["ll"]])
            yy = np.log(g["GrossOutputValue"].to_numpy())
            cy, *_ = np.linalg.lstsq(Xy, yy, rcond=None)
            resid_sd.append(float((yy - Xy @ cy).std()))
    def robust_sd(v):
        v = v.dropna().to_numpy()
        return float(1.4826 * np.median(np.abs(v - np.median(v))))

    sap = sa.merge(panel[["CountyCode", "Year", "ProvinceCode"]], on=["CountyCode", "Year"])
    shares = {}
    for y, g in sap.groupby("Year"):
        if len(g) >= 50:
            pm = g.groupby("ProvinceCode")["SAAI"].transform("mean")
            shares[int(y)] = float(np.var(pm) / np.var(g["SAAI"]))
    win = [s for y, s in shares.items() if FP_PERIODS["pre"][0] <= y <= FP_PERIODS["during"][1]]

    sp = sap.sort_values(["CountyCode", "Year"]).copy()
    sp["dxi"] = sp.groupby("CountyCode")["SAAI"].diff()
    sp["dy"] = sp.groupby("CountyCode")["Year"].diff()
    sp = sp[sp["dy"] == 1]
    dshare = {}
    for y, g in sp.groupby("Year"):
        if len(g) >= 50:
            dshare[int(y) - 1] = province_icc(g, "dxi")
    dwin = [s_ for t, s_ in dshare.items()
            if FP_PERIODS["pre"][0] <= t < FP_PERIODS["during"][1] and np.isfinite(s_)]

    years = sorted(sa["Year"].unique())
    by_year = pd.DataFrame([decompose_year(sa, t) for t in years[:-1]])
    leaver_means = {int(r.year_t): (None if pd.isna(r.mean_leave_t) else float(r.mean_leave_t))
                    for r in by_year.itertuples()}
    r10 = by_year.set_index("year_t").loc[2010]

    out = {
        "meta": {"git_hash": _git("rev-parse", "HEAD"), "date": date.today().isoformat(), "script": "scripts/k1_calibrate.py",
                 "input": panel_path, "input_sha256": panel_sha},
        "saai_spec": "multi",
        "source": "panel_scaling (pinned) via scripts/k1_calibrate.py; aggregates only",
        "v1_mle_fits": fits,
        "output_equation": {
            "coef_const_logpop_logland": [float(v) for v in coef],
            "yearly_resid_sd_mean": float(np.mean(resid_sd)),
            "county_mean_logx_mean": [float(v) for v in cmeans.mean()],
            "county_mean_logx_cov": np.cov(cmeans.to_numpy().T).tolist(),
            "within_county_dlogx_robust_sd": [robust_sd(d["lp"]), robust_sd(d["ll"])],
        },
        "province_share_by_year": shares,
        "province_share_mean_2000_2019": float(np.mean(win)),
        "province_dxi_icc_by_transition": dshare,
        "province_dxi_icc_mean_2000_2018": float(np.mean(dwin)),
        "leaver_mean_by_transition": leaver_means,
        "mech_c_2010": {"n_stay": int(r10.n_stay), "n_leave": int(r10.n_leave),
                        "mean_leave_t": float(r10.mean_leave_t),
                        "mean_stay_t": float(r10.mean_stay_t)},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
