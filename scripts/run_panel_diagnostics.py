"""
Three robustness analyses on the SAAI transition-pair pipeline.

Input: panel_scaling (the analysis panel).
  multi-factor SAAI  = compute_saai(panel, y_col='GrossOutputValue',
                                    x_cols=['PopNum','CultivateArea'])
  single-factor SAAI = compute_saai_single_factor(panel)
  FP periods: pre 2000-2012, during 2013-2019, post 2020-2023 (config.FP_PERIODS)

1. Increment autocorrelation (memoryless-noise check):
   period-pooled OLS of drift-adjusted noise residuals.
   Drift A(xi) per period = CUBIC OLS of delta_xi on xi (smoother than the
   25-bin KM interpolation; choice stated in output).
   delta_tilde = delta_xi - A_hat(xi); regress delta_tilde_{t+1} on
   delta_tilde_t (consecutive increments within county), NO county
   demeaning, two-way (county + year) clustered SEs via handwritten
   Cameron-Gelbach-Miller sandwich (V_county + V_year - V_intersection;
   intersection cells are singletons here so V_intersection = HC0-type).
   Robustness: cross-county distribution of per-county rho1 vs
   within-county time-permutation null (pre/during, 500 perms).
   -> results/tables/increment_autocorrelation.csv

2. Per-year sample table:
   -> results/tables/yearly_sample_structure.csv (rendered by tab_yearly() in
      scripts/make_tex_tables.py)

3. Excluding-2023 robustness:
   (a) pooled sectoral betas with/without 2023;
   (b) cross-sectional DFA trend (series ends 2021), trend slope
       with/without endpoint years 2020-2021.
   -> results/tables/excluding_2023_robustness.csv

Usage:  python3 scripts/run_panel_diagnostics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from src.data_merger import load_panel
from src.saai import compute_saai, compute_saai_single_factor
from src.fokker_planck import build_transition_pairs
from src.scaling import run_scaling_regression
from src.dfa import compute_cross_sectional_dfa_timeseries
from src.config import FP_PERIODS

TAB = "results/tables"
os.makedirs(TAB, exist_ok=True)

SEED = 20260806
N_PERM = 500
MIN_INCREMENTS = 5          # per-county eligibility for the permutation variant
MIN_PAIRS_TEST = 30         # below this, mark the pooled test "insufficient"

PINNED_PAIRS = {            # reference pair counts (panel_scaling pipeline)
    "multi":  {"pre": 5860, "during": 1734, "post": 91},
    "single": {"pre": 8696, "during": 4506, "post": 1331},
}


# ════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════

def cubic_drift_fit(pairs: pd.DataFrame) -> dict:
    """Cubic OLS of delta_xi on xi_t. Returns coefs + stable fixed point info."""
    xi = pairs["xi_t"].values
    d = pairs["delta_xi"].values
    X = np.column_stack([np.ones_like(xi), xi, xi**2, xi**3])
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    fitted = X @ coef

    # Stable zero of A(xi): real root with A'(root) < 0, inside data support,
    # closest to the median state (the attractor xi*).
    roots = np.roots(coef[::-1])            # numpy wants highest degree first
    real = roots[np.abs(roots.imag) < 1e-8].real
    lo, hi = np.percentile(xi, 1), np.percentile(xi, 99)
    aprime = lambda x: coef[1] + 2 * coef[2] * x + 3 * coef[3] * x**2
    stable = [r for r in real if lo <= r <= hi and aprime(r) < 0]
    if stable:
        xi_star = min(stable, key=lambda r: abs(r - np.median(xi)))
        a_prime_star = aprime(xi_star)
    else:
        xi_star, a_prime_star = np.nan, np.nan
    return {"coef": coef, "fitted": fitted, "xi_star": xi_star,
            "A_prime_at_xistar": a_prime_star}


def _cluster_meat(Xu: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sum of outer products of within-cluster score sums."""
    df = pd.DataFrame(Xu)
    sums = df.groupby(groups).sum().values
    return sums.T @ sums


def twoway_cluster_ols(y: np.ndarray, x: np.ndarray,
                       g_county: np.ndarray, g_year: np.ndarray) -> dict:
    """
    OLS y = a + rho*x, SE via Cameron-Gelbach-Miller two-way clustering
    (county + year - intersection).  Each (county, year) cell is a singleton
    here, so the intersection term is the White/HC0 meat.
    Small-sample factor G/(G-1) * (n-1)/(n-k) applied per component.
    Inference: t distribution with df = min(G_county, G_year) - 1.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    k = X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]

    def component(groups):
        G = len(pd.unique(groups))
        c = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
        return c * (XtX_inv @ _cluster_meat(Xu, groups) @ XtX_inv), G

    Vc, Gc = component(g_county)
    Vy, Gy = component(g_year)
    inter = pd.Series(list(zip(g_county, g_year))).astype(str).values
    Vi, _ = component(inter)          # singletons -> HC-type
    V = Vc + Vy - Vi

    var_rho = V[1, 1]
    method = "CGM two-way (county+year)"
    if var_rho <= 0:                  # non-PSD fallback (rare)
        var_rho = max(Vc[1, 1], Vy[1, 1])
        method = "fallback max(one-way county, one-way year)"
    se = float(np.sqrt(var_rho))
    dof = min(Gc, Gy) - 1
    t = beta[1] / se
    p = 2 * stats.t.sf(abs(t), df=dof)
    return {"rho": float(beta[1]), "se": se, "p": float(p), "df": int(dof),
            "G_county": int(Gc), "G_year": int(Gy), "se_method": method}


def consecutive_increment_pairs(pairs: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Join (delta_tilde at Year_t) with (delta_tilde at Year_t + lag), same county."""
    a = pairs[["CountyCode", "Year_t", "dtilde"]].copy()
    b = a.rename(columns={"Year_t": "Year_lead", "dtilde": "dtilde_lead"})
    a["Year_lead"] = a["Year_t"] + lag
    m = a.merge(b, on=["CountyCode", "Year_lead"], how="inner")
    return m.rename(columns={"dtilde": "dtilde_lag"})


# ════════════════════════════════════════════════════════════════════
# Increment autocorrelation
# ════════════════════════════════════════════════════════════════════

def run_r14(saai: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Drift-adjusted increment autocorrelation "
          "(drift = cubic OLS of delta_xi on xi; NO county demeaning)")
    print("=" * 70)

    rows = []
    perm_results = {}
    rng = np.random.default_rng(SEED)

    for spec in ("single", "multi"):
        for period in ("pre", "during", "post"):
            pairs = build_transition_pairs(saai[spec], period)
            n_raw = len(pairs)
            drift = cubic_drift_fit(pairs)
            pairs = pairs.copy()
            pairs["dtilde"] = pairs["delta_xi"].values - drift["fitted"]
            B_bar = float(np.mean(pairs["delta_xi"].values ** 2))

            cp1 = consecutive_increment_pairs(pairs, lag=1)
            cp2 = consecutive_increment_pairs(pairs, lag=2)
            row = {
                "period": period, "spec": spec,
                "n_transition_pairs": n_raw,
                "n_consec_pairs": len(cp1),
                "drift_method": "cubic OLS",
                "B_bar": B_bar,
                "xi_star": drift["xi_star"],
                "A_prime_at_xistar": drift["A_prime_at_xistar"],
            }
            if len(cp1) < MIN_PAIRS_TEST:
                row.update({"rho1": np.nan, "se": np.nan, "p": np.nan,
                            "status": f"insufficient ({len(cp1)} pairs)"})
                rows.append(row)
                print(f"  [{spec:6s}|{period:6s}] INSUFFICIENT: "
                      f"{len(cp1)} consecutive-increment pairs")
                continue

            res = twoway_cluster_ols(
                cp1["dtilde_lead"].values, cp1["dtilde_lag"].values,
                cp1["CountyCode"].values, cp1["Year_lead"].values)
            row.update({"rho1": res["rho"], "se": res["se"], "p": res["p"],
                        "ci_lo": res["rho"] - 1.96 * res["se"],
                        "ci_hi": res["rho"] + 1.96 * res["se"],
                        "se_method": res["se_method"],
                        "G_county": res["G_county"], "G_year": res["G_year"],
                        "status": "ok"})
            # lag-2 diagnostic (MA(1) measurement error predicts rho2 ~ 0)
            if len(cp2) >= MIN_PAIRS_TEST:
                res2 = twoway_cluster_ols(
                    cp2["dtilde_lead"].values, cp2["dtilde_lag"].values,
                    cp2["CountyCode"].values, cp2["Year_lead"].values)
                row.update({"rho2": res2["rho"], "se_rho2": res2["se"],
                            "p_rho2": res2["p"], "n_lag2_pairs": len(cp2)})
            # KM drift-slope contamination bound: |rho1| * B_bar
            row["drift_slope_bias"] = abs(res["rho"]) * B_bar
            row["bias_over_Aprime"] = (abs(res["rho"]) * B_bar
                                       / abs(drift["A_prime_at_xistar"])
                                       if np.isfinite(drift["A_prime_at_xistar"])
                                       else np.nan)
            rows.append(row)
            print(f"  [{spec:6s}|{period:6s}] rho1 = {res['rho']:+.4f} "
                  f"(SE {res['se']:.4f}, p = {res['p']:.4f}, "
                  f"n = {len(cp1):,}, {res['se_method']})"
                  f"   rho2 = {row.get('rho2', float('nan')):+.4f}")

            # Permutation robustness variant: single-factor, pre/during only
            if spec == "single" and period in ("pre", "during"):
                perm_results[period] = _county_perm_test(pairs, rng)

    df = pd.DataFrame(rows)
    for period, pr in perm_results.items():
        m = (df["period"] == period) & (df["spec"] == "single")
        for k, v in pr.items():
            df.loc[m, k] = v
    out = f"{TAB}/increment_autocorrelation.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # Bias note
    print("\n--- KM bias note (drift-slope contamination ~ |rho1| x B_bar) ---")
    for _, r in df[df["status"] == "ok"].iterrows():
        sig = "SIGNIFICANT" if r["p"] < 0.05 else "not significant"
        print(f"  {r['spec']:6s}|{r['period']:6s}: rho1 = {r['rho1']:+.4f} "
              f"({sig}, p = {r['p']:.4f}); B_bar = {r['B_bar']:.4f} -> "
              f"bias ~ {r['drift_slope_bias']:.5f} "
              f"vs |A'(xi*)| = {abs(r['A_prime_at_xistar']):.4f} "
              f"(ratio {r['bias_over_Aprime']:.3f})")
    return df


def _county_perm_test(pairs: pd.DataFrame, rng) -> dict:
    """
    Cross-county distribution of per-county rho1 (no demeaning:
    rho1_c = sum d_t d_{t+1} / sum d_t^2 over consecutive increments)
    vs within-county time-permutation null (shuffle each county's dtilde
    across its observed year slots).  Statistic = cross-county median.
    """
    counties = []
    for code, g in pairs.groupby("CountyCode"):
        g = g.sort_values("Year_t")
        if len(g) < MIN_INCREMENTS:
            continue
        years = g["Year_t"].values
        vals = g["dtilde"].values
        idx = np.where(np.diff(years) == 1)[0]     # consecutive-increment pairs
        if len(idx) < 2:
            continue
        counties.append((vals, idx))

    def stat(shuffle: bool) -> float:
        rhos = []
        for vals, idx in counties:
            v = rng.permutation(vals) if shuffle else vals
            denom = np.sum(v[idx] ** 2)
            if denom > 0:
                rhos.append(np.sum(v[idx] * v[idx + 1]) / denom)
        return float(np.median(rhos))

    T_obs = stat(shuffle=False)
    null = np.array([stat(shuffle=True) for _ in range(N_PERM)])
    p_lo = (1 + np.sum(null <= T_obs)) / (N_PERM + 1)
    p_hi = (1 + np.sum(null >= T_obs)) / (N_PERM + 1)
    p_perm = min(1.0, 2 * min(p_lo, p_hi))
    print(f"      permutation variant: median per-county rho1 = {T_obs:+.4f}, "
          f"null median = {np.median(null):+.4f}, p_perm = {p_perm:.4f} "
          f"({len(counties)} counties, {N_PERM} perms)")
    return {"perm_median_rho1": T_obs, "perm_null_median": float(np.median(null)),
            "perm_p": p_perm, "perm_n_counties": len(counties),
            "perm_n": N_PERM}


# ════════════════════════════════════════════════════════════════════
# Per-year sample table
# ════════════════════════════════════════════════════════════════════

def run_r22(panel_full: pd.DataFrame, panel: pd.DataFrame,
            saai: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Per-year sample table (2000-2023)")
    print("=" * 70)

    years = list(range(2000, 2024))

    def eligible(df, cols):
        m = pd.Series(True, index=df.index)
        for c in cols:
            m &= df[c].notna() & (df[c] > 0)
        return df[m].groupby("Year")["CountyCode"].nunique()

    n_master = panel_full.groupby("Year")["CountyCode"].nunique()
    n_scaling = panel.groupby("Year")["CountyCode"].nunique()
    n_multi = eligible(panel, ["GrossOutputValue", "PopNum", "CultivateArea"])
    n_single = eligible(panel, ["GrossOutputValue", "PopNum"])

    pair_year, pair_period = {}, {}
    for spec in ("multi", "single"):
        allp = []
        pair_period[spec] = {}
        for period in ("pre", "during", "post"):
            p = build_transition_pairs(saai[spec], period)
            pair_period[spec][period] = len(p)
            allp.append(p)
        pair_year[spec] = pd.concat(allp).groupby("Year_t").size()

    rows = []
    for y in years:
        rows.append({
            "row_type": "year", "Year": y,
            "n_master": int(n_master.get(y, 0)),
            "n_scaling": int(n_scaling.get(y, 0)),
            "n_saai_multi": int(n_multi.get(y, 0)),
            "n_saai_single": int(n_single.get(y, 0)),
            "pairs_multi": int(pair_year["multi"].get(y, 0)),
            "pairs_single": int(pair_year["single"].get(y, 0)),
        })
    labels = {"pre": "Pre (2000-2012)", "during": "During (2013-2019)",
              "post": "Post (2020-2023)"}
    for period in ("pre", "during", "post"):
        rows.append({
            "row_type": "period_total", "Year": labels[period],
            "n_master": "", "n_scaling": "", "n_saai_multi": "",
            "n_saai_single": "",
            "pairs_multi": pair_period["multi"][period],
            "pairs_single": pair_period["single"][period],
        })
    df = pd.DataFrame(rows)
    out = f"{TAB}/yearly_sample_structure.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")

    print("\n--- Pinned pair-count verification (panel_scaling) ---")
    all_ok = True
    for spec in ("multi", "single"):
        for period in ("pre", "during", "post"):
            got, want = pair_period[spec][period], PINNED_PAIRS[spec][period]
            ok = got == want
            all_ok &= ok
            print(f"  {spec:6s} {period:6s}: {got:>6,} "
                  f"(pinned {want:,})  {'PASS' if ok else 'FAIL'}")
    print(f"  => {'ALL PASS' if all_ok else 'MISMATCH — investigate'}")
    return df


# ════════════════════════════════════════════════════════════════════
# Excluding-2023 robustness
# ════════════════════════════════════════════════════════════════════

SECTOR_YCOL = {"Secondary": "RegGDP_Secondary", "Tertiary": "RegGDP_Tertiary",
               "NonAgri": "NonAgriGDP", "Primary": "RegGDP_Primary"}


def run_r26(panel: pd.DataFrame, saai_multi: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Excluding-2023 robustness")
    print("=" * 70)

    if "NonAgriGDP" not in panel.columns:
        panel = panel.copy()
        panel["NonAgriGDP"] = (panel["RegGDP_Secondary"].fillna(0)
                               + panel["RegGDP_Tertiary"].fillna(0))

    rows = []
    print("\n(a) pooled sectoral betas (y ~ PopNum, panel_scaling, HC1)")
    for sector, ycol in SECTOR_YCOL.items():
        for label, d in (("full", panel), ("excl2023", panel[panel["Year"] != 2023])):
            r = run_scaling_regression(d, y_col=ycol, x_cols=["PopNum"],
                                       year="pooled")
            rows.append({"analysis": "pooled_beta", "item": sector,
                         "sample": label, "estimate": r.betas["PopNum"],
                         "se": r.se["PopNum"], "n": int(r.n_obs),
                         "p": r.pvalue["PopNum"]})
        b_full = rows[-2]["estimate"]
        b_ex = rows[-1]["estimate"]
        print(f"  {sector:10s}: full = {b_full:.4f} (SE {rows[-2]['se']:.4f}, "
              f"n = {rows[-2]['n']:,}) | excl 2023 = {b_ex:.4f} "
              f"(SE {rows[-1]['se']:.4f}, n = {rows[-1]['n']:,}) | "
              f"shift = {b_ex - b_full:+.4f} "
              f"({abs(b_ex - b_full) / rows[-2]['se']:.2f} SE)")

    print("\n(b) cross-sectional DFA trend (multi-factor SAAI, CountyCode order)")
    dfa_ts = compute_cross_sectional_dfa_timeseries(saai_multi)
    y_min, y_max = int(dfa_ts["Year"].min()), int(dfa_ts["Year"].max())
    has_2023 = 2023 in set(dfa_ts["Year"])
    print(f"  DFA series spans {y_min}-{y_max} ({len(dfa_ts)} years); "
          f"contains 2023: {has_2023} "
          f"-> a 2023 endpoint effect on the DFA trend is "
          f"{'possible' if has_2023 else 'excluded (2023 not in series)'}")
    rows.append({"analysis": "dfa_series", "item": "year_range",
                 "sample": f"{y_min}-{y_max}", "estimate": len(dfa_ts),
                 "se": np.nan, "n": len(dfa_ts), "p": np.nan})

    def trend(d, label):
        sl, ic, rv, pv, se = stats.linregress(d["Year"], d["alpha"])
        rows.append({"analysis": "dfa_trend", "item": label,
                     "sample": f"{int(d['Year'].min())}-{int(d['Year'].max())}",
                     "estimate": sl, "se": se, "n": len(d), "p": pv})
        print(f"  alpha trend {label:28s}: slope = {sl:+.4f}/yr "
              f"(p = {pv:.4f}, n = {len(d)} years)")

    trend(dfa_ts, "full series")
    trend(dfa_ts[~dfa_ts["Year"].isin([2020, 2021])], "excl endpoints 2020-2021")
    trend(dfa_ts[dfa_ts["Year"] != 2021], "excl 2021 only")

    df = pd.DataFrame(rows)
    out = f"{TAB}/excluding_2023_robustness.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return df


# ════════════════════════════════════════════════════════════════════

def main():
    panel = load_panel("panel_scaling")
    panel_full = load_panel("panel_full")

    saai = {
        "multi": compute_saai(panel, y_col="GrossOutputValue",
                              x_cols=["PopNum", "CultivateArea"]),
        "single": compute_saai_single_factor(panel),
    }
    print(f"SAAI rows: multi = {len(saai['multi']):,}, "
          f"single = {len(saai['single']):,}")

    run_r14(saai)
    run_r22(panel_full, panel, saai)
    run_r26(panel, saai["multi"])
    print("\nDone.")


if __name__ == "__main__":
    main()
