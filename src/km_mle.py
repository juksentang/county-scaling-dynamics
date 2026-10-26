"""
Parametric Euler pseudo-likelihood estimation of the SAAI Langevin dynamics.

Upgrade path over binned Kramers-Moyal (Friedrich-Kleinhans lineage):
binned KM uses only local data per bin (high variance); the parametric MLE
pools the full sample through smooth coefficient functions, and regime
change is tested with a likelihood-ratio test on the full sample rather
than by comparing per-period derived quantities.

Model (Δt = 1 year, Euler transition density):

    Δξ | ξ  ~  N( A(ξ),  B(ξ) )
    A(ξ) = a0 + a1 ξ + a2 ξ² + a3 ξ³          (nests OU: a2 = a3 = 0)
    log B(ξ) = c0 + c1 ξ + c2 ξ²              (positivity guaranteed;
                                               c1 ≠ 0 captures the
                                               poverty-side asymmetry that
                                               the B-ratio measures)

Inference caveat (stated in Methods/SI): increments are not independent
within counties, so the χ² p-value of the LRT is approximate; the primary
uncertainty statements are county-block bootstrap CIs of the derived
quantities and of their period differences, which respect the dependence.
Both the binned-KM and MLE estimates are reported.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

N_PARAMS = 7


def _A(params: np.ndarray, xi: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = params[:4]
    return a0 + a1 * xi + a2 * xi**2 + a3 * xi**3


def _logB(params: np.ndarray, xi: np.ndarray) -> np.ndarray:
    c0, c1, c2 = params[4:]
    return np.clip(c0 + c1 * xi + c2 * xi**2, -20.0, 20.0)


def negloglik(params: np.ndarray, xi: np.ndarray, dxi: np.ndarray) -> float:
    mu = _A(params, xi)
    logB = _logB(params, xi)
    return float(np.sum(0.5 * (np.log(2 * np.pi) + logB
                               + (dxi - mu) ** 2 * np.exp(-logB))))


def init_params(xi: np.ndarray, dxi: np.ndarray) -> np.ndarray:
    """Moment-based warm start: OLS for A, log-squared-residual OLS for B."""
    X = np.column_stack([np.ones_like(xi), xi, xi**2, xi**3])
    a, *_ = np.linalg.lstsq(X, dxi, rcond=None)
    resid2 = np.clip((dxi - X @ a) ** 2, 1e-10, None)
    Xc = np.column_stack([np.ones_like(xi), xi, xi**2])
    c, *_ = np.linalg.lstsq(Xc, np.log(resid2), rcond=None)
    c = c.copy()
    c[0] += 1.27
    return np.concatenate([a, c])


def fit_mle(xi: np.ndarray, dxi: np.ndarray,
            x0: np.ndarray | None = None) -> dict:
    """Fit the 7-parameter Euler pseudo-likelihood model."""
    if x0 is None:
        x0 = init_params(xi, dxi)
    res = optimize.minimize(negloglik, x0, args=(xi, dxi), method="L-BFGS-B")
    if not res.success:
        res2 = optimize.minimize(negloglik, init_params(xi, dxi),
                                 args=(xi, dxi), method="Nelder-Mead",
                                 options={"maxiter": 5000, "xatol": 1e-8})
        if res2.fun < res.fun:
            res = res2
    return {"params": res.x, "nll": float(res.fun), "n": len(xi),
            "converged": bool(res.success)}


def _limit_observables(params, grid, V, B, xi_data, lo_s, hi_s,
                       xi_trap, xi_target, out):
    """degenerate="limit" branch of derived_observables (see its docstring).

    One formula for every path length: ΔV and τ are integrated over the
    node set {trap, grid nodes strictly inside the path, target}, so the end
    fractions are kept and a grid node entering the path adds a segment of
    zero width.  Every observable is therefore continuous in the trap
    position (up to the data granularity of mean_diffusion), on both sides
    of the one-grid-step threshold."""
    lo, hi = min(xi_trap, xi_target), max(xi_trap, xi_target)
    inside = grid[(grid > lo) & (grid < hi)]
    p = np.concatenate([[xi_trap], inside if xi_trap <= xi_target
                        else inside[::-1], [xi_target]])
    r = _A(params, p) / np.exp(_logB(params, p))
    v_rel = -np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(p))])
    out["xi_trap"] = float(xi_trap)
    out["delta_V"] = float(max(0.0, v_rel.max()))

    xi_w = xi_data[(xi_data >= lo_s) & (xi_data <= hi_s)]
    w_mask = (xi_w >= lo) & (xi_w <= hi)
    if w_mask.any():
        out["mean_diffusion"] = float(np.exp(_logB(params, xi_w[w_mask])).mean())
    elif hi > lo:
        q = np.concatenate([[lo], inside, [hi]])
        out["mean_diffusion"] = float(np.trapezoid(np.exp(_logB(params, q)), q)
                                      / (hi - lo))
    else:
        out["mean_diffusion"] = float(np.exp(_logB(params, np.array([lo])))[0])
    out["gamma"] = 2 * out["delta_V"] / out["mean_diffusion"]
    neg, pos = xi_w < 0, xi_w >= 0
    if neg.any() and pos.any():
        out["B_ratio_w"] = float(np.exp(_logB(params, xi_w[neg])).mean()
                                 / np.exp(_logB(params, xi_w[pos])).mean())

    if xi_trap <= xi_target:
        g, Vg, Bg, x0, xb = grid, V, B, xi_trap, xi_target
    else:
        g, Vg, Bg = -grid[::-1], V[::-1], B[::-1]
        x0, xb = -xi_trap, -xi_target
    f_in = np.exp(-2.0 * (Vg - Vg.min())) / Bg
    inner = np.concatenate([[0.0], np.cumsum(
        0.5 * (f_in[1:] + f_in[:-1]) * np.diff(g))])
    outer = 2.0 * np.exp(2.0 * (Vg - Vg.min())) * inner
    seg = (g > x0) & (g < xb)
    q = np.concatenate([[x0], g[seg], [xb]])
    oq = np.concatenate([[np.interp(x0, g, outer)], outer[seg],
                         [np.interp(xb, g, outer)]])
    out["tau_mfpt"] = float(np.trapezoid(oq, q))
    if (np.count_nonzero((grid >= lo) & (grid <= hi))) < 2:
        out["id_status"] = "subgrid_path"
    return out


def derived_observables(params: np.ndarray,
                        xi_data: np.ndarray,
                        xi_target: float = 0.0,
                        support: tuple[float, float] | None = None,
                        n_grid: int = 600,
                        degenerate: str = "nan") -> dict:
    """Escape observables from the fitted smooth A, B — same definitions as
    compute_escape_observables (trap = stable zero of A; ΔV = path barrier
    of V = −∫A/B toward ξ_target; Gardiner double-integral MFPT with the
    reflecting boundary at the far support edge), evaluated analytically on
    the fitted functions over the 1–99 percentile support of the data.
    """
    if degenerate not in ("nan", "limit"):
        raise ValueError("degenerate must be 'nan' or 'limit'")
    if support is None:
        support = (np.percentile(xi_data, 1), np.percentile(xi_data, 99))
    lo_s, hi_s = support
    grid = np.linspace(lo_s, hi_s, n_grid)
    A = _A(params, grid)
    B = np.exp(_logB(params, grid))

    out = {"xi_trap": np.nan, "delta_V": np.nan, "mean_diffusion": np.nan,
           "gamma": np.nan, "tau_mfpt": np.nan, "B_ratio_w": np.nan,
           "id_status": "failed"}

    sign_change = np.where(np.diff(np.sign(A)) != 0)[0]
    stable_roots = []
    for i in sign_change:
        if A[i] > 0 and A[i + 1] < 0:
            frac = A[i] / (A[i] - A[i + 1])
            stable_roots.append(grid[i] + frac * (grid[i + 1] - grid[i]))
    if stable_roots:
        xi_trap = stable_roots[0]
        out["id_status"] = "stable_zero"
    else:
        xi_trap = grid[np.argmin(np.abs(A))]
        out["id_status"] = "fallback_argminA"

    ratio = A / B
    V = -np.concatenate([[0.0], np.cumsum(
        0.5 * (ratio[1:] + ratio[:-1]) * np.diff(grid))])
    V -= V.min()

    lo, hi = (xi_trap, xi_target) if xi_trap <= xi_target else (xi_target, xi_trap)
    if degenerate == "limit":
        return _limit_observables(params, grid, V, B, xi_data, lo_s, hi_s,
                                  float(xi_trap), float(xi_target), out)
    path = (grid >= lo) & (grid <= hi)
    if path.sum() < 2:
        out["id_status"] = "degenerate_path"
        return out
    V_trap = np.interp(xi_trap, grid, V)
    out["xi_trap"] = float(xi_trap)
    out["delta_V"] = float(V[path].max() - V_trap)

    xi_w = xi_data[(xi_data >= lo_s) & (xi_data <= hi_s)]
    w_mask = (xi_w >= lo) & (xi_w <= hi)
    if w_mask.any():
        out["mean_diffusion"] = float(np.exp(_logB(params, xi_w[w_mask])).mean())
    else:
        out["mean_diffusion"] = float(B.mean())
    out["gamma"] = 2 * out["delta_V"] / out["mean_diffusion"]
    neg, pos = xi_w < 0, xi_w >= 0
    if neg.any() and pos.any():
        out["B_ratio_w"] = float(np.exp(_logB(params, xi_w[neg])).mean()
                                 / np.exp(_logB(params, xi_w[pos])).mean())

    if xi_trap <= xi_target:
        g, Vg, Bg, x0, xb = grid, V, B, xi_trap, xi_target
    else:
        g, Vg, Bg = -grid[::-1], V[::-1], B[::-1]
        x0, xb = -xi_trap, -xi_target
    f_in = np.exp(-2.0 * (Vg - Vg.min())) / Bg
    inner = np.concatenate([[0.0], np.cumsum(
        0.5 * (f_in[1:] + f_in[:-1]) * np.diff(g))])
    outer = 2.0 * np.exp(2.0 * (Vg - Vg.min())) * inner
    seg = (g >= x0) & (g <= xb)
    if seg.sum() >= 2:
        out["tau_mfpt"] = float(np.trapezoid(outer[seg], g[seg]))
    return out


def lrt_periods(pairs_by_period: dict[str, pd.DataFrame],
                periods: tuple[str, str] = ("pre", "during")) -> dict:
    """
    Likelihood-ratio tests for regime change between two periods.

    Fits: separate (7+7), fully shared (7), shared-drift (A common, B free:
    4 + 3 + 3), shared-diffusion (B common, A free: 3 + 4 + 4).
    Returns LR statistics with asymptotic χ² p-values (approximate under
    within-county dependence — see module docstring).
    """
    p1, p2 = periods
    d1, d2 = pairs_by_period[p1], pairs_by_period[p2]
    xi1, dx1 = d1["xi_t"].to_numpy(), d1["delta_xi"].to_numpy()
    xi2, dx2 = d2["xi_t"].to_numpy(), d2["delta_xi"].to_numpy()
    xi_all = np.concatenate([xi1, xi2])
    dx_all = np.concatenate([dx1, dx2])

    f1, f2 = fit_mle(xi1, dx1), fit_mle(xi2, dx2)
    nll_sep = f1["nll"] + f2["nll"]

    f_shared = fit_mle(xi_all, dx_all)

    def nll_shared_A(theta):
        pA1 = np.concatenate([theta[:4], theta[4:7]])
        pA2 = np.concatenate([theta[:4], theta[7:10]])
        return negloglik(pA1, xi1, dx1) + negloglik(pA2, xi2, dx2)

    x0 = np.concatenate([f_shared["params"][:4],
                         f1["params"][4:], f2["params"][4:]])
    rA = optimize.minimize(nll_shared_A, x0, method="L-BFGS-B")

    def nll_shared_B(theta):
        pB1 = np.concatenate([theta[3:7], theta[:3]])
        pB2 = np.concatenate([theta[7:11], theta[:3]])
        return negloglik(pB1, xi1, dx1) + negloglik(pB2, xi2, dx2)

    x0 = np.concatenate([f_shared["params"][4:],
                         f1["params"][:4], f2["params"][:4]])
    rB = optimize.minimize(nll_shared_B, x0, method="L-BFGS-B")

    def lr(nll_restricted, df):
        stat = 2.0 * (nll_restricted - nll_sep)
        return {"LR": float(stat), "df": df,
                "p_chi2": float(stats.chi2.sf(stat, df))}

    return {
        "fit_" + p1: f1, "fit_" + p2: f2, "fit_shared": f_shared,
        "any_change": lr(f_shared["nll"], N_PARAMS),
        "drift_change": lr(rA.fun, 4),
        "diffusion_change": lr(rB.fun, 3),
        "n_" + p1: len(xi1), "n_" + p2: len(xi2),
    }


def bootstrap_mle_quantities(saai_df: pd.DataFrame,
                             saai_col: str = "SAAI",
                             n_boot: int = 1000,
                             seed: int = 106,
                             periods: tuple[str, str] = ("pre", "during"),
                             xi_target: float = 0.0,
                             pairs: dict | None = None,
                             degenerate: str = "nan") -> tuple[pd.DataFrame, pd.DataFrame]:
    """County-block bootstrap of MLE-derived quantities and their paired
    period differences (same resampling contract as bootstrap_km_quantities;
    warm-started at the full-sample MLE for speed).
    """
    from src.fokker_planck import build_transition_pairs

    pairs_by_period, county_pairs = {}, {}
    counties_all: set = set()
    for period in periods:
        if pairs is not None:
            period_pairs = pairs[period]
        else:
            period_pairs = build_transition_pairs(saai_df, period, saai_col=saai_col)
        pairs_by_period[period] = period_pairs
        grouped = {code: g[["xi_t", "delta_xi"]].to_numpy()
                   for code, g in period_pairs.groupby("CountyCode")}
        county_pairs[period] = grouped
        counties_all.update(grouped.keys())
    counties = np.array(sorted(counties_all))
    n_c = len(counties)
    rng = np.random.default_rng(seed)

    warm = {p: fit_mle(pairs_by_period[p]["xi_t"].to_numpy(),
                       pairs_by_period[p]["delta_xi"].to_numpy())["params"]
            for p in periods}

    records = []
    for b in range(n_boot):
        drawn = counties[rng.integers(0, n_c, size=n_c)]
        rec = {"replicate": b}
        ok = True
        for p in periods:
            arrs = [county_pairs[p][c] for c in drawn if c in county_pairs[p]]
            if not arrs:
                ok = False
                break
            mat = np.concatenate(arrs, axis=0)
            fit = fit_mle(mat[:, 0], mat[:, 1], x0=warm[p])
            obs = derived_observables(fit["params"], mat[:, 0],
                                      xi_target=xi_target, degenerate=degenerate)
            for k in ("xi_trap", "delta_V", "mean_diffusion", "gamma",
                      "tau_mfpt", "B_ratio_w"):
                rec[f"{k}_{p}"] = obs[k]
            rec[f"id_{p}"] = obs["id_status"]
        if ok:
            records.append(rec)
    draws = pd.DataFrame(records)

    p1, p2 = periods
    rows = []
    for q in ("delta_V", "gamma", "mean_diffusion", "tau_mfpt", "B_ratio_w",
              "xi_trap"):
        v1, v2 = draws[f"{q}_{p1}"], draws[f"{q}_{p2}"]
        diff = (v2 - v1).dropna()
        rows.append({"quantity": f"{q} ({p2}-{p1})", "kind": "difference",
                     "median": diff.median(),
                     "ci_lo": diff.quantile(0.025),
                     "ci_hi": diff.quantile(0.975),
                     "P_gt0": float((diff > 0).mean()), "n_draws": len(diff)})
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (v2 / v1).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"quantity": f"{q} ({p2}/{p1})", "kind": "ratio",
                     "median": ratio.median(),
                     "ci_lo": ratio.quantile(0.025),
                     "ci_hi": ratio.quantile(0.975),
                     "P_gt1": float((ratio > 1).mean()), "n_draws": len(ratio)})
    summary = pd.DataFrame(rows)
    return draws, summary
