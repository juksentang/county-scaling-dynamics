"""The gate quantity is ``predict_bias`` (signature fixed by src/k1_contract.py,
contract version 6); the gate functions take ``predict_row`` outputs.  Its inputs are the *observed* composition columns of
one replicate, the replicate's complete-panel standardisation and the true
latent drift/diffusion of one period; it never sees an estimate.

Pipeline of the prediction (each step is a numbered result in the derivation):

1. Lemma 2' (exact, one transition, any Euler latent model): with yearly
   affine standardisation ξ = (x − μ_t)/σ_t, the stayers' increment law is
       E[Δξ | ξ] = A_t(ξ) = [x + A(x) − μ_{t+1}]/σ_{t+1} − ξ,  x = μ_t + σ_t ξ
       Var[Δξ | ξ] = B_t(ξ) = B(x)/σ_{t+1}²
   — a cubic / log-quadratic in ξ again for an unguarded chain, so the
   7-parameter family of src.km_mle is closed under the map
   (``map_params``); _pooled evaluates the map pointwise
   (``transition_moments``), which is exact for the guarded chain as well.  μ, σ are moments of
   the LATENT x (contract X_UNITS).
2. The constant of A_t is anchored on the observed mechanical term
   (Lemma 1: the stayers' mean Δξ equals ``mech_term`` exactly).
3. Stayer design per transition: the OBSERVED mean ``xi_bar_stay`` and
   variance ``var_stay`` of the stayers' ξ_t, shape a location–scale copy of
   the standardised stationary law.  (design="fallback" infers them from
   mech_term by a proportional split and equal within-group variance; it is
   a reported sensitivity row only, never gated.)
4. Pooling: the V0 estimate converges to the KL projection of the
   n_S-weighted mixture of the per-transition Gaussian laws onto the
   7-parameter family (``project``).  rho1 uses the linear OLS projection.
5. Observables of the projection go through src.k1_contract.observables
   (degenerate-path convention "limit").

Truth (``truth``, shared with k1_sim): the SAME pooled projection run on the
complete panel of the replicate — every county, no exit — with the
complete panel's yearly (mu_x, sigma_x).  So under no exit the predicted
estimate equals the truth identically, and predicted and simulated biases
differ only through their estimates.

B_ratio_w is predicted as NaN ("simulation only"); its truth is returned.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.config import FP_PERIODS
from src.k1_contract import (CALIBRATION_FILE, CONTRACT_VERSION, DESIGN_PROCESS_FILE,
                             EDGE_MASS_TOL, EDGE_SD, EXIT_DATINGS, GUARD_KEYS, LOGB_CLIP,
                             QUANTITIES, RE_PLACEMENTS, REGISTERED_CHECK,
                             REPORT_ROWS, ROW_DESIGN,
                             SIMULATION_ONLY, TRUE_PARAM_KEYS,
                             SHOCK_SHARE_COLUMN,
                             TRUE_PROCESS_KEYS, X_UNITS, XI_TARGET, cell_key,
                             check_cell_config, check_registered_true_params,
                             check_true_params, observables,
                             prediction_digest)
from src.km_mle import _logB

THEORY_VERSION = "k1-theory-14"
N_PSEUDO = 20000
REQUIRED = ("period", "year_t", "n_stay", "n_leave", "n_enter", "mu_x_t",
            "sigma_x_t", "mu_x_t1", "sigma_x_t1", "x_units", "mech_term",
            "re_placement", "exit_dating")
OBSERVED_DESIGN = ("xi_bar_stay", "var_stay")
STD_REQUIRED = ("year", "n", "mu_x", "sigma_x", "x_units", SHOCK_SHARE_COLUMN)
TRUTH_ABSORBS_PROVINCE_SHOCK_SHARE = True

REPO = Path(__file__).resolve().parents[1]
FROZEN_FILES = ("src/k1_theory.py", "src/k1_contract.py", "src/km_mle.py",
                "src/config.py",
                CALIBRATION_FILE,
                DESIGN_PROCESS_FILE,
                "scripts/k1_s0.py")
FREEZE_RECORD = "results/records/k1_theory_freeze_2.json"


def params_vector(true_params: dict) -> np.ndarray:
    """The 7 polynomial coefficients (the guard is checked and dropped; use
    ``process_spec`` wherever the chain itself is needed)."""
    return check_true_params(true_params)[0]


def process_spec(true_params: dict) -> tuple:
    """(theta, guard) of the latent chain (contract check_true_params)."""
    return check_true_params(true_params)


def _poly(theta, x):
    return theta[0] + theta[1] * x + theta[2] * x ** 2 + theta[3] * x ** 3


def _poly_slope(theta, x):
    return theta[1] + 2 * theta[2] * x + 3 * theta[3] * x ** 2


def _edge_slopes(theta, guard):
    lo, hi, fl = guard
    return min(_poly_slope(theta, lo), fl), min(_poly_slope(theta, hi), fl)


def drift(theta: np.ndarray, guard, x) -> np.ndarray:
    """A(x) of the simulated chain: the cubic, continued linearly outside the
    guard with slope min(A'(edge), slope_floor)."""
    x = np.asarray(x, dtype=float)
    if guard is None:
        return _poly(theta, x)
    lo, hi, _ = guard
    s_lo, s_hi = _edge_slopes(theta, guard)
    return (_poly(theta, np.clip(x, lo, hi)) + np.where(x > hi, s_hi * (x - hi), 0.0)
            + np.where(x < lo, s_lo * (x - lo), 0.0))


def drift_slope(theta: np.ndarray, guard, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if guard is None:
        return _poly_slope(theta, x)
    lo, hi, _ = guard
    s_lo, s_hi = _edge_slopes(theta, guard)
    return np.where(x > hi, s_hi, np.where(x < lo, s_lo, _poly_slope(theta, x)))


def log_diffusion(theta: np.ndarray, guard, x) -> np.ndarray:
    """log B(x) of the simulated chain: the quadratic, frozen at the edge
    value outside the guard, clipped to ±LOGB_CLIP."""
    x = np.asarray(x, dtype=float)
    if guard is not None:
        x = np.clip(x, guard[0], guard[1])
    return np.clip(theta[4] + theta[5] * x + theta[6] * x ** 2, -LOGB_CLIP, LOGB_CLIP)


def fixed_points(theta: np.ndarray, guard) -> list:
    """Real fixed points A(x) = 0 of the simulated chain, sorted: the cubic's
    real roots (inside the guard when there is one) and the roots of the
    linear continuations outside it."""
    coeffs = np.trim_zeros(np.asarray(theta[:4], float)[::-1], "f")
    roots = np.roots(coeffs) if len(coeffs) >= 2 else np.array([])
    roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    if guard is None:
        return sorted(roots)
    lo, hi, _ = guard
    out = [r for r in roots if lo <= r <= hi]
    s_lo, s_hi = _edge_slopes(theta, guard)
    a_lo, a_hi = float(_poly(theta, lo)), float(_poly(theta, hi))
    if a_hi * s_hi < 0:
        out.append(hi - a_hi / s_hi)
    if a_lo * s_lo > 0:
        out.append(lo - a_lo / s_lo)
    return sorted(out)


def _compose(coeffs, mu: float, sigma: float, length: int) -> np.ndarray:
    """Coefficients (low→high in ξ) of p(μ + σξ), p given low→high in x."""
    P = np.polynomial.Polynomial(np.asarray(coeffs, dtype=float))
    out = P(np.polynomial.Polynomial([mu, sigma])).coef
    res = np.zeros(length)
    res[:min(length, len(out))] = out[:length]
    return res


def map_params(theta_x: np.ndarray, mu_t: float, sigma_t: float,
               mu_t1: float, sigma_t1: float) -> np.ndarray:
    """Exact ξ-unit (a0..a3, c0..c2) of one transition (Lemma 2')."""
    a0, a1, a2, a3 = theta_x[:4]
    num = _compose([a0 - mu_t1, 1.0 + a1, a2, a3], mu_t, sigma_t, 4) / sigma_t1
    num[1] -= 1.0
    logb = _compose(theta_x[4:], mu_t, sigma_t, 3)
    logb[0] -= 2.0 * np.log(sigma_t1)
    return np.concatenate([num, logb])


def lemma2_linear(a0: float, a1: float, D: float, mu_t: float, sigma_t: float,
                  mu_t1: float, sigma_t1: float) -> dict:
    """Lemma 2 (linear latent, population OLS on stayers)."""
    return {"a0_hat": ((1 + a1) * mu_t + a0 - mu_t1) / sigma_t1,
            "a1_hat": (1 + a1) * sigma_t / sigma_t1 - 1,
            "D_hat": D / sigma_t1 ** 2}


def _fold_into(m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Reflect points into [lo, hi] at both edges, repeatedly (triangle
    wave of period 2(hi − lo)): the image of m under a reflecting boundary."""
    L = hi - lo
    y = np.mod(m - lo, 2.0 * L)
    return lo + np.where(y > L, 2.0 * L - y, y)


def _reflecting_kernel(grid: np.ndarray, mean_next: np.ndarray,
                       sb: np.ndarray) -> np.ndarray:
    lo, hi = float(grid[0]), float(grid[-1])
    mean_next = np.asarray(mean_next, float)
    sb = np.asarray(sb, float)
    if not (np.isfinite(mean_next).all() and np.isfinite(sb).all() and (sb > 0).all()):
        raise ValueError("Euler kernel: non-finite drift or diffusion on the grid; no "
                         "stationary law for this chain")
    m = _fold_into(mean_next, lo, hi)[:, None]
    s = sb[:, None]
    g = grid[None, :]
    K = (stats.norm.pdf(g, m, s) + stats.norm.pdf(g, 2.0 * lo - m, s)
         + stats.norm.pdf(g, 2.0 * hi - m, s))
    tot = K.sum(axis=1, keepdims=True)
    if not (np.isfinite(tot).all() and (tot > 0).all()):
        raise ValueError("Euler kernel: a grid row has no mass after reflection; no "
                         "stationary law for this chain")
    return K / tot


@lru_cache(maxsize=256)
def _stationary_cached(theta_t: tuple, guard_t) -> tuple:
    theta = np.array(theta_t)
    guard = None if guard_t is None else tuple(guard_t)
    fps = fixed_points(theta, guard)
    if not fps:
        raise ValueError("drift has no real fixed point")
    stable = [r for r in fps if -2.0 < float(drift_slope(theta, guard, r)) < 0.0]
    if not stable:
        raise ValueError("Euler chain has no stable fixed point")
    sd = [np.sqrt(np.exp(float(log_diffusion(theta, guard, r)))
                  / (1.0 - (1.0 + float(drift_slope(theta, guard, r))) ** 2))
          for r in stable]
    lo, hi = min(stable) - 12 * max(sd), max(stable) + 12 * max(sd)
    n = 801
    for _ in range(3):
        grid = np.linspace(lo, hi, n)
        dx = grid[1] - grid[0]
        sb = np.sqrt(np.exp(log_diffusion(theta, guard, grid)))
        if dx <= sb.min() / 4 or n >= 2401:
            break
        n = min(2401, int(np.ceil((hi - lo) / (sb.min() / 4))) + 1)
    K = _reflecting_kernel(grid, grid + drift(theta, guard, grid), sb)
    M = K.T - np.eye(n)
    M[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    p = np.clip(np.linalg.solve(M, rhs), 0.0, None)
    p /= p.sum()
    e_lo, e_hi = min(stable) - EDGE_SD * max(sd), max(stable) + EDGE_SD * max(sd)
    edge = float(p[(grid < e_lo) | (grid > e_hi)].sum())
    if not edge <= EDGE_MASS_TOL:
        raise ValueError(
            f"the latent chain is not confined: stationary mass {edge:.3g} lies more "
            f"than {EDGE_SD:g} local s.d. beyond the stable fixed points (tolerance "
            f"{EDGE_MASS_TOL:g}; guard {guard}).  The drift escapes through an "
            "unstable outer root; no truth or prediction exists for this chain "
            "unless it is guarded as simulated (contract GUARD_KEYS)")
    m = float(np.sum(p * grid))
    s = float(np.sqrt(np.sum(p * (grid - m) ** 2)))
    return grid, p, m, s, edge


def stationary_law(theta_x: np.ndarray, guard=None):
    """(grid, mass, mean, sd) of the stationary law of the simulated chain
    x_{t+1} = x + A(x) + √B(x) ε (guarded as in contract GUARD_KEYS when
    guard = (lo, hi, slope_floor) is given).  Raises if the chain is not
    confined (edge mass > EDGE_MASS_TOL) — never returns numbers for it."""
    g = None if guard is None else tuple(float(v) for v in guard)
    return _stationary_cached(tuple(float(v) for v in theta_x), g)[:4]


def stationary_edge_mass(theta_x: np.ndarray, guard=None) -> float:
    g = None if guard is None else tuple(float(v) for v in guard)
    return _stationary_cached(tuple(float(v) for v in theta_x), g)[4]


def _std_grid(grid, p, m, s, tol: float = 1e-12):
    """Standardised stationary grid and masses (mass < tol dropped), with the
    nodes re-standardised exactly under the masses."""
    keep = p > tol
    z, w = (grid[keep] - m) / s, p[keep] / p[keep].sum()
    z = z - np.sum(w * z)
    return z / np.sqrt(np.sum(w * z ** 2)), w


def _std_quantile_nodes(grid, p, m, s, n: int) -> np.ndarray:
    """n equal-mass quantile nodes of the standardised law, re-standardised
    exactly (mean 0, mean square 1 over the nodes, as ddof=0 in compute_saai)."""
    z = (grid - m) / s
    cdf = np.cumsum(p) - 0.5 * p
    u = (np.arange(n) + 0.5) / n
    nodes = np.interp(u, cdf, z)
    nodes = nodes - nodes.mean()
    return nodes / np.sqrt(np.mean(nodes ** 2))


def stayer_design(n_stay, n_leave, n_enter, mech) -> tuple[float, float, bool]:
    """FALLBACK (ξ̄_S, V_S, valid) of the stayers at t, inferred from mech_term."""
    n_S, n_L, n_E = float(n_stay), float(n_leave), float(n_enter)
    if n_L == 0:
        xbar = 0.0
    elif n_E == 0:
        xbar = -mech
    else:
        xbar = -mech * n_L / (n_L + n_E)
    m_L = -n_S * xbar / n_L if n_L > 0 else 0.0
    V = 1.0 - (n_S * xbar ** 2 + n_L * m_L ** 2) / (n_S + n_L)
    return xbar, V, bool(V > 0)


def _row_design(row, design: str) -> tuple[float, float, bool]:
    if design == "observed":
        xbar, V = float(row.xi_bar_stay), float(row.var_stay)
        return xbar, V, bool(np.isfinite(xbar) and np.isfinite(V) and V > 0)
    return stayer_design(row.n_stay, row.n_leave, row.n_enter, row.mech_term)


def project(nodes: np.ndarray, wts: np.ndarray, A_tgt: np.ndarray,
            B_tgt: np.ndarray) -> np.ndarray:
    """argmin_θ Σ w [½ log B_θ + (B_tgt + (A_tgt − A_θ)²)/(2 B_θ)] — the
    population limit of src.km_mle.fit_mle on the mixture."""
    wts = wts / wts.sum()
    X = np.column_stack([nodes ** k for k in range(4)])
    Xc = X[:, :3]
    sw = np.sqrt(wts)
    a, *_ = np.linalg.lstsq(X * sw[:, None], A_tgt * sw, rcond=None)
    v = B_tgt + (A_tgt - X @ a) ** 2
    c, *_ = np.linalg.lstsq(Xc * sw[:, None], np.log(v) * sw, rcond=None)

    def fun(th):
        r = A_tgt - X @ th[:4]
        lb = Xc @ th[4:]
        e = np.exp(-lb)
        f = np.sum(wts * 0.5 * (lb + (B_tgt + r ** 2) * e))
        g_a = -(wts * r * e) @ X
        g_c = (wts * 0.5 * (1.0 - (B_tgt + r ** 2) * e)) @ Xc
        return f, np.concatenate([g_a, g_c])

    res = optimize.minimize(fun, np.concatenate([a, c]), jac=True,
                            method="L-BFGS-B",
                            options={"ftol": 1e-15, "gtol": 1e-11,
                                     "maxiter": 20000})
    return res.x


def _linear_slope(nodes, wts, A_tgt) -> float:
    wts = wts / wts.sum()
    mx = np.sum(wts * nodes)
    my = np.sum(wts * A_tgt)
    return float(np.sum(wts * (nodes - mx) * (A_tgt - my))
                 / np.sum(wts * (nodes - mx) ** 2))


def transition_moments(theta_x: np.ndarray, guard, xi: np.ndarray, mu_t: float,
                       sigma_t: float, mu_t1: float, sigma_t1: float) -> tuple:
    """Lemma 2' evaluated pointwise (exact for ANY latent chain, guarded or
    not): with x = μ_t + σ_t ξ,
        A_t(ξ) = [x + A(x) − μ_{t+1}]/σ_{t+1} − ξ,   B_t(ξ) = B(x)/σ_{t+1}².
    For an unguarded chain this equals _A / _logB of map_params (test A4b)."""
    x = mu_t + sigma_t * np.asarray(xi, float)
    a = (x + drift(theta_x, guard, x) - mu_t1) / sigma_t1 - xi
    b = np.exp(log_diffusion(theta_x, guard, x)) / sigma_t1 ** 2
    return a, b


def absorbed_diffusion(b_t: np.ndarray, pq: np.ndarray, wH: float) -> np.ndarray:
    if not 0.0 <= wH < 1.0:
        raise ValueError(f"absorbed shock share w·H = {wH!r} outside [0, 1)")
    if wH == 0.0:
        return b_t
    sb = np.sqrt(b_t)
    m = float(np.sum(pq * sb) / np.sum(pq))
    return (1.0 - wH) * b_t + wH * (sb - m) ** 2


def expected_s2_ratio(theta_x: np.ndarray, guard, zq: np.ndarray, pq: np.ndarray,
                      mu_t: float, sigma_t: float, sigma_t1: float, wH: float) -> float:
    if not 0.0 <= wH < 1.0:
        raise ValueError(f"absorbed shock share w·H = {wH!r} outside [0, 1)")
    x = mu_t + sigma_t * np.asarray(zq, float)
    d = x + drift(theta_x, guard, x)
    md = float(np.sum(pq * d))
    bx = np.exp(log_diffusion(theta_x, guard, x))
    m = float(np.sum(pq * np.sqrt(bx)))
    e_s2 = float(np.sum(pq * (d - md) ** 2)) + float(np.sum(pq * bx)) - wH * m * m
    if not (np.isfinite(e_s2) and e_s2 > 0):
        raise ValueError(f"E_t[s²_t+1] = {e_s2!r} is not positive")
    return float(sigma_t1) ** 2 / e_s2


def _wH_of(s: pd.DataFrame, year_t1: int) -> float:
    """w·H of the standardisation (indexed by year) at the year t+1 of a
    transition — the year whose cross-sectional mean absorbs the common
    shock of t → t+1."""
    v = float(s.at[int(year_t1), SHOCK_SHARE_COLUMN])
    if not (np.isfinite(v) and 0.0 <= v < 1.0):
        raise ValueError(f"standardisation {SHOCK_SHARE_COLUMN} = {v!r} at {year_t1} is "
                         "not a share in [0, 1)")
    return v


def _pooled(theta_x: np.ndarray, rows: list[dict], guard=None) -> dict:
    grid, p, m_st, s_st = stationary_law(theta_x, guard)
    zq, pq = _std_grid(grid, p, m_st, s_st)
    n_tot = float(sum(r["n_stay"] for r in rows))
    nodes, wts, A_t, B_t, pseudo = [], [], [], [], []
    for r in rows:
        w = r["n_stay"] / n_tot
        xi = r["xbar"] + np.sqrt(r["V"]) * zq
        a_t, b_t = transition_moments(theta_x, guard, xi, r["mu_t"], r["sigma_t"],
                                      r["mu_t1"], r["sigma_t1"])
        k = expected_s2_ratio(theta_x, guard, zq, pq, *r["cp"], r["wH"])
        a_t = (a_t + xi) * np.sqrt(k) - xi
        a_t = a_t + (r["mech"] - np.sum(pq * a_t))
        b_t = absorbed_diffusion(b_t, pq, r["wH"])
        b_t = b_t * k
        nodes.append(xi)
        wts.append(w * pq)
        A_t.append(a_t)
        B_t.append(b_t)
        k = max(2, int(round(N_PSEUDO * w)))
        pseudo.append(r["xbar"] + np.sqrt(r["V"])
                      * _std_quantile_nodes(grid, p, m_st, s_st, k))
    nodes, wts = np.concatenate(nodes), np.concatenate(wts)
    A_t, B_t = np.concatenate(A_t), np.concatenate(B_t)
    theta = project(nodes, wts, A_t, B_t)
    pseudo = np.concatenate(pseudo)
    obs = observables(theta, pseudo)
    obs["rho1"] = 1.0 + _linear_slope(nodes, wts, A_t)
    return {"obs": obs, "theta": theta, "pseudo": pseudo}


def period_transitions(period: str) -> list[int]:
    """First years t of the transitions t → t+1 inside FP_PERIODS[period]
    (pre: 2000…2011, 12 transitions; during: 2013…2018, 6)."""
    y0, y1 = FP_PERIODS[period]
    return list(range(y0, y1))


def _const(df: pd.DataFrame, col: str):
    vals = df[col].unique()
    if len(vals) != 1:
        raise ValueError(f"{col} must be constant within one call, got {list(vals)}")
    return vals[0]


def _check(composition: pd.DataFrame, period: str) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED if c not in composition.columns]
    if missing:
        raise ValueError(f"composition lacks columns {missing} "
                         f"(contract version {CONTRACT_VERSION})")
    if "replicate" in composition.columns and composition["replicate"].nunique() > 1:
        raise ValueError("composition must hold one replicate")
    if "pipeline" in composition.columns and composition["pipeline"].nunique() > 1:
        raise ValueError("composition must hold one pipeline ('saai' or 'direct')")
    units = _const(composition, "x_units")
    if units != X_UNITS:
        raise ValueError(f"x_units is {units!r}; the mu_x/sigma_x columns must be "
                         f"moments of the latent x ({X_UNITS!r}), derivation section 4.2")
    rep, ex = _const(composition, "re_placement"), _const(composition, "exit_dating")
    if rep not in RE_PLACEMENTS or ex not in EXIT_DATINGS:
        raise ValueError(f"unknown re_placement {rep!r} or exit_dating {ex!r}")
    comp = composition[composition["period"] == period].sort_values("year_t")
    years = [int(y) for y in comp["year_t"]]
    want = period_transitions(period)
    bad = sorted(set(years) - set(want))
    if bad:
        raise ValueError(f"year_t {bad} outside the {period} transitions")
    dup = sorted({y for y in years if years.count(y) > 1})
    if dup:
        raise ValueError(f"year_t {dup} repeated in the {period} composition")
    miss = sorted(set(want) - set(years))
    if miss:
        raise ValueError(f"composition lacks the {period} transitions {miss}: "
                         "one row per transition is required (n_stay = 0 allowed)")
    flags = {"re_placement": rep, "exit_dating": ex,
             "H1": EXIT_DATINGS[ex], "H2": RE_PLACEMENTS[rep]}
    flags["valid_for_law"] = bool(flags["H1"] and flags["H2"])
    return comp, flags


def _match_cell(composition: pd.DataFrame, cell: dict | None) -> dict:
    if cell is None:
        return {"cell_checked": False, "cell_key": None}
    c = check_cell_config(cell)
    if "pipeline" not in composition.columns:
        raise ValueError("composition lacks the pipeline column needed to match the cell")
    for col in ("re_placement", "exit_dating", "pipeline"):
        v = _const(composition, col)
        if v != c[col]:
            raise ValueError(f"composition {col} {v!r} != cell {col} {c[col]!r}: the "
                             "composition and the cell meta disagree")
    return {"cell_checked": True, "cell_key": cell_key(c),
            "valid_for_law_cell": c["valid_for_law"]}


def _is_nonlinear(theta_x: np.ndarray, guard) -> bool:
    return guard is not None or bool(np.any(np.abs(theta_x[[2, 3, 5, 6]]) > 0))


def _check_location(comp: pd.DataFrame, theta_x: np.ndarray, m_st: float,
                    s_st: float, guard=None) -> None:
    """Belt and braces on top of the x_units tag: refuse all-zero means for a
    non-linear model whose stationary mean is not 0 (residual-unit input)."""
    nonlinear = _is_nonlinear(theta_x, guard)
    if (nonlinear and not comp.empty and abs(m_st) > 1e-6 * s_st
            and np.all(np.abs(comp["mu_x_t"].to_numpy(float)) < 1e-9)
            and np.all(np.abs(comp["mu_x_t1"].to_numpy(float)) < 1e-9)):
        raise ValueError(
            "mu_x_t/mu_x_t1 are all 0 but the latent chain's stationary mean is "
            f"{m_st:.4g}: the predictor needs latent-x moments over C_t, not "
            "the mean of the standardised residual (derivation section 4.2)")


def _check_standardisation(std: pd.DataFrame, period: str) -> pd.DataFrame:
    missing = [c for c in STD_REQUIRED if c not in std.columns]
    if missing:
        raise ValueError(f"standardisation lacks columns {missing}")
    if _const(std, "x_units") != X_UNITS:
        raise ValueError("standardisation must be in latent-x units")
    if std["year"].duplicated().any():
        raise ValueError("standardisation repeats a year")
    s = std.set_index("year")
    need = sorted({y for t in period_transitions(period) for y in (t, t + 1)})
    if not set(need) <= set(s.index):
        raise ValueError(f"standardisation lacks years {sorted(set(need) - set(s.index))}")
    if not (s.loc[need, "sigma_x"] > 0).all():
        raise ValueError("standardisation sigma_x must be > 0")
    return s


def _cell_windows(cell: dict) -> dict:
    """{window: range of years} of the cell meta's 'windows' string, e.g.
    'pre:2000-2012;during:2013-2019'."""
    out = {}
    for part in str(cell.get("windows", "")).split(";"):
        name, _, span = part.partition(":")
        lo, _, hi = span.partition("-")
        if not name or not lo.strip().isdigit() or not hi.strip().isdigit() \
                or int(hi) < int(lo):
            raise ValueError(f"cell windows {cell.get('windows')!r} are not "
                             "'name:YYYY-YYYY;...'")
        out[name.strip()] = range(int(lo), int(hi) + 1)
    return out


def _check_complete_panel(std: pd.DataFrame, cell: dict | None, period: str,
                          composition: pd.DataFrame | None = None) -> None:
    """Contracts 8, 9: with the cell meta given, the standardisation must be
    the replicate's COMPLETE panel — every simulated county of each window
    in every year of that window.  The cell's N is the sum over its windows
    of the window's county count (k1_sim contract_cell_meta; the windows of
    a K1a cell have different sizes, so no single per-year n can equal N).
    Refuses unless, for every window of cell['windows'], the table has
    exactly one row per year, n is the same in every year of the window (an
    observed panel loses counties at exits), the window sizes sum to N, and
    for every transition of the composition in this period n_t ≥ n_stay +
    n_leave and n_{t+1} ≥ n_stay + n_enter.  The observed panel's moments
    differ from the complete panel's by exactly the composition effect K1
    measures, so an observed-panel table would give a wrong truth and a
    wrong prediction without any other error.  Without a cell (unit tests,
    documentation tables) nothing is checked here."""
    if cell is None:
        return
    N = int(cell["N"])
    cols = [c for c in ("year", "n", "mu_x", "sigma_x") if c in std.columns]
    tab = std[cols].drop_duplicates()
    tab = tab.assign(year=pd.to_numeric(tab["year"], errors="coerce"),
                     n=pd.to_numeric(tab["n"], errors="coerce"))
    sizes = {}
    for w, years in _cell_windows(cell).items():
        rows = tab[tab["year"].isin(list(years))]
        if sorted(rows["year"].tolist()) != list(years):
            raise ValueError(f"standardisation is not the complete panel of the cell: window "
                             f"{w} needs exactly one row per year {years.start}–"
                             f"{years.stop - 1}, got years {sorted(rows['year'].tolist())}")
        vals = sorted(set(rows["n"].tolist()))
        if len(vals) != 1 or not np.isfinite(vals[0]) or vals[0] <= 0:
            raise ValueError(f"standardisation is not the complete panel of the cell: n must "
                             f"be the same in every year of window {w} (every simulated "
                             f"county), got {vals[:6]}")
        sizes[w] = int(vals[0])
    if sum(sizes.values()) != N:
        raise ValueError(f"standardisation is not the complete panel of the cell: the window "
                         f"sizes {sizes} sum to {sum(sizes.values())}, not the cell's N = {N}")
    if composition is None:
        return
    comp = composition[composition["period"] == period]
    ns = tab.set_index("year")["n"]
    for r in comp.itertuples(index=False):
        t = int(r.year_t)
        if int(r.n_stay) + int(r.n_leave) > int(ns.get(t, -1)) or \
                int(r.n_stay) + int(r.n_enter) > int(ns.get(t + 1, -1)):
            raise ValueError(f"composition {period} {t}: n_stay + n_leave = "
                             f"{int(r.n_stay) + int(r.n_leave)}, n_stay + n_enter = "
                             f"{int(r.n_stay) + int(r.n_enter)} exceed the complete "
                             "panel's n — the standardisation is not the complete panel")


def stationary_standardisation(true_params: dict, years=range(2000, 2024),
                               n: int = 1000, wH: float = 0.0) -> pd.DataFrame:
    _, _, m, s = stationary_law(*process_spec(true_params))
    return pd.DataFrame({"year": list(years), "n": n, "mu_x": m, "sigma_x": s,
                         "x_units": X_UNITS, SHOCK_SHARE_COLUMN: float(wH)})


def _complete_moments(s: pd.DataFrame, t: int) -> tuple:
    return (float(s.at[int(t), "mu_x"]), float(s.at[int(t), "sigma_x"]),
            float(s.at[int(t) + 1, "sigma_x"]))


def n_stable_fixed_points(theta_x: np.ndarray, guard=None) -> int:
    """Number of stable fixed points of the simulated chain (the criterion of
    _stationary_cached: −2 < A'(x*) < 0)."""
    return sum(1 for r in fixed_points(theta_x, guard)
               if -2.0 < float(drift_slope(theta_x, guard, r)) < 0.0)


def is_bistable(theta_x: np.ndarray, guard=None) -> bool:
    return n_stable_fixed_points(theta_x, guard) >= 2


def affine_invariant_two_delta_V(theta_x: np.ndarray, s: pd.DataFrame, period: str,
                                 guard=None) -> dict:
    _, _, m, sd = stationary_law(theta_x, guard)
    ts = period_transitions(period)
    rows = [{"n_stay": float(min(s.at[t, "n"], s.at[t + 1, "n"])),
             "mu_t": m, "sigma_t": sd, "mu_t1": m, "sigma_t1": sd,
             "mech": 0.0, "xbar": 0.0, "V": 1.0, "wH": _wH_of(s, t + 1),
             "cp": (m, sd, sd)} for t in ts]
    st = _pooled(theta_x, rows, guard)
    mu_bar = float(np.mean([float(s.at[t, "mu_x"]) for t in ts]))
    c = (mu_bar - m) / sd
    th = st["theta"]
    shifted = np.concatenate([_compose(th[:4], c, 1.0, 4), _compose(th[4:], c, 1.0, 3)])
    obs = observables(shifted, st["pseudo"] - c)
    return {"two_delta_V": float(obs["two_delta_V"]), "frame_shift_c": float(c),
            "mu_bar": mu_bar, "id_status": obs.get("id_status")}


def stationary_rho1(theta_x: np.ndarray, guard=None) -> float:
    grid, p, m, _ = stationary_law(theta_x, guard)
    nxt = grid + drift(theta_x, guard, grid)
    dx = grid - m
    return float(np.sum(p * dx * (nxt - np.sum(p * nxt))) / np.sum(p * dx * dx))


def _truth_full(theta_x: np.ndarray, std: pd.DataFrame, period: str,
                guard=None) -> dict:
    s = _check_standardisation(std, period)
    rows = [{"n_stay": float(min(s.at[t, "n"], s.at[t + 1, "n"])),
             "mu_t": float(s.at[t, "mu_x"]), "sigma_t": float(s.at[t, "sigma_x"]),
             "mu_t1": float(s.at[t + 1, "mu_x"]),
             "sigma_t1": float(s.at[t + 1, "sigma_x"]),
             "mech": 0.0, "xbar": 0.0, "V": 1.0, "wH": _wH_of(s, t + 1),
             "cp": _complete_moments(s, t)}
            for t in period_transitions(period)]
    out = _pooled(theta_x, rows, guard)
    out["rho1_complete_projection"] = float(out["obs"]["rho1"])
    out["obs"] = {**out["obs"], "rho1": stationary_rho1(theta_x, guard)}
    if is_bistable(theta_x, guard):
        ai = affine_invariant_two_delta_V(theta_x, s, period, guard)
        out["obs"] = {**out["obs"], "two_delta_V": ai["two_delta_V"]}
        out["two_delta_V_affine_invariant"] = ai
    return out


def truth(true_params: dict, standardisation: pd.DataFrame, period: str,
          cell: dict | None = None) -> dict:
    """Contract truth(): {quantity: value} in ξ units, plus 'id_status'.

    true_params must declare the guard (contract GUARD_KEYS); the chain is
    the one simulated, guard included.  Raises — never returns numbers — if
    that chain is not confined (stationary_law).
    """
    _check_complete_panel(standardisation, cell, period)
    theta, guard = process_spec(true_params)
    return dict(_truth_full(theta, standardisation, period, guard)["obs"])


def _nan_result() -> dict:
    return {q: np.nan for q in QUANTITIES}


def rho1_prediction(truth_rho1: float, open_rho1: float, complete_rho1: float) -> float:
    return float(truth_rho1) + (float(open_rho1) - float(complete_rho1))


def predict(composition: pd.DataFrame, true_params: dict, period: str,
            standardisation: pd.DataFrame, design: str = "observed",
            cell: dict | None = None) -> dict:
    """Predicted V0 estimate, truth and bias for one replicate and period."""
    if design not in ("observed", "fallback"):
        raise ValueError("design must be 'observed' or 'fallback'")
    link = _match_cell(composition, cell)
    comp, flags = _check(composition, period)
    _check_complete_panel(standardisation, cell, period, composition)
    if design == "observed":
        miss = [c for c in OBSERVED_DESIGN if c not in comp.columns]
        if miss:
            raise ValueError(f"design='observed' needs columns {miss}")
    theta_x, guard = process_spec(true_params)
    _, _, m_st, s_st = stationary_law(theta_x, guard)
    _check_location(comp, theta_x, m_st, s_st, guard)

    tr = _truth_full(theta_x, standardisation, period, guard)
    truth_obs = dict(tr["obs"])
    s_std = _check_standardisation(standardisation, period)
    diag = {"theory_version": THEORY_VERSION, "contract_version": CONTRACT_VERSION,
            "design": design, **flags, **link, "period": period,
            "replicate": (int(_const(composition, "replicate"))
                          if "replicate" in composition.columns else None),
            "truth_id_status": truth_obs.get("id_status"),
            "truth_path_length": float(abs(truth_obs["xi_trap"] - XI_TARGET)),
            "theta_true_xi": tr["theta"], "guard": guard,
            "fingerprint": SOURCE_FINGERPRINT}
    comp = comp[comp["n_stay"] > 0]
    if comp.empty:
        nan = _nan_result()
        return {"estimate": nan, "truth": truth_obs, "bias": dict(nan),
                "diag": {**diag, "n_transitions": 0}}

    rows, mixed_w = [], 0.0
    n_tot = float(comp["n_stay"].sum())
    for row in comp.itertuples(index=False):
        if not (row.sigma_x_t > 0 and row.sigma_x_t1 > 0):
            return {"estimate": _nan_result(), "truth": truth_obs,
                    "bias": _nan_result(), "diag": {**diag, "invalid": "sigma"}}
        xbar, V, ok = _row_design(row, design)
        if not ok:
            return {"estimate": _nan_result(), "truth": truth_obs,
                    "bias": _nan_result(),
                    "diag": {**diag, "invalid": "V_S<=0", "year_t": row.year_t}}
        if row.n_leave > 0 and row.n_enter > 0:
            mixed_w += row.n_stay / n_tot
        rows.append({"n_stay": float(row.n_stay), "mu_t": row.mu_x_t,
                     "sigma_t": row.sigma_x_t, "mu_t1": row.mu_x_t1,
                     "sigma_t1": row.sigma_x_t1, "mech": float(row.mech_term),
                     "xbar": xbar, "V": V,
                     "wH": _wH_of(s_std, int(row.year_t) + 1),
                     "cp": _complete_moments(s_std, int(row.year_t))})
    pooled = _pooled(theta_x, rows, guard)
    est = dict(pooled["obs"])
    est["rho1"] = rho1_prediction(truth_obs["rho1"], pooled["obs"]["rho1"],
                                  tr["rho1_complete_projection"])
    bias = {q: (est[q] - truth_obs[q]) if q not in SIMULATION_ONLY else np.nan
            for q in QUANTITIES}
    est = {q: (est[q] if q not in SIMULATION_ONLY else np.nan) for q in QUANTITIES} \
        | {"id_status": est["id_status"]}
    w = comp["n_stay"].to_numpy(float) / n_tot
    diag.update({
        "n_transitions": len(rows), "theta_app": pooled["theta"],
        "M_bar": float(np.sum(w * comp["mech_term"].to_numpy(float))),
        "xbar_stay": [r["xbar"] for r in rows], "V_stay": [r["V"] for r in rows],
        "mixed_weight": mixed_w,
        "rho1_open_projection": float(pooled["obs"]["rho1"]),
        "rho1_complete_projection": float(tr["rho1_complete_projection"]),
        "mu_offset_sd": float((np.sum(w * comp["mu_x_t"].to_numpy(float)) - m_st)
                              / s_st)})
    return {"estimate": est, "truth": truth_obs, "bias": bias, "diag": diag}


def predict_bias(composition: pd.DataFrame, true_params: dict, period: str,
                 standardisation: pd.DataFrame, *, design: str = "observed",
                 cell: dict | None = None) -> dict:
    """Contract entry point (src/k1_contract.py): {quantity: V0 estimate − truth}."""
    return predict(composition, true_params, period, standardisation,
                   design=design, cell=cell)["bias"]


def predict_row(row: str, composition: pd.DataFrame, true_params: dict, period: str,
                standardisation: pd.DataFrame, cell: dict) -> dict:
    """The output of one REPORT_ROWS row for one replicate and period, in
    the form the gate functions take (contract 6, pair_predictions):
    {"estimate", "truth", "bias", "diag"}, diag with cell_key, row, design,
    replicate, period and the digest over them and the estimate / truth.
      projection   : predict(design="observed");  fallback: design="fallback";
      closed_form  : closed_form_observables;  zeroth_order /
                     zeroth_order_mbar: zeroth_order_observables (form
                     'first_order' / 'mbar') — each with the truth (and the
                     cell link) of predict(design="observed").
    cell is required (the gates refuse anything not made for their cell)."""
    if row not in REPORT_ROWS:
        raise ValueError(f"unknown row {row!r}")
    if cell is None:
        raise ValueError("predict_row needs the cell meta (amendment 3 (e))")
    registered = check_registered_true_params(str(cell.get("process")), period,
                                              true_params)
    base = predict(composition, true_params, period, standardisation,
                   design=ROW_DESIGN[row], cell=cell)
    tru, diag = base["truth"], dict(base["diag"])
    if row in ("projection", "fallback"):
        est = base["estimate"]
    elif row == "closed_form":
        est = closed_form_observables(composition, true_params, period,
                                      standardisation, cell=cell)
    else:
        est = zeroth_order_observables(composition, true_params, period,
                                       standardisation, cell=cell,
                                       form=ZEROTH_ORDER_FORMS[row])
    est = {q: est[q] for q in QUANTITIES} | {"id_status": est.get("id_status")}
    bias = {q: (float(est[q]) - float(tru[q])) if q not in SIMULATION_ONLY else np.nan
            for q in QUANTITIES}
    diag.update(row=row, design=ROW_DESIGN[row], true_params=registered,
                true_params_check=REGISTERED_CHECK)
    if row in ZEROTH_ORDER_FORMS:
        diag["prl_condition_1_inputs"] = zeroth_order_inputs(
            composition, true_params, period, standardisation, cell)
    diag["digest"] = prediction_digest(diag, est, tru)
    return {"estimate": est, "truth": tru, "bias": bias, "diag": diag}


GOLDEN_RTOL, GOLDEN_ATOL = 1e-4, 1e-9


def _blob_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def source_fingerprint() -> dict:
    return {f: _blob_hash(REPO / f) for f in FROZEN_FILES}


SOURCE_FINGERPRINT = source_fingerprint()


def environment() -> dict:
    import scipy
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__}


NO_GUARD = dict.fromkeys(GUARD_KEYS)
GOLDEN_OU = dict(a0=0.02, a1=-0.125, a2=0.0, a3=0.0, c0=float(np.log(0.3)),
                 c1=0.0, c2=0.0, **NO_GUARD)
GOLDEN_CUBIC = dict(a0=0.05, a1=-0.2, a2=0.03, a3=-0.04, c0=float(np.log(0.3)),
                    c1=-0.1, c2=0.02, **NO_GUARD)
GOLDEN_GUARDED = dict(a0=-0.01, a1=-0.1, a2=0.008, a3=0.003, c0=float(np.log(0.17)),
                      c1=0.0, c2=0.0, guard_lo=-2.3, guard_hi=2.3, slope_floor=-0.09)


def _golden_composition(tp: dict) -> pd.DataFrame:
    """Fixed synthetic window: pre without exits; during with one block
    exit (192 of 385 at mech −0.45) and a σ path (golden outputs only)."""
    _, _, m, s = stationary_law(*process_spec(tp))
    rows = [("pre", t, 300, 0, 0, m, s, m, s, 0.0, 0.0, 1.0)
            for t in period_transitions("pre")]
    dur = period_transitions("during")
    spec = [(300, 0, 0, s, s, 0.0, 0.0, 1.0)] * 3 + [
        (193, 192, 0, s, 0.9 * s, -0.45, 0.45, 0.8),
        (193, 0, 0, 0.9 * s, 0.93 * s, 0.0, 0.0, 1.0),
        (193, 0, 0, 0.93 * s, 0.93 * s, 0.0, 0.0, 1.0)]
    rows += [("during", t, nS, nL, nE, m, st, m, st1, mech, xb, V)
             for t, (nS, nL, nE, st, st1, mech, xb, V) in zip(dur, spec)]
    cols = ["period", "year_t", "n_stay", "n_leave", "n_enter", "mu_x_t",
            "sigma_x_t", "mu_x_t1", "sigma_x_t1", "mech_term", "xi_bar_stay",
            "var_stay"]
    df = pd.DataFrame(rows, columns=cols)
    return df.assign(replicate=0, pipeline="direct", x_units=X_UNITS,
                     re_placement="none", exit_dating="info_t")


def golden_outputs() -> dict:
    out = {}
    for name, tp in (("ou", GOLDEN_OU), ("cubic", GOLDEN_CUBIC),
                     ("guarded", GOLDEN_GUARDED)):
        comp, std = _golden_composition(tp), stationary_standardisation(tp)
        for period in ("pre", "during"):
            rows = {"proj": predict(comp, tp, period, std)["estimate"],
                    "fallback": predict(comp, tp, period, std,
                                        design="fallback")["estimate"],
                    "truth": truth(tp, std, period),
                    "zeroth": zeroth_order_observables(comp, tp, period, std)}
            if name == "ou":
                rows["closed"] = closed_form_observables(comp, tp, period, std)
            for r, d in rows.items():
                for q in QUANTITIES:
                    out[f"{name}.{period}.{r}.{q}"] = float(d[q])
    return out


def _golden_mismatch(want: dict, got: dict) -> list:
    bad = []
    for k, w in want.items():
        g = got.get(k, np.nan)
        same_nan = np.isnan(w) and np.isnan(g)
        if not same_nan and not (abs(g - w) <= GOLDEN_ATOL + GOLDEN_RTOL * abs(w)):
            bad.append((k, w, g))
    return bad


def freeze_record(head: str) -> dict:
    """The theory-freeze record (written and committed by the authors' freeze script, which is not part of this archive)."""
    return {"frozen_commit": head, "theory_version": THEORY_VERSION,
            "contract_version": CONTRACT_VERSION,
            "fp_periods": {k: list(v) for k, v in FP_PERIODS.items()},
            "true_process_keys": list(TRUE_PROCESS_KEYS),
            "guard_constants": {"LOGB_CLIP": LOGB_CLIP, "EDGE_SD": EDGE_SD,
                                "EDGE_MASS_TOL": EDGE_MASS_TOL},
            "environment": environment(), "blobs": source_fingerprint(),
            "golden": golden_outputs()}


def assert_frozen(record_path: str | Path | None = None,
                  check_golden: bool = True) -> dict:
    """Refuse unless the committed freeze record exists and
      - every FROZEN_FILES blob hash, at import time and on disk now, equals
        the recorded one;
      - THEORY_VERSION, CONTRACT_VERSION and FP_PERIODS equal the recorded ones;
      - the golden outputs are reproduced to GOLDEN_RTOL / GOLDEN_ATOL in the
        running environment (a numpy / scipy version different from the
        recorded one is accepted only then, and reported).
    Returns the record plus 'environment_at_run' and 'environment_differs',
    for Step 2's meta."""
    path = Path(record_path) if record_path else REPO / FREEZE_RECORD
    if not path.exists():
        raise RuntimeError(f"no freeze record at {path}: k1_theory is not frozen")
    rec = json.loads(path.read_text())
    now = source_fingerprint()
    for f in FROZEN_FILES:
        want = rec["blobs"].get(f)
        if want is None or now[f] != want or SOURCE_FINGERPRINT[f] != want:
            raise RuntimeError(f"{f} differs from the freeze record "
                               f"(recorded {want}, on disk {now[f]}, imported "
                               f"{SOURCE_FINGERPRINT[f]}): re-run Step 2 in full")
    if rec.get("theory_version") != THEORY_VERSION:
        raise RuntimeError("THEORY_VERSION differs from the freeze record")
    if rec.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("CONTRACT_VERSION differs from the freeze record")
    if rec.get("fp_periods") != {k: list(v) for k, v in FP_PERIODS.items()}:
        raise RuntimeError("FP_PERIODS differs from the freeze record")
    env = environment()
    differs = env != rec.get("environment")
    if check_golden or differs:
        bad = _golden_mismatch(rec.get("golden", {}), golden_outputs())
        if bad or not rec.get("golden"):
            raise RuntimeError(f"golden outputs not reproduced in {env} "
                               f"(recorded in {rec.get('environment')}): {bad[:5]}")
    return {**rec, "environment_at_run": env, "environment_differs": differs}


def closed_form_ou(composition: pd.DataFrame, a1: float, D: float,
                   period: str, design: str = "observed", wH=None,
                   s2_ratio=None) -> dict:
    comp, _ = _check(composition, period)
    comp = comp[comp["n_stay"] > 0]
    w = comp["n_stay"].to_numpy(float)
    w = w / w.sum()
    mech = comp["mech_term"].to_numpy(float)
    k = np.ones(len(comp)) if s2_ratio is None else np.asarray(s2_ratio, float)
    r = (comp["sigma_x_t"] / comp["sigma_x_t1"]).to_numpy(float) * np.sqrt(k)
    a1_t = (1 + a1) * r - 1
    des = [_row_design(row, design) for row in comp.itertuples(index=False)]
    xb = np.array([d[0] for d in des])
    V = np.array([d[1] for d in des])
    xbar = np.sum(w * xb)
    M = np.sum(w * mech)
    Sxx = np.sum(w * (V + (xb - xbar) ** 2))
    Sxy = np.sum(w * (a1_t * V + (xb - xbar) * (mech - M)))
    b = Sxy / Sxx
    c = M - b * xbar
    wH = np.zeros(len(comp)) if wH is None else np.asarray(wH, float)
    B = np.sum(w * k * (1.0 - wH) * D / comp["sigma_x_t1"].to_numpy(float) ** 2) + np.sum(
        w * ((mech - c - b * xb) ** 2 + (a1_t - b) ** 2 * V))
    xi_star = xbar - M / b
    L = -xbar
    E = L - M
    return {"b": float(b), "xi_star_lin": float(xi_star), "B_hat": float(B),
            "two_delta_V_lin": float(abs(b) * (xi_star - XI_TARGET) ** 2 / B),
            "rho1_lin": float(1 + b), "M_bar": float(M), "L_bar": float(L),
            "E_bar": float(E),
            "xi_star_exact_LE": float(((1 + b) * L - E) / abs(b)),
            "zeroth_order_xi_star": float(((1 + a1) * L - E) / abs(a1)),
            "zeroth_order_xi_star_mbar": float(M / abs(a1)),
            "xbar_stay_pooled": float(xbar), "_w": w, "_xb": xb, "_V": V}


def closed_form_observables(composition: pd.DataFrame, true_params: dict,
                            period: str, standardisation: pd.DataFrame,
                            design: str = "observed", cell: dict | None = None) -> dict:
    _match_cell(composition, cell)
    _check_complete_panel(standardisation, cell, period, composition)
    th, guard = process_spec(true_params)
    if _is_nonlinear(th, guard):
        return _nan_result() | {"id_status": "not_defined_nonlinear"}
    s_std = _check_standardisation(standardisation, period)
    comp_k, _ = _check(composition, period)
    yrs = [int(t) for t in comp_k[comp_k["n_stay"] > 0]["year_t"]]
    wH = [_wH_of(s_std, t + 1) for t in yrs]
    grid, p, m_st, s_st = stationary_law(th, guard)
    zq, pq = _std_grid(grid, p, m_st, s_st)
    k = [expected_s2_ratio(th, guard, zq, pq, *_complete_moments(s_std, t), h)
         for t, h in zip(yrs, wH)]
    cf = closed_form_ou(composition, float(th[1]), float(np.exp(th[4])), period,
                        design=design, wH=wH, s2_ratio=k)
    if not (np.isfinite(cf["b"]) and cf["b"] < 0 and cf["B_hat"] > 0):
        return _nan_result() | {"id_status": "invalid_closed_form"}
    b = cf["b"]
    theta = np.array([-b * cf["xi_star_lin"], b, 0.0, 0.0, np.log(cf["B_hat"]),
                      0.0, 0.0])
    obs = observables(theta, _gaussian_pseudo(cf["_w"], cf["_xb"], cf["_V"]))
    tr = _truth_full(th, standardisation, period, guard)
    obs["rho1"] = rho1_prediction(tr["obs"]["rho1"], cf["rho1_lin"],
                                  tr["rho1_complete_projection"])
    obs["B_ratio_w"] = np.nan
    return obs


ZEROTH_ORDER_FORMS = {"zeroth_order": "first_order", "zeroth_order_mbar": "mbar"}


def zeroth_order_inputs(composition: pd.DataFrame, true_params: dict,
                        period: str, standardisation: pd.DataFrame,
                        cell: dict | None = None) -> dict:
    _match_cell(composition, cell)
    _check_complete_panel(standardisation, cell, period, composition)
    comp, _ = _check(composition, period)
    comp = comp[comp["n_stay"] > 0]
    th, guard = process_spec(true_params)
    tr = _truth_full(th, standardisation, period, guard)
    t_obs, t_th = tr["obs"], tr["theta"]
    s = _check_standardisation(standardisation, period)
    if comp.empty or not np.isfinite(t_obs["xi_trap"]):
        return {"invalid": "not_defined"}
    if "xi_bar_stay" not in comp.columns:
        raise ValueError("the zeroth-order row needs the observed xi_bar_stay (L_t = −ξ̄_S,t)")
    w = comp["n_stay"].to_numpy(float)
    w = w / w.sum()
    yrs = period_transitions(period)
    wc = np.array([min(s.at[t, "n"], s.at[t + 1, "n"]) for t in yrs], float)
    wc = wc / wc.sum()
    sc_t = np.array([s.at[t, "sigma_x"] for t in yrs], float)
    sc_t1 = np.array([s.at[t + 1, "sigma_x"] for t in yrs], float)
    x0 = float(t_obs["xi_trap"])
    a = float(np.polynomial.Polynomial(t_th[:4]).deriv()(x0))
    B = float(np.exp(_logB(t_th, np.array([x0])))[0])
    if not a < 0:
        return {"invalid": "not_defined", "a1": a}
    mech = comp["mech_term"].to_numpy(float)
    L_t = -comp["xi_bar_stay"].to_numpy(float)
    E_t = L_t - mech
    st, st1 = comp["sigma_x_t"].to_numpy(float), comp["sigma_x_t1"].to_numpy(float)
    return {"xi_star_true": x0, "a1": a, "rho": 1.0 + a, "B": B,
            "L_bar": float(np.sum(w * L_t)), "E_bar": float(np.sum(w * E_t)),
            "M_bar": float(np.sum(w * mech)),
            "B_sigma_factor": float(np.sum(w / st1 ** 2) / np.sum(wc / sc_t1 ** 2)),
            "rho_sigma_factor": float(np.sum(w * st / st1) / np.sum(wc * sc_t / sc_t1)),
            "rho1_true": float(t_obs["rho1"])}


def zeroth_order_observables(composition: pd.DataFrame, true_params: dict,
                             period: str, standardisation: pd.DataFrame,
                             cell: dict | None = None, form: str = "first_order") -> dict:
    if form not in ZEROTH_ORDER_FORMS.values():
        raise ValueError(f"form must be one of {sorted(ZEROTH_ORDER_FORMS.values())}")
    inp = zeroth_order_inputs(composition, true_params, period, standardisation, cell)
    if "invalid" in inp:
        return _nan_result() | {"id_status": "not_defined"}
    a = inp["a1"]
    shift = ((inp["rho"] * inp["L_bar"] - inp["E_bar"]) / abs(a) if form == "first_order"
             else inp["M_bar"] / abs(a))
    xi0 = inp["xi_star_true"] + shift
    B0 = inp["B"] * inp["B_sigma_factor"]
    rho = inp["rho1_true"] * inp["rho_sigma_factor"]
    theta = np.array([-a * xi0, a, 0.0, 0.0, np.log(B0), 0.0, 0.0])
    obs = observables(theta, _gaussian_pseudo(np.array([1.0]), np.array([0.0]),
                                              np.array([1.0])))
    obs["rho1"] = float(rho)
    obs["B_ratio_w"] = np.nan
    return obs


def _gaussian_pseudo(w, xb, V) -> np.ndarray:
    """Gaussian quantile pseudo-samples, N_PSEUDO in total, in proportion to
    the weights w, location xb, variance V (as _pooled does for OU)."""
    w = np.asarray(w, float) / np.sum(w)
    out = []
    for wi, m, v in zip(w, xb, V):
        k = max(2, int(round(N_PSEUDO * wi)))
        z = stats.norm.ppf((np.arange(k) + 0.5) / k)
        z = (z - z.mean()) / np.sqrt(np.mean((z - z.mean()) ** 2))
        out.append(m + np.sqrt(v) * z)
    return np.concatenate(out)
