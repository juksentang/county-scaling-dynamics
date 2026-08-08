"""
Landau potential fitting: V(ξ) = (a/2)ξ² + (b/4)ξ⁴ - hξ + V₀

The Landau model provides a parametric theory for phase transitions:
- a < 0: bistable (two potential wells)
- a > 0: monostable (single well)
- h: asymmetry (external field breaking symmetry)
- Critical point: a = 0 (second-order transition) or h = h_c (first-order)
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from src.distribution import compute_effective_potential


def landau_potential(xi: np.ndarray, a: float, b: float,
                     h: float, V0: float) -> np.ndarray:
    """Landau potential: V = (a/2)ξ² + (b/4)ξ⁴ - hξ + V₀ (Eq 12)."""
    return (a / 2) * xi**2 + (b / 4) * xi**4 - h * xi + V0


def fit_landau_to_veff(xi_grid: np.ndarray,
                       V_eff: np.ndarray,
                       n_restarts: int = 10) -> dict:
    """
    Fit Landau potential to empirical V_eff using nonlinear least squares.

    Constraint: b > 0 (ensures bounded potential).
    Multiple random restarts for robustness.

    Returns dict with fitted parameters, R², and classification.
    """
    if len(xi_grid) < 10:
        return {"status": "insufficient_data"}

    best_result = None
    best_cost = np.inf

    rng = np.random.RandomState(42)

    for _ in range(n_restarts):
        # Random initial guesses
        a0 = rng.uniform(-2, 2)
        b0 = rng.uniform(0.01, 2)
        h0 = rng.uniform(-1, 1)
        V0_0 = V_eff.min()

        try:
            popt, pcov = curve_fit(
                landau_potential, xi_grid, V_eff,
                p0=[a0, b0, h0, V0_0],
                bounds=([-np.inf, 0.001, -np.inf, -np.inf],
                        [np.inf, np.inf, np.inf, np.inf]),
                maxfev=10000,
            )

            residuals = V_eff - landau_potential(xi_grid, *popt)
            cost = np.sum(residuals**2)

            if cost < best_cost:
                best_cost = cost
                best_result = {
                    "a": popt[0],
                    "b": popt[1],
                    "h": popt[2],
                    "V0": popt[3],
                    "cov": pcov,
                }
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return {"status": "fit_failed"}

    a, b, h = best_result["a"], best_result["b"], best_result["h"]

    # R²
    V_pred = landau_potential(xi_grid, a, b, h, best_result["V0"])
    ss_res = np.sum((V_eff - V_pred)**2)
    ss_tot = np.sum((V_eff - V_eff.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Classification
    # Critical field: h_c = (2/3)√(-a/3) × a (for a < 0)
    if a < 0:
        is_bistable = True
        h_c = (2/3) * np.sqrt(-a/3) * abs(a) if a != 0 else 0
        barrier_height = a**2 / (4 * b) if b > 0 else 0
    else:
        is_bistable = False
        h_c = 0
        barrier_height = 0

    best_result.update({
        "status": "success",
        "R2": r2,
        "is_bistable": is_bistable,
        "h_c": h_c,
        "barrier_height": barrier_height,
    })

    return best_result


def fit_landau_timeseries(df: pd.DataFrame,
                          saai_col: str = "SAAI") -> pd.DataFrame:
    """
    Fit Landau potential year-by-year.

    Returns DataFrame with Year, a, b, h, h_c, is_bistable, barrier_height, R2.
    """
    records = []
    for year, grp in df.groupby("Year"):
        xi = grp[saai_col].dropna().values
        if len(xi) < 30:
            continue

        xi_grid, V_eff = compute_effective_potential(xi)
        if len(xi_grid) == 0:
            continue

        result = fit_landau_to_veff(xi_grid, V_eff)

        if result.get("status") == "success":
            records.append({
                "Year": year,
                "a": result["a"],
                "b": result["b"],
                "h": result["h"],
                "h_c": result["h_c"],
                "is_bistable": result["is_bistable"],
                "barrier_height": result["barrier_height"],
                "R2": result["R2"],
                "N": len(xi),
            })

    return pd.DataFrame(records)


def compute_critical_exponents(landau_ts: pd.DataFrame) -> dict:
    """
    Estimate critical exponents from Landau time series.

    Near the critical point (a → 0):
    - Order parameter: m ~ |a|^{β} with β = 1/2
    - Susceptibility: χ ~ |a|^{-γ} with γ = 1
    - Barrier height: ΔV ~ |a|^{2}

    Returns dict with estimated exponents and fit quality.
    """
    # Find transition year: where a crosses zero
    a_vals = landau_ts["a"].values
    years = landau_ts["Year"].values

    transition_idx = None
    for i in range(len(a_vals) - 1):
        if a_vals[i] < 0 and a_vals[i+1] > 0:
            transition_idx = i
            break

    if transition_idx is None:
        return {"status": "no_transition_found"}

    transition_year = years[transition_idx]

    # Use pre-transition data (a < 0) to estimate exponents
    pre = landau_ts[landau_ts["a"] < 0].copy()
    if len(pre) < 3:
        return {"status": "insufficient_pre_transition_data"}

    # Log-log fit: barrier_height vs |a|
    log_a = np.log(np.abs(pre["a"].values))
    log_barrier = np.log(pre["barrier_height"].values + 1e-10)

    from scipy.stats import linregress
    slope, intercept, r, p, se = linregress(log_a, log_barrier)

    return {
        "status": "success",
        "transition_year": transition_year,
        "barrier_exponent": slope,
        "barrier_r2": r**2,
        "expected_exponent": 2.0,  # Landau theory predicts 2
    }
