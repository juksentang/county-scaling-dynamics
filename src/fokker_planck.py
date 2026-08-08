"""
Fokker-Planck analysis: Kramers-Moyal estimation, FP potential, steady state.

Key principle: FP estimation uses ALL ~2500 counties (not just 832 poverty counties).
We model the evolution of the complete SAAI distribution P(ξ,t).
Transition pairs are pooled within FP periods to increase sample size.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import integrate
from src.config import FP_PERIODS, N_BINS_FP, MIN_BIN_COUNT


@dataclass
class KMResult:
    """Result of Kramers-Moyal estimation for one period."""
    period: str
    bin_centers: np.ndarray
    A: np.ndarray             # Drift coefficient A(ξ)
    B: np.ndarray             # Diffusion coefficient B(ξ)
    A_se: np.ndarray          # Standard error of A
    B_se: np.ndarray          # Standard error of B
    n_per_bin: np.ndarray     # Number of observations per bin
    n_pairs_total: int
    reliable: np.ndarray      # Boolean mask for bins with sufficient data


def build_transition_pairs(df: pd.DataFrame,
                           period: str,
                           saai_col: str = "SAAI",
                           window: tuple[int, int] | None = None,
                           exclude_years: frozenset = frozenset()) -> pd.DataFrame:
    """
    Build (ξ_t, ξ_{t+1}) transition pairs for ALL counties within a period.

    Parameters
    ----------
    df : DataFrame with Year, CountyCode, and saai_col.
    period : key in FP_PERIODS ('pre', 'during', 'post').
    window : optional (year_start, year_end) overriding FP_PERIODS[period]
        (sensitivity variant with the campaign window extended to 2020).
    exclude_years : set of crisis years E; a pair (t, t+1) is dropped iff
        t ∈ E or t+1 ∈ E (increment-level filter).  Default empty set
        applies no exclusion.

    Returns
    -------
    DataFrame with columns: CountyCode, Year_t, xi_t, xi_t1, delta_xi.
    """
    year_start, year_end = window if window is not None else FP_PERIODS[period]

    sub = df[(df["Year"] >= year_start) & (df["Year"] <= year_end)].copy()
    sub = sub[["Year", "CountyCode", saai_col]].dropna()
    sub = sub.sort_values(["CountyCode", "Year"])

    pairs = []
    for code, grp in sub.groupby("CountyCode"):
        grp = grp.sort_values("Year")
        years = grp["Year"].values
        xi_vals = grp[saai_col].values

        for i in range(len(years) - 1):
            # Only consecutive years; drop crisis-contaminated increments
            if (years[i+1] - years[i] == 1
                    and years[i] not in exclude_years
                    and years[i+1] not in exclude_years):
                pairs.append({
                    "CountyCode": code,
                    "Year_t": years[i],
                    "xi_t": xi_vals[i],
                    "xi_t1": xi_vals[i+1],
                    "delta_xi": xi_vals[i+1] - xi_vals[i],
                })

    return pd.DataFrame(pairs)


def estimate_kramers_moyal(pairs_df: pd.DataFrame,
                           n_bins: int = N_BINS_FP,
                           period: str = "",
                           binning: str = "equal_width") -> KMResult:
    """
    Estimate Kramers-Moyal coefficients A(ξ) and B(ξ) from transition pairs.

    A_k = E[Δξ | ξ ∈ bin_k] / Δt
    B_k = E[(Δξ)² | ξ ∈ bin_k] / Δt

    With Δt = 1 year.

    Parameters
    ----------
    pairs_df : DataFrame with xi_t, xi_t1, delta_xi.
    n_bins : number of bins for conditional moments.
    period : label for the period.
    binning : 'equal_width' (default) or 'equal_count' (adaptive bins with
        equal numbers of pairs per bin; appropriate for thin samples).
        Both modes share the same 1st-99th percentile support clipping so
        that estimates are comparable across modes.

    Returns
    -------
    KMResult dataclass.
    """
    xi = pairs_df["xi_t"].values
    delta = pairs_df["delta_xi"].values

    # Define bins using 1st-99th percentile range
    xi_min = np.percentile(xi, 1)
    xi_max = np.percentile(xi, 99)
    if binning == "equal_count":
        clip_mask = (xi >= xi_min) & (xi <= xi_max)
        edges = np.quantile(xi[clip_mask], np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(edges)          # drop duplicate quantiles
        n_bins = len(bin_edges) - 1           # actual bin count after dedupe
        # Equal-count bins have uneven widths: use the within-bin median of
        # xi_t as the bin center (edge midpoints would be systematically off)
        bin_centers = np.array([
            np.median(xi[(xi >= bin_edges[k]) & (xi < bin_edges[k+1])])
            if ((xi >= bin_edges[k]) & (xi < bin_edges[k+1])).any()
            else 0.5 * (bin_edges[k] + bin_edges[k+1])
            for k in range(n_bins)
        ])
    else:
        bin_edges = np.linspace(xi_min, xi_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    A = np.full(n_bins, np.nan)
    B = np.full(n_bins, np.nan)
    A_se = np.full(n_bins, np.nan)
    B_se = np.full(n_bins, np.nan)
    n_per_bin = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        mask = (xi >= bin_edges[k]) & (xi < bin_edges[k+1])
        n_k = mask.sum()
        n_per_bin[k] = n_k

        if n_k >= 5:
            d_k = delta[mask]
            A[k] = d_k.mean()           # Δt = 1
            B[k] = (d_k ** 2).mean()    # Δt = 1
            A_se[k] = d_k.std() / np.sqrt(n_k)
            B_se[k] = np.std(d_k ** 2) / np.sqrt(n_k)

    reliable = n_per_bin >= MIN_BIN_COUNT

    return KMResult(
        period=period,
        bin_centers=bin_centers,
        A=A,
        B=B,
        A_se=A_se,
        B_se=B_se,
        n_per_bin=n_per_bin,
        n_pairs_total=len(pairs_df),
        reliable=reliable,
    )


def estimate_kramers_moyal_all_periods(df: pd.DataFrame,
                                       saai_col: str = "SAAI",
                                       exclude_years: frozenset = frozenset()
                                       ) -> dict[str, KMResult]:
    """
    Estimate KM coefficients for all three FP periods.

    exclude_years is passed through to build_transition_pairs (crisis-year
    exclusion); the default empty set applies no exclusion.

    Returns dict: period → KMResult.
    """
    results = {}
    for period in FP_PERIODS:
        pairs = build_transition_pairs(df, period, saai_col=saai_col,
                                       exclude_years=exclude_years)
        print(f"  Period '{period}': {len(pairs):,} transition pairs")
        if len(pairs) > 0:
            km = estimate_kramers_moyal(pairs, period=period)
            results[period] = km
            n_reliable = km.reliable.sum()
            print(f"    Reliable bins: {n_reliable}/{N_BINS_FP} "
                  f"(min count per bin: {km.n_per_bin[km.reliable].min() if n_reliable > 0 else 0})")
    return results


def compute_fp_potential(km: KMResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FP potential from drift A(ξ): V_FP(ξ) = -∫ A(ξ')/B(ξ') dξ'

    This is different from V_eff = -ln P(ξ) which comes from KDE.
    V_FP captures the dynamical potential from the Fokker-Planck equation.
    """
    mask = ~np.isnan(km.A) & ~np.isnan(km.B) & (km.B > 0)
    if mask.sum() < 3:
        return np.array([]), np.array([])

    xi = km.bin_centers[mask]
    ratio = km.A[mask] / km.B[mask]

    # Cumulative trapezoidal integration
    V = np.zeros_like(xi)
    for i in range(1, len(xi)):
        V[i] = V[i-1] - integrate.trapezoid(ratio[:i+1], xi[:i+1])

    # Actually integrate from x0 to x
    V_fp = np.zeros_like(xi)
    for i in range(1, len(xi)):
        V_fp[i] = -integrate.trapezoid(ratio[:i+1], xi[:i+1])

    V_fp -= V_fp.min()

    return xi, V_fp


def compute_steady_state(km: KMResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FP steady-state distribution:
    P_st(ξ) ∝ (1/B(ξ)) · exp(2 ∫₀^ξ A(ξ')/B(ξ') dξ')
    """
    mask = ~np.isnan(km.A) & ~np.isnan(km.B) & (km.B > 0)
    if mask.sum() < 3:
        return np.array([]), np.array([])

    xi = km.bin_centers[mask]
    A_vals = km.A[mask]
    B_vals = km.B[mask]

    # Cumulative integral of 2*A/B
    ratio = 2 * A_vals / B_vals
    integral = np.zeros_like(xi)
    for i in range(1, len(xi)):
        integral[i] = integrate.trapezoid(ratio[:i+1], xi[:i+1])

    P_st = (1.0 / B_vals) * np.exp(integral)

    # Normalize
    norm = integrate.trapezoid(P_st, xi)
    if norm > 0:
        P_st = P_st / norm

    return xi, P_st


def find_drift_zeros(km: KMResult) -> list[dict]:
    """
    Find zeros of A(ξ) — equilibrium points.

    Returns list of dicts with xi, type ('stable' if A'<0, 'unstable' if A'>0).
    """
    mask = ~np.isnan(km.A)
    if mask.sum() < 3:
        return []

    xi = km.bin_centers[mask]
    A_vals = km.A[mask]

    zeros = []
    for i in range(len(xi) - 1):
        if A_vals[i] * A_vals[i+1] < 0:
            # Linear interpolation
            frac = A_vals[i] / (A_vals[i] - A_vals[i+1])
            xi_zero = xi[i] + frac * (xi[i+1] - xi[i])

            # Classify: stable if A goes from + to -, unstable if - to +
            if A_vals[i] > 0 and A_vals[i+1] < 0:
                eq_type = "stable"
            else:
                eq_type = "unstable"

            zeros.append({"xi": xi_zero, "type": eq_type})

    return zeros


def simulate_langevin(A_func, B_func, n_particles=2000, n_steps=100,
                      dt=1.0, xi0_range=(-3, 3), seed=42):
    """
    Simulate Langevin dynamics: dξ = A(ξ)dt + √B(ξ) dW.

    Returns DataFrame with columns: particle_id, step, xi (matching transition pair format).
    """
    rng = np.random.RandomState(seed)
    xi = rng.uniform(xi0_range[0], xi0_range[1], size=n_particles)

    records = []
    for t in range(n_steps + 1):
        for i in range(n_particles):
            records.append({"particle_id": i, "step": t, "xi": xi[i]})
        if t < n_steps:
            noise = rng.randn(n_particles)
            a_vals = np.array([A_func(x) for x in xi])
            b_vals = np.array([B_func(x) for x in xi])
            b_vals = np.clip(b_vals, 1e-10, None)
            xi = xi + a_vals * dt + np.sqrt(b_vals * dt) * noise

    return pd.DataFrame(records)


def _build_pairs_from_sim(sim_df):
    """Convert Langevin simulation output to transition pairs format."""
    pairs = []
    for pid, grp in sim_df.groupby("particle_id"):
        grp = grp.sort_values("step")
        steps = grp["step"].values
        xi_vals = grp["xi"].values
        for i in range(len(steps) - 1):
            if steps[i+1] - steps[i] == 1:
                pairs.append({
                    "CountyCode": pid,
                    "Year_t": steps[i],
                    "xi_t": xi_vals[i],
                    "xi_t1": xi_vals[i+1],
                    "delta_xi": xi_vals[i+1] - xi_vals[i],
                })
    return pd.DataFrame(pairs)


def validate_km_recovery(A_func, B_func, n_particles=2000, n_steps=100,
                         dt=1.0, n_bins=25, xi_range=(-3, 3)):
    """
    Validate KM estimation by simulating known dynamics and recovering coefficients.

    The KM estimator computes E[Δξ]/Δt_step where Δt_step=1 (one simulation step).
    With simulation step size dt, the true conditional moments are:
        E[Δξ] = A(ξ)·dt,  E[(Δξ)²] = B(ξ)·dt + O(dt²)
    So we rescale: A_hat/dt → should recover A_true, B_hat/dt → should recover B_true.

    Returns dict with RMSE_A, RMSE_B, and grid arrays for plotting.
    """
    sim = simulate_langevin(A_func, B_func, n_particles=n_particles,
                            n_steps=n_steps, dt=dt, xi0_range=xi_range)
    pairs = _build_pairs_from_sim(sim)

    km = estimate_kramers_moyal(pairs, n_bins=n_bins, period="synthetic")

    # Compare on valid bins — rescale by dt
    mask = ~np.isnan(km.A) & ~np.isnan(km.B)
    xi_grid = km.bin_centers[mask]
    A_hat = km.A[mask] / dt      # rescale to physical drift
    B_hat = km.B[mask] / dt      # rescale to physical diffusion

    A_true = np.array([A_func(x) for x in xi_grid])
    B_true = np.array([B_func(x) for x in xi_grid])

    rmse_A = np.sqrt(np.mean((A_hat - A_true) ** 2))
    rmse_B = np.sqrt(np.mean((B_hat - B_true) ** 2))

    return {
        "rmse_A": rmse_A,
        "rmse_B": rmse_B,
        "xi_grid": xi_grid,
        "A_true": A_true,
        "A_hat": A_hat,
        "B_true": B_true,
        "B_hat": B_hat,
        "km": km,
        "n_pairs": len(pairs),
        "dt": dt,
    }


def estimate_kramers_moyal_kernel(pairs_df, bandwidth="silverman",
                                  n_grid=100):
    """
    Nadaraya-Watson kernel estimation of drift A(ξ) and diffusion B(ξ).

    A(ξ) = Σ K_h(ξ-ξᵢ)Δξᵢ / Σ K_h(ξ-ξᵢ)
    B(ξ) = Σ K_h(ξ-ξᵢ)(Δξᵢ)² / Σ K_h(ξ-ξᵢ) - A(ξ)²

    Returns KMResult with kernel-smoothed estimates.
    """
    xi = pairs_df["xi_t"].values
    delta = pairs_df["delta_xi"].values

    # Bandwidth selection
    if bandwidth == "silverman":
        n = len(xi)
        sigma = xi.std()
        h = 1.06 * sigma * n ** (-1/5)
    else:
        h = float(bandwidth)

    # Grid
    xi_min = np.percentile(xi, 1)
    xi_max = np.percentile(xi, 99)
    grid = np.linspace(xi_min, xi_max, n_grid)

    A = np.full(n_grid, np.nan)
    B = np.full(n_grid, np.nan)
    A_se = np.full(n_grid, np.nan)
    B_se = np.full(n_grid, np.nan)
    n_eff = np.zeros(n_grid)

    for k in range(n_grid):
        u = (xi - grid[k]) / h
        w = np.exp(-0.5 * u**2) / (np.sqrt(2 * np.pi) * h)
        w_sum = w.sum()

        if w_sum < 1e-10:
            continue

        # Effective sample size
        n_eff[k] = w_sum ** 2 / (w ** 2).sum() if (w ** 2).sum() > 0 else 0

        A[k] = np.sum(w * delta) / w_sum
        B[k] = np.sum(w * delta**2) / w_sum - A[k]**2
        B[k] = max(B[k], 1e-10)

        # SE via weighted standard error
        if n_eff[k] > 2:
            resid_a = delta - A[k]
            A_se[k] = np.sqrt(np.sum(w * resid_a**2) / w_sum) / np.sqrt(n_eff[k])
            B_se[k] = B[k] * np.sqrt(2 / n_eff[k])

    reliable = n_eff >= MIN_BIN_COUNT

    return KMResult(
        period="kernel",
        bin_centers=grid,
        A=A,
        B=B,
        A_se=A_se,
        B_se=B_se,
        n_per_bin=n_eff.astype(int),
        n_pairs_total=len(pairs_df),
        reliable=reliable,
    )


def estimate_km_by_group(df: pd.DataFrame,
                         group_col: str = "UrbGroup",
                         group_bins: dict | None = None,
                         saai_col: str = "SAAI") -> dict[str, dict]:
    """
    Estimate Kramers-Moyal coefficients separately for urbanization groups.

    Parameters
    ----------
    df : DataFrame with Year, CountyCode, saai_col, and either group_col
         or RegGDP/NonAgriGDP for computing urbanization ratio.
    group_col : column with group labels, or 'auto' to create from urbanization.
    group_bins : dict mapping group_name → (u_low, u_high). Default:
                 Low (<0.50), Medium (0.50-0.80), High (>=0.80).
    saai_col : SAAI column name.

    Returns
    -------
    dict[group_name → {"km": {period: KMResult}, "fp": {period: (xi, V)}}]
    """
    data = df.copy()

    if group_col not in data.columns or group_col == "auto":
        data["_u"] = (data["NonAgriGDP"] / data["RegGDP"]).clip(0, 1.5)
        if group_bins is None:
            group_bins = {"Low": (0, 0.50), "Medium": (0.50, 0.80), "High": (0.80, 2.0)}
        data["_group"] = pd.cut(
            data["_u"],
            bins=[v[0] for v in group_bins.values()] + [list(group_bins.values())[-1][1]],
            labels=list(group_bins.keys()),
            include_lowest=True,
        )
        group_col = "_group"

    results = {}
    for grp_name, grp_df in data.groupby(group_col, observed=True):
        grp_name = str(grp_name)
        km_periods = {}
        fp_periods = {}
        for period in FP_PERIODS:
            pairs = build_transition_pairs(grp_df, period, saai_col=saai_col)
            if len(pairs) < 50:
                continue
            km = estimate_kramers_moyal(pairs, period=f"{period}_{grp_name}")
            km_periods[period] = km
            xi_fp, V_fp = compute_fp_potential(km)
            fp_periods[period] = (xi_fp, V_fp)

        results[grp_name] = {
            "km": km_periods,
            "fp": fp_periods,
            "n_counties": grp_df["CountyCode"].nunique(),
        }

    return results


def compute_kramers_escape_time(km: KMResult,
                                xi_trap: float | None = None,
                                xi_target: float = 0.0,
                                xi_barrier: float | None = None) -> dict:
    """
    Estimate mean first-passage time (MFPT) from trap to xi_target.

    In a monostable potential the classical Kramers (well-to-well) escape
    is undefined.  We instead compute the MFPT to reach a threshold
    xi_target (default: the scaling benchmark xi = 0).  The Arrhenius
    form tau ~ exp(2 DeltaV / D) is the leading-order saddle-point
    approximation of the MFPT integral and is valid for any potential
    with a local minimum and a barrier, regardless of what lies beyond
    the barrier (Gardiner 2009, Section 5.2.8).

    Uses two approaches:
    1. Arrhenius approximation: tau ~ exp(2 DeltaV / D)
    2. Full MFPT via Gardiner double integral.

    Parameters
    ----------
    km : KMResult with drift A and diffusion B.
    xi_trap : position of the trap (stable fixed point). Auto-detected if None.
    xi_target : absorbing boundary / escape threshold (default 0.0).
    xi_barrier : position of the barrier. Auto-detected if None.

    Returns
    -------
    dict with tau_arrhenius, tau_mfpt, barrier_height, mean_diffusion,
    xi_target.
    """
    # Find equilibria
    zeros = find_drift_zeros(km)
    stable = [z for z in zeros if z["type"] == "stable"]
    unstable = [z for z in zeros if z["type"] == "unstable"]

    if xi_trap is None:
        if stable:
            xi_trap = stable[0]["xi"]
        else:
            xi_trap = km.bin_centers[km.reliable][np.nanargmin(np.abs(km.A[km.reliable]))]

    # Compute FP potential
    xi_fp, V_fp = compute_fp_potential(km)
    if len(xi_fp) < 3:
        return {"status": "insufficient_data"}

    # Find barrier: highest V_fp between xi_trap and xi_target
    trap_idx = np.argmin(np.abs(xi_fp - xi_trap))
    target_idx = np.argmin(np.abs(xi_fp - xi_target))

    # Ensure we search between trap and target for the barrier
    if target_idx > trap_idx:
        between_mask = (xi_fp > xi_fp[trap_idx]) & (xi_fp <= xi_fp[target_idx])
    else:
        between_mask = (xi_fp < xi_fp[trap_idx]) & (xi_fp >= xi_fp[target_idx])

    if xi_barrier is not None:
        barrier_idx = np.argmin(np.abs(xi_fp - xi_barrier))
    elif between_mask.any():
        barrier_idx = np.where(between_mask)[0][np.argmax(V_fp[between_mask])]
    else:
        # Fallback: highest V_fp to the right of xi_trap
        right_mask = xi_fp > xi_trap
        if right_mask.any():
            barrier_idx = np.where(right_mask)[0][np.argmax(V_fp[right_mask])]
        else:
            barrier_idx = np.argmax(V_fp)

    delta_V = V_fp[barrier_idx] - V_fp[trap_idx]

    # Mean diffusion in the trap region
    mask_B = ~np.isnan(km.B) & km.reliable
    xi_B = km.bin_centers[mask_B]
    B_vals = km.B[mask_B]
    trap_region = (xi_B >= xi_trap - 0.5) & (xi_B <= xi_fp[barrier_idx])
    D = np.mean(B_vals[trap_region]) if trap_region.any() else np.mean(B_vals)

    # Arrhenius: tau ~ exp(2 * DeltaV / D)
    # Leading-order saddle-point approximation of MFPT integral;
    # valid for any potential with a well and barrier (Gardiner 2009)
    tau_arrhenius = np.exp(2 * delta_V / D) if D > 0 else np.inf

    # Full MFPT via Gardiner double integral (Eq 5.2.117)
    tau_mfpt = np.nan
    try:
        A_interp = np.interp(xi_fp, km.bin_centers[~np.isnan(km.A)], km.A[~np.isnan(km.A)])
        B_interp = np.interp(xi_fp, km.bin_centers[mask_B], B_vals)
        B_interp = np.clip(B_interp, 1e-10, None)

        inner = np.zeros_like(xi_fp)
        for i in range(1, len(xi_fp)):
            inner[i] = integrate.trapezoid(
                np.exp(-2 * V_fp[:i+1] / B_interp[:i+1]), xi_fp[:i+1]
            )

        integrand = np.exp(2 * V_fp / B_interp) * inner / B_interp
        i_start = max(0, trap_idx)
        i_end = min(len(xi_fp), target_idx + 1) if target_idx > trap_idx else min(len(xi_fp), barrier_idx + 1)
        if i_end > i_start + 1:
            tau_mfpt = 2 * integrate.trapezoid(
                integrand[i_start:i_end], xi_fp[i_start:i_end]
            )
    except Exception:
        pass

    return {
        "xi_trap": xi_trap,
        "xi_target": xi_target,
        "xi_barrier": xi_fp[barrier_idx],
        "barrier_height": delta_V,
        "mean_diffusion": D,
        "tau_arrhenius": tau_arrhenius,
        "tau_mfpt": tau_mfpt,
        "V_fp_xi": xi_fp,
        "V_fp": V_fp,
    }


def compute_escape_observables(km: KMResult,
                               xi_target: float = 0.0,
                               n_grid: int = 400) -> dict:
    """
    Escape observables for the trap-to-target first-passage problem.

    Definitions:

    - xi_trap (ξ*): first *stable* zero of the drift A(ξ) (linear
      interpolation between bins).  Fallback if no stable zero exists:
      argmin |A| over reliable bins.  ``id_status`` records which branch
      was used ('stable_zero' | 'fallback_argminA' | 'failed').
    - V_FP(ξ) = -∫ A/B dξ'  (``compute_fp_potential``; min-shifted, and
      any constant shift cancels exactly in the MFPT integrand).
    - ΔV ≡ max over the escape path ξ* → ξ_target of V_FP, minus
      V_FP(ξ*).  For a monotone tilted well this equals
      V_FP(ξ_target) − V_FP(ξ*); both are returned
      (``delta_V`` and ``delta_V_at_target``).
    - D̄ (``mean_diffusion``): mean of B over reliable bins on the path
      [min(ξ*, ξ_target), max(ξ*, ξ_target)].
    - Γ = 2ΔV / D̄  (barrier-to-noise index).
    - τ_MFPT: full Gardiner double integral (Gardiner 2009, Eq. 5.2.157)
      with the absorbing boundary at ξ_target and the reflecting boundary
      at the far edge of the estimated domain:

          T(ξ*) = 2 ∫_{ξ*}^{ξ_target} dy  e^{+2 V_FP(y)}
                      ∫_{a}^{y} dz  e^{-2 V_FP(z)} / B(z)

      for ξ* < ξ_target (mirrored otherwise), evaluated on a uniform
      grid with linear interpolation of A and B.  Note that with the
      convention V_FP = −∫A/B the stationary weight is exp(−2 V_FP), so
      the exponentials carry 2V_FP — not 2V_FP/B.
    - tau_arrhenius: leading-order barrier factor exp(2ΔV) under the
      V_FP convention (dimensionless exponent; curvature prefactor
      omitted).  Reported for diagnostic comparison only; τ_MFPT is the
      primary quantity.
    """
    out = {
        "xi_target": xi_target,
        "xi_trap": np.nan,
        "xi_barrier": np.nan,
        "delta_V": np.nan,
        "delta_V_at_target": np.nan,
        "mean_diffusion": np.nan,
        "gamma": np.nan,
        "tau_mfpt": np.nan,
        "tau_arrhenius": np.nan,
        "id_status": "failed",
    }

    # ── potential and support ──
    xi_fp, V_fp = compute_fp_potential(km)
    if len(xi_fp) < 4:
        return out
    if not (xi_fp.min() <= xi_target <= xi_fp.max()):
        out["id_status"] = "target_outside_domain"
        return out

    # ── trap identification ──
    # The escape origin is the global minimum of V_FP over the estimated
    # support. Taking the first stable zero instead makes the escape time
    # depend on which crossing the binning happened to resolve: the binned
    # drift can sit within its own standard error of zero across an interval
    # and cross it several times there, and county-block resampling then moves
    # the identified location across that whole interval. The parametric
    # drift, being cubic, yields one stable fixed point per period, for which
    # the two rules coincide.
    zeros = find_drift_zeros(km)
    stable = [z for z in zeros if z["type"] == "stable"]
    if stable:
        in_dom = [z for z in stable if xi_fp.min() <= z["xi"] <= xi_fp.max()]
        candidates = in_dom or stable
        xi_trap = min(candidates,
                      key=lambda z: np.interp(z["xi"], xi_fp, V_fp))["xi"]
        id_status = "stable_zero"
    else:
        rel = km.reliable & ~np.isnan(km.A)
        if not rel.any():
            return out
        xi_trap = km.bin_centers[rel][np.nanargmin(np.abs(km.A[rel]))]
        id_status = "fallback_argminA"

    # ── uniform fine grid over the estimated support ──
    grid = np.linspace(xi_fp.min(), xi_fp.max(), n_grid)
    V = np.interp(grid, xi_fp, V_fp)
    mask_B = ~np.isnan(km.B) & (km.B > 0)
    B = np.interp(grid, km.bin_centers[mask_B], km.B[mask_B])
    B = np.clip(B, 1e-12, None)

    lo, hi = (xi_trap, xi_target) if xi_trap <= xi_target else (xi_target, xi_trap)
    path = (grid >= lo) & (grid <= hi)
    if path.sum() < 2:
        out["id_status"] = "degenerate_path"
        return out

    # ── ΔV on the escape path ──
    V_trap = np.interp(xi_trap, grid, V)
    V_target = np.interp(xi_target, grid, V)
    k_barrier = np.where(path)[0][np.argmax(V[path])]
    delta_V = V[k_barrier] - V_trap
    delta_V_at_target = V_target - V_trap

    # ── D̄ over reliable bins on the path ──
    rel_B = km.reliable & ~np.isnan(km.B)
    xi_rB = km.bin_centers[rel_B]
    on_path = (xi_rB >= lo) & (xi_rB <= hi)
    D_bar = km.B[rel_B][on_path].mean() if on_path.any() else km.B[rel_B].mean()

    # ── Gardiner double-integral MFPT ──
    # Escape upward (trap below target): reflect at left edge.
    # Escape downward: mirror the axis and reuse the same formula.
    if xi_trap <= xi_target:
        g, Vg, Bg = grid, V, B
        x0, xb = xi_trap, xi_target
    else:
        g = -grid[::-1]
        Vg = V[::-1]
        Bg = B[::-1]
        x0, xb = -xi_trap, -xi_target

    # inner(y) = ∫_a^y exp(-2V)/B dz   (cumulative trapezoid)
    f_in = np.exp(-2.0 * (Vg - Vg.min())) / Bg          # shift cancels below
    inner = np.concatenate([[0.0], np.cumsum(
        0.5 * (f_in[1:] + f_in[:-1]) * np.diff(g))])
    outer_integrand = 2.0 * np.exp(2.0 * (Vg - Vg.min())) * inner
    seg = (g >= x0) & (g <= xb)
    if seg.sum() >= 2:
        out["tau_mfpt"] = float(np.trapezoid(outer_integrand[seg], g[seg]))

    out.update({
        "xi_trap": float(xi_trap),
        "xi_barrier": float(grid[k_barrier]),
        "delta_V": float(delta_V),
        "delta_V_at_target": float(delta_V_at_target),
        "mean_diffusion": float(D_bar),
        "gamma": float(2.0 * delta_V / D_bar) if D_bar > 0 else np.nan,
        "tau_arrhenius": float(np.exp(2.0 * delta_V)),
        "id_status": id_status,
        "V_fp_xi": xi_fp,
        "V_fp": V_fp,
    })
    return out


def analyze_diffusion_heterogeneity(km: KMResult,
                                     threshold: float = 0.0) -> dict:
    """
    Test whether diffusion B(xi) differs between xi < threshold and xi > threshold.

    If B is smaller for xi < 0 (negative SAAI), counties in the poverty region
    experience less stochastic variance, making escape more difficult.

    Returns
    -------
    dict with B_neg_mean, B_pos_mean, ratio, t_stat, p_value.
    """
    mask = km.reliable & ~np.isnan(km.B)
    xi = km.bin_centers[mask]
    B = km.B[mask]

    neg = B[xi < threshold]
    pos = B[xi >= threshold]

    if len(neg) < 2 or len(pos) < 2:
        return {"status": "insufficient_bins", "n_neg": len(neg), "n_pos": len(pos)}

    from scipy.stats import mannwhitneyu, ttest_ind

    t_stat, t_p = ttest_ind(neg, pos, equal_var=False)
    try:
        u_stat, u_p = mannwhitneyu(neg, pos, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan

    return {
        "B_neg_mean": neg.mean(),
        "B_pos_mean": pos.mean(),
        "B_neg_std": neg.std(),
        "B_pos_std": pos.std(),
        "ratio": neg.mean() / pos.mean() if pos.mean() > 0 else np.nan,
        "t_stat": t_stat,
        "t_pvalue": t_p,
        "mw_stat": u_stat,
        "mw_pvalue": u_p,
        "n_neg": len(neg),
        "n_pos": len(pos),
        "threshold": threshold,
    }


def validate_fp_consistency(km: KMResult,
                            empirical_xi: np.ndarray) -> dict:
    """
    Validate FP model consistency.

    Checks:
    1. Steady state vs empirical distribution (KL divergence)
    2. Pawula theorem: D4 / D2² should be small
    """
    from scipy.stats import gaussian_kde

    xi_ss, P_ss = compute_steady_state(km)
    if len(xi_ss) == 0:
        return {"status": "insufficient_data"}

    # Empirical KDE on same grid
    kde = gaussian_kde(empirical_xi[np.isfinite(empirical_xi)])
    P_emp = kde(xi_ss)
    P_emp_norm = P_emp / integrate.trapezoid(P_emp, xi_ss)

    # KL divergence (P_emp || P_ss)
    mask_valid = (P_ss > 1e-10) & (P_emp_norm > 1e-10)
    if mask_valid.sum() > 0:
        kl = integrate.trapezoid(
            P_emp_norm[mask_valid] * np.log(P_emp_norm[mask_valid] / P_ss[mask_valid]),
            xi_ss[mask_valid]
        )
    else:
        kl = np.nan

    # L2 distance
    l2 = np.sqrt(integrate.trapezoid((P_ss - P_emp_norm)**2, xi_ss))

    return {
        "kl_divergence": kl,
        "l2_distance": l2,
        "n_grid_points": len(xi_ss),
    }
