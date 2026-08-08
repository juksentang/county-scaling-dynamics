"""
Spatial analysis: Moran's I, spatial correlation function, correlation length.

Available for 2017-2023 only (coordinate data limitation).
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform


def compute_distance_matrix(df: pd.DataFrame,
                            lng_col: str = "CountyLng",
                            lat_col: str = "CountyLat") -> np.ndarray:
    """
    Compute Haversine great-circle distance matrix (km).

    Parameters
    ----------
    df : DataFrame with lng_col and lat_col columns.

    Returns
    -------
    N×N distance matrix in kilometers.
    """
    coords = df[[lat_col, lng_col]].dropna().values
    n = len(coords)

    # Convert to radians
    coords_rad = np.radians(coords)

    # Haversine pairwise
    R = 6371.0  # Earth radius in km

    lat = coords_rad[:, 0]
    lng = coords_rad[:, 1]

    dlat = lat[:, None] - lat[None, :]
    dlng = lng[:, None] - lng[None, :]

    a = np.sin(dlat / 2)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlng / 2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distances = R * c

    return distances


def compute_morans_i(values: np.ndarray,
                     distances: np.ndarray,
                     distance_threshold: float = 200.0,
                     n_permutations: int = 999) -> dict:
    """
    Compute Moran's I statistic with permutation test (Eq 15).

    I = (N/W) × Σᵢ Σⱼ wᵢⱼ(xᵢ-x̄)(xⱼ-x̄) / Σᵢ(xᵢ-x̄)²

    where wᵢⱼ = 1 if d(i,j) < threshold, else 0.

    Parameters
    ----------
    values : 1D array of values.
    distances : N×N distance matrix.
    distance_threshold : km, for binary weight matrix.
    n_permutations : number of permutations for p-value.

    Returns
    -------
    dict with I, E_I, z_score, p_value.
    """
    mask = np.isfinite(values)
    vals = values[mask]
    dist = distances[np.ix_(mask, mask)]

    N = len(vals)
    if N < 10:
        return {"I": np.nan, "p_value": np.nan, "status": "insufficient_data"}

    # Weight matrix
    W = (dist < distance_threshold).astype(float)
    np.fill_diagonal(W, 0)
    W_sum = W.sum()

    if W_sum == 0:
        return {"I": np.nan, "p_value": np.nan, "status": "no_neighbors"}

    # Moran's I
    x_dev = vals - vals.mean()
    numerator = N * np.dot(x_dev, W @ x_dev)
    denominator = W_sum * np.sum(x_dev**2)
    I = numerator / denominator if denominator > 0 else 0

    # Expected value under null
    E_I = -1 / (N - 1)

    # Permutation test
    if n_permutations > 0:
        rng = np.random.RandomState(42)
        I_perm = np.zeros(n_permutations)
        for p in range(n_permutations):
            perm_vals = rng.permutation(vals)
            perm_dev = perm_vals - perm_vals.mean()
            num_p = N * np.dot(perm_dev, W @ perm_dev)
            den_p = W_sum * np.sum(perm_dev**2)
            I_perm[p] = num_p / den_p if den_p > 0 else 0

        p_value = (np.sum(np.abs(I_perm) >= np.abs(I)) + 1) / (n_permutations + 1)
        z_score = (I - I_perm.mean()) / I_perm.std() if I_perm.std() > 0 else 0
    else:
        p_value = np.nan
        z_score = np.nan

    return {
        "I": I,
        "E_I": E_I,
        "z_score": z_score,
        "p_value": p_value,
        "N": N,
        "W_sum": W_sum,
    }


def compute_spatial_correlation_function(values: np.ndarray,
                                         distances: np.ndarray,
                                         r_bins: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial correlation function C(r,t) (Eq 13).

    C(r) = <δξᵢ · δξⱼ>_{d(i,j)∈[r, r+Δr]} / σ²

    Parameters
    ----------
    values : 1D array of values.
    distances : N×N distance matrix.
    r_bins : distance bin edges in km. Default: 0 to 500 km in 50 km steps.

    Returns
    -------
    (r_centers, C_r): distance bin centers and correlation values.
    """
    if r_bins is None:
        r_bins = np.arange(0, 550, 50)

    mask = np.isfinite(values)
    vals = values[mask]
    dist = distances[np.ix_(mask, mask)]

    N = len(vals)
    x_dev = vals - vals.mean()
    var = np.var(vals)

    if var == 0 or N < 10:
        return np.array([]), np.array([])

    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    C_r = np.full(len(r_centers), np.nan)

    # Upper triangle indices
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dist[triu_idx]
    pair_prods = x_dev[triu_idx[0]] * x_dev[triu_idx[1]]

    for k in range(len(r_centers)):
        bin_mask = (pair_dists >= r_bins[k]) & (pair_dists < r_bins[k+1])
        n_pairs = bin_mask.sum()
        if n_pairs >= 10:
            C_r[k] = pair_prods[bin_mask].mean() / var

    return r_centers, C_r


def fit_correlation_length(r_centers: np.ndarray,
                           C_r: np.ndarray) -> dict:
    """
    Fit correlation length from C(r) (Eq 20):
    C(r) = C₀ · r^{-ν} · exp(-r/λ)

    Returns dict with λ (correlation length), ν, C0, R².
    """
    mask = np.isfinite(C_r) & (C_r > 0) & (r_centers > 0)
    if mask.sum() < 3:
        return {"lambda": np.nan, "status": "insufficient_data"}

    r = r_centers[mask]
    C = C_r[mask]

    def model(r, C0, nu, lam):
        return C0 * r**(-nu) * np.exp(-r / lam)

    try:
        popt, pcov = curve_fit(
            model, r, C,
            p0=[C[0] * r[0], 0.5, 200],
            bounds=([0, 0, 10], [np.inf, 5, 5000]),
            maxfev=5000,
        )
        C0, nu, lam = popt

        C_pred = model(r, *popt)
        ss_res = np.sum((C - C_pred)**2)
        ss_tot = np.sum((C - C.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "C0": C0, "nu": nu, "lambda": lam, "R2": r2,
            "status": "success",
        }
    except (RuntimeError, ValueError):
        return {"lambda": np.nan, "status": "fit_failed"}


def correlation_length_timeseries(df: pd.DataFrame,
                                  saai_col: str = "SAAI") -> pd.DataFrame:
    """
    Compute correlation length λ(t) for each year with spatial data.

    Only works for years with coordinate data (2017-2023).
    """
    records = []
    for year, grp in df.groupby("Year"):
        sub = grp.dropna(subset=[saai_col, "CountyLng", "CountyLat"]).copy()
        if len(sub) < 30:
            continue

        dist = compute_distance_matrix(sub)
        r_centers, C_r = compute_spatial_correlation_function(
            sub[saai_col].values, dist
        )

        if len(r_centers) > 0:
            fit = fit_correlation_length(r_centers, C_r)

            # Also compute Moran's I
            moran = compute_morans_i(sub[saai_col].values, dist)

            records.append({
                "Year": year,
                "N": len(sub),
                "Morans_I": moran["I"],
                "Morans_p": moran.get("p_value", np.nan),
                "lambda": fit.get("lambda", np.nan),
                "lambda_R2": fit.get("R2", np.nan),
            })

    return pd.DataFrame(records)
