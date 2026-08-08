"""
Distribution analysis: GMM fitting, Boltzmann-Gibbs, KDE, effective potential V_eff.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture


@dataclass
class GMMResult:
    """Result of a Gaussian Mixture Model fit."""
    n_components: int
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    bic: float
    aic: float
    log_likelihood: float


def fit_gmm(xi: np.ndarray, n_components: int = 2,
            n_init: int = 20, random_state: int = 42) -> GMMResult:
    """
    Fit Gaussian Mixture Model to SAAI values.

    Parameters
    ----------
    xi : 1D array of SAAI values.
    n_components : number of mixture components.
    n_init : number of random initializations.
    random_state : random seed.

    Returns
    -------
    GMMResult dataclass.
    """
    xi = xi[np.isfinite(xi)].reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        random_state=random_state,
    )
    gmm.fit(xi)

    return GMMResult(
        n_components=n_components,
        weights=gmm.weights_.flatten(),
        means=gmm.means_.flatten(),
        covariances=gmm.covariances_.flatten(),
        bic=gmm.bic(xi),
        aic=gmm.aic(xi),
        log_likelihood=gmm.score(xi) * len(xi),
    )


def fit_gmm_timeseries(df: pd.DataFrame,
                       saai_col: str = "SAAI",
                       max_k: int = 3) -> pd.DataFrame:
    """
    Fit GMM year-by-year with BIC-based model selection.

    Returns DataFrame with Year, BestK, weights, means, stds, BIC for each K.
    """
    records = []
    for year, grp in df.groupby("Year"):
        xi = grp[saai_col].dropna().values
        if len(xi) < 30:
            continue

        best_bic = np.inf
        best_k = 1
        all_results = {}

        for k in range(1, max_k + 1):
            try:
                result = fit_gmm(xi, n_components=k)
                all_results[k] = result
                if result.bic < best_bic:
                    best_bic = result.bic
                    best_k = k
            except Exception:
                pass

        record = {"Year": year, "N": len(xi), "BestK": best_k}

        # Store BIC for each K
        for k, res in all_results.items():
            record[f"BIC_K{k}"] = res.bic
            record[f"AIC_K{k}"] = res.aic

        # Store best model parameters
        best = all_results.get(best_k)
        if best is not None:
            # Sort components by mean
            order = np.argsort(best.means)
            for i, idx in enumerate(order):
                record[f"w{i+1}"] = best.weights[idx]
                record[f"mu{i+1}"] = best.means[idx]
                record[f"sigma{i+1}"] = np.sqrt(best.covariances[idx])

        records.append(record)

    return pd.DataFrame(records)


def fit_boltzmann_gibbs(xi: np.ndarray) -> dict:
    """
    Fit exponential (Boltzmann-Gibbs) distribution to positive SAAI tail.

    P(ξ) ∝ exp(-ξ/T) for ξ > 0, where T is the "temperature".
    """
    xi_pos = xi[xi > 0]
    if len(xi_pos) < 10:
        return {"T": np.nan, "n": 0}

    T = xi_pos.mean()  # MLE for exponential rate = 1/mean
    # KS test against exponential
    ks_stat, ks_p = stats.kstest(xi_pos, "expon", args=(0, T))

    return {
        "T": T,
        "n": len(xi_pos),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
    }


def compute_kde(xi: np.ndarray, bandwidth: str | float = "silverman",
                n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute KDE density estimate.

    Parameters
    ----------
    xi : 1D array of values.
    bandwidth : 'silverman', 'scott', or float.
    n_points : number of evaluation points.

    Returns
    -------
    (xi_grid, density): arrays for the KDE curve.
    """
    xi = xi[np.isfinite(xi)]
    if len(xi) < 5:
        return np.array([]), np.array([])

    if isinstance(bandwidth, str):
        kde = stats.gaussian_kde(xi, bw_method=bandwidth)
    else:
        kde = stats.gaussian_kde(xi, bw_method=bandwidth)

    xi_min = np.percentile(xi, 1) - 0.5
    xi_max = np.percentile(xi, 99) + 0.5
    xi_grid = np.linspace(xi_min, xi_max, n_points)
    density = kde(xi_grid)

    return xi_grid, density


def compute_effective_potential(xi: np.ndarray,
                               bandwidth: str | float = "silverman",
                               n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute effective potential V_eff(ξ) = -ln P̂(ξ) from KDE.

    This is Eq 7/16 in the framework. Minima in V_eff correspond to
    stable states; maxima correspond to unstable barriers.

    Returns
    -------
    (xi_grid, V_eff): arrays for the potential landscape.
    """
    xi_grid, density = compute_kde(xi, bandwidth=bandwidth, n_points=n_points)
    if len(density) == 0:
        return np.array([]), np.array([])

    # Avoid log(0)
    density_safe = np.maximum(density, 1e-10)
    V_eff = -np.log(density_safe)

    # Normalize: set minimum to 0
    V_eff -= V_eff.min()

    return xi_grid, V_eff


def find_potential_extrema(xi_grid: np.ndarray,
                           V_eff: np.ndarray) -> dict:
    """
    Find minima (stable states) and maxima (barriers) of V_eff.

    Returns dict with:
    - minima: list of (xi, V) for local minima
    - maxima: list of (xi, V) for local maxima
    - barrier_height: height of barrier between minima (if bimodal)
    """
    if len(xi_grid) < 5:
        return {"minima": [], "maxima": [], "barrier_height": 0}

    # Smooth V_eff slightly for robustness
    from scipy.ndimage import uniform_filter1d
    V_smooth = uniform_filter1d(V_eff, size=5)

    min_idx = argrelextrema(V_smooth, np.less, order=10)[0]
    max_idx = argrelextrema(V_smooth, np.greater, order=10)[0]

    minima = [(xi_grid[i], V_eff[i]) for i in min_idx]
    maxima = [(xi_grid[i], V_eff[i]) for i in max_idx]

    # Barrier height: maximum V between the two deepest minima
    barrier = 0.0
    if len(minima) >= 2:
        # Sort minima by V (deepest first)
        sorted_mins = sorted(minima, key=lambda x: x[1])
        m1_xi, m1_v = sorted_mins[0]
        m2_xi, m2_v = sorted_mins[1]
        # Find max V between them
        left = min(m1_xi, m2_xi)
        right = max(m1_xi, m2_xi)
        between = (xi_grid >= left) & (xi_grid <= right)
        if between.any():
            barrier = V_eff[between].max() - max(m1_v, m2_v)

    return {
        "minima": minima,
        "maxima": maxima,
        "n_minima": len(minima),
        "n_maxima": len(maxima),
        "barrier_height": barrier,
    }


def compute_veff_timeseries(df: pd.DataFrame,
                            saai_col: str = "SAAI",
                            n_points: int = 200) -> dict:
    """
    Compute V_eff for each year.

    Returns dict of year → (xi_grid, V_eff).
    """
    result = {}
    for year, grp in df.groupby("Year"):
        xi = grp[saai_col].dropna().values
        if len(xi) < 30:
            continue
        xi_grid, veff = compute_effective_potential(xi, n_points=n_points)
        result[year] = (xi_grid, veff)
    return result


def compare_distribution_models(xi: np.ndarray, max_k: int = 3) -> pd.DataFrame:
    """
    Compare 1, 2, 3-component GMM by BIC.

    Returns DataFrame with K, BIC, AIC, delta_BIC.
    """
    xi = xi[np.isfinite(xi)]
    records = []
    for k in range(1, max_k + 1):
        try:
            result = fit_gmm(xi, n_components=k)
            records.append({
                "K": k,
                "BIC": result.bic,
                "AIC": result.aic,
                "LogLik": result.log_likelihood,
            })
        except Exception:
            pass

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["delta_BIC"] = df["BIC"] - df["BIC"].min()
    return df
