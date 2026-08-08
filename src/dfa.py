"""
Detrended Fluctuation Analysis (DFA) and Multifractal DFA (MFDFA).

Supports both temporal DFA (per-county time series) and cross-sectional DFA
(SAAI values across counties in a given year, ordered spatially).
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_dfa(series: np.ndarray,
                scales: np.ndarray | None = None,
                order: int = 1) -> dict:
    """
    Standard Detrended Fluctuation Analysis.

    Parameters
    ----------
    series : 1-D array of values.
    scales : array of window sizes. Default: log-spaced from 4 to N//4.
    order : polynomial order for detrending (1 = linear).

    Returns
    -------
    dict with scales, F (fluctuation function), alpha (Hurst exponent),
    alpha_se, intercept, r_squared.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)

    if N < 16:
        return {"status": "insufficient_data", "n": N}

    # Integrate: cumulative sum of deviations from mean
    profile = np.cumsum(x - x.mean())

    if scales is None:
        min_scale = max(4, order + 2)
        max_scale = N // 4
        if max_scale <= min_scale:
            max_scale = N // 2
        n_scales = min(20, max_scale - min_scale + 1)
        scales = np.unique(np.logspace(
            np.log10(min_scale), np.log10(max_scale), n_scales
        ).astype(int))

    F = np.full(len(scales), np.nan)

    for i, s in enumerate(scales):
        s = int(s)
        if s > N or s < order + 2:
            continue

        n_segments = N // s
        if n_segments < 1:
            continue

        var_sum = 0.0
        count = 0
        for seg in range(n_segments):
            start = seg * s
            end = start + s
            segment = profile[start:end]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            var_sum += np.mean((segment - trend) ** 2)
            count += 1

        # Also from the end (standard DFA practice)
        for seg in range(n_segments):
            start = N - (seg + 1) * s
            end = start + s
            if start < 0:
                break
            segment = profile[start:end]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            var_sum += np.mean((segment - trend) ** 2)
            count += 1

        if count > 0:
            F[i] = np.sqrt(var_sum / count)

    valid = np.isfinite(F) & (F > 0) & (scales > 0)
    if valid.sum() < 3:
        return {"status": "insufficient_scales", "n_valid": int(valid.sum())}

    log_s = np.log(scales[valid])
    log_F = np.log(F[valid])
    slope, intercept, r_val, p_val, se = stats.linregress(log_s, log_F)

    return {
        "scales": scales[valid],
        "F": F[valid],
        "alpha": slope,
        "alpha_se": se,
        "intercept": intercept,
        "r_squared": r_val ** 2,
        "p_value": p_val,
        "n": N,
    }


def compute_cross_sectional_dfa(df: pd.DataFrame,
                                year: int,
                                saai_col: str = "SAAI",
                                sort_col: str = "CountyCode",
                                order: int = 1) -> dict:
    """
    Cross-sectional DFA: treat SAAI values across ~1800 counties as a
    spatial series ordered by sort_col (default: CountyCode).

    Parameters
    ----------
    df : DataFrame with Year, CountyCode, saai_col.
    year : year to analyze.
    saai_col : SAAI column.
    sort_col : column for spatial ordering (CountyCode or CountyLng).

    Returns
    -------
    dict from compute_dfa plus year and n_counties.
    """
    sub = df[df["Year"] == year].dropna(subset=[saai_col])
    if sort_col in sub.columns:
        sub = sub.sort_values(sort_col)
    else:
        sub = sub.sort_values("CountyCode")

    series = sub[saai_col].values
    result = compute_dfa(series, order=order)
    result["year"] = year
    result["n_counties"] = len(series)
    return result


def compute_cross_sectional_dfa_timeseries(df: pd.DataFrame,
                                            saai_col: str = "SAAI",
                                            sort_col: str = "CountyCode",
                                            order: int = 1) -> pd.DataFrame:
    """
    Run cross-sectional DFA for each year. Returns DataFrame with
    Year, alpha, alpha_se, r_squared, n_counties.
    """
    records = []
    for year in sorted(df["Year"].unique()):
        result = compute_cross_sectional_dfa(df, year, saai_col, sort_col, order)
        if "alpha" in result:
            records.append({
                "Year": year,
                "alpha": result["alpha"],
                "alpha_se": result["alpha_se"],
                "r_squared": result["r_squared"],
                "n_counties": result["n_counties"],
            })

    return pd.DataFrame(records)


def compute_mfdfa(series: np.ndarray,
                  scales: np.ndarray | None = None,
                  q_range: np.ndarray | None = None,
                  order: int = 1) -> dict:
    """
    Multifractal Detrended Fluctuation Analysis.

    Parameters
    ----------
    series : 1-D array.
    scales : window sizes.
    q_range : array of q values (moments). Default: [-5, -3, -1, 0, 1, 2, 3, 5].
    order : detrending polynomial order.

    Returns
    -------
    dict with q, h_q (generalized Hurst exponents), tau_q (scaling exponents),
    D_q (multifractal spectrum), alpha_q (singularity strengths).
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)

    if N < 32:
        return {"status": "insufficient_data", "n": N}

    profile = np.cumsum(x - x.mean())

    if scales is None:
        min_scale = max(4, order + 2)
        max_scale = N // 4
        n_scales = min(20, max_scale - min_scale + 1)
        scales = np.unique(np.logspace(
            np.log10(min_scale), np.log10(max_scale), n_scales
        ).astype(int))

    if q_range is None:
        q_range = np.array([-5, -3, -1, 0, 1, 2, 3, 5], dtype=float)

    # Compute segment variances for each scale
    Fq = np.full((len(q_range), len(scales)), np.nan)

    for j, s in enumerate(scales):
        s = int(s)
        if s > N or s < order + 2:
            continue

        n_segments = N // s
        if n_segments < 1:
            continue

        seg_vars = []
        for seg in range(n_segments):
            start = seg * s
            segment = profile[start:start + s]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            v = np.mean((segment - trend) ** 2)
            if v > 0:
                seg_vars.append(v)

        # From the end
        for seg in range(n_segments):
            start = N - (seg + 1) * s
            if start < 0:
                break
            segment = profile[start:start + s]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            v = np.mean((segment - trend) ** 2)
            if v > 0:
                seg_vars.append(v)

        if len(seg_vars) < 2:
            continue

        seg_vars = np.array(seg_vars)
        for i, q in enumerate(q_range):
            if q == 0:
                Fq[i, j] = np.exp(0.5 * np.mean(np.log(seg_vars)))
            else:
                Fq[i, j] = np.mean(seg_vars ** (q / 2)) ** (1.0 / q)

    # Fit h(q) for each q
    h_q = np.full(len(q_range), np.nan)
    h_q_se = np.full(len(q_range), np.nan)

    for i in range(len(q_range)):
        valid = np.isfinite(Fq[i]) & (Fq[i] > 0)
        if valid.sum() < 3:
            continue
        log_s = np.log(scales[valid])
        log_F = np.log(Fq[i, valid])
        slope, _, _, _, se = stats.linregress(log_s, log_F)
        h_q[i] = slope
        h_q_se[i] = se

    # Multifractal spectrum
    tau_q = q_range * h_q - 1
    # Legendre transform for singularity spectrum
    alpha_mf = np.gradient(tau_q, q_range)
    f_alpha = q_range * alpha_mf - tau_q

    return {
        "q": q_range,
        "h_q": h_q,
        "h_q_se": h_q_se,
        "tau_q": tau_q,
        "alpha_mf": alpha_mf,
        "f_alpha": f_alpha,
        "scales": scales,
        "Fq": Fq,
        "n": N,
    }


def _phase_randomize(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a phase-randomized surrogate preserving the power spectrum."""
    n = len(series)
    ft = np.fft.rfft(series)
    phases = rng.uniform(0, 2 * np.pi, len(ft))
    phases[0] = 0  # preserve DC component
    if n % 2 == 0:
        phases[-1] = 0  # Nyquist component stays real
    ft_surr = np.abs(ft) * np.exp(1j * phases)
    return np.fft.irfft(ft_surr, n=n)


def compute_mfdfa_surrogate_test(series: np.ndarray,
                                  n_surrogates: int = 999,
                                  q_range: np.ndarray | None = None,
                                  order: int = 1,
                                  seed: int = 42) -> dict:
    """
    Surrogate data test for multifractality.

    Generates phase-randomized surrogates that preserve the power spectrum
    but destroy nonlinear structure, then compares the observed multifractal
    width Δh with the surrogate distribution.

    Parameters
    ----------
    series : 1-D array of SAAI values (spatially ordered).
    n_surrogates : number of surrogate realisations.
    q_range : moment orders (default same as compute_mfdfa).
    order : detrending order.
    seed : random seed for reproducibility.

    Returns
    -------
    dict with delta_h_obs, delta_h_surrogates, z_score, p_value,
    mean_surr, std_surr.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    if q_range is None:
        q_range = np.array([-5, -3, -1, 0, 1, 2, 3, 5], dtype=float)

    # Observed Δh
    obs = compute_mfdfa(x, q_range=q_range, order=order)
    if "h_q" not in obs or np.any(np.isnan(obs["h_q"])):
        return {"status": "mfdfa_failed"}

    h_q_obs = obs["h_q"]
    delta_h_obs = h_q_obs[0] - h_q_obs[-1]  # h(q_min) - h(q_max)

    # Surrogate distribution
    rng = np.random.default_rng(seed)
    delta_h_surr = np.full(n_surrogates, np.nan)

    for i in range(n_surrogates):
        surr = _phase_randomize(x, rng)
        res = compute_mfdfa(surr, q_range=q_range, order=order)
        if "h_q" in res and not np.any(np.isnan(res["h_q"])):
            delta_h_surr[i] = res["h_q"][0] - res["h_q"][-1]

    valid = np.isfinite(delta_h_surr)
    delta_h_surr = delta_h_surr[valid]
    mean_surr = np.mean(delta_h_surr)
    std_surr = np.std(delta_h_surr, ddof=1)

    z_score = (delta_h_obs - mean_surr) / std_surr if std_surr > 0 else np.inf
    p_value = np.mean(delta_h_surr >= delta_h_obs)  # one-sided

    return {
        "delta_h_obs": delta_h_obs,
        "delta_h_surrogates": delta_h_surr,
        "z_score": z_score,
        "p_value": p_value,
        "mean_surr": mean_surr,
        "std_surr": std_surr,
        "n_valid_surrogates": int(valid.sum()),
        "h_q_obs": h_q_obs,
        "q_range": q_range,
    }


# ════════════════════════════════════════════════════════════
# Geographic ordering robustness
# ════════════════════════════════════════════════════════════

def _hilbert_index(x: float, y: float, order: int = 16) -> int:
    """Map (x, y) in [0,1]^2 to Hilbert curve index (rotate-and-flip)."""
    n = 1 << order
    ix = int(np.clip(x * n, 0, n - 1))
    iy = int(np.clip(y * n, 0, n - 1))
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (ix & s) > 0 else 0
        ry = 1 if (iy & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate quadrant
        if ry == 0:
            if rx == 1:
                ix = s - 1 - ix
                iy = s - 1 - iy
            ix, iy = iy, ix
        s >>= 1
    return d


def compute_cross_sectional_dfa_geographic(
    df: pd.DataFrame,
    year: int,
    saai_col: str = "SAAI",
    lng_col: str = "CountyLng",
    lat_col: str = "CountyLat",
    ordering: str = "hilbert",
    order: int = 1,
) -> dict:
    """
    Cross-sectional DFA with geographic ordering.

    Parameters
    ----------
    ordering : 'hilbert' (Hilbert curve), 'longitude' (west→east),
               'latitude' (south→north).
    """
    sub = df[df["Year"] == year].dropna(subset=[saai_col, lng_col, lat_col]).copy()
    if len(sub) < 20:
        return {"year": year, "ordering": ordering, "status": "insufficient_data"}

    if ordering == "hilbert":
        # Normalise coordinates to [0,1]
        lng = sub[lng_col].values
        lat = sub[lat_col].values
        lng_n = (lng - lng.min()) / max(lng.max() - lng.min(), 1e-10)
        lat_n = (lat - lat.min()) / max(lat.max() - lat.min(), 1e-10)
        sub["_hilbert"] = [_hilbert_index(x, y) for x, y in zip(lng_n, lat_n)]
        sub = sub.sort_values("_hilbert")
    elif ordering == "longitude":
        sub = sub.sort_values(lng_col)
    elif ordering == "latitude":
        sub = sub.sort_values(lat_col)
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    series = sub[saai_col].values
    result = compute_dfa(series, order=order)
    result["year"] = year
    result["ordering"] = ordering
    result["n_counties"] = len(series)
    return result


def dfa_ordering_robustness(
    df: pd.DataFrame,
    saai_col: str = "SAAI",
    lng_col: str = "CountyLng",
    lat_col: str = "CountyLat",
) -> pd.DataFrame:
    """
    Compare DFA exponent across three spatial orderings for years
    with coordinate data.

    Returns DataFrame with Year, Ordering, alpha, alpha_se, r_squared, n.
    """
    # Identify years with coordinate data
    has_coords = df.dropna(subset=[lng_col, lat_col])
    years_with_coords = sorted(has_coords["Year"].unique())

    records = []
    for year in years_with_coords:
        # AdminCode ordering
        res_admin = compute_cross_sectional_dfa(df, year, saai_col, sort_col="CountyCode")
        if "alpha" in res_admin:
            records.append({
                "Year": year, "Ordering": "AdminCode",
                "alpha": res_admin["alpha"], "alpha_se": res_admin["alpha_se"],
                "r_squared": res_admin["r_squared"], "n": res_admin["n_counties"],
            })

        # Geographic orderings
        for ordering in ["longitude", "hilbert"]:
            res = compute_cross_sectional_dfa_geographic(
                df, year, saai_col, lng_col, lat_col, ordering=ordering
            )
            if "alpha" in res:
                records.append({
                    "Year": year, "Ordering": ordering.capitalize(),
                    "alpha": res["alpha"], "alpha_se": res["alpha_se"],
                    "r_squared": res["r_squared"], "n": res["n_counties"],
                })

    return pd.DataFrame(records)
