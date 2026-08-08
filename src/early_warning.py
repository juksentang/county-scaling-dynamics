"""
Early warning signals for phase transitions:
variance, lag-1 autocorrelation, skewness time series.

Critical slowing down theory predicts that near a tipping point:
- Variance increases (system explores wider state space)
- Autocorrelation increases (recovery rate slows)
- Skewness changes sign (asymmetric fluctuations)
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_early_warning_signals(df: pd.DataFrame,
                                  saai_col: str = "SAAI",
                                  poverty_only: bool = False) -> pd.DataFrame:
    """
    Compute early warning signals from SAAI time series.

    For each year t, compute across all counties:
    - Variance σ²(t)
    - Lag-1 autocorrelation ρ(t) (spatial, across counties)
    - Skewness γ₁(t)
    - Kurtosis κ(t)

    Parameters
    ----------
    df : DataFrame with Year, CountyCode, and saai_col.
    poverty_only : if True, restrict to poverty counties.

    Returns
    -------
    DataFrame with Year and EWS metrics.
    """
    if poverty_only and "IsPoverty" in df.columns:
        data = df[df["IsPoverty"] == True].copy()
    else:
        data = df.copy()

    records = []
    for year, grp in data.groupby("Year"):
        xi = grp[saai_col].dropna().values
        if len(xi) < 10:
            continue

        record = {
            "Year": year,
            "N": len(xi),
            "variance": np.var(xi),
            "std": np.std(xi),
            "skewness": stats.skew(xi),
            "kurtosis": stats.kurtosis(xi),
            "mean": np.mean(xi),
        }

        records.append(record)

    ews_df = pd.DataFrame(records).sort_values("Year")

    # Compute temporal lag-1 autocorrelation (across years for same county)
    # This requires panel structure
    if "CountyCode" in data.columns:
        autocorrs = []
        for year in ews_df["Year"].values:
            next_year = year + 1
            current = data[data["Year"] == year][["CountyCode", saai_col]]
            future = data[data["Year"] == next_year][["CountyCode", saai_col]]

            merged = current.merge(future, on="CountyCode", suffixes=("_t", "_t1"))
            if len(merged) >= 10:
                rho = merged[f"{saai_col}_t"].corr(merged[f"{saai_col}_t1"])
                autocorrs.append(rho)
            else:
                autocorrs.append(np.nan)

        ews_df["autocorrelation"] = autocorrs

    return ews_df


def detect_transition_point(ews_df: pd.DataFrame) -> dict:
    """
    Estimate the transition year from EWS signals.

    Methods:
    1. Variance peak
    2. Autocorrelation peak
    3. Skewness sign change
    """
    result = {}

    # Variance peak
    if "variance" in ews_df.columns:
        peak_idx = ews_df["variance"].idxmax()
        result["variance_peak_year"] = ews_df.loc[peak_idx, "Year"]
        result["variance_peak_value"] = ews_df.loc[peak_idx, "variance"]

    # Autocorrelation peak
    if "autocorrelation" in ews_df.columns:
        ac = ews_df["autocorrelation"].dropna()
        if len(ac) > 0:
            peak_idx = ac.idxmax()
            result["autocorr_peak_year"] = ews_df.loc[peak_idx, "Year"]
            result["autocorr_peak_value"] = ews_df.loc[peak_idx, "autocorrelation"]

    # Skewness sign change
    if "skewness" in ews_df.columns:
        sk = ews_df["skewness"].values
        years = ews_df["Year"].values
        for i in range(len(sk) - 1):
            if sk[i] < 0 and sk[i+1] > 0:
                result["skewness_sign_change_year"] = years[i]
                break

    return result


def critical_slowing_down_test(ews_df: pd.DataFrame,
                               pre_period: tuple = (2000, 2015)) -> dict:
    """
    Kendall τ trend test for critical slowing down.

    Tests whether variance and autocorrelation show increasing trends
    in the pre-transition period.
    """
    pre = ews_df[
        (ews_df["Year"] >= pre_period[0]) & (ews_df["Year"] <= pre_period[1])
    ].copy()

    if len(pre) < 5:
        return {"status": "insufficient_data"}

    result = {}
    for metric in ["variance", "autocorrelation", "skewness"]:
        if metric in pre.columns:
            vals = pre[metric].dropna().values
            if len(vals) >= 5:
                tau, p = stats.kendalltau(range(len(vals)), vals)
                result[f"{metric}_tau"] = tau
                result[f"{metric}_p"] = p
                result[f"{metric}_trend"] = (
                    "increasing" if tau > 0 and p < 0.05 else
                    "decreasing" if tau < 0 and p < 0.05 else
                    "no_trend"
                )

    return result
