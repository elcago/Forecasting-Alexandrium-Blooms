"""Sections 2-4: exceedance prevalence (Table 1), monthly exceedance counts (Figure 2),
missingness-bias check (Table 2), and Mood's median tests (Tables 3A/3B)."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, median_test

from config import ALL_THRESHOLDS, LAG_CANDIDATES, THRESHOLD_ALERT, THRESHOLD_ELEVATED
from data_prep import add_lag_features


def table1_exceedance_prevalence(sw, thresholds=ALL_THRESHOLDS):
    n = len(sw)
    rows = []
    for thr in thresholds:
        exceed = sw[sw["alexandrium_spp_cells_l"] >= thr]
        rows.append({
            "threshold": thr,
            "station_weeks_ge_threshold": len(exceed),
            "prevalence_pct": len(exceed) / n * 100,
            "stations_reporting_ge1": exceed["location_code"].nunique(),
        })
    return pd.DataFrame(rows)


def figure2_monthly_exceedances(sw, thresholds=(THRESHOLD_ALERT, THRESHOLD_ELEVATED)):
    month = sw["time_utc"].dt.month
    out = {}
    for thr in thresholds:
        counts = month[sw["alexandrium_spp_cells_l"] >= thr].value_counts().reindex(range(1, 13), fill_value=0)
        out[thr] = counts.sort_index()
    return out

# ===== 3. Missingness-bias check (Mann-Whitney U) =====

def table2_missingness_bias(sw, predictors=LAG_CANDIDATES, lags=(0, 1, 2, 3), min_n=5, alpha=0.05):
    sw = sw.sort_values(["location_code", "time_utc"]).reset_index(drop=True)
    for lag in lags:
        sw[f"_abundance_lead{lag}"] = sw.groupby("location_code")["alexandrium_spp_cells_l"].shift(-lag)

    rows = []
    for pred in predictors:
        measured = sw[pred].notna()
        row = {"predictor": pred, "weeks_missing": int((~measured).sum())}
        pvals = []
        for lag in lags:
            lead_col = f"_abundance_lead{lag}"
            present_vals = sw.loc[measured, lead_col].dropna()
            missing_vals = sw.loc[~measured, lead_col].dropna()
            if len(present_vals) < min_n or len(missing_vals) < min_n:
                p = np.nan
            else:
                _, p = mannwhitneyu(present_vals, missing_vals, alternative="two-sided")
            row[f"lag{lag}_p"] = p
            pvals.append(p)
        valid = [p for p in pvals if not np.isnan(p)]
        row["result"] = "biased" if valid and all(p < alpha for p in valid) else "no evidence of bias"
        rows.append(row)
    return pd.DataFrame(rows)


# ===== 4. Mood's median test =====

def _moods_p(a, b, min_n=5):
    if len(a) < min_n or len(b) < min_n:
        return np.nan
    try:
        _, p, _, _ = median_test(a, b)
        return p
    except Exception:
        return np.nan


def table3_moods_median_tests(sw, predictors=LAG_CANDIDATES + ["salinity"], lags=(0, 1, 2, 3)):
    sw = add_lag_features(sw, predictors=predictors, max_lag=max(lags))

    # lags 0-3, elevated + alert thresholds
    table3a_rows = {}
    for pred in predictors:
        row = {}
        for thr, thr_name in [(THRESHOLD_ELEVATED, "elev"), (THRESHOLD_ALERT, "alert")]:
            for lag in lags:
                col = pred if lag == 0 else f"{pred}_lag{lag}"
                sub = sw[["alexandrium_spp_cells_l", col]].dropna()
                elev = sub.loc[sub["alexandrium_spp_cells_l"] >= thr, col]
                non_elev = sub.loc[sub["alexandrium_spp_cells_l"] < thr, col]
                row[f"{thr_name}_lag{lag}"] = _moods_p(elev, non_elev)
        table3a_rows[pred] = row

    # lag 0, all five thresholds
    table3b_rows = {}
    for pred in predictors:
        row = {}
        for thr in ALL_THRESHOLDS:
            sub = sw[["alexandrium_spp_cells_l", pred]].dropna()
            elev = sub.loc[sub["alexandrium_spp_cells_l"] >= thr, pred]
            non_elev = sub.loc[sub["alexandrium_spp_cells_l"] < thr, pred]
            row[f"p_ge_{thr}"] = _moods_p(elev, non_elev)
        table3b_rows[pred] = row

    return pd.DataFrame(table3a_rows).T, pd.DataFrame(table3b_rows).T


