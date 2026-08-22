"""Section 1: load raw CalHABMAP data, build the station-week table, add lag features,
horizon targets, and the cross-validation fold splitter used by every other module."""

import numpy as np
import pandas as pd

from config import (
    RAW_PREDICTOR_COLS, EXCLUDE_STATIONS, LAG_CANDIDATES, HORIZONS,
    THRESHOLD_ELEVATED, THRESHOLD_ALERT, N_FOLDS,
)


def load_raw(data_path, exclude_stations=EXCLUDE_STATIONS):
    raw = pd.read_csv(data_path, low_memory=False, parse_dates=["time_utc"])
    raw = raw[~raw["location_code"].isin(exclude_stations)].copy()
    raw = raw.dropna(subset=["alexandrium_spp_cells_l"])
    raw["iso_year"] = raw["time_utc"].dt.isocalendar().year
    raw["iso_week"] = raw["time_utc"].dt.isocalendar().week
    raw["station_week"] = (
        raw["location_code"] + "_" + raw["iso_year"].astype(str) + "_" + raw["iso_week"].astype(str)
    )
    return raw


def build_station_week_table(raw, predictor_cols=RAW_PREDICTOR_COLS):
    agg_target = raw.groupby("station_week")["alexandrium_spp_cells_l"].max().reset_index()
    agg_preds = raw.groupby("station_week")[list(predictor_cols.values())].mean().reset_index()
    agg_time = raw.groupby("station_week")["time_utc"].min().reset_index()
    agg_station = raw.groupby("station_week")["location_code"].first().reset_index()

    sw = (
        agg_target.merge(agg_preds, on="station_week")
        .merge(agg_time, on="station_week")
        .merge(agg_station, on="station_week")
    )
    sw = sw.rename(columns={v: k for k, v in predictor_cols.items()})
    sw = sw.sort_values(["time_utc", "location_code"]).reset_index(drop=True)
    return sw


def add_lag_features(sw, predictors=LAG_CANDIDATES, max_lag=1):
    for p in predictors:
        for lag in range(1, max_lag + 1):
            sw[f"{p}_lag{lag}"] = sw.groupby("location_code")[p].shift(lag)
    return sw


def add_horizon_targets(sw, horizons=HORIZONS, thresholds=(("elevated", THRESHOLD_ELEVATED), ("alert", THRESHOLD_ALERT))):
    for h in horizons:
        target_col = f"target_h{h}"
        sw[target_col] = (
            sw["alexandrium_spp_cells_l"] if h == 1
            else sw.groupby("location_code")["alexandrium_spp_cells_l"].shift(-(h - 1))
        )
        sw[f"log_target_h{h}"] = np.log1p(sw[target_col])
        for name, thr in thresholds:
            sw[f"label_{name}_h{h}"] = (sw[target_col] >= thr).astype(int)
    return sw


def make_folds(n, k=N_FOLDS):
    bounds = np.linspace(0, n, k + 1, dtype=int)
    return [(np.arange(0, bounds[i + 1]), np.arange(bounds[i + 1], bounds[i + 2])) for i in range(k - 1)]

