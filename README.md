# Forecasting-Alexandrium-Blooms

"""Config: constants and hyperparameters shared across all modules."""

import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ===== CONFIG =====

EXCLUDE_STATIONS = ["HSB", "HUM", "TP"]  # stations with 0% Alexandrium data

RAW_PREDICTOR_COLS = {
    "temp": "temp_degree_c",
    "chl": "avg_chloro_mg_m3",
    "nitrate": "nitrate_um",
    "nitrite": "nitrite_um",
    "ammonium": "ammonium_um",
    "phosphate": "phosphate_um",
    "silicate": "silicate_um",
    "salinity": "salinity",
}
LAG_CANDIDATES = ["temp", "chl", "nitrate", "nitrite", "ammonium", "phosphate", "silicate"]  # no salinity

THRESHOLD_ELEVATED = 1_000
THRESHOLD_ALERT = 10_000
ALL_THRESHOLDS = [100, 1_000, 5_000, 10_000, 20_000]

N_FOLDS = 8
STOP_MARGIN = 0.002
CV_XGB_PARAMS = dict(n_estimators=150, max_depth=4)  # scoring model for feature selection

FINAL_FEATURES = ["temp", "chl", "nitrite", "silicate"]  # using feature set found above

# Classification hyperparameters
FINAL_XGB_PARAMS = dict(
    n_estimators=30, max_depth=2, learning_rate=0.03,
    subsample=0.85, colsample_bytree=1.0, min_child_weight=3,
)
FINAL_RF_PARAMS = dict(n_estimators=300, max_depth=1, min_samples_leaf=10, max_features=None)
FINAL_BOUQUET_PARAMS = dict(max_depth=2, min_samples_split=5, min_samples_leaf=5, criterion="gini")
FINAL_LSTM_PARAMS = dict(hidden=64, dropout=0.3, lr=0.001, batch_size=64)
SEQ_LEN = 8
N_LSTM_SEEDS = 5

# Regression hyperparameters
FINAL_XGB_REG_PARAMS = dict(n_estimators=20, max_depth=2, learning_rate=0.1, subsample=0.7)
FINAL_RF_REG_PARAMS = dict(n_estimators=200, max_depth=2, min_samples_leaf=10)
BOUQUET_REG_MAX_DEPTH = 5

HORIZONS = (1, 2, 3)
