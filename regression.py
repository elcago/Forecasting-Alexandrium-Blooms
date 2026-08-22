"""Section 9: log-abundance regression (Table 8) at h=1,2,3 — XGBoost, Random
Forest, the Bouquet-style tree, the LSTM, and a null (historical-mean) baseline."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import (
    BOUQUET_REG_MAX_DEPTH, FINAL_FEATURES, FINAL_LSTM_PARAMS, FINAL_RF_REG_PARAMS,
    FINAL_XGB_REG_PARAMS, HORIZONS, N_FOLDS, RANDOM_STATE, SEQ_LEN,
)
from data_prep import make_folds


def evaluate_regressor(model_fn, data, features, target_col, k=N_FOLDS):
    d = data.dropna(subset=features + [target_col]).reset_index(drop=True)
    rmses, maes, r2s = [], [], []
    for tr_idx, te_idx in make_folds(len(d), k):
        tr, te = d.iloc[tr_idx], d.iloc[te_idx]
        Xtr, Xte = tr[features].values, te[features].values
        ytr, yte = tr[target_col].values, te[target_col].values
        m = model_fn()
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
        maes.append(mean_absolute_error(yte, pred))
        r2s.append(r2_score(yte, pred))
    return np.mean(rmses), np.mean(maes), np.mean(r2s)


def build_lstm_regression_sequences(data, features, target_col, seq_len=SEQ_LEN):
    d = data.dropna(subset=features + [target_col]).reset_index(drop=True)
    X, y, end_times = [], [], []
    for _, grp in d.groupby("location_code"):
        grp = grp.sort_values("time_utc").reset_index(drop=True)
        if len(grp) <= seq_len:
            continue
        mat = StandardScaler().fit_transform(grp[features].values.astype(float))
        target = grp[target_col].values
        times = grp["time_utc"].values
        for i in range(seq_len, len(grp)):
            X.append(mat[i - seq_len:i])
            y.append(target[i])
            end_times.append(times[i])
    X, y, end_times = np.array(X), np.array(y), np.array(end_times)
    order = np.argsort(end_times)
    return X[order], y[order]


def evaluate_lstm_regressor(data, features, target_col, hidden, dropout, lr, batch_size, k=N_FOLDS, seq_len=SEQ_LEN):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    X, y = build_lstm_regression_sequences(data, features, target_col, seq_len)
    rmses, maes, r2s = [], [], []
    for tr_idx, te_idx in make_folds(len(X), k):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        tf.random.set_seed(42)
        model = Sequential([
            KerasLSTM(hidden, input_shape=(seq_len, len(features))),
            Dropout(dropout),
            Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
        model.fit(Xtr, ytr, epochs=15, batch_size=batch_size, verbose=0,
                  callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        pred = model.predict(Xte, verbose=0).flatten()
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
        maes.append(mean_absolute_error(yte, pred))
        r2s.append(r2_score(yte, pred))
    return np.mean(rmses), np.mean(maes), np.mean(r2s)


def table8_regression(sw, features=None, horizons=HORIZONS, skip_lstm=False):
    features = features or [f"{f}_lag1" for f in FINAL_FEATURES]
    rows = []
    for h in horizons:
        target_col = f"log_target_h{h}"

        rmse, mae, r2 = evaluate_regressor(
            lambda: xgb.XGBRegressor(**FINAL_XGB_REG_PARAMS, verbosity=0, random_state=RANDOM_STATE),
            sw, features, target_col)
        rows.append({"horizon": h, "model": "XGBoost", "rmse": rmse, "mae": mae, "r2": r2})

        rmse, mae, r2 = evaluate_regressor(
            lambda: RandomForestRegressor(**FINAL_RF_REG_PARAMS, random_state=RANDOM_STATE),
            sw, features, target_col)
        rows.append({"horizon": h, "model": "Random Forest", "rmse": rmse, "mae": mae, "r2": r2})

        rmse, mae, r2 = evaluate_regressor(
            lambda: DecisionTreeRegressor(max_depth=BOUQUET_REG_MAX_DEPTH, random_state=RANDOM_STATE),
            sw, features, target_col)
        rows.append({"horizon": h, "model": "Bouquet-style tree", "rmse": rmse, "mae": mae, "r2": r2})

        rmse, mae, r2 = evaluate_regressor(lambda: DummyRegressor(strategy="mean"), sw, features, target_col)
        rows.append({"horizon": h, "model": "Null baseline", "rmse": rmse, "mae": mae, "r2": r2})

        if not skip_lstm:
            rmse, mae, r2 = evaluate_lstm_regressor(sw, features, target_col, **FINAL_LSTM_PARAMS)
            rows.append({"horizon": h, "model": "LSTM", "rmse": rmse, "mae": mae, "r2": r2})
    return pd.DataFrame(rows)


