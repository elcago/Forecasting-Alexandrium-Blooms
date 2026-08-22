"""Section 7: classification models (Table 6) at h=1,2,3 for the elevated and
alert thresholds — XGBoost, Random Forest, the Bouquet-style tree, and the LSTM."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    FINAL_BOUQUET_PARAMS, FINAL_FEATURES, FINAL_LSTM_PARAMS, FINAL_RF_PARAMS,
    FINAL_XGB_PARAMS, HORIZONS, N_FOLDS, N_LSTM_SEEDS, RANDOM_STATE, SEQ_LEN,
)
from data_prep import make_folds


def evaluate_sklearn_classifier(model_fn, data, label_col, features, k=N_FOLDS):
    d = data.dropna(subset=features + [label_col]).reset_index(drop=True)
    precs, recs, aurocs = [], [], []
    for tr_idx, te_idx in make_folds(len(d), k):
        tr, te = d.iloc[tr_idx], d.iloc[te_idx]
        if tr[label_col].sum() < 3 or te[label_col].sum() < 1:
            continue
        Xtr, Xte = tr[features].values, te[features].values
        ytr, yte = tr[label_col].values, te[label_col].values
        m = model_fn(ytr)
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        precs.append(precision_score(yte, pred, zero_division=0))
        recs.append(recall_score(yte, pred, zero_division=0))
        if len(np.unique(yte)) > 1:
            aurocs.append(roc_auc_score(yte, prob))
    return np.mean(precs), np.mean(recs), (np.mean(aurocs) if aurocs else np.nan)


def build_lstm_sequences(data, features, label_col, seq_len=SEQ_LEN):
    d = data.dropna(subset=features + [label_col]).reset_index(drop=True)
    X, y, end_times = [], [], []
    for _, grp in d.groupby("location_code"):
        grp = grp.sort_values("time_utc").reset_index(drop=True)
        if len(grp) <= seq_len:
            continue
        mat = StandardScaler().fit_transform(grp[features].values.astype(float))
        lb = grp[label_col].values
        times = grp["time_utc"].values
        for i in range(seq_len, len(grp)):
            X.append(mat[i - seq_len:i])
            y.append(lb[i])
            end_times.append(times[i])
    X, y, end_times = np.array(X), np.array(y), np.array(end_times)
    order = np.argsort(end_times)
    return X[order], y[order]


def evaluate_lstm_classifier(data, features, label_col, hidden, dropout, lr, batch_size,
                              n_seeds=N_LSTM_SEEDS, k=N_FOLDS, seq_len=SEQ_LEN):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    X, y = build_lstm_sequences(data, features, label_col, seq_len)
    seed_p, seed_r, seed_a = [], [], []
    for seed in range(n_seeds):
        p_list, r_list, a_list = [], [], []
        for tr_idx, te_idx in make_folds(len(X), k):
            ytr, yte = y[tr_idx], y[te_idx]
            if ytr.sum() < 3 or yte.sum() < 1:
                continue
            wts = np.where(ytr == 1, (ytr == 0).sum() / max(ytr.sum(), 1), 1.0)
            tf.random.set_seed(seed * 7 + 42)
            model = Sequential([
                KerasLSTM(hidden, input_shape=(seq_len, len(features))),
                Dropout(dropout),
                Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
            model.fit(X[tr_idx], ytr, sample_weight=wts, epochs=15, batch_size=batch_size,
                      verbose=0, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
            prob = model.predict(X[te_idx], verbose=0).flatten()
            pred = (prob >= 0.5).astype(int)
            p_list.append(precision_score(yte, pred, zero_division=0))
            r_list.append(recall_score(yte, pred, zero_division=0))
            if len(np.unique(yte)) > 1:
                a_list.append(roc_auc_score(yte, prob))
        if p_list:
            seed_p.append(np.mean(p_list))
            seed_r.append(np.mean(r_list))
            seed_a.append(np.mean(a_list) if a_list else np.nan)
    return np.mean(seed_p), np.mean(seed_r), np.nanmean(seed_a)


def _classifier_fns(horizon):
    if horizon == 1:
        bouquet_fn = lambda ytr: DecisionTreeClassifier(
            **FINAL_BOUQUET_PARAMS, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        bouquet_fn = lambda ytr: DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)

    return {
        "XGBoost": lambda ytr: xgb.XGBClassifier(
            **FINAL_XGB_PARAMS, scale_pos_weight=(ytr == 0).sum() / max(ytr.sum(), 1),
            verbosity=0, eval_metric="logloss", random_state=RANDOM_STATE),
        "Random Forest": lambda ytr: RandomForestClassifier(
            **FINAL_RF_PARAMS, class_weight="balanced", random_state=RANDOM_STATE),
        "Bouquet-style tree": bouquet_fn,
    }


def table6_classification(sw, features=None, horizons=HORIZONS, skip_lstm=False):
    features = features or [f"{f}_lag1" for f in FINAL_FEATURES]
    rows = []
    for h in horizons:
        for thr_name in ["elevated", "alert"]:
            label_col = f"label_{thr_name}_h{h}"
            for model_name, fn in _classifier_fns(h).items():
                p, r, a = evaluate_sklearn_classifier(fn, sw, label_col, features)
                rows.append({"horizon": h, "threshold": thr_name, "model": model_name,
                             "precision": p, "recall": r, "auroc": a})
            if not skip_lstm:
                p, r, a = evaluate_lstm_classifier(sw, features, label_col, **FINAL_LSTM_PARAMS)
                rows.append({"horizon": h, "threshold": thr_name, "model": "LSTM",
                             "precision": p, "recall": r, "auroc": a})
    return pd.DataFrame(rows)


