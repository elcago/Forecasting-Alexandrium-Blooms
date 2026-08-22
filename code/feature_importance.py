"""Section 8: feature importance (Table 7) for the tree models via bootstrapped
gain/impurity scores, and for the LSTM via permutation of the most recent week."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from config import FINAL_BOUQUET_PARAMS, FINAL_FEATURES, FINAL_LSTM_PARAMS, FINAL_RF_PARAMS, FINAL_XGB_PARAMS, N_FOLDS, RANDOM_STATE, SEQ_LEN
from data_prep import make_folds
from classification import build_lstm_sequences


def _unweighted_classifier_fns():
    return {
        "XGBoost": lambda yb: xgb.XGBClassifier(
            **FINAL_XGB_PARAMS, verbosity=0, eval_metric="logloss", random_state=RANDOM_STATE),
        "Random Forest": lambda yb: RandomForestClassifier(
            **FINAL_RF_PARAMS, random_state=RANDOM_STATE),
        "Bouquet-style tree": lambda yb: DecisionTreeClassifier(
            **FINAL_BOUQUET_PARAMS, random_state=RANDOM_STATE),
    }


def table7_tree_importance(sw, features=None, n_boot=30):
    features = features or [f"{f}_lag1" for f in FINAL_FEATURES]
    fns = _unweighted_classifier_fns()
    rows = []
    for thr_name in ["elevated", "alert"]:
        label_col = f"label_{thr_name}_h1"
        d = sw.dropna(subset=features + [label_col]).reset_index(drop=True)
        tr_idx, _ = make_folds(len(d))[-1]
        tr = d.iloc[tr_idx].reset_index(drop=True)

        for model_name, fn in fns.items():
            rng = np.random.default_rng(RANDOM_STATE)
            boot_importances = {f: [] for f in features}
            for _ in range(n_boot):
                boot_idx = rng.choice(len(tr), size=len(tr), replace=True)
                Xb = tr[features].values[boot_idx]
                yb = tr[label_col].values[boot_idx]
                if len(np.unique(yb)) < 2:
                    continue  # skip a bootstrap draw that happened to have only one class
                m = fn(yb)
                m.fit(Xb, yb)
                for f, imp in zip(features, m.feature_importances_):
                    boot_importances[f].append(imp)
            for f, vals in boot_importances.items():
                rows.append({"model": model_name, "threshold": thr_name, "feature": f,
                             "importance": np.mean(vals)})
    return pd.DataFrame(rows)


def table7_lstm_importance(sw, features=None, n_repeats=30, seq_len=SEQ_LEN):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    features = features or [f"{f}_lag1" for f in FINAL_FEATURES]
    rows = []
    for thr_name in ["elevated", "alert"]:
        label_col = f"label_{thr_name}_h1"
        X, y = build_lstm_sequences(sw, features, label_col, seq_len)
        split = int(len(X) * (1 - 1 / N_FOLDS))
        Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

        tf.random.set_seed(42)
        model = Sequential([
            KerasLSTM(FINAL_LSTM_PARAMS["hidden"], input_shape=(seq_len, len(features))),
            Dropout(FINAL_LSTM_PARAMS["dropout"]),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(FINAL_LSTM_PARAMS["lr"]), loss="binary_crossentropy")
        wts = np.where(ytr == 1, (ytr == 0).sum() / max(ytr.sum(), 1), 1.0)
        model.fit(Xtr, ytr, sample_weight=wts, epochs=15, batch_size=FINAL_LSTM_PARAMS["batch_size"],
                  verbose=0, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

        baseline_ap = average_precision_score(yte, model.predict(Xte, verbose=0).flatten())
        for feat_idx, feat_name in enumerate(features):
            rng = np.random.default_rng(0)
            drops = []
            for _ in range(n_repeats):
                Xte_shuf = Xte.copy()
                last_step = Xte_shuf[:, -1, feat_idx].copy()
                rng.shuffle(last_step)
                Xte_shuf[:, -1, feat_idx] = last_step
                ap = average_precision_score(yte, model.predict(Xte_shuf, verbose=0).flatten())
                drops.append(baseline_ap - ap)
            rows.append({"model": "LSTM", "threshold": thr_name, "feature": feat_name,
                         "importance": np.mean(drops)})
    return pd.DataFrame(rows)


