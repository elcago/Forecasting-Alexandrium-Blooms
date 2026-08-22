"""Section 6: hyperparameter search (Table 5) for XGBoost, Random Forest,
the Bouquet-style tree, and the LSTM. Slow — re-derives the values already
hardcoded in config.py, as a check rather than something to run every time."""

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE
from classification import evaluate_lstm_classifier


def tune_xgb_and_rf(sw, features, label_col="label_elevated_h1", tune_frac=0.6, n_iter=50):
    d = sw.dropna(subset=features + [label_col]).reset_index(drop=True)
    tune_df = d.iloc[: int(tune_frac * len(d))]
    X_tune, y_tune = tune_df[features].values, tune_df[label_col].values
    scale = (y_tune == 0).sum() / max(y_tune.sum(), 1)
    tscv = TimeSeriesSplit(n_splits=3)

    xgb_grid = {
        "n_estimators": [20, 30, 50, 100, 150, 200, 300, 500],
        "max_depth": [1, 2, 3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.10, 0.20],
        "subsample": [0.60, 0.70, 0.85, 1.0],
        "colsample_bytree": [0.60, 0.70, 0.85, 1.0],
        "min_child_weight": [1, 3, 5, 7],
    }
    rf_grid = {
        "n_estimators": [50, 100, 150, 200, 300, 500],
        "max_depth": [1, 2, 3, 5, 10, 20, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }

    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(scale_pos_weight=scale, verbosity=0, eval_metric="aucpr", random_state=RANDOM_STATE),
        xgb_grid, n_iter=n_iter, cv=tscv, scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_search.fit(X_tune, y_tune)

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        rf_grid, n_iter=n_iter, cv=tscv, scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_search.fit(X_tune, y_tune)
    return xgb_search.best_params_, rf_search.best_params_


def tune_bouquet(sw, features, label_col="label_elevated_h1", tune_frac=0.6, n_iter=40):
    d = sw.dropna(subset=features + [label_col]).reset_index(drop=True)
    tune_df = d.iloc[: int(tune_frac * len(d))]
    X_tune, y_tune = tune_df[features].values, tune_df[label_col].values
    tscv = TimeSeriesSplit(n_splits=3)

    grid = {
        "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    search = RandomizedSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        grid, n_iter=n_iter, cv=tscv, scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_tune, y_tune)
    return search.best_params_


def tune_lstm(sw, features, label_col="label_elevated_h1", n_random_combos=20):
    grid = {
        "hidden": [8, 16, 32, 64, 128],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "lr": [0.0005, 0.001, 0.005, 0.01],
        "batch_size": [16, 32, 64, 128],
    }
    import random
    random.seed(RANDOM_STATE)
    all_combos = [(h, d, lr, bs) for h in grid["hidden"] for d in grid["dropout"]
                  for lr in grid["lr"] for bs in grid["batch_size"]]
    sampled = random.sample(all_combos, n_random_combos)

    results = []
    for hidden, dropout, lr, bs in sampled:
        _, _, auroc = evaluate_lstm_classifier(sw, features, label_col, hidden, dropout, lr, bs, n_seeds=1)
        results.append((hidden, dropout, lr, bs, auroc))
    results.sort(key=lambda x: (x[4] if not np.isnan(x[4]) else -1), reverse=True)
    hidden, dropout, lr, bs, _ = results[0]
    return dict(hidden=hidden, dropout=dropout, lr=lr, batch_size=bs)


