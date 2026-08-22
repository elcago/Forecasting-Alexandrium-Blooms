"""Section 5: the six independent feature-selection methods (Table 4) and the
Spearman correlation check among the final four predictors (Figure 3)."""

from collections import Counter

import numpy as np
import xgboost as xgb
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from config import CV_XGB_PARAMS, FINAL_FEATURES, LAG_CANDIDATES, N_FOLDS, RANDOM_STATE, STOP_MARGIN
from data_prep import make_folds


def cv_average_precision(features, data, label_col="label_elevated_h1", k=N_FOLDS):
    d = data.dropna(subset=features + [label_col]).reset_index(drop=True)
    if d[label_col].sum() < 15 or len(d) < 50:
        return np.nan
    scores = []
    for tr_idx, te_idx in make_folds(len(d), k):
        tr, te = d.iloc[tr_idx], d.iloc[te_idx]
        if tr[label_col].sum() < 3 or te[label_col].sum() < 1:
            continue
        Xtr, Xte = tr[features].values, te[features].values
        ytr, yte = tr[label_col].values, te[label_col].values
        scale = (ytr == 0).sum() / max(ytr.sum(), 1)
        m = xgb.XGBClassifier(**CV_XGB_PARAMS, scale_pos_weight=scale, verbosity=0,
                               eval_metric="logloss", random_state=RANDOM_STATE)
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        scores.append(average_precision_score(yte, prob))
    return np.mean(scores) if scores else np.nan


def forward_stepwise_selection(candidates, sw, label_col="label_elevated_h1", stop_margin=STOP_MARGIN):
    selected, remaining = [], list(candidates)
    for step in range(len(candidates)):
        results = [(c, cv_average_precision(selected + [c], sw, label_col)) for c in remaining]
        results = [(c, s) for c, s in results if not np.isnan(s)]
        if not results:
            break
        results.sort(key=lambda x: x[1], reverse=True)
        best_c, best_s = results[0]
        prev_best = cv_average_precision(selected, sw, label_col) if selected else -np.inf
        if best_s > prev_best + stop_margin or step == 0:
            selected.append(best_c)
            remaining.remove(best_c)
        else:
            break
    return selected


def backward_elimination_selection(candidates, sw, label_col="label_elevated_h1", stop_margin=STOP_MARGIN):
    current = list(candidates)
    current_ap = cv_average_precision(current, sw, label_col)
    while len(current) > 1:
        results = [(f, cv_average_precision([x for x in current if x != f], sw, label_col)) for f in current]
        results.sort(key=lambda x: x[1], reverse=True)
        best_remove, best_ap_after = results[0]
        if best_ap_after >= current_ap - stop_margin:
            current.remove(best_remove)
            current_ap = best_ap_after
        else:
            break
    return current


def lasso_selection(candidates, X, y):
    X_scaled = StandardScaler().fit_transform(X)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                class_weight="balanced", random_state=RANDOM_STATE)
    lasso.fit(X_scaled, y)
    return [f for f, c in zip(candidates, lasso.coef_[0]) if abs(c) > 1e-6]


def rfe_selection(candidates, X, y, n_features=4):
    base_model = xgb.XGBClassifier(**CV_XGB_PARAMS, verbosity=0, eval_metric="logloss", random_state=RANDOM_STATE)
    rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    return [f for f, s in zip(candidates, rfe.support_) if s]


def mutual_info_selection(candidates, X, y, n_features=4):
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    ranked = sorted(zip(candidates, mi), key=lambda x: -x[1])
    return [f for f, _ in ranked[:n_features]]


def xgb_importance_selection(candidates, X, y, n_features=4):
    model = xgb.XGBClassifier(**CV_XGB_PARAMS, verbosity=0, eval_metric="logloss", random_state=RANDOM_STATE)
    model.fit(X, y)
    ranked = sorted(zip(candidates, model.feature_importances_), key=lambda x: -x[1])
    return [f for f, _ in ranked[:n_features]]


def table4_feature_selection(sw, candidates=None, label_col="label_elevated_h1", consensus_min_count=4):
    candidates = candidates or [f"{p}_lag1" for p in LAG_CANDIDATES]
    d_all = sw.dropna(subset=candidates + [label_col]).reset_index(drop=True)
    X_all, y_all = d_all[candidates].values, d_all[label_col].values

    methods = {
        "Forward stepwise": forward_stepwise_selection(candidates, sw, label_col),
        "Backward elimination": backward_elimination_selection(candidates, sw, label_col),
        "LASSO": lasso_selection(candidates, X_all, y_all),
        "RFE": rfe_selection(candidates, X_all, y_all),
        "Mutual information": mutual_info_selection(candidates, X_all, y_all),
        "XGBoost importance": xgb_importance_selection(candidates, X_all, y_all),
    }
    counts = Counter(f for feats in methods.values() for f in feats)
    consensus = [f for f, c in counts.items() if c >= consensus_min_count]
    return methods, counts, consensus


def figure4_correlation(sw, features=FINAL_FEATURES):
    lag_cols = [f"{f}_lag1" for f in features]
    return sw[lag_cols].corr(method="spearman")

