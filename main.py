"""Entry point: runs the full pipeline and prints every table and figure
from the paper, in order. Run with: python main.py"""

from config import FINAL_FEATURES
from data_prep import load_raw, build_station_week_table, add_lag_features, add_horizon_targets
from descriptive_stats import table1_exceedance_prevalence, figure2_monthly_exceedances, table2_missingness_bias, table3_moods_median_tests
from feature_selection import table4_feature_selection, figure4_correlation
from tuning import tune_xgb_and_rf, tune_bouquet, tune_lstm
from classification import table6_classification
from feature_importance import table7_tree_importance, table7_lstm_importance
from regression import table8_regression


DATA_PATH = "calhabmap.csv"
SKIP_LSTM = False
RUN_TUNE = True


def print_header(title):
    line = "=" * 70
    print(line)
    print(title)
    print(line)


def main():
    print(f"Loading {DATA_PATH} ...")
    raw = load_raw(DATA_PATH)
    sw = build_station_week_table(raw)
    sw = add_lag_features(sw, max_lag=1)
    sw = add_horizon_targets(sw)
    print(f"Total station-weeks: {len(sw)}\n")

    print_header("TABLE 1 - Exceedance prevalence")
    print(table1_exceedance_prevalence(sw).to_string(index=False))

    print()
    print_header("FIGURE 2 - Monthly exceedance counts")
    for thr, counts in figure2_monthly_exceedances(sw).items():
        print(f"\n>= {thr} cells/L:")
        print(counts.to_string())

    print()
    print_header("TABLE 2 - Missingness bias (Mann-Whitney U)")
    print(table2_missingness_bias(sw).to_string(index=False))

    print()
    print_header("TABLE 3A/3B - Mood's median tests")
    t3a, t3b = table3_moods_median_tests(sw)
    print("\nTable 3A (lag 0-3, elevated & alert):")
    print(t3a.to_string())
    print("\nTable 3B (lag 0, all thresholds):")
    print(t3b.to_string())

    print()
    print_header("TABLE 4 - Feature selection (6 methods)")
    methods, counts, consensus = table4_feature_selection(sw)
    for name, feats in methods.items():
        print(f"  {name:<24} {feats}")
    print(f"\nFeature frequency: {dict(counts)}")
    print(f"Consensus set (needs at least 4 of 6 methods to agree): {consensus}")

    print("\nFigure 4 - Spearman correlation among the final four features:")
    print(figure4_correlation(sw).round(2).to_string())

    if RUN_TUNE:
        print()
        print_header("TABLE 5 - Re-running hyperparameter search")
        print("This part is slow. It re-derives the hyperparameters that are")
        print("already hardcoded above in FINAL_XGB_PARAMS etc, as a way to check")
        print("them, not because you need to run this every time.\n")
        features = [f"{f}_lag1" for f in FINAL_FEATURES]
        xgb_best, rf_best = tune_xgb_and_rf(sw, features)
        bouquet_best = tune_bouquet(sw, features)
        print(f"XGBoost best:  {xgb_best}")
        print(f"RF best:       {rf_best}")
        print(f"Bouquet best:  {bouquet_best}")
        if not SKIP_LSTM:
            lstm_best = tune_lstm(sw, features)
            print(f"LSTM best:     {lstm_best}")
        print("\nNote: this does not automatically update FINAL_XGB_PARAMS and the")
        print("others above. If these come out different, update them by hand.")

    print()
    print_header("TABLE 6 - Classification, h=1/2/3")
    t6 = table6_classification(sw, skip_lstm=SKIP_LSTM)
    print(t6.to_string(index=False))

    print()
    print_header("TABLE 7 - Feature importance (h=1)")
    print("Note: the importance scores below may shift slightly between runs, since the random seed does not guarantee identical draws every time.\n")
    t7_tree = table7_tree_importance(sw)
    print(t7_tree.to_string(index=False))
    if not SKIP_LSTM:
        t7_lstm = table7_lstm_importance(sw)
        print(t7_lstm.to_string(index=False))

    print()
    print_header("TABLE 8 - Regression, h=1/2/3")
    t8 = table8_regression(sw, skip_lstm=SKIP_LSTM)
    print(t8.to_string(index=False))

    print("\nDone.")

if __name__ == "__main__":
    main()