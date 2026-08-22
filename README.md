# Forecasting Toxic Alexandrium Blooms Along the California Coast

Machine learning models that forecast *Alexandrium* harmful algal bloom exceedances 1 to 3 weeks ahead, using shore-station data from California's CalHABMAP program.

## Summary

- **Data**: 6,481 station-weeks from 14 CalHABMAP stations (2005–2026)
- **Predictors**: chlorophyll-a, silicate, nitrite, temperature (selected via 6 feature-selection methods)
- **Models**: XGBoost, Random Forest, a Bouquet-style decision tree, and an LSTM
- **Thresholds**: elevated (≥1,000 cells/L) and alert (≥10,000 cells/L)
- **Result**: XGBoost gets the highest AUROC in 5 of 6 threshold/horizon combos. All models fail to beat a naive baseline on regression, meaning current predictors flag risky weeks but cannot estimate bloom size.

## Repo structure

```
alexandrium-hab-forecasting/
├── code/
│   ├── config.py              # constants and final hyperparameters
│   ├── data_prep.py           # load CSV, build station-week table, lag features, CV folds
│   ├── descriptive_stats.py   # Table 1, Figure 2, Table 2, Tables 3A/3B
│   ├── feature_selection.py   # Table 4 (6 methods), Figure 3 (correlation)
│   ├── tuning.py              # Table 5 (hyperparameter search)
│   ├── classification.py      # Table 6 (XGBoost, RF, Bouquet tree, LSTM)
│   ├── feature_importance.py  # Table 7
│   ├── regression.py          # Table 8
│   └── main.py                # runs the full pipeline, prints every table
├── requirements.txt
└── README.md
```

## Data

Data comes from SCCOOS/CalHABMAP via their ERDDAP server:
https://erddap.sccoos.org/erddap/

Download the station CSV and save it as `calhabmap.csv` in the repo root, or update `DATA_PATH` in `code/main.py`.

## Running

```bash
pip install -r requirements.txt
cd code
python main.py
```

This prints every table and figure data in the paper: exceedance prevalence, missingness-bias tests, Mood's median tests, feature selection, hyperparameter tuning, classification results, feature importance, and regression results.

## Citation

If you use this code, cite the paper: *Forecasting Toxic Alexandrium Blooms Along the California Coast*.
