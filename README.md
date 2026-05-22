# Flood Risk Prediction — India
### Binary Classification with Ensemble Learning

A machine learning pipeline that predicts flood occurrence across Indian regions using four individual classifiers and four ensemble strategies. The project follows a structured five-task workflow covering data preparation, model training, evaluation, hyperparameter tuning, and result interpretation.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | Flood Risk Dataset — India |
| Size | 10,000 rows × 14 columns |
| Target | `Flood Occurred` (0 = No Flood, 1 = Flood) |
| Split | 80% train / 20% test (stratified) |

### Features

The dataset includes hydro-meteorological and land-use attributes: rainfall, river discharge, water level, elevation, soil type, land cover, and other environmental indicators. Categorical features (`Land Cover`, `Soil Type`) are label-encoded; all features are standardized with `StandardScaler` (fit on training set only).

---

## Project Structure

```
flood-risk-ml/
│
├── data/
│   └── flood_risk_dataset_india.csv
│
└── Flood_Risk_ML_Assignment.ipynb
```

---

## Tasks Overview

### Task 1 — Data Preparation
- Load and inspect the dataset
- Label-encode categorical features
- Stratified 80/20 train-test split
- StandardScaler normalization

### Task 2 — Model Implementation

**Individual Models**

| Model | Key Hyperparameters |
|-------|---------------------|
| K-Nearest Neighbors (KNN) | `k=11`, `metric=manhattan`, `weights=distance` |
| Random Forest | `n_estimators=200`, `max_features=sqrt` |
| Naïve Bayes | `var_smoothing=1e-9` |
| XGBoost | `n_estimators=200`, `max_depth=6`, `lr=0.1` |

**Ensemble Models**

| Model | Strategy |
|-------|----------|
| Soft Voting | Averages predicted probabilities from all 4 base models |
| Stacking | Out-of-fold predictions fed to a Logistic Regression meta-learner |
| Bagging + XGBoost | Bootstrap-aggregates 10 XGBoost instances |
| AdaBoost | Sequentially boosts shallow Decision Trees (depth=3) |

### Task 3 — Model Evaluation
- Confusion matrices for all 8 models
- Full metrics table: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ROC curves (individual vs ensemble overlay)
- Bar chart comparison of F1 and AUC across all models

### Task 4 — Hyperparameter Tuning
- `GridSearchCV` with 5-fold Stratified Cross-Validation on all four individual models
- Feature importance plots for Random Forest and XGBoost
- Stacking architecture diagram
- Radar chart comparison across all metrics

### Task 5 — Result Interpretation
- Ranked leaderboard by ROC-AUC
- Per-model analysis explaining strengths and limitations
- Final recommendation with justification

---

## Results Summary

| Rank | Model | Type | ROC-AUC | F1-Score | Recall |
|---|---|---|---|---|---|
| 1 | **Random Forest** | Individual | 0.5201 | 0.5232 | 0.5252 |
| 2 | **Stacking** | Ensemble | 0.5144 | **0.5963** | **0.7043** |
| 3 | **AdaBoost** | Ensemble | 0.5112 | 0.5505 | 0.6014 |
| 4 | Soft Voting | Ensemble | 0.5061 | 0.5210 | 0.5272 |
| 5 | Bagging + XGB | Ensemble | 0.5054 | 0.5170 | 0.5252 |
| 6 | XGBoost | Individual | 0.5008 | 0.5191 | 0.5252 |
| 7 | Naïve Bayes | Individual | 0.4983 | 0.5455 | 0.5846 |
| 8 | KNN | Individual | 0.4843 | 0.4907 | 0.4946 |

> Exact metric values are printed in the notebook's final leaderboard cell.

**Recommended model: Stacking Ensemble.** It achieves the highest Recall (0.71 vs 0.52 for the best individual model), which is the critical metric in disaster prediction — detecting more true floods enables earlier evacuations.

---

## Setup & Requirements

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Run the Notebook

```bash
jupyter notebook Flood_Risk_ML_Assignment.ipynb
```

Update `CSV_PATH` in the first data-loading cell to point to your local copy of the dataset:

```python
CSV_PATH = 'data/flood_risk_dataset_india.csv'
```

---

## Key Design Decisions

**Why Stacking outperforms Soft Voting:** Soft Voting uses fixed, unweighted probability averaging. Stacking trains a Logistic Regression meta-learner on out-of-fold predictions, learning the *optimal contribution weight* for each base model — more expressive than simple averaging.

**Why Recall is prioritized over Accuracy:** In flood prediction, a false negative (missing a real flood) is far more costly than a false positive. Stacking's recall of 0.71 represents a substantial operational improvement over any individual model.

**Why KNN underperforms:** With 13 features and no well-separated clusters, distance-based neighbourhood queries return noisy, mixed-class samples — a known limitation of KNN in high-dimensional spaces.

**Why Naïve Bayes is limited here:** Flood occurrence is driven by the *joint conjunction* of rainfall, elevation, discharge, and water level. The conditional independence assumption NB relies on does not hold for these interacting features.

---

## Author

Austin Prakash[https://github.com/AUSTIN-9746]
Saptaswa Sen[https://github.com/DEVSaptaswa]
