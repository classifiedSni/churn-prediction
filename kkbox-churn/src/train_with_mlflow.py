#KKBox MLflow Experiment Tracking

import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score,
                             average_precision_score,
                             precision_recall_curve)
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR     = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / 'data' / 'processed' / 'features.csv'
MODELS_DIR   = BASE_DIR / 'models'

#Load data
print("Loading data...")
df           = pd.read_csv(FEATURES_CSV)
feature_cols = json.load(open(MODELS_DIR / 'feature_columns.json'))

X = df[feature_cols].values
y = df['is_churn'].values

# Temporal split (same as modeling file)
cutoff_value = df['days_since_last_txn'].quantile(0.20)
train_mask   = df['days_since_last_txn'] >= cutoff_value
test_mask    = df['days_since_last_txn'] <  cutoff_value

X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, 'is_churn'].values
X_test  = df.loc[test_mask,  feature_cols].values
y_test  = df.loc[test_mask,  'is_churn'].values

print(f"Train: {X_train.shape[0]:,} rows — churn: {y_train.mean():.2%}")
print(f"Test:  {X_test.shape[0]:,}  rows — churn: {y_test.mean():.2%}")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

# Metric helpe
def get_optimal_metrics(y_true, y_prob):
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    f1_arr    = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)
    best_idx  = f1_arr.argmax()
    threshold = thresholds[best_idx]
    y_pred    = (y_prob >= threshold).astype(int)
    return {
        'roc_auc'          : round(roc_auc_score(y_true, y_prob), 4),
        'avg_precision'    : round(average_precision_score(y_true, y_prob), 4),
        'f1'               : round(float(f1_arr[best_idx]), 4),
        'precision'        : round(float(precision_arr[best_idx]), 4),
        'recall'           : round(float(recall_arr[best_idx]), 4),
        'optimal_threshold': round(float(threshold), 4),
    }

#Set experiment
mlflow.set_experiment("kkbox-churn-prediction")

#1. Logistic Regression
with mlflow.start_run(run_name="LogisticRegression"):
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    params = {
        "C"            : 0.1,
        "class_weight" : "balanced",
        "max_iter"     : 1000,
        "solver"       : "lbfgs"
    }
    model = LogisticRegression(**params)
    model.fit(Xtr_s, y_train)
    probs   = model.predict_proba(Xte_s)[:, 1]
    metrics = get_optimal_metrics(y_test, probs)

    mlflow.log_params(params)
    mlflow.log_params({"split": "temporal", "n_features": len(feature_cols)})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    print(f"LR       → AUC={metrics['roc_auc']}  F1={metrics['f1']}  "
          f"threshold={metrics['optimal_threshold']}")

# 2. LightGBM
with mlflow.start_run(run_name="LightGBM"):
    params = {
        "n_estimators"     : 1000,
        "learning_rate"    : 0.05,
        "num_leaves"       : 31,
        "max_depth"        : -1,
        "min_child_samples": 50,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "reg_alpha"        : 0.5,
        "reg_lambda"       : 0.5,
        "class_weight"     : "balanced",
        "random_state"     : 42,
        "n_jobs"           : -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set  = [(X_test, y_test)],
        callbacks = [lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(period=-1)]
    )
    probs   = model.predict_proba(X_test)[:, 1]
    metrics = get_optimal_metrics(y_test, probs)

    mlflow.log_params(params)
    mlflow.log_params({"split": "temporal", "n_features": len(feature_cols),
                       "best_iteration": model.best_iteration_})
    mlflow.log_metrics(metrics)
    mlflow.lightgbm.log_model(model, "model")
    print(f"LightGBM → AUC={metrics['roc_auc']}  F1={metrics['f1']}  "
          f"threshold={metrics['optimal_threshold']}")

#3. XGBoost
with mlflow.start_run(run_name="XGBoost"):
    params = {
        "n_estimators"         : 1000,
        "learning_rate"        : 0.05,
        "max_depth"            : 5,
        "subsample"            : 0.8,
        "colsample_bytree"     : 0.8,
        "scale_pos_weight"     : round(scale_pos_weight, 2),
        "eval_metric"          : "auc",
        "early_stopping_rounds": 50,
        "random_state"         : 42,
        "n_jobs"               : -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False
    )
    probs   = model.predict_proba(X_test)[:, 1]
    metrics = get_optimal_metrics(y_test, probs)

    mlflow.log_params(params)
    mlflow.log_params({"split": "temporal", "n_features": len(feature_cols),
                       "best_iteration": model.best_iteration})
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "model")
    print(f"XGBoost  → AUC={metrics['roc_auc']}  F1={metrics['f1']}  "
          f"threshold={metrics['optimal_threshold']}")

print("\n✅ All runs logged.")
print("Launch UI: mlflow ui --backend-store-uri sqlite:///src/mlflow.db")
