import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, average_precision_score)
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


FEATURES_CSV = Path(r'/Users/siddharthm/Desktop/churn-prediction/data/processed/features.csv')
MODELS_DIR   = Path(r'/Users/siddharthm/Desktop/churn-prediction/models')

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv(FEATURES_CSV)
feature_cols = json.load(open(MODELS_DIR/'feature_columns.json'))

X = df[feature_cols].values
y = df['churn_binary'].values

# Time-based split: tenure ≤ 36 = train, > 36 = test
tenure_col = df['tenure'].values
train_mask = tenure_col <= 36
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── Metric helper ──────────────────────────────────────────
def log_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":        round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":  round(average_precision_score(y_true, y_prob), 4),
        "precision":      round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":         round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":             round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

# ── Set experiment ─────────────────────────────────────────
mlflow.set_experiment("churn-prediction")

# ── 1. Logistic Regression ─────────────────────────────────
with mlflow.start_run(run_name="LogisticRegression"):
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    params = {"C": 0.1, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}
    model  = LogisticRegression(**params)
    model.fit(Xtr_s, y_train)
    probs  = model.predict_proba(Xte_s)[:, 1]
    metrics = log_metrics(y_test, probs)

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    print(f"LR      → AUC={metrics['roc_auc']}  F1={metrics['f1']}")

# ── 2. LightGBM ────────────────────────────────────────────
with mlflow.start_run(run_name="LightGBM"):
    params = {
        "n_estimators": 300, "learning_rate": 0.05,
        "num_leaves": 31, "max_depth": -1,
        "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.1,
        "reg_lambda": 0.1, "class_weight": "balanced",
        "random_state": 42, "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    probs   = model.predict_proba(X_test)[:, 1]
    metrics = log_metrics(y_test, probs)

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_param("best_iteration", model.best_iteration_)
    mlflow.lightgbm.log_model(model, "model")
    print(f"LightGBM→ AUC={metrics['roc_auc']}  F1={metrics['f1']}")

# ── 3. XGBoost ─────────────────────────────────────────────
with mlflow.start_run(run_name="XGBoost"):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    params = {
        "n_estimators": 300, "learning_rate": 0.05,
        "max_depth": 5, "subsample": 0.8,
        "colsample_bytree": 0.8, "scale_pos_weight": neg / pos,
        "eval_metric": "auc", "early_stopping_rounds": 50,
        "random_state": 42, "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    probs   = model.predict_proba(X_test)[:, 1]
    metrics = log_metrics(y_test, probs)

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_param("best_iteration", model.best_iteration)
    mlflow.xgboost.log_model(model, "model")
    print(f"XGBoost → AUC={metrics['roc_auc']}  F1={metrics['f1']}")
