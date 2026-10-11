# Customer Churn Prediction System

An end-to-end machine learning system that predicts customer churn probability for a telecom company. Built on real Telco data enriched with synthetic multi-table behavioral data (transactions, web sessions, support tickets), served via a production-ready REST API, containerized with Docker, and tracked with MLflow.

---

## Project Overview

Churn prediction is one of the highest-ROI applications of ML in industry. Retaining an existing customer costs 5–7× less than acquiring a new one. This project builds a complete pipeline — from raw data to a live prediction API — that identifies at-risk customers before they leave.

**Key results:**
| Model | AUC | F1 | Recall |
|---|---|---|---|
| XGBoost | **0.9983** | **0.8815** | **0.9972** |
| LightGBM | 0.9976 | 0.8729 | 0.9972 |
| Logistic Regression | 0.9772 | 0.7807 | 0.7458 |

---

## Architecture

```
Raw Data (Telco CSV + Synthetic Tables)
        │
        ▼
  Data Cleaning & EDA
        │
        ▼
  Feature Engineering (49 features across 4 tables)
        │
        ▼
  Model Training & Selection (LR / LightGBM / XGBoost)
        │
        ▼
  MLflow Experiment Tracking
        │
        ▼
  FastAPI REST API  ──▶  Docker Container
```

---

## Dataset

- **Base:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 26.5% churn rate
- **Synthetic extensions** (generated to simulate real multi-table environments):
  - `transactions.csv` — 202,377 rows (RFM + payment failure signals)
  - `sessions.csv` — 1,218,789 rows (engagement and activity trends)
  - `support_tickets.csv` — 41,190 rows (satisfaction scores, ticket categories)

**Observation cutoff:** 2023-09-30 (strict cutoff to prevent data leakage)

**Train/test split:** Time-based using tenure as proxy — customers with tenure ≤ 36 months for training, > 36 months for testing (4,042 / 3,001 rows).

---

## Feature Engineering

49 features engineered across four domains:

| Domain | Features | Examples |
|---|---|---|
| Customer | 22 | contract_risk, is_fiber_optic, protection_score |
| Transactions | 7 | failed_payment_rate, avg_spend, recency_days |
| Sessions | 8 | sessions_last_30d, activity_trend_30_90 |
| Support Tickets | 12 | avg_satisfaction, low_satisfaction_tickets |

---

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                    # Original Telco CSV
│   ├── processed/              # Cleaned data + master feature table
│   └── synthetic/              # Generated transaction/session/ticket data
├── models/
│   ├── lightgbm/               # lgbm_churn_model.pkl
│   ├── xgboost/                # xgb_churn_model.pkl
│   ├── logistic_regression/    # lr_churn_model.pkl + scaler.pkl
│   ├── feature_columns.json    # Ordered feature list (49 features)
│   └── model_metadata.json     # AUC, lift, ROI metrics
├── notebooks/                  # EDA, feature engineering, modeling notebooks
├── src/
│   ├── api.py                  # FastAPI prediction service
│   └── train_with_mlflow.py    # MLflow experiment tracking script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/classifiedSni/churn-prediction.git
cd churn-prediction
docker compose up
```

API is live at `http://localhost:8000`. No Python setup required.

### Option 2 — Local

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### `GET /health`
```json
{"status": "ok", "model": "LightGBM", "version": "1.0.0"}
```

### `GET /model-info`
Returns model metadata: AUC, lift curve, training date, feature count.

### `POST /predict`
Single customer prediction.

**Request body (49 features):**
```json
{
  "tenure": 2,
  "MonthlyCharges": 85.0,
  "contract_risk": 2,
  "is_fiber_optic": 1,
  "sessions_last_30d": 1.0,
  "avg_satisfaction": 2.1,
  "failed_payment_rate": 0.33,
  ...
}
```

**Response:**
```json
{
  "churn_probability": 0.9964,
  "churn_prediction": 1,
  "risk_tier": "HIGH",
  "top_risk_drivers": [
    "Month-to-month contract",
    "Very low recent session activity",
    "Low support satisfaction score"
  ]
}
```

**Risk tiers:** `HIGH` (≥ 70%), `MEDIUM` (40–70%), `LOW` (< 40%)

### `POST /batch_predict`
Batch predictions for a list of customers. Returns high-risk count and per-customer results.

Interactive docs: `http://localhost:8000/docs`

---

## MLflow Experiment Tracking

All three models are tracked with parameters, metrics, and artifacts.

```bash
cd churn-prediction
python src/train_with_mlflow.py
mlflow ui --backend-store-uri sqlite:///src/mlflow.db
```

Open `http://127.0.0.1:5000` → churn-prediction experiment → Training runs → Compare.

---

## Model Performance

### Why XGBoost wins on recall
Recall of 0.9972 means the model catches 99.7% of actual churners. In a business context, missing a churner (false negative) is far more costly than a false alarm (false positive), so recall is the primary optimization target.

### Business impact (10,000 customer deployment)
- Customers flagged as HIGH risk: ~2,000
- True churners caught: ~1,980 (at 0.9972 recall)
- Revenue retained at $50 retention cost vs $500 lost CLV: **~$870,000 net value**

---

## Tech Stack

- **ML:** scikit-learn, LightGBM, XGBoost, SHAP
- **API:** FastAPI, Pydantic, Uvicorn
- **Tracking:** MLflow
- **Containerization:** Docker, docker-compose
- **Data:** pandas, numpy, scipy

---

## Author

Siddharth M — Master's in Data Science
