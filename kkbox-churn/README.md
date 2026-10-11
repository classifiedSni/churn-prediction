# KKBox Music Streaming Churn Prediction

An end-to-end machine learning system that predicts subscriber churn for KKBox, a music streaming service. Built on the WSDM KKBox Churn Prediction Challenge dataset — real subscription transactions and listening behavior from 970,000+ subscribers — served via a production-ready REST API and containerized with Docker.

---

## Why This Problem Is Hard

KKBox's churn rate is only 8.99% — a severe class imbalance where a naive model predicting "no churn" gets 91% accuracy while being completely useless. The challenge is identifying the 9% who will leave before they do, using behavioral signals from subscription history and listening patterns.

**Key modeling decisions:**
- Temporal train/test split (not random) to simulate real deployment conditions
- Removed `days_until_expiry` after identifying it as a near-leaky feature
- Threshold optimized for F1 rather than defaulting to 0.5

---

## Results

| Model | AUC | F1 | Recall | Precision |

| **LightGBM** | **0.9686** | **0.8191** | **0.8298** | 0.8087 |
| XGBoost | 0.9599 | 0.8218 | 0.8216 | 0.8220 |
| Logistic Regression | 0.9407 | 0.5773 | 0.5322 | 0.6307 |

Evaluated on a temporal holdout (most recent 20% by last transaction date). AUC is comparable to top solutions on the original Kaggle competition leaderboard (~0.91–0.93).


## Dataset

**Source:** [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)

| File | Rows | Description |

| `train_v2.csv` | 970,960 | Churn labels per subscriber |
| `members_v3.csv` | 6,769,473 | Demographics: city, age, gender, registration |
| `transactions_v2.csv` | 1,431,009 | Subscription history: plan, payment, cancellations |
| `user_logs_v2.csv` | 18,396,362 | Daily listening behavior (March 2017) |

**Churn definition:** Subscriber did not renew within 30 days after membership expiry.



## Feature Engineering

35 features across three behavioral domains:

| Domain | Features | Key signals |

| Member | 8 | Age, city, registration channel, account age |
| Transactions | 12 | Cancellation history, auto-renew rate, plan type, payment patterns |
| Listening logs | 15 | Daily songs, completion rate, skip rate, active days |

**Notable decisions:**
- `user_logs_v2` covers only March 2017 — aggregated per user across the full month
- Age cleaned: 67% of raw values are 0 (unreported) — treated as missing with `age_known` flag
- 22.3% of training users had zero listening activity — captured as `has_log_data = 0`
- `days_until_expiry` dropped from final model after feature importance analysis confirmed leakage



## Project Structure


kkbox-churn/
├── data/
│   ├── raw/                    # Original CSVs from Kaggle
│   └── processed/              # Master feature table + feature_columns.json
├── models/
│   ├── lightgbm/               # lgbm_model.pkl (final model)
│   ├── xgboost/                # xgb_model.pkl
│   ├── logistic_regression/    # lr_model.pkl + scaler.pkl
│   ├── feature_columns.json    # Ordered feature list (34 features)
│   └── model_metadata.json     # AUC, threshold, split method
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   └── api.py                  # FastAPI prediction service
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/classifiedSni/churn-prediction.git
cd churn-prediction/kkbox-churn
docker compose up
```

API live at `http://localhost:8001`. No Python setup required.

### Option 2 — Local

```bash
pip install -r requirements.txt
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8001
```

---

## API Reference

### `GET /health`
```json
{"status": "ok", "model": "LightGBM", "version": "1.0.0"}
```

### `POST /predict`

```json
{
  "city": 1,
  "age": 25.0,
  "age_known": 1,
  "ever_cancelled": 1,
  "auto_renew_rate": 0.2,
  "skip_rate": 0.75,
  "completion_rate": 0.05,
  "has_log_data": 1,
  ...
}
```

**Response:**
```json
{
  "churn_probability": 0.9995,
  "churn_prediction": 1,
  "risk_tier": "HIGH",
  "top_risk_drivers": [
    "Previously cancelled subscription",
    "Low auto-renewal rate",
    "High song skip rate"
  ]
}
```

**Risk tiers:** `HIGH` (≥ 70%), `MEDIUM` (40–70%), `LOW` (< 40%)  
**Prediction threshold:** 0.8226 (optimized for F1 on temporal holdout)

### `POST /batch_predict`
Accepts a list of subscribers, returns per-subscriber predictions and aggregate high-risk count.

Interactive docs: `http://localhost:8001/docs`


## Modeling Notes

### Why temporal split matters
A random split would let the model train and test on subscribers from the same time window — artificially inflating AUC because there's no distribution shift to handle. The temporal split trains on users with older transaction history and tests on those with more recent activity, simulating actual production deployment.

### Why days_until_expiry was removed
Initial feature importance showed `days_until_expiry` was 3–4× more important than any other feature. Since churn is defined as not renewing after expiry, this feature was nearly encoding the churn event timing directly. Removing it dropped AUC from 0.9950 → 0.9686 — a more honest and generalizable result.

### Optimal threshold
With only 3.14% churn rate in the temporal test set, the default 0.5 threshold predicted almost no churners (F1 = 0.33). Optimizing threshold on the precision-recall curve found 0.8226 as the F1-maximizing threshold (F1 = 0.8191).



## Tech Stack

- **ML:** scikit-learn, LightGBM, XGBoost
- **API:** FastAPI, Pydantic, Uvicorn
- **Containerization:** Docker, docker-compose
- **Data:** pandas, numpy (18M+ row chunked processing)


## Author

Siddharth Munagala — Master's in Data Science
