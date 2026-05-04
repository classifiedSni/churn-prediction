#KKBox Churn Prediction API

import json
import joblib
import numpy as np
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#Paths
BASE_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR    = BASE_DIR / 'models' / 'lightgbm'
FEATURE_FILE  = BASE_DIR / 'models' / 'feature_columns.json'
METADATA_FILE = BASE_DIR / 'models' / 'model_metadata.json'

#Load artifact
print("Loading model artifacts...")
model    = joblib.load(MODELS_DIR / 'lgbm_model.pkl')
features = json.load(open(FEATURE_FILE))
metadata = json.load(open(METADATA_FILE))
threshold = metadata['optimal_threshold']
print(f"Model loaded — {len(features)} features | "
      f"AUC={metadata['roc_auc']} | threshold={threshold}")

#FastAPI app
app = FastAPI(
    title       = "KKBox Churn Prediction API",
    description = "Predicts music streaming subscriber churn using LightGBM "
                  "trained on KKBox subscription and listening behavior data.",
    version     = "1.0.0"
)

#Request schema
class SubscriberFeatures(BaseModel):
    # Member features
    city:                    int   = Field(..., ge=0,  description="City code")
    age:                     float = Field(..., ge=0,  le=120, description="User age")
    age_known:               int   = Field(..., ge=0,  le=1,   description="1 if age is valid")
    gender_female:           int   = Field(..., ge=0,  le=1)
    gender_male:             int   = Field(..., ge=0,  le=1)
    gender_unknown:          int   = Field(..., ge=0,  le=1)
    registered_via:          int   = Field(..., ge=0,  description="Registration channel")
    days_since_registration: float = Field(..., ge=0,  description="Days since account creation")
    # Transaction features
    txn_count:               float = Field(..., ge=0,  description="Total transactions")
    cancel_count:            float = Field(..., ge=0,  description="Number of cancellations")
    ever_cancelled:          int   = Field(..., ge=0,  le=1,   description="Ever cancelled subscription")
    auto_renew_rate:         float = Field(..., ge=0,  le=1,   description="Fraction of auto-renewals")
    avg_plan_days:           float = Field(..., ge=0,  description="Average subscription plan length")
    most_common_plan_days:   float = Field(..., ge=0,  description="Most frequent plan length")
    avg_price_paid:          float = Field(..., ge=0,  description="Average amount paid per transaction")
    total_paid:              float = Field(..., ge=0,  description="Total amount paid")
    discount_rate:           float = Field(..., ge=0,  le=1,   description="Fraction of discounted transactions")
    unique_payment_methods:  int   = Field(..., ge=0,  description="Number of distinct payment methods used")
    days_since_last_txn:     float = Field(..., ge=0,  description="Days since last transaction")
    cancel_ratio:            float = Field(..., ge=0,  le=1,   description="Cancellations / total transactions")
    # Listening log features (March 2017)
    total_days:              float = Field(..., ge=0,  description="Active listening days in March 2017")
    num_25:                  float = Field(..., ge=0,  description="Songs listened to 25% completion")
    num_50:                  float = Field(..., ge=0,  description="Songs listened to 50% completion")
    num_75:                  float = Field(..., ge=0,  description="Songs listened to 75% completion")
    num_985:                 float = Field(..., ge=0,  description="Songs listened to 98.5% completion")
    num_100:                 float = Field(..., ge=0,  description="Songs listened to 100% completion")
    num_unq:                 float = Field(..., ge=0,  description="Unique songs played")
    total_secs:              float = Field(..., ge=0,  description="Total seconds listened")
    avg_daily_songs:         float = Field(..., ge=0,  description="Average songs per active day")
    avg_daily_secs:          float = Field(..., ge=0,  description="Average seconds listened per day")
    avg_daily_unq:           float = Field(..., ge=0,  description="Average unique songs per day")
    completion_rate:         float = Field(..., ge=0,  le=1,   description="Fraction of songs fully completed")
    skip_rate:               float = Field(..., ge=0,  le=1,   description="Fraction of songs skipped early")
    has_log_data:            int   = Field(..., ge=0,  le=1,   description="1 if user has March 2017 activity")

# Response schemas
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  int
    risk_tier:         str
    top_risk_drivers:  List[str]

class BatchPredictionResponse(BaseModel):
    total_subscribers: int
    high_risk_count:   int
    predictions:       List[dict]

# Helpers
def get_risk_tier(prob: float) -> str:
    if prob >= 0.70: return "HIGH"
    if prob >= 0.40: return "MEDIUM"
    return "LOW"

def get_risk_drivers(f: dict) -> List[str]:
    drivers = []
    if f.get('ever_cancelled', 0) == 1:
        drivers.append("Previously cancelled subscription")
    if f.get('auto_renew_rate', 1) < 0.5:
        drivers.append("Low auto-renewal rate")
    if f.get('has_log_data', 1) == 0:
        drivers.append("No listening activity in last month")
    if f.get('skip_rate', 0) > 0.4:
        drivers.append("High song skip rate")
    if f.get('completion_rate', 1) < 0.5:
        drivers.append("Low song completion rate")
    if f.get('cancel_ratio', 0) > 0.1:
        drivers.append("High cancellation ratio")
    if f.get('total_days', 30) < 5:
        drivers.append("Very few active listening days")
    if f.get('days_since_last_txn', 0) > 30:
        drivers.append("No recent transaction activity")
    return drivers[:3] if drivers else ["No strong individual risk drivers identified"]

#Endpoints
@app.get("/health")
def health_check():
    return {"status": "ok", "model": "LightGBM", "version": "1.0.0"}

@app.get("/model-info")
def model_info():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(subscriber: SubscriberFeatures):
    try:
        feat_dict   = subscriber.model_dump()
        feat_values = [feat_dict[f] for f in features]
        X           = np.array(feat_values).reshape(1, -1)
        prob        = float(model.predict_proba(X)[0][1])
        pred        = int(prob >= threshold)
        return PredictionResponse(
            churn_probability = round(prob, 4),
            churn_prediction  = pred,
            risk_tier         = get_risk_tier(prob),
            top_risk_drivers  = get_risk_drivers(feat_dict)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(subscribers: List[SubscriberFeatures]):
    try:
        rows = []
        for s in subscribers:
            feat_dict = s.model_dump()
            rows.append([feat_dict[f] for f in features])
        X     = np.array(rows)
        probs = model.predict_proba(X)[:, 1]
        predictions = []
        for i, (prob, subscriber) in enumerate(zip(probs, subscribers)):
            feat_dict = subscriber.model_dump()
            predictions.append({
                "subscriber_index" : i,
                "churn_probability": round(float(prob), 4),
                "churn_prediction" : int(prob >= threshold),
                "risk_tier"        : get_risk_tier(float(prob)),
                "top_risk_drivers" : get_risk_drivers(feat_dict)
            })
        high_risk = sum(1 for p in predictions if p['risk_tier'] == 'HIGH')
        return BatchPredictionResponse(
            total_subscribers = len(subscribers),
            high_risk_count   = high_risk,
            predictions       = predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
