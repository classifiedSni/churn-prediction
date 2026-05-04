# src/api.py
# ── Churn Prediction FastAPI Service ──────────────────────

import json
import joblib
import numpy as np
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR    = BASE_DIR / 'models' / 'lightgbm'
FEATURE_FILE  = BASE_DIR / 'models' / 'feature_columns.json'
METADATA_FILE = BASE_DIR / 'models' / 'model_metadata.json'

# ── Load model artifacts at startup ───────────────────────
print("Loading model artifacts...")
model    = joblib.load(MODELS_DIR / 'lgbm_churn_model.pkl')
features = json.load(open(FEATURE_FILE))
metadata = json.load(open(METADATA_FILE))
print(f"✅ Model loaded — {len(features)} features, AUC={metadata['roc_auc']}")

# ── FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title       = "Churn Prediction API",
    description = "Predicts customer churn probability using LightGBM trained on multi-table telecom data.",
    version     = "1.0.0"
)

# ── Request schema ─────────────────────────────────────────
class CustomerFeatures(BaseModel):
    # Customer base features
    tenure:                float = Field(..., ge=0,  description="Months as customer")
    MonthlyCharges:        float = Field(..., ge=0,  description="Monthly bill ($)")
    TotalCharges:          float = Field(..., ge=0,  description="Total billed ($)")
    SeniorCitizen:         int   = Field(..., ge=0, le=1)
    contract_risk:         int   = Field(..., ge=0, le=2, description="0=Two year, 1=One year, 2=Month-to-month")
    is_fiber_optic:        int   = Field(..., ge=0, le=1)
    is_dsl:                int   = Field(..., ge=0, le=1)
    pays_electronic_check: int   = Field(..., ge=0, le=1)
    pays_mailed_check:     int   = Field(..., ge=0, le=1)
    pays_bank_transfer:    int   = Field(..., ge=0, le=1)
    Partner_bin:           int   = Field(..., ge=0, le=1)
    Dependents_bin:        int   = Field(..., ge=0, le=1)
    PhoneService_bin:      int   = Field(..., ge=0, le=1)
    MultipleLines_bin:     int   = Field(..., ge=0, le=1)
    OnlineSecurity_bin:    int   = Field(..., ge=0, le=1)
    OnlineBackup_bin:      int   = Field(..., ge=0, le=1)
    DeviceProtection_bin:  int   = Field(..., ge=0, le=1)
    TechSupport_bin:       int   = Field(..., ge=0, le=1)
    StreamingTV_bin:       int   = Field(..., ge=0, le=1)
    StreamingMovies_bin:   int   = Field(..., ge=0, le=1)
    PaperlessBilling_bin:  int   = Field(..., ge=0, le=1)
    protection_score:      int   = Field(..., ge=0, le=4)
    # Transaction features
    recency_days:            float = Field(..., ge=0)
    transaction_count:       float = Field(..., ge=0)
    total_spend:             float = Field(..., ge=0)
    avg_spend:               float = Field(..., ge=0)
    spend_std:               float = Field(..., ge=0)
    failed_payment_rate:     float = Field(..., ge=0, le=1)
    has_ever_failed_payment: int   = Field(..., ge=0, le=1)
    # Session features
    total_sessions:          float = Field(..., ge=0)
    avg_session_duration:    float = Field(..., ge=0)
    avg_pages_viewed:        float = Field(..., ge=0)
    days_since_last_session: float = Field(..., ge=0)
    sessions_last_30d:       float = Field(..., ge=0)
    sessions_last_60d:       float = Field(..., ge=0)
    sessions_last_90d:       float = Field(..., ge=0)
    activity_trend_30_90:    float = Field(..., ge=0)
    # Ticket features
    total_tickets:            float = Field(..., ge=0)
    avg_satisfaction:         float = Field(..., ge=1, le=5)
    min_satisfaction:         float = Field(..., ge=1, le=5)
    avg_resolution_hrs:       float = Field(..., ge=0)
    days_since_last_ticket:   float = Field(..., ge=0)
    tickets_last_30d:         float = Field(..., ge=0)
    tickets_last_90d:         float = Field(..., ge=0)
    tickets_account:          float = Field(..., ge=0)
    tickets_billing:          float = Field(..., ge=0)
    tickets_service:          float = Field(..., ge=0)
    tickets_technical:        float = Field(..., ge=0)
    low_satisfaction_tickets: float = Field(..., ge=0)

# ── Response schemas ───────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  int
    risk_tier:         str
    top_risk_drivers:  List[str]

class BatchPredictionResponse(BaseModel):
    total_customers: int
    high_risk_count: int
    predictions:     List[dict]

# ── Helpers ────────────────────────────────────────────────
def get_risk_tier(prob: float) -> str:
    if prob >= 0.70: return "HIGH"
    if prob >= 0.40: return "MEDIUM"
    return "LOW"

def get_risk_drivers(feat_values: dict) -> List[str]:
    drivers = []
    if feat_values.get('contract_risk', 0) == 2:
        drivers.append("Month-to-month contract")
    if feat_values.get('sessions_last_30d', 999) < 3:
        drivers.append("Very low recent session activity")
    if feat_values.get('avg_satisfaction', 5) < 2.5:
        drivers.append("Low support satisfaction score")
    if feat_values.get('failed_payment_rate', 0) > 0.05:
        drivers.append("High failed payment rate")
    if feat_values.get('is_fiber_optic', 0) == 1:
        drivers.append("Fiber optic customer (high churn segment)")
    if feat_values.get('tenure', 999) < 12:
        drivers.append("New customer (under 12 months)")
    if feat_values.get('low_satisfaction_tickets', 0) > 3:
        drivers.append("Multiple low-satisfaction support tickets")
    if feat_values.get('activity_trend_30_90', 1) < 0.2:
        drivers.append("Declining engagement trend")
    return drivers[:3] if drivers else ["No strong individual risk drivers identified"]

# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "model": "LightGBM", "version": "1.0.0"}

@app.get("/model-info")
def model_info():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        feat_dict   = customer.model_dump()
        feat_values = [feat_dict[f] for f in features]
        X           = np.array(feat_values).reshape(1, -1)
        prob        = float(model.predict_proba(X)[0][1])
        pred        = int(prob >= 0.5)
        return PredictionResponse(
            churn_probability = round(prob, 4),
            churn_prediction  = pred,
            risk_tier         = get_risk_tier(prob),
            top_risk_drivers  = get_risk_drivers(feat_dict)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(customers: List[CustomerFeatures]):
    try:
        rows = []
        for c in customers:
            feat_dict = c.model_dump()
            rows.append([feat_dict[f] for f in features])
        X     = np.array(rows)
        probs = model.predict_proba(X)[:, 1]
        predictions = []
        for i, (prob, customer) in enumerate(zip(probs, customers)):
            feat_dict = customer.model_dump()
            predictions.append({
                "customer_index"   : i,
                "churn_probability": round(float(prob), 4),
                "churn_prediction" : int(prob >= 0.5),
                "risk_tier"        : get_risk_tier(float(prob)),
                "top_risk_drivers" : get_risk_drivers(feat_dict)
            })
        high_risk = sum(1 for p in predictions if p['risk_tier'] == 'HIGH')
        return BatchPredictionResponse(
            total_customers = len(customers),
            high_risk_count = high_risk,
            predictions     = predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
