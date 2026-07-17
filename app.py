"""
Bank Marketing Campaign - Term Deposit Prediction API
FastAPI application for serving the trained ML model.

Endpoints:
    GET  /         - Health check
    POST /predict  - Single prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# ============================================================
# Load Model
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Bank Term Deposit Prediction API",
    description="Predicts whether a bank client will subscribe to a term deposit based on marketing campaign data.",
    version="1.0.0"
)

# ============================================================
# Request/Response Models
# ============================================================
class CustomerInput(BaseModel):
    """Input schema for a single customer prediction."""
    age: int = Field(..., example=35, description="Age of the client")
    job: str = Field(..., example="admin.", description="Client's job type")
    marital: str = Field(..., example="married", description="Marital status")
    education: str = Field(..., example="university.degree", description="Education level")
    default: str = Field(..., example="no", description="Has credit in default?")
    housing: str = Field(..., example="yes", description="Has housing loan?")
    loan: str = Field(..., example="no", description="Has personal loan?")
    contact: str = Field(..., example="cellular", description="Contact communication type")
    month: str = Field(..., example="may", description="Last contact month")
    day_of_week: str = Field(..., example="mon", description="Last contact day of week")
    duration: int = Field(..., example=200, description="Last contact duration in seconds")
    campaign: int = Field(..., example=2, description="Number of contacts during this campaign")
    pdays: int = Field(..., example=999, description="Days since last contact from previous campaign")
    previous: int = Field(..., example=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., example="nonexistent", description="Outcome of previous campaign")
    emp_var_rate: float = Field(..., alias="emp.var.rate", example=-1.8, description="Employment variation rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx", example=92.893, description="Consumer price index")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", example=-46.2, description="Consumer confidence index")
    euribor3m: float = Field(..., example=1.313, description="Euribor 3 month rate")
    nr_employed: float = Field(..., alias="nr.employed", example=5099.1, description="Number of employees")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    prediction: str
    probability_yes: float
    probability_no: float
    risk_level: str


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "API is running",
        "model": "Bank Term Deposit Prediction - RandomForest",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """
    Predict whether a customer will subscribe to a term deposit.
    
    Returns prediction (yes/no), probabilities, and risk level.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check model.pkl file.")

    try:
        # Convert input to DataFrame with correct column names
        input_data = {
            "age": [customer.age],
            "job": [customer.job],
            "marital": [customer.marital],
            "education": [customer.education],
            "default": [customer.default],
            "housing": [customer.housing],
            "loan": [customer.loan],
            "contact": [customer.contact],
            "month": [customer.month],
            "day_of_week": [customer.day_of_week],
            "duration": [customer.duration],
            "campaign": [customer.campaign],
            "pdays": [customer.pdays],
            "previous": [customer.previous],
            "poutcome": [customer.poutcome],
            "emp.var.rate": [customer.emp_var_rate],
            "cons.price.idx": [customer.cons_price_idx],
            "cons.conf.idx": [customer.cons_conf_idx],
            "euribor3m": [customer.euribor3m],
            "nr.employed": [customer.nr_employed],
        }
        
        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]

        pred_label = "yes" if prediction == 1 else "no"
        prob_yes = float(round(probabilities[1], 4))
        prob_no = float(round(probabilities[0], 4))

        # Determine risk level
        if prob_yes >= 0.7:
            risk = "HIGH - Very likely to subscribe"
        elif prob_yes >= 0.4:
            risk = "MEDIUM - Moderate chance of subscribing"
        else:
            risk = "LOW - Unlikely to subscribe"

        return PredictionResponse(
            prediction=pred_label,
            probability_yes=prob_yes,
            probability_no=prob_no,
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# ============================================================
# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
