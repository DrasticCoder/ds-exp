from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
import json
import os

APP_DIR = os.path.dirname(__file__)
ART_DIR = os.path.join(APP_DIR, "artifacts")

PIPE_PATH = os.path.join(ART_DIR, "inference_pipeline.joblib")
COLS_PATH = os.path.join(ART_DIR, "expected_columns.json")

# load pipeline + expected columns on startup
inference_pipeline = joblib.load(PIPE_PATH)
expected = json.load(open(COLS_PATH))
EXPECTED_COLS = expected["expected_input_cols"]

app = FastAPI(title="Disease Outbreak Risk Prediction API", version="1.0")


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    labels: List[str]
    probabilities: Optional[List[float]] = None


LABEL_MAP = {0: "Low Risk", 1: "High Risk"}


@app.get("/health")
def health():
    return {
        "status": "ok", 
        "expected_features": len(EXPECTED_COLS),
        "description": "Disease Outbreak Risk Prediction API",
        "model_features": EXPECTED_COLS
    }


@app.get("/")
def root():
    return {
        "message": "Disease Outbreak Risk Prediction API",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Predict disease outbreak risk",
            "/docs": "API documentation"
        }
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict disease outbreak risk based on input features.
    
    Expected features:
    - Population: Population size
    - Cases_Reported: Number of reported cases
    - Deaths_Reported: Number of deaths
    - Recovered: Number of recovered cases
    - Vaccination_Coverage_Pct: Vaccination coverage percentage
    - Healthcare_Expenditure_PctGDP: Healthcare expenditure as % of GDP
    - Urbanization_Rate_Pct: Urbanization rate percentage
    - Avg_Temperature_C: Average temperature in Celsius
    - Avg_Humidity_Pct: Average humidity percentage
    - case_fatality_rate: Case fatality rate
    - cases_per_100k: Cases per 100,000 population
    - recovery_rate: Recovery rate
    - healthcare_vaccination_score: Healthcare-vaccination interaction score
    - Country: Country name
    - Disease_Name: Disease name
    
    Returns:
    - 0 (Low Risk): Case fatality rate ≤ 1% AND cases per 100k ≤ 100
    - 1 (High Risk): Case fatality rate > 1% OR cases per 100k > 100
    """
    if not req.records:
        raise HTTPException(400, "No records provided.")
    df = pd.DataFrame(req.records)

    # align columns: keep only expected, add missing as NaN
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[EXPECTED_COLS]  # reorder & drop extras

    # run inference
    try:
        proba = None
        if hasattr(inference_pipeline.named_steps["model"], "predict_proba"):
            proba = inference_pipeline.predict_proba(df)[:, 1].tolist()
        preds = inference_pipeline.predict(df).tolist()
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]
    return PredictResponse(
        predictions=[int(p) for p in preds], labels=labels, probabilities=proba
    )
