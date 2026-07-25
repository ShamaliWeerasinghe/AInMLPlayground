# FastAPI documentation: https://fastapi.tiangolo.com/
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Industrial Robot Predictive Maintenance API",
    version="1.0.0",
    description="Predicts whether a robot is likely to fail within 7 days.",
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "ml_models" / "random_forest_failure_predictor.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train model first.")

prediction_model = joblib.load(MODEL_PATH)


class SensorInput(BaseModel):
    timestamp: str = Field(..., example="2026-01-01 00:00:00")
    robot_id: str = Field(..., example="RB-101")
    operating_hours: float = Field(..., example=125.7)
    temp_c: float = Field(..., example=73.2)
    vibration_mm_s: float = Field(..., example=3.4)
    motor_current_a: float = Field(..., example=18.1)
    hydraulic_pressure_bar: float = Field(..., example=109.5)
    ambient_humidity_pct: float = Field(..., example=46.3)
    spindle_rpm: float = Field(..., example=1510.0)
    input_voltage_v: float = Field(..., example=401.2)
    error_code: str = Field(..., example="E00")
    maintenance_due_days: int = Field(..., example=14)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "model_path": str(MODEL_PATH)}


@app.post("/predict")
def predict(payload: SensorInput):
    try:
        input_df = pd.DataFrame([payload.model_dump()])
        prediction = int(prediction_model.predict(input_df)[0])
        failure_probability = float(prediction_model.predict_proba(input_df)[0][1])

        return {
            "prediction": prediction,
            "failure_probability": round(failure_probability, 6),
            "label": "likely_failure_within_7d" if prediction == 1 else "unlikely_failure_within_7d",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
