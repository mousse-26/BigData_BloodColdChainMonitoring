from fastapi import FastAPI
from pathlib import Path
import joblib
from pydantic import BaseModel
import pandas as pd
from utils.preprocessing import engineer_features
from monitoring.monitoring import log_prediction

# ============================================================
# Load Trained Model
# ============================================================

MODEL_PATH = Path("models/blood_bag_classifier.pkl")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Blood Bag Safety Prediction API",
    version="1.0.0"
)

ENCODER_PATH = Path("artifacts/label_encoders.pkl")
label_encoders = joblib.load(ENCODER_PATH)

# ============================================================
# Request Schema
# ============================================================


class BloodBagRequest(BaseModel):

    temp_mean: float
    temp_min: float
    temp_max: float
    temp_std: float

    frac_temp_above_6: float
    frac_temp_above_8: float

    hum_mean: float
    hum_std: float

    door_count: int
    light_mean_abs: float
    accel_rms: float
    handling_stress: float

    route: str
    blood_type: str
    product_type: str


@app.get("/")
def home():
    return {
        "message": "Blood Bag Safety Prediction API is running.",
        "model_version": "1.0.0"
    }


@app.post("/predict")
def predict(data: BloodBagRequest):

    # Convert request to dictionary
    input_data = data.model_dump()

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply feature engineering
    input_df = engineer_features(input_df)

    # Encode categorical features
    for column in ["route", "blood_type", "product_type"]:
        input_df[column] = label_encoders[column].transform(input_df[column])

    # Ensure correct feature order
    feature_order = [
        "route",
        "blood_type",
        "product_type",
        "temp_mean",
        "temp_min",
        "temp_max",
        "temp_std",
        "frac_temp_above_6",
        "frac_temp_above_8",
        "hum_mean",
        "hum_std",
        "door_count",
        "light_mean_abs",
        "accel_rms",
        "handling_stress",
        "temp_range",
        "temp_cv",
        "temp_deviation",
        "humidity_per_temp",
        "handling_per_door"
    ]

    input_df = input_df[feature_order]

    # Make prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": "Safe" if prediction == 1 else "Unsafe",
        "prediction_code": int(prediction),
        "probability_safe": round(probabilities[1], 4),
        "probability_unsafe": round(probabilities[0], 4),
        "model_version": "1.0.0"
    }


log_prediction(
    input_data,
    "Safe" if prediction == 1 else "Unsafe",
    round(max(probability), 4)
)
