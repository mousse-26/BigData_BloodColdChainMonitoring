import csv
import os
from datetime import datetime

LOG_FILE = "monitoring/prediction_logs.csv"


def log_prediction(input_data, prediction, confidence):
    """
    Logs each prediction made by the FastAPI service.

    Parameters:
        input_data (dict): Input features received by the API.
        prediction (str): Model prediction ("Safe" or "Unsafe").
        confidence (float): Prediction confidence score.
    """

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "confidence",
                "route",
                "blood_type",
                "product_type",
                "temp_mean",
                "hum_mean"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prediction,
            confidence,
            input_data.get("route"),
            input_data.get("blood_type"),
            input_data.get("product_type"),
            input_data.get("temp_mean"),
            input_data.get("hum_mean")
        ])
