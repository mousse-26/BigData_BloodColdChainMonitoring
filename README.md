# 🩸 Blood Cold Chain Monitoring System

An end-to-end production-inspired Machine Learning system for predicting whether a blood bag remains **Safe** or **Unsafe** during transportation using IoT telemetry data.

The project demonstrates the complete ML lifecycle—from data processing and feature engineering to model training, deployment with FastAPI, and production monitoring.

---

## 📌 Problem Statement

Blood products must be transported within strict environmental conditions to preserve their quality and safety.

During transportation, factors such as:

- Temperature
- Humidity
- Door openings
- Light exposure
- Movement and handling

can affect blood quality.

The objective of this project is to build an ML system capable of predicting whether a blood bag is **Safe** or **Unsafe** based on telemetry data collected during transport.

---

## 🚀 Project Highlights

- Data validation and preprocessing
- Feature engineering
- Baseline model (Logistic Regression)
- Candidate model (Random Forest)
- Model evaluation and comparison
- Production model selection
- Model persistence using Joblib
- REST API using FastAPI
- Swagger UI for testing
- Prediction logging
- Monitoring and retraining strategy
- Production architecture documentation

---

## 🏗️ System Architecture

<img width="522" height="923" alt="Screenshot 2026-08-06 at 16 20 44" src="https://github.com/user-attachments/assets/f71c7ddc-3177-496a-8fa6-83f5d88a25aa" />


The workflow follows a complete production-inspired ML pipeline:

1. Raw telemetry data ingestion
2. Data validation and cleaning
3. Feature engineering
4. Train/Test split
5. Baseline model training
6. Candidate model training
7. Model evaluation
8. Production model selection
9. Model persistence
10. FastAPI inference service
11. Prediction logging
12. Monitoring and retraining

---

## 📂 Project Structure

```text
BigData_BloodColdChainMonitoring/
│
├── architecture/
├── artifacts/
├── configs/
├── datasets/
├── ingestion/
├── models/
├── monitoring/
├── notebooks/
├── output/
├── serving/
├── tests/
├── utils/
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project uses simulated healthcare IoT telemetry collected from blood bag transportation.

Each observation contains transportation statistics including:

- Route
- Blood Type
- Product Type
- Temperature statistics
- Humidity statistics
- Door count
- Light exposure
- Acceleration
- Handling stress

Target Variable:

- Safe
- Unsafe

---

## ⚙️ Feature Engineering

The following engineered features were created:

- Temperature Range
- Temperature Coefficient of Variation
- Temperature Deviation
- Humidity per Temperature
- Handling per Door Opening

These features improve the model's ability to detect unsafe transportation conditions.

---

## 🤖 Machine Learning Models

### Baseline Model

- Logistic Regression

### Candidate Model

- Random Forest Classifier

The candidate model outperformed the baseline and was promoted to production.

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Random Forest achieved superior predictive performance and was selected as the production model.

---

## 💾 Model Persistence

The production model is serialized using Joblib.

Saved artifacts include:

- Trained Random Forest model
- Label Encoders
- Evaluation metrics

These artifacts allow the inference service to reuse the trained model without retraining.

---

## 🌐 FastAPI Inference Service

The production model is deployed using FastAPI.

API Features:

- REST endpoint
- Automatic request validation
- Swagger documentation
- Confidence score
- Model version information

Example Response

```json
{
  "prediction": "Safe",
  "prediction_code": 1,
  "confidence": 0.72,
  "model_version": "1.0.0"
}
```
## API Demo

### Swagger UI
<img width="1050" height="570" alt="Screenshot 2026-08-06 at 16 24 50" src="https://github.com/user-attachments/assets/2cd25f26-86ad-4f4c-9707-c61708f433e1" />

### Sample Prediction
<img width="1044" height="567" alt="Screenshot 2026-08-06 at 16 26 59" src="https://github.com/user-attachments/assets/b22af3e1-5d50-4328-9382-eb32d8e0469d" />


---

## 📋 Monitoring

The system includes a production monitoring component that logs every prediction for future analysis.

Monitoring tracks:

- Prediction history
- Timestamp
- Model version
- Confidence score

These logs can be used for:

- Drift detection
- Model auditing
- Performance monitoring

---

## 🔄 Retraining Strategy

The project also documents a retraining plan.

Retraining may be triggered when:

- Prediction accuracy decreases
- Data drift is detected
- New transportation routes are introduced
- Sufficient new labeled data becomes available

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PySpark
- FastAPI
- Uvicorn
- Joblib
- Jupyter Notebook

---

## ▶️ Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn serving.app:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 📌 Future Improvements

- Docker deployment
- CI/CD pipeline
- Cloud deployment
- Real-time streaming inference
- Automated retraining pipeline
- MLflow experiment tracking

---

## 👩‍💻 Author

**Khushi Mishra**

M.Sc. Artificial Intelligence & Data Science  
BITS Pilani
