# 🩸 Blood Cold Chain Monitoring System

An end-to-end production-inspired Machine Learning system for predicting whether a blood bag remains **Safe** or **Unsafe** during transportation using IoT telemetry data.

The project demonstrates the complete ML lifecycle—from data processing and feature engineering to model training, deployment with FastAPI, an operator dashboard, and production monitoring.

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

The objective of this project is to build an ML system capable of predicting whether a blood bag is Safe or Unsafe based on telemetry data collected during transport.

---

## 🚀 Project Highlights

- Data validation and preprocessing
- Distributed processing with PySpark
- Feature engineering
- Baseline model (Logistic Regression)
- Candidate model (Random Forest)
- Model evaluation and comparison
- Production model selection
- Model persistence using Joblib
- REST API using FastAPI
- Responsive web dashboard
- Prediction logging
- Diagnostic tooling for model behaviour
- Monitoring and retraining strategy
- Production architecture documentation

---

## 🏗️ System Architecture

![Architecture](architecture/architecture.png)

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
11. Web dashboard
12. Prediction logging
13. Monitoring and retraining

---

## 📂 Project Structure

```
BigData_BloodColdChainMonitoring/
│
├── architecture/
├── artifacts/          # fitted label encoders
├── configs/
├── datasets/
├── frontend/           # operator dashboard (single HTML file)
├── ingestion/
├── models/             # persisted Random Forest
├── monitoring/         # prediction logging
├── notebooks/
├── output/
├── serving/            # FastAPI inference service
├── tests/
├── utils/              # feature engineering shared by training and inference
├── diagnose.py         # model behaviour diagnostic
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Simulated healthcare IoT telemetry collected from blood bag transportation — **288,000 shipments**.

Each observation contains transportation statistics including route, blood type, product type, temperature statistics, humidity statistics, door count, light exposure, acceleration, and handling stress.

**Target variable:** derived from a continuous `health_index`, binarized at 0.05.

| Class | Share |
|---|---|
| Safe | 21.6% |
| Unsafe | 78.4% |

The class imbalance matters: predicting "Unsafe" for every shipment would score 78.4% accuracy. Accuracy alone is therefore not a meaningful metric for this problem.

---

## ⚙️ Feature Engineering

Five engineered features were derived from the raw telemetry:

| Feature | Definition |
|---|---|
| `temp_range` | `temp_max − temp_min` |
| `temp_cv` | `temp_std / temp_mean` |
| `temp_deviation` | deviation from the 4 °C storage midpoint |
| `humidity_per_temp` | `hum_mean / temp_mean` |
| `handling_per_door` | `handling_stress / door_count` |

`temp_range` became the single most important feature in the final model.

---

## 🤖 Machine Learning Models

**Baseline:** Logistic Regression
**Candidate:** Random Forest Classifier (100 trees)

The candidate outperformed the baseline and was promoted to production.

---

## 📈 Model Evaluation

Stratified 80/20 split. Metrics computed on **57,600 held-out rows**.

| Metric | Logistic Regression | Random Forest ✅ |
|---|---|---|
| Accuracy | 0.918 | **0.941** |
| Precision | 0.741 | **0.811** |
| Recall | 0.951 | 0.950 |
| F1 Score | 0.833 | **0.875** |
| ROC-AUC | 0.973 | **0.987** |

Recall is prioritised over raw accuracy: releasing a compromised unit is far costlier than quarantining a good one.

**Top features by importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `temp_range` | 0.239 |
| 2 | `frac_temp_above_6` | 0.190 |
| 3 | `frac_temp_above_8` | 0.120 |
| 4 | `temp_min` | 0.110 |
| 5 | `door_count` | 0.068 |

The model separates the classes along physically coherent lines:

| Feature | Unsafe mean | Safe mean |
|---|---|---|
| `temp_min` | 1.72 | 3.77 |
| `temp_std` | 0.176 | 0.124 |
| `frac_temp_above_8` | 0.058 | 0.001 |
| `door_count` | 0.853 | 0.253 |
| `handling_stress` | 0.993 | 0.680 |

Safe consignments were colder-stable, never heat-excursed, opened less often, and handled more gently — indicating the model learned a genuine signal rather than a dataset artifact.

---

## 💾 Model Persistence

The production model is serialized using Joblib. Saved artifacts include the trained Random Forest, label encoders, and evaluation metrics, allowing the inference service to reuse the trained model without retraining.

> **Note:** `scikit-learn` must match the version used to pickle the model, or joblib raises `InconsistentVersionWarning` and predictions cannot be trusted. The version is pinned in `requirements.txt`.

---

## 🌐 FastAPI Inference Service

**Endpoints**

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | health check and model version |
| `POST` | `/predict` | classify a consignment |
| `GET` | `/docs` | Swagger UI |

**Example response**

```json
{
  "prediction": "Safe",
  "prediction_code": 1,
  "probability_safe": 0.8412,
  "probability_unsafe": 0.1588,
  "model_version": "1.0.0"
}
```

Returning both class probabilities rather than a single confidence figure lets the caller apply its own decision threshold — necessary here, because the cost of the two error types is not symmetric.

---

## 🖥️ Dashboard

A single self-contained HTML file (`frontend/index.html`) providing a form interface to the classifier. No build step; works on desktop and mobile.

- All 15 model inputs, grouped by category
- Category values sourced from the fitted label encoders, so unseen labels cannot reach the model
- Each field displays its training range and flags out-of-range entries
- Comparison of the submitted shipment against the training distribution
- Preset cases drawn from real dataset rows
- Session-level prediction history

---

## 🔍 Engineering Finding: Training–Serving Skew

Building the dashboard surfaced a defect that testing through Swagger alone had not.

**The API request schema implies physical units. The model expects transformed ones.**

Inspecting the training distribution:

| Column | Min in training data | Why it cannot be a physical reading |
|---|---|---|
| `frac_temp_above_6` | −0.408 | a fraction of time cannot be negative |
| `light_mean_abs` | −14.24 | an absolute value cannot be negative |
| `door_count` | −0.531 | a count cannot be negative or fractional |
| `temp_std` | −0.012 | a standard deviation cannot be negative |

Individual rows are also internally inconsistent — one sample carries `temp_mean` 5.22 alongside `temp_max` 3.70 — indicating each column was independently scaled or noised upstream of the published dataset.

**Consequence:** a client posting genuine sensor readings receives an unreliable prediction, because a random forest cannot extrapolate beyond its training range.

**Diagnosis:** `diagnose.py` runs four synthetic shipments of increasing severity through the exact inference path and reports feature ordering, engineered-feature variance, predictions, importances, and tree geometry. Predictions did not move monotonically with input severity, isolating the problem to the input space rather than the model.

**Mitigation:** the dashboard presents these fields as normalized telemetry indices, displays training ranges, flags out-of-range input, and marks affected predictions as unreliable.

**Proper fix:** recover the original transformation, persist it as a fitted artifact alongside the label encoders, and apply it in `serving/app.py`.

---

## ⚠️ Known Limitations

- **The 0.05 threshold is arbitrary.** Chosen for workable class proportions, not from a clinical standard. `health_index` is heavily right-skewed (median 0.0016), so the cut sits in a dense region and samples near it are effectively unseparable. Misclassifications concentrate there.
- **The decision threshold is 0.5**, implicitly weighting a false Safe and a false Unsafe equally. In transfusion medicine they are not equal.
- **`product_type` carries no information.** The dataset contains only `RBC`; measured importance is exactly 0.000.
- **Dataset is synthetic.** Results require revalidation before any real-world claim.
- **Monitoring logs two features**, not the full vector — insufficient for drift detection.

---

## 📋 Monitoring

Every prediction is logged with timestamp, predicted class, confidence, and model version, supporting drift detection, model auditing, and performance monitoring.

---

## 🔄 Retraining Strategy

Retraining may be triggered when prediction accuracy decreases, data drift is detected, new transportation routes are introduced, or sufficient new labeled data becomes available.

---

## 🛠️ Technologies Used

Python · Pandas · NumPy · Scikit-learn · PySpark · FastAPI · Uvicorn · Joblib · Jupyter

---

## ▶️ Running the Project

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API
uvicorn serving.app:app --reload --port 8000

# Dashboard, in a second terminal
cd frontend && python3 -m http.server 5500
```

Swagger UI: http://127.0.0.1:8000/docs
Dashboard: http://localhost:5500

---

## 📌 Future Improvements

- Cost-aware decision threshold with a reported precision–recall trade-off
- Recover and persist the input transformation so the API accepts physical units
- Log the full feature vector and add distribution-drift checks
- Model `health_index` as a regression, letting the operator choose the cut point
- Docker deployment, CI/CD, cloud hosting
- Real-time streaming inference
- MLflow experiment tracking

---

## 👩‍💻 Author

**Khushi Mishra**
M.Sc. Artificial Intelligence & Data Science, BITS Pilani
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
