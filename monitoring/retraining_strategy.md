# Model Retraining Strategy

## Overview

Machine learning models can become less accurate over time as production data changes. To maintain reliable performance, the blood bag safety classification model should be periodically evaluated and retrained when necessary.

The retraining strategy combines scheduled retraining with performance-based triggers.

---

## Retraining Triggers

The model should be retrained if one or more of the following conditions occur:

| Trigger | Action |
|---------|--------|
| Significant data drift detected | Retrain the model using updated data |
| Model accuracy drops below 90% | Retrain and re-evaluate the model |
| New hospitals or routes are introduced | Update label encoders and retrain |
| New product types become available | Retrain the model with updated categories |
| Large volume of newly labelled data collected | Retrain to improve generalization |
| Scheduled interval (every 6 months) | Perform routine retraining |

---

## Retraining Workflow

The retraining process consists of the following steps:

1. Collect newly labelled telemetry data.
2. Validate and clean the new dataset.
3. Apply the same feature engineering pipeline used during training.
4. Retrain the baseline and candidate models.
5. Evaluate both models using the existing evaluation metrics.
6. Promote the best-performing model to production.
7. Save the updated model and preprocessing artifacts.
8. Deploy the new model through the FastAPI inference service.

---

## Model Versioning

Each production model should be assigned a version number (for example, v1.0.0, v1.1.0, or v2.0.0).

Model artifacts, evaluation metrics, and preprocessing objects should be stored together so that previous versions can be restored if required.

---

## Rollback Strategy

If a newly deployed model performs worse than the previous production model, the system should immediately roll back to the last stable version.

Keeping previous model versions ensures reliable deployment and minimizes production risk.