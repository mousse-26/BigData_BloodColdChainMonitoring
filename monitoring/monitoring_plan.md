# Monitoring Plan

## Overview

After deployment, the machine learning model should be continuously monitored to ensure that it maintains reliable predictive performance and that incoming production data remains similar to the data used during training.

The monitoring strategy focuses on four key areas:

1. Input Data Monitoring
2. Prediction Monitoring
3. Model Performance Monitoring
4. API Health Monitoring

---

## 1. Input Data Monitoring

The following input features should be monitored regularly:

- Temperature Mean
- Temperature Standard Deviation
- Humidity Mean
- Door Count
- Handling Stress
- Route Distribution

Significant changes in the distribution of these features may indicate **data drift**, meaning the production data no longer resembles the training data.

---

## 2. Prediction Monitoring

The following prediction statistics should be tracked:

- Total predictions
- Percentage of Safe predictions
- Percentage of Unsafe predictions
- Average confidence score

Unexpected changes in prediction distribution may indicate model degradation.

---

## 3. Model Performance Monitoring

When true labels become available, the following metrics should be monitored:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

A noticeable decrease in these metrics indicates that the model should be reviewed.

---

## 4. API Health Monitoring

The production API should monitor:

- Response latency
- Failed requests
- Server uptime
- Prediction throughput

These metrics ensure the inference service remains reliable and responsive.